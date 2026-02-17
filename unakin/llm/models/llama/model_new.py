import math
from dataclasses import dataclass
import os
import random
import time
from typing import Any, List, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from xformers.ops.fmha import AttentionBias
import xformers.ops as xops

from unakin.llm.models.llama._cache import CacheMetadata, SimpleMetadda, CacheView, BufferCache, FixedSizeMetadata

@dataclass
class ModelArgs:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = -1  # defined later by tokenizer
  multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
  ffn_dim_multiplier: Optional[float] = None
  norm_eps: float = 1e-5
  rope_theta: Optional[float] = 10000.0

CODE_LLAMA_34B_ARGS = ModelArgs(
  dim=8192,
  n_layers=48,
  n_heads=64,
  n_kv_heads=8,
  multiple_of=256,
  ffn_dim_multiplier=1.0,
  norm_eps=1e-5,
  vocab_size=32_000,
  rope_theta=1_000_000,
)

CODE_LLAMA_13B_ARGS = ModelArgs(
  dim=5120,
  n_layers=40,
  n_heads=40,
  multiple_of=256,
  ffn_dim_multiplier=1.0,
  norm_eps=1e-5,
  vocab_size=32_016,
  rope_theta=1_000_000,
)

CODE_LLAMA_7B_ARGS = ModelArgs(
  dim=4096,
  multiple_of=256,
  n_heads=32,
  n_layers=32,
  norm_eps=1e-6,
  vocab_size=32_016,
)

LLAMA_13B_ARGS = ModelArgs(
  dim=5120,
  n_layers=40,
  n_heads=40,
  multiple_of=256,
  norm_eps=1e-5,
  vocab_size=32_000,
)

LLAMA_7B_ARGS = ModelArgs(
  dim=4096,
  multiple_of=256,
  n_heads=32,
  n_layers=32,
  norm_eps=1e-6,
  vocab_size=32_000,
)

TEST_CONFIG = ModelArgs(
  dim=256,
  multiple_of=256,
  n_heads=4,
  n_layers=4,
  norm_eps=1e-6,
  vocab_size=32_000,
)

@dataclass
class ModelWeights:
  ckpt_name: str
  args: ModelArgs

CODE_LLAMA_13B_WEIGHTS = ModelWeights(
  ckpt_name="code-llama-13b-instruct.pth",
  args=CODE_LLAMA_13B_ARGS,
)

CODE_LLAMA_34B_WEIGHTS = ModelWeights(
  ckpt_name="code-llama-34b-instruct.pth",
  args=CODE_LLAMA_34B_ARGS,
)

def polar_precompute_freqs_cis(
  dim: int,
  end: int,
  theta: float = 10000.0
):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                 [: (dim // 2)].float() / dim))
  
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis

def precompute_freqs_cis(
  dim: int,
  end: int,
  theta: float = 10000.0,
  device: Optional[torch.device] = None,
) -> torch.Tensor:
  device = device or torch.device("cpu")
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)
                           [: (dim // 2)].float() / dim))
  t = torch.arange(end, device=device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_sin = torch.sin(freqs)
  freqs_cos = torch.cos(freqs)
  freqs_cis = torch.stack([freqs_sin, freqs_cos], dim=-1)
  return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [1, x.shape[1], 1, x.shape[-1]]
  return freqs_cis.view(shape)

def _rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
  x_real = x[..., 0::2]
  x_imag = x[..., 1::2]

  freq_cos = reshape_for_broadcast(cos, x_real).float()
  freq_sin = reshape_for_broadcast(sin, x_imag).float()

  x_out_real = x_real * freq_cos - x_imag * freq_sin
  x_out_imag = x_real * freq_sin + x_imag * freq_cos
  x_out = torch.stack([x_out_real, x_out_imag], dim=-1).flatten(-2)
  return x_out.type_as(x)

def cos_sin_apply_rotary_emb(
  xq: torch.Tensor,
  xk: torch.Tensor,
  freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  sin, cos = freqs_cis[..., 0], freqs_cis[..., 1]
  return _rotate(xq, sin, cos), _rotate(xk, sin, cos)

def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    output = _rms_norm(x, self.eps).type_as(x)
    return output * self.weight

  def __repr__(self):
    return f"RMSNorm({self.weight.shape[0]})"


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
  bs, slen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:, :, :, None, :]
      .expand(bs, slen, n_kv_heads, n_rep, head_dim)
      .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
  )

class Attention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    self.n_local_heads = args.n_heads
    self.n_local_kv_heads = self.n_kv_heads
    self.n_rep = self.n_local_heads // self.n_local_kv_heads
    self.head_dim = args.dim // args.n_heads

    self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
    self.seq_size = 0

  def forward(
    self,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask: Optional[AttentionBias] = None,
    cache: Optional['CacheView'] = None,
  ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)    

    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xq, xk = cos_sin_apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    if cache is None:
      keys, values = xk, xv
      mask = xops.fmha.LowerTriangularMask()
    elif cache.metadata.prefill:
      keys, values = cache.interleave_kv(xk, xv)
      cache.update(xk, xv)
      keys = keys.view(bsz, -1, self.n_local_kv_heads, self.head_dim)
      values = values.view(bsz, -1, self.n_local_kv_heads, self.head_dim)
      mask = cache.metadata.mask
    else:
      keys, values = cache.interleave_kv(xk, xv)
      cache.update(xk, xv)
      keys = keys.view(bsz, -1, self.n_local_kv_heads, self.head_dim)
      values = values.view(bsz, -1, self.n_local_kv_heads, self.head_dim)      
      mask = cache.metadata.mask

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
    values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

    # print(f"Doing attention with keys: {keys.shape}, values: {values.shape}, query: {xq.shape}")
    attn_output: torch.Tensor = xops.memory_efficient_attention(xq, keys, values, attn_bias=mask)
    attn_output = attn_output.reshape(bsz, seqlen, -1).contiguous()
    attn_output = self.wo(attn_output)
    return attn_output
    
class FeedForward(nn.Module):
  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
      ffn_dim_multiplier: Optional[float],
  ):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
  
class TransformerBlock(nn.Module):
  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.attention = Attention(args)
    self.feed_forward = FeedForward(
      dim=args.dim,
      hidden_dim=4 * args.dim,
      multiple_of=args.multiple_of,
      ffn_dim_multiplier=args.ffn_dim_multiplier,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(
    self,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    cache: Optional[CacheView] = None,
    mask: Optional[AttentionBias] = None,
  ):
    
    h = x + self.attention.forward(
      self.attention_norm(x),
      freqs_cis,
      mask=mask,
      cache=cache,
    )

    out = h + self.feed_forward.forward(
      self.ffn_norm(h)
    )

    return out
  
@dataclass
class RuntimeOptions:
  max_seq_len: Optional[int] = None,
 
DEFAULT_RUNTIME_OPTS = RuntimeOptions(max_seq_len=32_768)

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.dim = params.dim
    self.theta = params.rope_theta
    self.params = params
    self.n_layers = params.n_layers
    self.n_kv_heads = params.n_kv_heads if params.n_kv_heads is not None \
                      else params.n_heads
    self.head_dim = params.dim // params.n_heads

    self.layers = nn.ModuleList(
      [TransformerBlock(i, params) for i in range(params.n_layers)]
    )

    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

  @property
  def device(self):
    return self.tok_embeddings.weight.device

  def _register_freqs_cis(self, runtime_options: RuntimeOptions):
    self.register_buffer(
      "freqs_cis",
      precompute_freqs_cis(
        self.params.dim // self.params.n_heads,
        runtime_options.max_seq_len,
        self.theta,
        self.device,
      ),
      persistent=False,
    )

  def _get_freqs_cis(self, runtime_options: RuntimeOptions):
    if not hasattr(self, "freqs_cis") or self.freqs_cis.shape[-2] < runtime_options.max_seq_len:
      self._register_freqs_cis(runtime_options)
    return self.freqs_cis

  def forward(
    self,
    tokens: torch.Tensor,
    cache: Optional[BufferCache] = None,
    metadata: Optional[SimpleMetadda | CacheMetadata] = None,
    runtime_opts: Optional[RuntimeOptions] = DEFAULT_RUNTIME_OPTS,
  ):
    _bsz, _seqlen = tokens.shape    
    freqs_cis = self._get_freqs_cis(runtime_opts)
    
    if metadata is not None:
      freqs_cis = freqs_cis[metadata.positions]
      mask = metadata.mask
    else:
      freqs_cis = freqs_cis[:_seqlen]
      mask = xops.LowerTriangularMask()
      
    x = self.tok_embeddings(tokens)

    for idx, layer in enumerate(self.layers):
      cache_kv = None if cache is None else cache.get_cache_view(idx, metadata)
      x = layer(x, freqs_cis, cache_kv, mask)

    x = self.norm(x)
    return self.output(x).float()
  
  def generate(
    self,
    tokens: torch.Tensor,
    cache: BufferCache,
    seq_lens: List[int],
  ):
    metadata = cache.get_input_metadata(seqlens=seq_lens)
    output = self(tokens, cache=cache, metadata=metadata)
    cache.update_seqlens(seqlens=metadata.seqlens)
    return output

  @property
  def device(self):
    return self.tok_embeddings.weight.device
    
  @staticmethod
  def from_base_model(
    model_weights: ModelWeights,
    base_path: Optional[str] = None,
    device: Optional[torch.device] = torch.device("cpu")):
    with torch.device("meta"):
      transformer = Transformer(model_weights.args)

    ckpt_path = model_weights.ckpt_name if base_path is None else \
                os.path.join(base_path, model_weights.ckpt_name)
    
    ckpt = torch.load(ckpt_path, map_location=device, mmap=True)
    transformer.load_state_dict(ckpt, strict=True, assign=True)
    return transformer
  
  @staticmethod
  def from_args(args: ModelArgs):
    return Transformer(args)

def model_forward_with_cache(
  model: Transformer,
  tokens: torch.Tensor,
  cache: BufferCache,
  seq_lens: List[int],
):
  metadata = cache.get_input_metadata(seqlens=seq_lens)
  output = model(tokens, cache=cache, metadata=metadata)
  cache.update_seqlens(seqlens=seq_lens)
  return output


if __name__ == "__main__":
  pass
  """
  start_time = time.time()
  base_path = "/home/ted/fsx/llama_base/"
  transformer = Transformer.from_base_model(
    CODE_LLAMA_13B_WEIGHTS,
    base_path=base_path,
  )
  transformer = transformer.cuda().half()
  print(f"Time taken: {time.time() - start_time}")
  """
  
  