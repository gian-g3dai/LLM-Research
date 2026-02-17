import math
from dataclasses import dataclass
import os
import pickle
import random
import time
from typing import Any, List, Optional, Tuple
import numpy as np
import torch.amp
import torch.distributed
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn

from xformers.ops.fmha import AttentionBias
from xformers.ops.swiglu_op import SwiGLU
import xformers.ops as xops
import xformers.ops as xops


from unakin.llm.distributed.utils import print_rank0
from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer
from unakin.llm.kernels.norms import fused_rms_norm_fn
from unakin.llm.kernels.fused_proj import FusedProjectionPlusCrossEntropyLoss, FusedCrossEntropyOutput, FusedLossArgs


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

def write_to_kv_cache_from_mask(
  mask: xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
  cache_k: torch.Tensor,
  cache_v: torch.Tensor,
  new_k: torch.Tensor,
  new_v: torch.Tensor,
):
  q_seqinfo = mask.q_seqinfo
  k_seqinfo = mask.k_seqinfo
  q_seqlen  = q_seqinfo.seqstart[1:] - q_seqinfo.seqstart[:-1]
  k_seqlen  = k_seqinfo.seqlen
  kv_start_write_pos = k_seqinfo.seqstart[:-1] + k_seqlen - q_seqlen  

  positions = [
    torch.arange(
      kv_start_write_pos[i],
      kv_start_write_pos[i]+q_seqlen[i],
      device=q_seqinfo.seqstart.device)
    for i in range(kv_start_write_pos.shape[0])
  ]
  return torch.cat(positions)

def get_position_ids_from_mask(
  mask: xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
  device: torch.device = torch.device("cuda")
):
  q_seqinfo = mask.q_seqinfo
  q_seqlen  = q_seqinfo.seqstart[1:] - q_seqinfo.seqstart[:-1]
  k_seqlen  = mask.k_seqinfo.seqlen
  kv_start_write_pos = k_seqlen - q_seqlen
  
  positions = [
    torch.arange(start, start+slen, device=device)
    for start, slen in zip(kv_start_write_pos, q_seqlen)
  ]
  
  return torch.cat(positions)

def tqdm_range(*args):
  if torch.distributed.get_rank() != 0:
    return range(*args)
  
  return tqdm(range(*args), total=len(range(*args)))

def get_cache_positions_from_mask(
  mask: xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
  device: torch.device = torch.device("cuda")
):
  q_seqinfo = mask.q_seqinfo
  k_seqinfo = mask.k_seqinfo
  q_seqlen  = q_seqinfo.seqstart[1:] - q_seqinfo.seqstart[:-1]
  k_seqlen  = k_seqinfo.seqlen
  kv_start_write_pos = k_seqinfo.seqstart[:-1] + k_seqlen - q_seqlen  

  positions = [
    torch.arange(
      kv_start_write_pos[i],
      kv_start_write_pos[i]+q_seqlen[i],
      device=device)
    for i in range(kv_start_write_pos.shape[0])
  ]
  return torch.cat(positions)

class KVSeqInfo(nn.Module):
  def __init__(self, start: int, stop: int) -> None:
    super().__init__()
    self.register_buffer('start', torch.tensor(start, dtype=torch.long), persistent=False)
    self.register_buffer('stop', torch.tensor(stop, dtype=torch.long), persistent=False)

class KVCacheLayer(nn.Module):
  def __init__(
    self,
    n_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    max_bsz: int) -> None:
    super().__init__()
    
    # trailing batch dimension
    cache_k = torch.zeros(size=(1, max_bsz * max_seq_len, n_kv_heads, head_dim))
    cache_v = torch.zeros(size=(1, max_bsz * max_seq_len, n_kv_heads, head_dim))

    self.register_buffer("cache_k", cache_k)
    self.register_buffer("cache_v", cache_v)

  def update_kv(self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor) -> None:
    self.cache_k.index_copy_(1, positions, k)
    self.cache_v.index_copy_(1, positions, v)
  
  @property
  def device(self) -> torch.device:
    return self.cache_k.device

class KVCache(nn.Module):
  def __init__(
    self,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    max_bsz: int
  ) -> None:
    
    super().__init__()
    self.n_layers = n_layers
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.max_seq_len = max_seq_len
    self.max_bsz = max_bsz
    self.kv_seqlen = torch.zeros(max_bsz)

    self.caches = nn.ModuleList(
      [
        KVCacheLayer(
          n_kv_heads=n_kv_heads,
          head_dim=head_dim,
          max_seq_len=max_seq_len,
          max_bsz=max_bsz,
        )
        for idx in range(n_layers)
      ]
    )

  def __getitem__(self, idx: int) -> KVCacheLayer:
    return self.caches[idx]

  def reset(self) -> None:
    self.kv_seq_info.start.zero_()
    self.kv_seq_info.stop.zero_()
    for cache in self.caches:
      cache.cache_k.zero_()
      cache.cache_v.zero_()

  @property
  def kv_start(self) -> torch.Tensor:
    return self.kv_seq_info.start
  
  @property
  def kv_stop(self) -> torch.Tensor:
    return self.kv_seq_info.stop

  def inc_kv_seqlen(self, kv_seqlen: torch.Tensor) -> None:
    self.kv_seqlen.add_(kv_seqlen)

  @property
  def device(self) -> torch.device:
    return self.caches[0].device

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
  pretrain_max_len: int = None

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
  pretrain_max_len=16_384,
  rope_theta=1_000_000,
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

LLAMA3_70B_ARGS = ModelArgs(
  dim=8192,
  ffn_dim_multiplier=1.3,
  multiple_of=4096,
  n_heads=64,
  n_kv_heads=8,
  n_layers=80,
  norm_eps=1e-05,
  vocab_size=128_256,
  rope_theta=500000.0
)

LLAMA3_8B_ARGS = ModelArgs(
  dim=4096,
  ffn_dim_multiplier=1.3,
  multiple_of=1024,
  n_heads=32,
  n_kv_heads=8,
  n_layers=32,
  norm_eps=1e-05,
  vocab_size=128_256,
  rope_theta=500000.0,
  pretrain_max_len=8192,
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
  max_seq_len: int = None

  def load(self, base_path: str):
    return Transformer.from_base_model(self, base_path=base_path)

LLAMA3_70B_WEIGHTS = ModelWeights(
  ckpt_name="llama3-70b-instruct.pth",
  args=LLAMA3_70B_ARGS
)

LLAMA3_8B_CKPT_WEIGHTS = ModelWeights(
  ckpt_name="llama3-8b-ckpt.pth",
  args=LLAMA3_8B_ARGS
)

LLAMA3_8B_WEIGHTS = ModelWeights(
  ckpt_name="llama3-8b-instruct.pth",
  args=LLAMA3_8B_ARGS
)

CODE_LLAMA_13B_WEIGHTS = ModelWeights(
  ckpt_name="code-llama-13b-instruct.pth",
  args=CODE_LLAMA_13B_ARGS,
)

CODE_LLAMA_7B_WEIGHTS = ModelWeights(
  ckpt_name="code-llama-7b-instruct.pth",
  args=CODE_LLAMA_7B_ARGS,
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
  freqs_cis = freqs_cis.squeeze()
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [1, x.shape[1], 1, x.shape[-1]]
  return freqs_cis.view(shape)

# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#   ndim = x.ndim
#   freqs_cis = freqs_cis.squeeze() if freqs_cis.shape[0] != 1 else freqs_cis

#   assert 0 <= 1 < ndim
#   # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#   shape = [1, x.shape[1], 1, x.shape[-1]]
#   return freqs_cis.view(shape)

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

# TODO, do this in Triton
def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  x = x.float() # float32 cast
  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    return fused_rms_norm_fn(x, self.weight, self.eps)

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
    self.dim = args.dim

    self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

  def forward(
    self,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask: Optional[AttentionBias] = None,
    cache: Optional[KVCacheLayer] = None,
    write_cache_pos: Optional[torch.Tensor] = None,
  ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xq, xk = cos_sin_apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # 1 write + 1 read
    if cache is not None:
      cache.update_kv(xk, xv, write_cache_pos)
      keys, values = cache.cache_k, cache.cache_v
    else:
      keys, values = xk, xv

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
    out = self.w2(F.silu(self.w1(x)) * self.w3(x))
    return out
  
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
    mask: Optional[AttentionBias] = None,
    cache: Optional[KVCacheLayer] = None,
    write_cache_pos: Optional[torch.Tensor] = None,    
  ):
    
    h = x + self.attention.forward(
      self.attention_norm(x),
      freqs_cis,
      mask=mask,
      cache=cache,
      write_cache_pos=write_cache_pos,
    )

    out = h + self.feed_forward.forward(
      self.ffn_norm(h)
    )

    return out

def scale_rope_theta(theta_p, t_new, t_pretrain):
  return theta_p ** (math.log(t_new / (2*math.pi), t_pretrain / (2*math.pi)))

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs, max_seq_len: int = None):
    super().__init__()
    self.dim = params.dim
    self.params = params
    self.n_layers = params.n_layers
    self.n_heads = params.n_heads
    self.n_kv_heads = params.n_kv_heads if params.n_kv_heads is not None \
                      else params.n_heads
  
    self.head_dim = params.dim // params.n_heads

    self.layers = nn.ModuleList(
      [TransformerBlock(i, params) for i in range(params.n_layers)]
    )

    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
    if max_seq_len and params.pretrain_max_len != max_seq_len:
      self.theta = scale_rope_theta(params.rope_theta, max_seq_len, params.pretrain_max_len)
    else:
      self.theta = params.rope_theta
    self._register_freqs_cis(max_seq_len)

  @property
  def device(self):
    return self.tok_embeddings.weight.device

  def _register_freqs_cis(self, max_seq_len: int):
    self.register_buffer(
      "freqs_cis",
      precompute_freqs_cis(
        self.params.dim // self.params.n_heads,
        max_seq_len,
        self.theta,
        device="cpu",
      ),
      persistent=False,
    )

  def forward(
    self,
    tokens: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    mask: Optional[xops.fmha.AttentionBias] = None,
    cache: Optional[KVCache] = None,
    cache_pos: Optional[torch.Tensor] = None,
    **kwargs,
  ):
    
    if isinstance(self.output, FusedProjectionPlusCrossEntropyLoss):
      assert "loss_args" in kwargs, "FusedProjectionPlusCrossEntropyLoss requires loss_args" 
 
    _bsz, _seqlen = tokens.shape    
    self.freqs_cis = self.freqs_cis.to(tokens.device)
    freqs_cis = self.freqs_cis[positions]
    
    x = self.tok_embeddings(tokens)

    for idx, layer in enumerate(self.layers):
      cache_kv = cache[idx] if cache is not None else None
      x = layer(x, freqs_cis, mask, cache_kv, cache_pos)

    x = self.norm(x)

    # fused path
    if isinstance(self.output, FusedProjectionPlusCrossEntropyLoss):
      return self.output(x, kwargs.pop("loss_args"))

    # regular path    
    return self.output(x)


  @property
  def device(self):
    return self.tok_embeddings.weight.device
  
  @property
  def dtype(self):
    return self.tok_embeddings.weight.dtype

  def apply_fused_loss(self):
    self.output = FusedProjectionPlusCrossEntropyLoss.from_linear_projection(self.output)
    return self
  
  def unapply_fused_loss(self):
    output_layer = torch.nn.Linear(in_features=self.output.proj_weight.shape[1],
                                   out_features=self.output.proj_weight.shape[0], device=self.output.proj_weight.device,
                                   bias=False)
    output_layer.weight.data.copy_(self.output.proj_weight.data)
    self.output = output_layer
    return self

  @staticmethod
  def from_base_model(
    model_weights: ModelWeights,
    base_path: Optional[str] = None,
    device: Optional[torch.device] = torch.device("cpu")):
    with torch.device("meta"):
      transformer = Transformer(model_weights.args, model_weights.max_seq_len)

    ckpt_path = model_weights.ckpt_name if base_path is None else \
                os.path.join(base_path, model_weights.ckpt_name)
    
    ckpt = torch.load(ckpt_path, map_location=device, mmap=True)
    transformer.load_state_dict(ckpt, strict=False, assign=True)
    return transformer
  
  @staticmethod
  def from_args(args: ModelArgs):
    return Transformer(args)

if __name__ == "__main__": 
  start_time = time.time()
  base_path = "/mnt/raid0/llamas/llama_base"
  transformer = Transformer.from_base_model(
    CODE_LLAMA_7B_WEIGHTS,
    base_path=base_path,
  )