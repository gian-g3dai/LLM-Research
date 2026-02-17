import math
from dataclasses import dataclass
import os
import random
import time
from typing import Any, List, Optional, Tuple
import numpy as np
import torch.amp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn

from xformers.ops.fmha import AttentionBias
import xformers.ops as xops
import xformers.ops as xops

# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

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
):
  q_seqinfo = mask.q_seqinfo
  q_seqlen  = q_seqinfo.seqstart[1:] - q_seqinfo.seqstart[:-1]
  k_seqlen  = mask.k_seqinfo.seqlen
  kv_start_write_pos = k_seqlen - q_seqlen
  
  positions = [
    torch.arange(start, start+slen, device=mask.q_seqinfo.seqstart.device)
    for start, slen in zip(kv_start_write_pos, q_seqlen)
  ]
  
  return torch.cat(positions)

def get_cache_positions_from_mask(
  mask: xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
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
    max_bsz: int,
    kv_seq_info: KVSeqInfo) -> None:
    super().__init__()
    
    # trailing batch dimension
    cache_k = torch.zeros(size=(1, max_bsz * max_seq_len, n_kv_heads, head_dim))
    cache_v = torch.zeros(size=(1, max_bsz * max_seq_len, n_kv_heads, head_dim))

    self.register_buffer("cache_k", cache_k)
    self.register_buffer("cache_v", cache_v)
    self.kv_seq_info = kv_seq_info
  
  def update_kv(self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor) -> None:
    self.cache_k[:, positions].copy_(k)
    self.cache_v[:, positions].copy_(v)
  
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
    self.kv_seq_info = KVSeqInfo(start=0, stop=0)

    self.caches = nn.ModuleList(
      [
        KVCacheLayer(
          n_kv_heads=n_kv_heads,
          head_dim=head_dim,
          max_seq_len=max_seq_len,
          max_bsz=max_bsz,
          kv_seq_info=self.kv_seq_info
        )
        for _ in range(n_layers)
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

  def inc_kv_end(self, kv_end: int) -> None:
    self.kv_seq_info.stop.add_(kv_end)
  
  def inc_kv_start(self, kv_start: int) -> None:
    self.kv_seq_info.start.add_(kv_start)

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
  x = x.float() # float32 cast
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
    self._register_freqs_cis(128 * 1024)

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
    positions: torch.Tensor,
    mask: Optional[xops.fmha.AttentionBias] = None,
    cache: Optional[KVCache] = None,
    cache_pos: Optional[torch.Tensor] = None,
  ):
    _bsz, _seqlen = tokens.shape    
    freqs_cis = self.freqs_cis[positions]
      
    x = self.tok_embeddings(tokens)

    for idx, layer in enumerate(self.layers):
      cache_kv = cache[idx] if cache is not None else None
      x = layer(x, freqs_cis, mask, cache_kv, cache_pos)

    x = self.norm(x)
    return self.output(x)

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
    transformer.load_state_dict(ckpt, strict=False, assign=True)
    return transformer
  
  @staticmethod
  def from_args(args: ModelArgs):
    return Transformer(args)

@torch.no_grad()
def decode_tokens(
  model: Transformer,
  tokens: torch.Tensor,
  positions: torch.Tensor,
  cache: Optional[KVCache] = None,
):
  
  N_WARMUP = 3
  bsz = tokens.shape[0]
  mask = xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
    q_seqlen=[1024] * bsz,
    kv_seqlen=[1024] * bsz,
    kv_padding=4096,    
  )

  k_seqinfo = mask.k_seqinfo
  q_seqinfo = mask.q_seqinfo

  q_seqinfo.seqstart.to(model.device)
  k_seqinfo.seqstart.to(model.device)
  k_seqinfo.seqlen.to(model.device)

  cache = KVCache(
    n_layers=model.params.n_layers,
    max_bsz=tokens.shape[0],
    max_seq_len=4_096,
    n_kv_heads=model.n_kv_heads,
    head_dim=model.head_dim,
  ).to(model.device, model.tok_embeddings.weight.dtype)

  tokens = tokens.to(model.device).flatten().view(1, -1)
  positions = get_position_ids_from_mask(mask)
  cache_positions = get_cache_positions_from_mask(mask)
  
  graph = torch.cuda.CUDAGraph()
  #model = torch.compile(model)

  #for _ in range(3):
  for niter in tqdm(range(2048), total=2048):
    if niter < N_WARMUP:
      # graph is not recorded for prefill round
      # or for warmup rounds
      output = model(tokens, positions, mask, cache, cache_positions)
    elif niter == N_WARMUP:
      recording_kwargs = {}
      if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
        recording_kwargs["capture_error_mode"] = "thread_local"
      with torch.cuda.graph(graph, **recording_kwargs):
        output = model(tokens, positions, mask, cache, cache_positions)
      graph.replay()
      torch.cuda.synchronize()
    else:
      # output = model(tokens, positions, mask, cache, cache_positions)
      graph.replay()

    logits = output[:, q_seqinfo.seqstart[1:] - 1, :]
    next_token = torch.argmax(logits, dim=-1)
    next_token = next_token.reshape(1, bsz)

    # we are always updating the kv_cache size
    k_seqinfo.seqlen.add_(k_seqinfo.seqlen < 2048)

    # at prefill time, we have to reset attention mask object
    # this also means we have to allocate new tensor pointers
    # for positional ids and for caching overwrite positions
    if niter == 0:
      q_seqinfo.seqstart.copy_(torch.arange(bsz + 1, dtype=torch.int))
      q_seqinfo.min_seqlen = 1
      q_seqinfo.seqstart_py = q_seqinfo.seqstart.tolist()
      tokens = tokens[:, :bsz]

      cache_positions = get_cache_positions_from_mask(mask)
      positions = get_position_ids_from_mask(mask)
    
    # at decoding time, since we are using CUDAGraphs, we want to
    # copy the updated values in the already allocated pointer 
    # so the CUDAGraph can preserve the compiled reference
    else:
      cache_positions.copy_(get_cache_positions_from_mask(mask))
      positions.copy_(get_position_ids_from_mask(mask))

    # we are always copying the next token to the input tensor    
      tokens.copy_(next_token)

if __name__ == "__main__": 
  start_time = time.time()
  base_path = "/mnt/raid0/llamas/llama_base"
  transformer = Transformer.from_base_model(
    CODE_LLAMA_13B_WEIGHTS,
    base_path=base_path,
  )
  transformer = transformer.cuda().half()
  print(f"Time taken: {time.time() - start_time}")

  indices = torch.randint(0, 32000, (1, 4096 * 4), device="cuda")
  positions = torch.cat([torch.arange(0, 1024, device="cuda") for _ in range(16)])
  cache = KVCache(
    n_layers=transformer.params.n_layers,
    max_bsz=4,
    max_seq_len=2048,
    n_kv_heads=transformer.n_kv_heads,
    head_dim=transformer.head_dim,
  )

  tokens = torch.randint(0, 32000, (8, 1024), device="cuda")
  decode_tokens(transformer, tokens, positions)