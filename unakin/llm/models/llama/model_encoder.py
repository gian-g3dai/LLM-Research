import math
import os
from typing import Any, List, Optional, Tuple
import torch.amp
import torch.distributed

import torch
import torch.nn.functional as F
from torch import nn
from unakin.llm.kernels.norms import fused_rms_norm_fn
from unakin.llm.models.llama.model_compile import ModelArgs, ModelWeights
from unakin.llm.models.llama.model_compile import CODE_LLAMA_7B_WEIGHTS

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
  ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xq, xk = cos_sin_apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    keys, values = xk, xv

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
    values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

    # reshapes for torch.compile
    # (bs, n_local_heads, seqlen, head_dim)
    xq, keys, values = list(map(lambda t: t.transpose(1, 2), (xq, keys, values)))

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
      attn_output: torch.Tensor = F.scaled_dot_product_attention(
        query=xq, key=keys, value=values, is_causal=True
      )

    # contiguous for ffn / linear
    attn_output = attn_output.transpose(1, 2).reshape(bsz, seqlen, -1).contiguous()
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
  ):
    
    h = x + self.attention.forward(self.attention_norm(x), freqs_cis)
    out = h + self.feed_forward.forward(self.ffn_norm(h))

    return out

def scale_rope_theta(theta_p, t_new, t_pretrain):
  return theta_p ** (math.log(t_new / (2*math.pi), t_pretrain / (2*math.pi)))

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs, max_seq_len: int = None, n_layers: Optional[int] = None):
    super().__init__()
    self.dim = params.dim
    self.params = params
    self.n_layers = params.n_layers
    self.n_heads = params.n_heads
    self.n_kv_heads = params.n_kv_heads if params.n_kv_heads is not None \
                      else params.n_heads
  
    self.head_dim = params.dim // params.n_heads

    used_layers = n_layers if n_layers is not None \
                  else params.n_layers
    
    self.layers = nn.ModuleList(
      [TransformerBlock(i, params) for i in range(used_layers)]
    )

    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    
    self.norm = RMSNorm(params.dim, eps=params.norm_eps) 
    
    if max_seq_len and params.pretrain_max_len != max_seq_len:
      self.theta = scale_rope_theta(
        params.rope_theta, max_seq_len, params.pretrain_max_len)
    else:
      self.theta = params.rope_theta
    
    max_seq_len = max_seq_len or params.pretrain_max_len
    self._register_freqs_cis(max_seq_len)

  def forward(self, x, pos):
    
    self.freqs_cis = self.freqs_cis.to(x.device)
    freqs_cis = self.freqs_cis[pos]

    x = self.tok_embeddings(x)
    for layer in self.layers:
      x = layer(x, freqs_cis)
    x = self.norm(x)
    return x

  @property
  def device(self):
    return self.tok_embeddings.weight.device
  
  @property
  def dtype(self):
    return self.tok_embeddings.weight.dtype

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

  @staticmethod
  def from_base_model(
    model_weights: ModelWeights,
    base_path: Optional[str] = None,
    device: Optional[torch.device] = torch.device("cpu"),
    n_layers: Optional[int] = None,):
    with torch.device("meta"):
      transformer = Transformer(model_weights.args, model_weights.max_seq_len, n_layers=n_layers)

    ckpt_path = model_weights.ckpt_name if base_path is None else \
                os.path.join(base_path, model_weights.ckpt_name)
    
    ckpt = torch.load(ckpt_path, map_location=device, mmap=True)
    transformer.load_state_dict(ckpt, strict=False, assign=True)
    return transformer
  
if __name__ == "__main__":
  transformer = Transformer.from_base_model(
    model_weights=CODE_LLAMA_7B_WEIGHTS,
    base_path="/mnt/raid0/llamas/llama_base",
  )

  BSZ    = 4
  SEQLEN = 8192

  transformer = transformer.to(dtype=torch.bfloat16, device="cuda")
  transformer = transformer.eval()
  transformer = torch.compile(transformer)

  from tqdm import tqdm
  with torch.no_grad():
    for i in tqdm(list(range(750)), total=750):
      x = torch.randint(0, 32016, (BSZ, SEQLEN))
      pos = torch.arange(0, SEQLEN).unsqueeze(0)
      x = x.to("cuda")
      pos = pos.to("cuda")
      with torch.inference_mode():
        out = transformer(x, pos)
   