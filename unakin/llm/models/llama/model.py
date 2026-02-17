import functools
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch
import xformers.ops as xops
from torch import nn
from torch.nn import functional as F
from xformers.ops.fmha import AttentionBias

from unakin.llm.models import save_utils
from unakin.llm import BaseLLM


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

  max_batch_size: int = 32
  max_seq_len: int = 2048

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
  max_seq_len=32768,
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
  max_seq_len=32768,
)

CODE_LLAMA_7B_ARGS = ModelArgs(
  dim=4096,
  multiple_of=256,
  n_heads=32,
  n_layers=32,
  norm_eps=1e-6,
  vocab_size=32_016,
  max_seq_len=32768,
)


LLAMA_13B_ARGS = ModelArgs(
  dim=5120,
  n_layers=40,
  n_heads=40,
  multiple_of=256,
  norm_eps=1e-5,
  vocab_size=32_000,
  max_seq_len=32768,
)

LLAMA_7B_ARGS = ModelArgs(
  dim=4096,
  multiple_of=256,
  n_heads=32,
  n_layers=32,
  norm_eps=1e-6,
  vocab_size=32_000,
  max_seq_len=32768,
)

TEST_CONFIG = ModelArgs(
  dim=256,
  multiple_of=256,
  n_heads=4,
  n_layers=4,
  norm_eps=1e-6,
  vocab_size=32_000,
  max_seq_len=2048,
)


class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

  def __repr__(self):
    return f"RMSNorm({self.weight.shape[0]})"

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                 [: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} != {(x.shape[1], x.shape[-1])}"
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


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


class Attention(torch.nn.Module):

  _DEFAULT_BSZ_CACHE = 1
  _DEFAULT_SEQ_LEN_CACHE = 2048

  def __init__(self, args: ModelArgs):
    super().__init__()
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    self.n_local_heads = args.n_heads
    self.n_local_kv_heads = self.n_kv_heads
    self.n_rep = self.n_local_heads // self.n_local_kv_heads
    self.head_dim = args.dim // args.n_heads
    self.params = args

    self.wq = nn.Linear(
        args.dim,
        args.n_heads * self.head_dim,
        bias=False,

    )
    self.wk = nn.Linear(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
    )
    self.wv = nn.Linear(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
    )
    self.wo = nn.Linear(
        args.n_heads * self.head_dim,
        args.dim,
        bias=False,
    )

    self.has_soft_prompt = False

  def register_soft_prompt(self, soft_prompt_size: int):
    self.soft_prompt_size = soft_prompt_size
    self.soft_embedding_gate = nn.Parameter(
      torch.zeros(1, 1, self.n_local_kv_heads, 1)
    )
    self.soft_prompt_embedding = nn.Parameter(torch.randn(
        self.soft_prompt_size, self.params.dim
    ))

    self.has_soft_prompt = True

  def register_cache_kv(self, max_bsz: int = 1, max_seq_len: int = 2048):
    self.cache_k = torch.zeros(
        max_bsz, max_seq_len, self.n_local_kv_heads, self.head_dim
    ).to(self.wq.weight)

    self.cache_v = torch.zeros(
        max_bsz, max_seq_len, self.n_local_kv_heads, self.head_dim
    ).to(self.wq.weight)

  def  _attn_dispatch(self,
                     q: torch.Tensor,
                     k: torch.Tensor,
                     v: torch.Tensor,
                     mask: Optional[torch.Tensor],
                     q_seq_lens: List[int] = None,
                     kv_seq_lens: List[int] = None,
                     decoding: bool = False,
                     should_be_causal: bool = True):
    bsz = q.shape[0]

    if not decoding:
      # this is when we train with xformers (flashattn)
      if bsz == 1:
        assert mask is None, "mask must be None when batch size is 1 during training"
        # this is a full attention path in xformers (using it only for soft prompts)
        if q_seq_lens is not None and kv_seq_lens is not None:
          mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
              q_seq_lens, kv_seq_lens
          )
        # this is training time standard causal mask in xformers
        elif q_seq_lens is not None and kv_seq_lens is None:
          mask = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
            q_seq_lens, q_seq_lens
          )

        elif q_seq_lens is None and kv_seq_lens is None:
          mask = xops.fmha.attn_bias.LowerTriangularMask()

        # this is already in the right shape
        output = xops.memory_efficient_attention(q, k, v, mask)

      else:
        # what might happen here is that we get q_seq_lens, or kv_seq_lens
        # which are needed for block diagonal masking. However block diagonal
        # masking does not work with batch size > 1. So we need to use normal masks
        # realistically, we get bsz != 1 is during inference time.
        # this means we receive a precomputed mask of type Tensor
        if mask is None and should_be_causal:
          mask = xops.fmha.attn_bias.LowerTriangularMask()
        
        output = xops.memory_efficient_attention(q, k, v, mask)

    # inference path
    else:
      # this won't work under any circumstances with xformers
      # several cases here:
      #   1. full-attention (soft-prompts) - doesn't need mask
      #   2. causal-attention (normal-decoding) - needs mask

      if should_be_causal and q.shape[1] != 1:
        return xops.memory_efficient_attention(q, k, v, xops.fmha.attn_bias.LowerTriangularMask())

      return xops.memory_efficient_attention(q, k, v, None)

    return output

  def forward(
      self,
      x: torch.Tensor,
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      q_seq_lens: List[int] = None,
      is_decoding: bool = False,
  ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    if is_decoding:
      if start_pos != 0:
        if not hasattr(self, "cache_k"):
          print(
            f"WARNING: Registering cache of default max size: "
            f"{self._DEFAULT_BSZ_CACHE} x {self._DEFAULT_SEQ_LEN_CACHE}. Please call "
            f"`register_cache_kv(bsz, seqlen)` with desired max sizes to avoid this warning."
          )
          # before ()
          self.register_cache_kv(bsz, seqlen)

      if bsz > self.cache_k.shape[0] or start_pos + seqlen > self.cache_k.shape[1]:
        raise ValueError(
          f"Cache size {self.cache_k.shape} is too small for the current batch size or sequence size.")

      self.cache_k = self.cache_k.to(xq)
      self.cache_v = self.cache_v.to(xq)

      self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
      self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

      keys = self.cache_k[:, :start_pos + seqlen]
      values = self.cache_v[:, :start_pos + seqlen]
    
    else:
      keys, values = xk, xv

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
    # (bs, seqlen, n_local_heads, head_dim)
    values = repeat_kv(values, self.n_rep)

    attention_values = self._attn_dispatch(
      xq, keys, values, mask,
      q_seq_lens=q_seq_lens,
      decoding=is_decoding,
      should_be_causal=True
    )

    if self.has_soft_prompt:
      repeat_skv = len(q_seq_lens) if q_seq_lens is not None else 1

      # repeat after forward pass to reduce ops usage
      sk = self.wk(self.soft_prompt_embedding) \
          .reshape(1, self.soft_prompt_size, self.n_local_kv_heads, self.head_dim) \
          .repeat(bsz, repeat_skv, 1, 1)
      sv = self.wv(self.soft_prompt_embedding) \
          .reshape(1, self.soft_prompt_size, self.n_local_kv_heads, self.head_dim) \
          .repeat(bsz, repeat_skv, 1, 1)

      # k/v is constant, so we can just repeat it
      kv_seq_lens = [self.soft_prompt_size for _ in range(len(q_seq_lens))]

      soft_attention_values = self._attn_dispatch(
        xq, sk, sv, None,
        q_seq_lens=q_seq_lens, kv_seq_lens=kv_seq_lens,
        decoding=(start_pos != 0), should_be_causal=False
      )

      # gate the scores
      soft_attention_values = torch.tanh(self.soft_embedding_gate) * soft_attention_values
      attention_values = attention_values + soft_attention_values

    attention_values = attention_values.contiguous().view(bsz, seqlen, -1)
    output = self.wo(attention_values)

    return output


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

    self.w1 = nn.Linear(dim, hidden_dim, bias=False,)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False,)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False,)

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
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      q_seq_lens: List[int],
      is_decoding: bool = False,
  ):
    
    h = x + self.attention.forward(
        self.attention_norm(x), start_pos, freqs_cis, mask,
        q_seq_lens, is_decoding
    )
    out = h + self.feed_forward.forward(self.ffn_norm(h))

    return out

class Transformer(BaseLLM):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    # make sure to move the desired device
    self.tok_embeddings = nn.Embedding(
        params.vocab_size, params.dim,
    )

    self.layers: List[TransformerBlock] = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(layer_id, params))

    self.norm = RMSNorm(params.dim, eps=params.norm_eps)

    self.output = nn.Linear(
        params.dim, params.vocab_size, bias=False
    )

    self.use_xops = False

  def enable_xops(self, enable: bool = True):
    self.use_xops = enable

  def post_init_meta_buffers(self):
    self.freqs_cis = precompute_freqs_cis(
      self.params.dim // self.params.n_heads,
      self.params.max_seq_len * 2,
      theta=self.params.rope_theta,
    ).to(self.tok_embeddings.weight.device)

    # NOTE: The max_seq_len


  def forward(
          self,
          tokens: torch.Tensor,
          start_pos: int,
          positional_ids: Optional[torch.Tensor] = None,
          q_seq_lens: List[int] = None,
          mask: Optional[torch.Tensor | AttentionBias] = None,
          is_decoding: bool = False,):

    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    
    # small check to make sure we're not going through an
    # unintialized buffer
    if not hasattr(self, "freqs_cis"):
      self.post_init_meta_buffers()

    self.freqs_cis = self.freqs_cis.to(h.device)

    if positional_ids is not None:
      freqs_cis = self.freqs_cis[positional_ids]
      freqs_cis = freqs_cis.squeeze(0)
    else:
      freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

    # symbolizes that we're not collating a batch
    if q_seq_lens is not None and len(q_seq_lens) == 1:
      if is_decoding:
        q_seq_lens = None

    if mask is None and is_decoding:
      # symbolizes that we have diferent len(q) and len(kv)
      if seqlen > 1:
        mask = torch.full(
            (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
        )
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

    tensors = []
    import pickle
    for idx, layer in enumerate(self.layers): 
      h = layer.forward(
        x=h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask,
        q_seq_lens=q_seq_lens, is_decoding=is_decoding,
      )
      tensors.append(h.detach().cpu().numpy())

    with open("old.pkl", "wb") as f:
      pickle.dump(tensors, f)

    h = self.norm(h)
    output = self.output(h).float()

    return output

  def register_cache_kv(self, *args, max_bsz: int = 1, max_seq_len: int = 2048, **kwargs):
    print(f"INFO: Registering cache of size {max_bsz} x {max_seq_len}")
    for layer in self.layers:
      layer.attention.register_cache_kv(max_bsz, max_seq_len)

  def register_soft_prompt(self, *args, soft_prompt_size: int, n_layers: int, **kwargs):
    for idx, layer in enumerate(self.layers[::-1][:n_layers]):

      layer: TransformerBlock
      layer.attention.register_soft_prompt(soft_prompt_size)

  @classmethod
  def from_pretrained(cls, path: str, args: Optional[ModelArgs] = None):
    import safetensors.torch
    
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)

    print(path)
    if args is not None:
      model_constructor = functools.partial(cls, args)
      if path.endswith("st"):
        state_dict = safetensors.torch.load_file(path, device="cpu")
      else:
        state_dict = torch.load(path, map_location="cpu")
    else:
      ckpt = torch.load(path, map_location="cpu")
      model_args = ckpt["args"]
      if not isinstance(model_args, ModelArgs):
        raise ValueError(
            f"Expected ModelArgs, got {type(model_args)} instead."
        )
      model_constructor = functools.partial(cls, model_args)
      state_dict = ckpt["model_state_dict"]

    model, _ = save_utils.load_virtual_checkpoint(model_constructor, state_dict,
                                                  ignore_keys=["freqs_cis"],
                                                  load_state=True)
    
    # TODO: Fix this hack. It is what it is
    model.post_init_meta_buffers()
    # model.register_cache_kv(max_bsz=1, max_seq_len=16384)
    torch.set_default_dtype(prev_dtype)
    return model

  #### TODO: modify config 
  @classmethod
  def from_json_file(cls, path_or_json: str | Dict[str, Any]):
    if isinstance(path_or_json, str):
      with open(path_or_json, "r") as f:
        config = json.load(f)
    else:
      config = path_or_json

    # extract model arguments
    model_args = ModelArgs(**config['model_args']) if \
        "model_args" in config else None

    if "ckpt" in config:
      return cls.from_pretrained(config['ckpt']['model_path'], model_args)
    elif "base" in config:

      # get home dir
      # home = os.path.expanduser("~")
      fstack_home = "/mnt/raid0/llamas/" # "/home/ted/fsx/"
      base = os.path.join(fstack_home, "llama_base")
      
      BASE_ARG_MAP = {
        "llama-13b-chat": LLAMA_13B_ARGS,
        "llama-13b": LLAMA_13B_ARGS,
        "code-llama-34b-instruct": CODE_LLAMA_34B_ARGS,
        "code-llama-13b-instruct": CODE_LLAMA_13B_ARGS,
        "code-llama-7b-instruct":  CODE_LLAMA_7B_ARGS,
        "code-llama-34b": CODE_LLAMA_34B_ARGS,
        "code-llama-13b": CODE_LLAMA_13B_ARGS,
        "code-llama-7b": CODE_LLAMA_7B_ARGS,
        "test": TEST_CONFIG,
      }

      BASE_LOC_MAP = {k: os.path.join(base, f"{k}.pth")
                      for k in BASE_ARG_MAP.keys()}
      
      for k, v in BASE_LOC_MAP.items():
        if not os.path.exists(v):
          BASE_LOC_MAP[k] = v.replace(".st", ".pth")
      
      model_name = config['base']
      model_base = BASE_LOC_MAP[config['base']]
      print(model_base, model_name)
      return cls.from_pretrained(model_base, BASE_ARG_MAP[model_name])
    else:
      return cls(model_args)


if __name__ == "__main__":
  # time the load
  import time
  start = time.time()
  model = Transformer(CODE_LLAMA_7B_ARGS)
  print(f"Loaded model in {time.time() - start:.2f}s")
  model.register_cache_kv(max_bsz=1, max_seq_len=16384)
  model = torch.compile(model)
  model = model.cuda()

  random_decode_string = ["Hello there my beautiful soup"]
  
  from unakin.llm.dialogue.base import DialogueModel
  from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

  tokenizer = LlamaTokenizer(model_path="/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  model = DialogueModel(model, tokenizer)
  model.chat_completion(
    [[{"role": "user", "content": random_decode_string[0]}]]
  )

  