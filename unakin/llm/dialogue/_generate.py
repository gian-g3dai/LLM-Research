import torch
from typing import List, Optional
import xformers.ops as xops
from unakin.llm.tokenizers import TokenizerBase
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from unakin.llm.distributed.utils import print_rank0
from unakin.llm.models.llama.model_compile import (
  Transformer,
  KVCache,
  ModelArgs,
  get_cache_positions_from_mask,
  get_position_ids_from_mask,
  tqdm_range
)

@torch.jit.script
def sample_top_p(logits: torch.Tensor, p: float, temperature: float = 0.0):
  if temperature:
    logits /= temperature

  logits.squeeze_(0)
  with torch.no_grad():
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token).squeeze(1)
    return next_token

# TODO: check if it works out of the box with FSDP
@torch.no_grad()
def decode_tokens(
  model: Transformer,
  tokens: List[List[int]],
  tokenizer: TokenizerBase,
  cache: Optional[KVCache] = None,
  stop_tokens: Optional[set] = {-1},
  max_gen_len: int = 4096,
  temperature: float = 0.0,
  top_p: float = 0.95,
):
  N_WARMUP = 3  
  _DEVICE, _DTYPE = model.device, model.dtype
  bsz = len(tokens)
  prompt_tokens = tokens.copy()
  max_sample_len = max([len(t) for t in tokens])

  print("Longest prompt length", max_sample_len)

  if cache is None:

    cache = KVCache(
      n_layers=model.n_layers,
      max_bsz=bsz,
      max_seq_len=max_gen_len + max_sample_len,
      n_kv_heads=model.n_kv_heads,
      head_dim=model.dim // model.n_heads,
    ).to(_DEVICE, _DTYPE)

  mask = xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
    q_seqlen=[len(t) for t in tokens],
    kv_seqlen=[len(t) + kv_len.item() for (t, kv_len) in zip(tokens, cache.kv_seqlen)],
    kv_padding=max_gen_len + max_sample_len,
  )

  k_seqinfo = mask.k_seqinfo
  q_seqinfo = mask.q_seqinfo
  
  k_seqinfo.min_seqlen = 1
  k_seqinfo.max_seqlen = int(1e6)

  q_seqinfo.seqstart = q_seqinfo.seqstart.to(_DEVICE)
  k_seqinfo.seqstart = k_seqinfo.seqstart.to(_DEVICE)
  k_seqinfo.seqlen   = k_seqinfo.seqlen.to(_DEVICE)

  assert k_seqinfo.seqstart.device == _DEVICE, f"Expected device: {_DEVICE} got device: {k_seqinfo.seqstart.device}"
  assert q_seqinfo.seqstart.device == _DEVICE, f"Expected device: {_DEVICE} got device: {q_seqinfo.seqstart.device}"
  assert k_seqinfo.seqlen.device   == _DEVICE, f"Expected device: {_DEVICE} got device: {k_seqinfo.seqlen.device}"

  write_tokens: torch.Tensor = torch.full(
    size=(bsz, max_gen_len,), fill_value=int(next(iter(stop_tokens))),
    device=torch.cuda.current_device(), dtype=torch.long
  )

  # write toks to saved tokens
  # for i_bsz, i_toks in enumerate(tokens):
  #   write_tokens[i_bsz, :len(i_toks)].copy_(i_toks)

  # create them into tokens
  tokens = torch.tensor(sum(tokens, []), device="cuda")
  tokens = tokens.to(_DEVICE).flatten().view(1, -1)

  eos_reached = torch.tensor([False] * bsz, device="cuda")
  eos_reached_tmp = torch.tensor([False] * bsz, device="cuda")
  eos_reached_idx = torch.tensor([0] * bsz, device="cuda")
  positions = get_position_ids_from_mask(mask, device=torch.cuda.current_device())
  cache_positions = get_cache_positions_from_mask(mask, device=torch.cuda.current_device())
  
  graph = torch.cuda.CUDAGraph()

  for niter in tqdm_range(max_gen_len):
    if niter < N_WARMUP:
      # graph is not recorded for prefill round
      # or for warmup rounds
      output = model.forward(tokens, positions, mask, cache, cache_positions)
    elif niter == N_WARMUP:
      recording_kwargs = {}
      if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
        recording_kwargs["capture_error_mode"] = "thread_local"
      with torch.cuda.graph(graph, **recording_kwargs):
        output = model.forward(tokens, positions, mask, cache, cache_positions)
      graph.replay()
      torch.cuda.synchronize()
    else:
      graph.replay()

    logits = output[:, q_seqinfo.seqstart[1:] - 1, :]

    assert(logits.shape[:2] == (1, bsz)), f"Expected sampled logits of shape: {(bsz, 1)} got shape: {logits.shape[:2]}"
    next_token = sample_top_p(logits, top_p, temperature)

    # we are always updating the kv_cache size
    # this comes before recomputing query & cache positions
    # due to the fact those are dependendent on how many keys we have currently seen
    k_seqinfo.seqlen.add_(k_seqinfo.seqlen < max_gen_len)
    k_seqinfo.seqlen_py = k_seqinfo.seqlen.tolist()

    # at prefill time, we have to reset attention mask object
    # this also means we have to allocate new tensor pointers
    # for positional ids and for caching overwrite positions
    if niter == 0:
      q_seqinfo.seqstart.copy_(torch.arange(bsz + 1, dtype=torch.int))
      q_seqinfo.min_seqlen = 1
      q_seqinfo.max_seqlen = 1
      q_seqinfo.seqstart_py = q_seqinfo.seqstart.tolist()
      tokens = tokens[:, :bsz]

      cache_positions = get_cache_positions_from_mask(mask).to(tokens.device)
      positions = get_position_ids_from_mask(mask).to(tokens.device)

    # at decoding time, since we are using CUDAGraphs, we want to
    # copy the updated values in the already allocated pointer 
    # so the CUDAGraph can preserve the compiled reference
    else:
      cache_positions.copy_(get_cache_positions_from_mask(mask))
      positions.copy_(get_position_ids_from_mask(mask))

    # we are always copying the next token to the input tensor    
    tokens.copy_(next_token)
    
    # positions will be updated to contain the next input position
    # this is equivalent to the current copy location we have to add
    # in the current write buffer (which is the full generation buffer)
    write_tokens[:, niter].copy_(next_token)

    if stop_tokens != {-1}:
      # use an or mask to check that we have reached end of sequence
      # first time one of next token will be step token, we can safely assume
      # that whatever comes next is not a valid generation

      # NOTE: this is just going to make sure once the longest subsequence is 
      # finished decoding, we will not do any additional computation up to MAX_SEQ_LEN
      for stop_token in stop_tokens:
        eos_reached |= (next_token.view(-1) == stop_token)

      if not torch.equal(eos_reached, eos_reached_tmp):
        eos_reached_idx += (eos_reached_tmp != eos_reached) * (niter + 1)
        eos_reached_tmp.copy_(eos_reached)
      
      # eos_reached |= (k_seqinfo.seqlen.view(-1) == max_gen_len)
      if all(eos_reached):
        write_tokens = write_tokens[:, :niter+5]
        break

  # eos not reached -> 
  eos_reached_idx += (eos_reached == False) * max_gen_len

  out_tokens = tokenizer.decode(
        [p + gen[:eos_idx] for p, gen, eos_idx in zip(prompt_tokens, write_tokens.tolist(), eos_reached_idx)]
      )

  return out_tokens