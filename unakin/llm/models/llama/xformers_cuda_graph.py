from typing import List
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import (
  AttentionBias,
  BlockDiagonalCausalMask,
)

import torch
import xformers
from torch import Tensor

class Attention(torch.nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def forward(
    self, q: Tensor, k: Tensor, v: Tensor, mask: AttentionBias) -> Tensor:
    return xops.memory_efficient_attention(
      q, k, v, mask
    )

def eager_from_seqlens(fn: Attention, q: Tensor, k: Tensor, v: Tensor, seqlens: List[int]):
  o = fn.forward(q=q, k=k, v=v, mask=BlockDiagonalCausalMask.from_seqlens(seqlens))
  return o

def eager_with_mask(attn: Attention, q: Tensor, k: Tensor, v: Tensor, mask: AttentionBias,):
  o = attn.forward(q=q, k=k, v=v, mask=mask)
  return o

def set_mask_seqlens(mask: AttentionBias, seqlens: List[int]):
  min_seqlen = min(seqlens)
  max_seqlen = max(seqlens)
  
  new_seqstart = torch.zeros_like(mask.q_seqinfo.seqstart)
  len_seqstart = len(mask.q_seqinfo.seqstart)
  
  seqlen_padded = seqlens + [0] * (len_seqstart - len(seqlens) - 1)
  cum_sum_seqlens = torch.cumsum(torch.tensor(seqlen_padded, dtype=torch.long), dim=0)

  new_seqstart[1:len(seqlen_padded)+1] = cum_sum_seqlens
  new_seqstart_py = new_seqstart.tolist()   

  mask.q_seqinfo.seqstart.copy_(new_seqstart)
  mask.q_seqinfo.seqstart_py = new_seqstart_py
  mask.q_seqinfo.seqlens_py = seqlens
  
  mask.q_seqinfo.max_seqlen = 32_768 * 2
  mask.q_seqinfo.min_seqlen = 1

def graph_from_seqlens(attn: Attention, q: Tensor, k: Tensor, v: Tensor, seqlens: List[int]):
  mask = BlockDiagonalCausalMask.from_seqlens([0] * MAX_MASK_LEN)
  mask.q_seqinfo.to(device=torch.device("cuda"))
  mask.k_seqinfo.to(device=torch.device("cuda"))
  set_mask_seqlens(mask, seqlens)

  graph = torch.cuda.CUDAGraph()
  for _ in range(3):
    o = attn.forward(x, x, x, mask)

  with torch.cuda.graph(graph):
    o = attn.forward(x, x, x, mask)

  graph.replay()
  torch.cuda.synchronize()
  return o

def graph_from_subsequent_seqlens(attn: Attention, q: Tensor, k: Tensor, v: Tensor, seqlens: List[int],):
  mask = BlockDiagonalCausalMask.from_seqlens([0] * MAX_MASK_LEN)
  mask.q_seqinfo.to(device=torch.device("cuda"))
  mask.k_seqinfo.to(device=torch.device("cuda"))

  # run some dummy mask
  set_mask_seqlens(mask, [2] * (MAX_MASK_LEN // 2)) 

  graph = torch.cuda.CUDAGraph()
  for _ in range(3):
    o = attn.forward(x, x, x, mask)

  with torch.cuda.graph(graph):
    o = attn.forward(x, x, x, mask)

  # dummy forward
  graph.replay()
  torch.cuda.synchronize()

  set_mask_seqlens(mask, seqlens)
  graph.replay()
  torch.cuda.synchronize()

  return o

if __name__ == "__main__":
  print(torch.__version__)
  print(xformers.__version__)

  MAX_MASK_LEN = 16_384
  HALF_MASK_LEN = MAX_MASK_LEN // 2

  torch.set_default_device("cuda")
  x = torch.randn(1, MAX_MASK_LEN, 8, 32)
  x = x.half()

  attn = Attention()

  # normal seqlens list
  o1 = eager_from_seqlens(fn=attn, q=x, k=x, v=x,
                          seqlens=[HALF_MASK_LEN, HALF_MASK_LEN])

  # seqlens list with 0 padding  
  o2 = eager_from_seqlens(fn=attn, q=x, k=x, v=x,
                          seqlens=[HALF_MASK_LEN, HALF_MASK_LEN] + [0] * (MAX_MASK_LEN - 2))


  # mask with edited seqinfo tensors
  fixed_seqlens = [0] * MAX_MASK_LEN
  mask = BlockDiagonalCausalMask.from_seqlens(fixed_seqlens)
  set_mask_seqlens(mask, [HALF_MASK_LEN, HALF_MASK_LEN])

  o3 = eager_with_mask(attn=attn, q=x, k=x, v=x,
                       mask=mask)
  
  # graph mode with 1 single replay
  o4 = graph_from_seqlens(attn=attn, q=x, k=x, v=x,
                          seqlens=[HALF_MASK_LEN, HALF_MASK_LEN])
  
  # graph mode with where the last graph replay uses the provided seqlens
  o5 = graph_from_subsequent_seqlens(attn=attn, q=x, k=x, v=x,
                                     seqlens=[HALF_MASK_LEN, HALF_MASK_LEN])

  # eager mode embedding
  print(o1[0][-1][0])

  # graph mode with mask edditing
  print(o5[0][-1][0])

  assert torch.allclose(o1, o2, rtol = 1e-8)
  assert torch.allclose(o1, o3, rtol = 1e-8)
  assert torch.allclose(o1, o4, rtol = 1e-8)
  assert torch.allclose(o1, o5, rtol = 1e-8)

  