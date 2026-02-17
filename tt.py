import time
from typing import List
import torch
import xformers.ops as xops
import matplotlib.pyplot as plt
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True
torch.set_default_device("cuda")

mask = xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
  q_seqlen=[2, 3, 4],
  kv_padding=12,
  kv_seqlen=[6, 6, 6],
)


def write_from_mask(mask, cache, value):
  q_seqinfo = mask.q_seqinfo
  k_seqinfo = mask.k_seqinfo
  q_seqlen  = q_seqinfo.seqstart[1:] - q_seqinfo.seqstart[:-1]
  start, stop = q_seqinfo.seqstart[0], q_seqinfo.seqstart[-1]
  kv_start = k_seqinfo.seqstart[:-1]
  kv_stop  = kv_start + k_seqinfo.seqlen

  offset = 0
  for i in range(kv_start.shape[0]):
    start = kv_start[i]
    stop = kv_stop[i]
    slen = q_seqlen[i]
    print(f"start: {start}, stop: {stop}, slen: {slen}")
    cache[:, stop:stop+slen].copy_(value[:, offset:offset+slen])
    offset += slen  


compiled = torch.compile(write_from_mask, fullgraph=False, dynamic=True)  

times = []
for i in range(10):  
  cache = torch.zeros(1, 2048 * 16, 1, 8)
  value = torch.ones(1, 16, 1, 8)
  mask = xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
    q_seqlen=[1] * 16,
    kv_padding=2048,
    kv_seqlen=[1024] * 16,
  )
  start = time.time()
  write_from_mask(mask, cache, value)
  times.append(time.time() - start)
  print(cache.sum())
