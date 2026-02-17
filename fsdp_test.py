from functools import partial
import time
from typing import List
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
  ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
  transformer_auto_wrap_policy
)

from pprint import pprint
import numpy as np

import torch.distributed as dist
import torch

def setup_dist():
  dist.init_process_group(backend='nccl', init_method='env://')

class LoraLinear(torch.nn.Module):
  def __init__(self, in_feats, out_feats, rank) -> None:
    super().__init__()
    self.linear = torch.nn.Linear(in_feats, out_feats)
    self.linear.requires_grad_(False)
    self.rank = rank
    self.A = torch.nn.Parameter(torch.randn(rank, in_feats))
    self.B = torch.nn.Parameter(torch.randn(out_feats, rank))

  def forward(self, x):
    dw = torch.matmul(self.B, self.A)
    w  = self.linear.weight + dw
    y = torch.nn.functional.linear(x, w)
    return y / y.norm()

if __name__ == "__main__":
  SLEEP_TIME = 2

  setup_dist()
  torch.cuda.set_device(dist.get_rank())
  
  wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LoraLinear}
  )

  model = torch.nn.Sequential(*[
    LoraLinear(8192, 8192, 16) for _ in range(16)
  ])
  
  print(torch.cuda.reset_peak_memory_stats())

  model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
    auto_wrap_policy=wrap_policy
  )

  optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

  time.sleep(SLEEP_TIME)
  dist.barrier()

  if dist.get_rank() == 0:
    print(model)
    print(torch.cuda.memory_summary())

  time.sleep(SLEEP_TIME)
  dist.barrier()
  torch.cuda.reset_accumulated_memory_stats()
  start_events: List[torch.cuda.Event] = []
  end_events: List[torch.cuda.Event] = []
  for _ in range(1000):
    x = torch.randn(4, 8192).cuda()
    start_events.append(torch.cuda.Event(enable_timing=True))
    end_events.append(torch.cuda.Event(enable_timing=True))
    torch.cuda.synchronize()
    start_events[-1].record()
    y = model(x)
    torch.abs(y.mean()).backward()
    # collect gradient magnitudes
    norms = []
    
    optim.step()
    end_events[-1].record()
    with torch.no_grad():
      with model.summon_full_params(model, with_grads=True):
        for n, p in model.named_parameters():
          if dist.get_rank() == 0:
            if p.requires_grad == True:
              print(n, p.grad.norm(), p.norm(), p.grad.shape)
              norms.append(p.grad.norm().item())
      if dist.get_rank() == 0:
        print(f"Norms: {np.mean(norms)}")

    dist.barrier()
    optim.zero_grad()

  time.sleep(SLEEP_TIME)
  dist.barrier()
  if dist.get_rank() == 0:
    print(f"Average time: {sum([start.elapsed_time(end) for start, end in zip(start_events, end_events)]) / len(start_events)} ms")
    # print(torch.cuda.memory_summary())