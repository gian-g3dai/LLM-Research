from functools import partial
import math
import torch

import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
  CPUOffload,
  BackwardPrefetch,
  ShardingStrategy,
  MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
  CustomPolicy,
  transformer_auto_wrap_policy,
  lambda_auto_wrap_policy,
  _or_policy
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper)

class CustomLinear(torch.nn.Module):
  def __init__(self, in_features, out_features, bias=True):
    super(CustomLinear, self).__init__()
    self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

  def forward(self, x):
    return self.linear(x)
  

class StrangeModule(torch.nn.Module):
  def __init__(self):
    super(StrangeModule, self).__init__()
    self.features = torch.nn.Sequential(
      CustomLinear(4096, 4096),
      CustomLinear(4096, 4096),
    )
    self.classifier = torch.nn.Linear(4096, 32)

  def forward(self, x, as_features: bool = True, as_logits: bool = False):
    if as_features:
      return self.features(x)
    
    if as_logits:
      return self.classifier(x)

def setup_dist():
  torch.distributed.init_process_group(backend="nccl")

def run_chunk_fwd_bwd(
  model: FSDP,
  inputs: torch.Tensor,
  targets: torch.Tensor,
  block_size: int,
):
  features = model.forward(inputs, as_features=True)
  N_BLOCKS = math.ceil(features.size(0) / block_size)
  prelogits = []
  
  for i in range(N_BLOCKS):
    block = features[i * block_size: (i + 1) * block_size]
    block = block.detach()
    block.requires_grad = True
    logits = model.forward(block, as_logits=True)
    label_block = targets[i * block_size: (i + 1) * block_size]
    loss = torch.nn.functional.cross_entropy(logits, label_block, reduction='none')
    loss = loss.mean()
    loss.backward()
    prelogits.append(block.grad.detach())
  
  grads = torch.cat(prelogits, dim=0)
  print(f"{torch.distributed.get_rank()} grads: {grads.device}")
  features.backward(grads)

if __name__ == "__main__":

  setup_dist()
  RANK, WORLD_SIZE = torch.distributed.get_rank(), torch.distributed.get_world_size()

  with torch.device(f"cuda:{RANK}"):
    model = StrangeModule()

  model.features = FSDP(
    model.features,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={CustomLinear}),
    use_orig_params=True,
    device_id=RANK,
  )

  if RANK == 0:
    print(model)

  inputs = torch.randn(1024, 4096, device=f"cuda:{RANK}")
  targets = torch.randint(0, 32, (1024,), device=f"cuda:{RANK}")
  run_chunk_fwd_bwd(model, inputs, targets, 512)