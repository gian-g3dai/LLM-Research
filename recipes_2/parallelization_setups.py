import contextlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
from omegaconf import OmegaConf, DictConfig

import yaml
import torch
import warnings

from unakin.llm.distributed.fsdp import MP_POLICY as FSDP_1_MP_POLICY, TRAIN_FSDP_WRAP as TRAIN_FSDP_1_WRAP, get_lora_wrapping_policy as fsdp_1_lora_wrapping_policy
# from unakin.llm.distributed.fsdp_2 import MP_POLICY as FSDP_2_MP_POLICY
# from unakin.llm.distributed.fsdp_2 import ReshardStrategy
# from unakin.llm.distributed.fsdp_2 import TRAIN_FSDP_WRAP as TRAIN_FSDP_2_WRAP
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP_1
# from torch.distributed._composable.fsdp.fully_shard import FSDP as FSDP_2

__TYPE_FSDP_UNION__ = FSDP_1 | None # FSDP_2

class ParallelStrategy(Enum):
  FSDP_1 = "FSDP_1"
  FSDP_2 = "FSDP_2"
  # TODO: No Tensor-Parallelism yet because of Adapters

__TYPE_EMPTY_LIST_DATACLASS__ = field(default_factory=lambda: [])

@dataclass
class FSDP_1_Config:
  mixed_precision: FSDP_1_MP_POLICY = FSDP_1_MP_POLICY.BF_16,
  cpu_offload: bool = False,
  wrapping_policy: dict = field(default_factory=lambda: {}),
  forward_prefetch: bool = True,
  backward_prefetch: bool = True,
  act_ckpt_layers: List[type] = __TYPE_EMPTY_LIST_DATACLASS__,

  def __post_init__(self):
    from unakin.llm.distributed.fsdp import MP_POLICY
    # print(type(self.mixed_precision))
    if isinstance(self.mixed_precision, str):
      self.mixed_precision = eval(self.mixed_precision)
    else:
      raise ValueError("MP_POLICY.xx_xx should be the mixed_precision value.")

# @dataclass
# class FSDP_2_Config:
#   wrapping_policy: List = __TYPE_EMPTY_LIST_DATACLASS__,
#   reshard_strategy: ReshardStrategy = ReshardStrategy.DEFAULT_TRANSFORMER,
#   mixed_precision: FSDP_2_MP_POLICY = FSDP_2_MP_POLICY.DEFAULT,
#   act_ckpt_layers: List[type] = __TYPE_EMPTY_LIST_DATACLASS__,

#   def __post_init__(self):
#     from unakin.llm.distributed.fsdp_2 import MP_POLICY
#     if isinstance(self.mixed_precision, str):
#       self.mixed_precision = eval(self.mixed_precision)
#     else:
#       raise ValueError("Check FSDP_2 mixed precision values.")

__TYPE_FSDP_CONFIG_UNION__ = FSDP_1_Config | None # FSDP_2_Config

@dataclass
class ParallelizationConfig:
  strategy: ParallelStrategy
  fsdp_settings: __TYPE_FSDP_CONFIG_UNION__

  def __post_init__(self):
    # arguments are going to come in as a dict
    self.strategy = ParallelStrategy(self.strategy)
    self.fsdp_settings = FSDP_1_Config(**self.fsdp_settings) \
                  if self.strategy == ParallelStrategy.FSDP_1 \
                  else FSDP_2_Config(**self.fsdp_settings)  
    
  def has_cpu_offload(self) -> bool:
    if self.strategy == ParallelStrategy.FSDP_2:
      return False
    elif self.strategy == ParallelStrategy.FSDP_1:
      return self.fsdp_settings.cpu_offload

def parallelize_model(
  model: torch.nn.Module,
  parallel_config: ParallelizationConfig
) -> __TYPE_FSDP_UNION__:
  
  fsdp_config = parallel_config.fsdp_settings

  if parallel_config.strategy == ParallelStrategy.FSDP_1:
    return TRAIN_FSDP_1_WRAP(
      module=model,
      precision=fsdp_config.mixed_precision,
      cpu_offload=fsdp_config.cpu_offload,
      forward_prefetch=fsdp_config.forward_prefetch,
      backward_prefetch=fsdp_config.backward_prefetch,
      wrapping_policy=fsdp_config.wrapping_policy,
      act_ckpt_layers=fsdp_config.act_ckpt_layers,
    )
  elif parallel_config.strategy == ParallelStrategy.FSDP_2:
    return TRAIN_FSDP_2_WRAP(
      module=model,
      mp_policy=fsdp_config.mixed_precision,
      shard_layer_types=fsdp_config.wrapping_policy,
      checkpoint_layer_types=fsdp_config.act_ckpt_layers,
      reshard_strategy=fsdp_config.reshard_strategy
    )
  else:
    raise ValueError("Invalid parallelization strategy")

def grad_sync_context_manager(module: __TYPE_FSDP_UNION__, enabled: bool = False) -> contextlib.AbstractContextManager:
  if not enabled:
    return contextlib.nullcontext()
  
  if hasattr(module, "no_sync"):
    return module.no_sync() 
  
  warnings.warn(f"Module of type {type(module)} does not support no_sync. Ignoring...")
  return contextlib.nullcontext()