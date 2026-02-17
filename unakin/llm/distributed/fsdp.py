from enum import Enum
from functools import partial
from typing import List, Optional
import torch
from torch import nn

import omegaconf
from omegaconf import OmegaConf, DictConfig
import torch.distributed

# TODO: Make this less explicit 
from unakin.llm.adapters.llm_adapter_new import LowRankFactor, LoraLayer
from unakin.llm.models.llama.model_compile import TransformerBlock, Attention, FeedForward
from unakin.llm.distributed.utils import (
  print_rank0
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

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload, FullOptimStateDictConfig,
    FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType)

from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

def get_lora_wrapping_policy(
  lora_cls: Optional[type] = None,
  wrap_active_params: bool = False,
  individual_cls: Optional[List[type]] = [],
  transformer_block_cls: Optional[List[type]] = []
) -> partial:

  assert any([lora_cls, individual_cls, transformer_block_cls]), \
  f"At least of (lora_cls={lora_cls}, individual_cls={individual_cls}, transformer_block_cls={transformer_block_cls}) must be provided"

  policies = []

  if lora_cls:
    def lora_lambda_fn(m: nn.Module) -> bool:
      # TODO: lora_cls must be LRF from yaml
      return isinstance(m, eval(lora_cls))
  
    lora_policy = partial(lambda_auto_wrap_policy, lambda_fn=lora_lambda_fn)
    policies.append(lora_policy)

  if wrap_active_params:
    def active_param_lambda_fn(m: nn.Module) -> bool:
      return (
        len(list(m.children())) == 0
        and getattr(m, 'weight', None) is not None
        and m.weight.requires_grad
      )
    
    active_param_policy = partial(lambda_auto_wrap_policy, lambda_fn=active_param_lambda_fn)
    policies.append(active_param_policy)

  # TODO: transformer_block_cls from yaml
  transformer_block_cls = [eval(cls) for cls in transformer_block_cls]
  individual_cls = tuple(eval(cls) for cls in individual_cls)

  if len(individual_cls) > 0:
    def individual_cls_fn(m: nn.Module) -> bool:
      return isinstance(m, individual_cls)
  else:
    def individual_cls_fn(m: nn.Module) -> bool:
      return False
    
  # higher level policies which have lower precedence
  individual_cls_policies = [partial(lambda_auto_wrap_policy, lambda_fn=individual_cls_fn)]
  transformer_block_policies = [partial(transformer_auto_wrap_policy, transformer_layer_cls={cls}) for cls in transformer_block_cls]
  
  # expand the policies
  policies = policies + individual_cls_policies + transformer_block_policies
  return partial(_or_policy, policies=policies)

def get_block_wrapping_policy(block_cls: Optional[List[type]] = []):
  """Simple wrapper for easier access to get_lora_wrapping_policy"""
  return get_lora_wrapping_policy(transformer_block_cls=block_cls)

def get_precision_policy(
  param_dtype: str, reduce_dtype: str, buffer_dtype: str, low_precision_grad: bool = False
) -> MixedPrecision:
  """Get a precision policy for FSDP"""

  precision_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,}

  param_dtype = precision_map.get(param_dtype, torch.float32)
  reduce_dtype = precision_map.get(reduce_dtype, torch.float32)
  buffer_dtype = precision_map.get(buffer_dtype, torch.float32)

  return MixedPrecision(
    param_dtype=param_dtype,
    reduce_dtype=reduce_dtype,
    buffer_dtype=buffer_dtype,
    keep_low_precision_grads=low_precision_grad,
    cast_forward_inputs=True
  )

FP_16_MP_POLICY = partial(
  get_precision_policy, param_dtype="float16", reduce_dtype="float16", buffer_dtype="float16")
BF_16_MP_POLICY = partial(
  get_precision_policy, param_dtype="bfloat16", reduce_dtype="bfloat16", buffer_dtype="bfloat16")
FP_32_MP_POLICY = partial(
  get_precision_policy, param_dtype="float32", reduce_dtype="float32", buffer_dtype="float32")

class MP_POLICY(Enum):
  FP_16 = FP_16_MP_POLICY
  BF_16 = BF_16_MP_POLICY
  FP_32 = FP_32_MP_POLICY

def get_act_ckpt_policy(enabled: bool = True, checkpoint_impl: Optional[CheckpointImpl] = CheckpointImpl.NO_REENTRANT):
  """Get an activation checkpointing policy for FSDP"""
  if enabled:
    return partial(checkpoint_wrapper, checkpoint_impl=checkpoint_impl)
  else: 
    return None

def apply_act_ckpt_policy(model: nn.Module, ckpt_policy: partial | None):
  """Apply an activation checkpointing policy to a model"""
  if ckpt_policy:
    apply_activation_checkpointing(model=model, auto_wrap_policy=ckpt_policy,
                                   checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT))
    return model
  else:
    return model

def validate_wrapping_policy_kwargs(wrapping_policy: dict):
  """Validate the wrapping policy kwargs"""
  if isinstance(wrapping_policy, DictConfig): 
    wrapping_policy = OmegaConf.to_container(wrapping_policy)

  assert wrapping_policy, "wrapping_policy must be provided"
  assert isinstance(wrapping_policy, dict), "wrapping_policy must be a dictionary"
  
  has_lora_cls = wrapping_policy.get("lora_cls", None)
  has_individual_cls = wrapping_policy.get("individual_cls", None)
  has_transformer_block_cls = wrapping_policy.get("transformer_block_cls", None)
  assert any([has_lora_cls, has_individual_cls, has_transformer_block_cls]), \
    "At least one of lora_cls, individual_cls, or transformer_block_cls must be provided"

def wrap_fsdp_model(
  model: nn.Module,
  mixed_precision: MixedPrecision,
  sharding_config: ShardingStrategy,
  wrapping_policy: dict,
  offload_config: bool,
  forward_prefetch: bool = True,
  backward_prefetch: BackwardPrefetch | None = BackwardPrefetch.BACKWARD_PRE,  
  act_ckpt_layers: List[type] = [],
):
  
  print(f"Number of optimizable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
  
  validate_wrapping_policy_kwargs(wrapping_policy)
  wrapping_policy = get_lora_wrapping_policy(**wrapping_policy)

  """Wrap a model with FSDP"""
  model = FSDP(
    module=model,
    auto_wrap_policy=wrapping_policy,
    sharding_strategy=sharding_config,
    mixed_precision=mixed_precision,
    cpu_offload=CPUOffload(offload_params=offload_config),
    forward_prefetch=forward_prefetch,
    backward_prefetch=backward_prefetch,
    device_id=torch.distributed.get_rank(),
    use_orig_params=True
  )

  if act_ckpt_layers:
    ackt_ckpt_policy = get_block_wrapping_policy(block_cls=act_ckpt_layers)
    model = apply_act_ckpt_policy(model, ckpt_policy=ackt_ckpt_policy)

  return model

def TRAIN_FSDP_WRAP(
  module: nn.Module,
  wrapping_policy: dict,
  precision: MP_POLICY = MP_POLICY.BF_16,
  cpu_offload: bool = False,
  forward_prefetch: bool = True,
  backward_prefetch: bool = True,
  act_ckpt_layers: List[type] = []
) -> FSDP:
  assert isinstance(precision, MP_POLICY), f"precision must be one of {list(MP_POLICY.__members__.keys())}"

  return wrap_fsdp_model(
    model=module,
    mixed_precision=precision.value(),
    sharding_config=ShardingStrategy.FULL_SHARD,
    # if a wrapping policy is not provided, we'll use the default one
    wrapping_policy=wrapping_policy,
    offload_config=CPUOffload(offload_params=cpu_offload),
    # we want to checkpoint activations only during training
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE if backward_prefetch else None,
    forward_prefetch=forward_prefetch,
    act_ckpt_layers=act_ckpt_layers,
  )

def INFER_FSDP_WRAP(
  module: nn.Module,
  wrapping_policy: dict,
  precision: MP_POLICY = MP_POLICY.FP_16,
  cpu_offload: bool = False
) -> FSDP:
  assert isinstance(precision, MP_POLICY), f"precision must be one of {list(MP_POLICY.__members__.keys())}"
  return wrap_fsdp_model(
    model=module,
    mixed_precision=precision.value(),
    sharding_config=ShardingStrategy.FULL_SHARD,
    wrapping_policy=wrapping_policy,
    offload_config=CPUOffload(offload_params=cpu_offload),
    # no need for backward prefetching
    backward_prefetch=None,
    forward_prefetch=False,
    # we don't actually need to checkpoint activations here
    # because we're not saving gradients or intermediate activations
  )