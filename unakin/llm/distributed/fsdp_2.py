import torch

from enum import Enum
from typing import Dict, List
from functools import partial

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)

from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)

from unakin.llm.models.llama.model_compile import Transformer, CODE_LLAMA_7B_WEIGHTS, TransformerBlock
from unakin.llm.distributed.utils import print_rank0, destroy_process_group, setup_async_distributed_process_group, get_dist_info

class ReshardStrategy(Enum):
  DEFAULT = "default"
  DEFAULT_TRANSFORMER = "default_transformer"

class CheckpointStrategy(Enum):
  FULL = "full"
  FULL_RNG = "full_rng"

def checkpoint_wrapper(
  module: torch.nn.Module, strategy: CheckpointStrategy=CheckpointStrategy.FULL) -> torch.nn.Module:

  # TODO: add support for selective checkpointing
  __STRATEGY_CONFIG_MAP__ = {
    CheckpointStrategy.FULL: {
      "checkpoint_fn": checkpoint, "use_reentrant": False, "preserver_rng_state": False, "checkpoint_impl": CheckpointImpl.NO_REENTRANT},
    CheckpointStrategy.FULL_RNG: {
      "checkpoint_fn": checkpoint, "use_reentrant": False, "preserver_rng_state": True, "checkpoint_impl": CheckpointImpl.NO_REENTRANT},
  }

  return ptd_checkpoint_wrapper(module=module, **__STRATEGY_CONFIG_MAP__[strategy])

  

def get_mp_policy(param_dtype: str="bf16", reduce_dtype="fp32"):
  __ALLOWED_DTYPE_MAP__ = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
  }

  assert param_dtype in __ALLOWED_DTYPE_MAP__, f"param_dtype must be one of {__ALLOWED_DTYPE_MAP__.keys()}"
  assert reduce_dtype in __ALLOWED_DTYPE_MAP__, f"reduce_dtype must be one of {__ALLOWED_DTYPE_MAP__.keys()}"

  # TODO: output_dtype ?
  return MixedPrecisionPolicy(
    param_dtype=__ALLOWED_DTYPE_MAP__[param_dtype],
    reduce_dtype=__ALLOWED_DTYPE_MAP__[reduce_dtype],
  )

DEFAULT_MP_POLICY = partial(get_mp_policy, param_dtype="bf16", reduce_dtype="fp32")
FULL_BF16_MP_POLICY = partial(get_mp_policy, param_dtype="bf16", reduce_dtype="bf16")
FULL_FP32_MP_POLICY = partial(get_mp_policy, param_dtype="fp32", reduce_dtype="fp32")

class MP_POLICY(Enum):
  DEFAULT = DEFAULT_MP_POLICY
  BF_16 = FULL_BF16_MP_POLICY
  FP_32 = FULL_FP32_MP_POLICY

class _ModuleAttributeRef:
  # a class that holds a reference to a module attribute, which can be used to modify the attribute
  def __init__(self, parent: torch.nn.Module, child: torch.nn.Module, attribute_name: str):
    self.parent = parent
    self.child  = child
    self.attribute_name = attribute_name

  def _set_attribute(self, value):
    setattr(self.parent, self.attribute_name, value)
  
  def __eq__(self, other: "_ModuleAttributeRef"):
    assert isinstance(other, _ModuleAttributeRef), f"expected {type(self)}, got {type(other)}"

    return all([
      id(self.parent) == id(other.parent),
      id(self.child) == id(other.child),
      self.attribute_name == other.attribute_name,
    ])

  def __hash__(self):
    return hash((id(self.parent), id(self.child), self.attribute_name))

  def __repr__(self) -> str:
    return f"_ModuleAttributeRef(parent={id(self.parent)}, child={id(self.child)}, attribute_name={self.attribute_name})"

def filter_layers_by_predicate(module: torch.nn.Module, predicate: callable) -> list[_ModuleAttributeRef]:
  # recursively filter layers by predicate
  def _filter(module: torch.nn.Module, predicate: callable):
    for name, child in module.named_children():
      if predicate(child):
        yield _ModuleAttributeRef(module, child, name)
      else:
        yield from _filter(child, predicate)
    
  return list(_filter(module, predicate))
  
def apply_checkpoint_policy(
  module: torch.nn.Module,
  target_layer_refs: list[_ModuleAttributeRef] = [],
  checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.FULL,
) -> torch.nn.Module:
  """Apply checkpointing policy to a model. The application is inplace, however we preserve a functional API"""
  if len(target_layer_refs) == 0:
    return checkpoint_wrapper(module, strategy=checkpoint_strategy)
  
  for layer_ref in target_layer_refs:
    parent, layer, attribute_name = layer_ref
    layer = checkpoint_wrapper(layer, strategy=checkpoint_strategy)
    layer_ref._set_attribute(layer)
  return module

def find_layer_types(
  module: torch.nn.Module, match_strs: List[str]
) -> List[type]:

  def is_match(class_name: str):
    if not match_strs: 
      return True
    return class_name in match_strs

  layer_types = {}
  for n, m in module.named_modules():
    class_name = m.__class__.__name__
    if class_name not in layer_types and is_match(class_name):
      layer_types[class_name] = type(m)
  return list(layer_types.values())

def apply_fully_sharded_policy(
  module: torch.nn.Module,
  mesh: DeviceMesh,
  mp_policy: MixedPrecisionPolicy,
  shard_layer_types: list[str] = [],
  checkpoint_layer_types: list[str] = [],
  reshard_strategy: ReshardStrategy = ReshardStrategy.DEFAULT_TRANSFORMER,
) -> torch.nn.Module:
  
  shard_layer_types = find_layer_types(module, shard_layer_types)
  checkpoint_layer_types = find_layer_types(module, checkpoint_layer_types)

  shard_layer_refs = list(set(sum(
    [
      filter_layers_by_predicate(module, lambda x: isinstance(x, layer_type)) 
      for layer_type in shard_layer_types
    ],
    []
  )))

  checkpoint_layer_refs = list(set(sum(
    [
      filter_layers_by_predicate(module, lambda x: isinstance(x, layer_type))
      for layer_type in checkpoint_layer_types
    ],
    []
  )))

  """Apply fully sharded policy to a model. The application is inplace, however we preserve a functional API"""
  fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}
  for idx, layer_ref in enumerate(shard_layer_refs):
    parent, layer, attribute_name = vars(layer_ref).values()
    # TODO: this is gonna have to happen somewhere else
    reshard_after_forward = idx < len(shard_layer_refs) - 1 if \
                            reshard_strategy == ReshardStrategy.DEFAULT_TRANSFORMER \
                            else False
    
    # TODO: somehow these refs should contain the strategy? I don't know man...
    # TODO: this doesn't work with in syntax?
    if checkpoint_layer_refs.__contains__(layer_ref):
      layer = checkpoint_wrapper(layer, strategy=CheckpointStrategy.FULL)

    fully_shard(
      module=layer,
      **fsdp_config,
      reshard_after_forward=reshard_after_forward,
    )

    # edit the parent module to point to the sharded layer
    # TODO: misterious reasons, reasoning taken from torchtitan
    shard_layer_refs[idx]._set_attribute(layer)

  # TODO: one final shard for misterious reasons
  module = fully_shard(module=module, **fsdp_config,)    
  return module

def TRAIN_FSDP_WRAP(
  module: torch.nn.Module,
  mp_policy: MixedPrecisionPolicy,
  shard_layer_types: list[str] = [],
  checkpoint_layer_types: list[str] = [],
  reshard_strategy: ReshardStrategy = ReshardStrategy.DEFAULT_TRANSFORMER,
) -> torch.nn.Module:
  rank, world_size = get_dist_info()
  # TODO: this is going to have to support some funky kind of dimensioning or partitioning at some point
  mesh = init_device_mesh("cuda", (world_size,))

  return apply_fully_sharded_policy(
    module=module,
    mesh=mesh,
    mp_policy=mp_policy,
    shard_layer_types=shard_layer_types,
    checkpoint_layer_types=checkpoint_layer_types,
    reshard_strategy=reshard_strategy,
  )
  
if __name__ == "__main__":
  setup_async_distributed_process_group()
  base_path = "/mnt/raid0/llamas/llama_base"
  transformer = Transformer.from_base_model(model_weights=CODE_LLAMA_7B_WEIGHTS, base_path=base_path)
  mesh = init_device_mesh("cuda", (8,))
  mp_policy = get_mp_policy(param_dtype="bf16", reduce_dtype="fp32")
  TRAIN_FSDP_WRAP(transformer, mp_policy, shard_layer_types=["TransformerBlock"])
  # transformer = apply_checkpoint_policy(transformer, target_layer_refs, CheckpointStrategy.FULL)
  print_rank0(transformer)
  destroy_process_group()