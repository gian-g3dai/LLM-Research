import collections
import functools
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import hydra
import safetensors.torch
import torch
import torch.distributed
import tqdm
import wandb
import pickle
import numpy as np
import pandas as pd
from utils import tensor_to_numpy
from enum import Enum
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
    offload_wrapper)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload, FullOptimStateDictConfig,
    FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from unakin.llm.adapters.llm_adapter import LlmAdapter
from unakin.llm.adapters.utils import AdapterConfigFactory
from unakin.llm.data.dataset_factory import DatasetFactory
from unakin.llm.models.utils import ModelFactory
from unakin.llm.optimizers.utils import OptimizerFactory
from unakin.llm.tokenizers import LlamaTokenizer, TokenizerBase


class ComparisonOp(Enum):
  LT = 0
  LE = 1
  GT = 2
  GE = 3
  EQ = 4
  NE = 5

  def __call__(self, a: float, b: float) -> bool:
    return apply_op(self, a, b)

def apply_op(op: ComparisonOp, a: float, b: float) -> bool:
  if op == ComparisonOp.LT:
    return a < b
  elif op == ComparisonOp.LE:
    return a <= b
  elif op == ComparisonOp.GT:
    return a > b
  elif op == ComparisonOp.GE:
    return a >= b
  elif op == ComparisonOp.EQ:
    return a == b
  elif op == ComparisonOp.NE:
    return a != b
  else:
    raise ValueError(f"Invalid operator: {op}")
  
def compare_metric_dicts(dict_new: dict, dict_old: dict, op_map: dict) -> bool:
  for metric, op in op_map.items():
    if apply_op(op, dict_new[metric], dict_old[metric]):
      return True
  return False

def get_improved_metrics(dict_new: dict, dict_old: dict, op_map: dict) -> dict:
  improved_metrics = {}
  for metric, op in op_map.items():
    if apply_op(op, dict_new[metric], dict_old[metric]):
      improved_metrics[metric] = dict_new[metric]
    else:
      improved_metrics[metric] = dict_old[metric]
  return improved_metrics

def print_rank0(*args, **kwargs):
  if torch.distributed.get_rank() == 0:
    print(*args, **kwargs)

@dataclass
class FineTuningConfig:
  # debug mode
  debug: bool = True
  seed: int = 42

  # layer names to wrap
  layer_classes: List[str] = field(
    default_factory=lambda: ["TransformerBlock"])

  # activation / gradient checkpointing & offloading
  enabled_act_ckpt: bool = True
  enabled_act_offload: bool = False

  # parameter offloading
  enabled_param_offload: bool = False

  # mixed precision
  param_dtype: str = "bfloat16"
  reduce_dtype: str = "bfloat16"
  buffer_dtype: str = "bfloat16"
  low_precision_grad: bool = True

  # sharding
  sharding_policy: str = "full"

  # distributed training
  enabled_fsdp: bool = True
  enabled_ddp: bool = False

  # checkpointing
  save_interval: int = 128

  # save path
  save_path: str | None = None

  # logging
  log_interval: int = 2
  save_per_token_loss: bool = True

  # evaluation
  eval_interval: int = 32

  # checkpointing
  ckpt_dir: str = "checkpoints"

  # wandb
  wandb: bool = True
  wandb_key: str | None = "REMOVED" # REMOVED # ted
  wandb_experiment_name: str = "dataset_r_a_fft"

  # scaler
  scaler_init_scale: float = 2 ** 16
  scaler_backoff_factor: float = 0.5
  scaler_growth_factor: float = 2.0
  scaler_growth_interval: int = 2000

  # gradient clipping & optimization
  num_training_tokens: int = 120_000_000 # Prev: 60M
  ref_batch_size: int = 1
  gpu_batch_size: int = 1
  clip_grad_norm: float = 1.0
  warm_up: float = 0.02

  # loss protection policy
  skip_malformed_batches: bool = True
  malformed_batch_ppl_threshold: float = 32.0

  soften_loss_spikes: bool = True
  apply_per_token_softening: bool = True
  loss_spike_threshold: float = 3.5
  loss_spike_factor: float = 0.7
  malformed_batch_loss_threshold: float = 5

  # pseudo-lora policy
  n_steps_apply_partial_lora: int = 16
  partial_lora_decay: float = 0.1

  # resume checkpoint
  resume_ckpt: str | None = None # "checkpoints/copper-water-252/000016"
  resume_training: bool = False

  # optimizer config
  # TODO: this is rather dynnamic, maybe best to leave like this?
  optimizer_config_path: str = "configs/optimizers/adamw_base.json"

  # dataset config
  # TODO: this is rather dynnamic, maybe best to leave like this?
  dataset_config_path: str = "configs/datasets/dataset_r_a_01.json"

  # test dataset config
  test_dataset_config_path: str = "configs/datasets/benchmark-01.json"
  # test_batch_size for inference > 1
  test_batch_size: int = 1

  # model config 
  # TODO: this is rather dynnamic, maybe best to leave like this?
  model_config_path: str = "configs/models/default_conf.json"

  # adapter config
  # TODO: this is rather dynnamic, maybe best to leave like this?
  adapter_config_path: str = "configs/adapters/full_finetune.json"


#########################################
#       Distributed utils               #
#########################################

def setup_dist():
  """Setup distributed process group"""
  torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
  )


def cleanup_dist():
  """Cleanup distributed process group"""
  if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()


def setup_print(rank: int, debug: bool = False):
  """Setup stdout/stderr redirection"""
  if debug:
    return

  if rank != 0:
    f = open(os.devnull, "w")
    sys.stdout = f
    sys.stderr = f

#########################################
#            FSDP wrappers              #
#########################################


def get_layer_wrap_policy(m: nn.Module, layer_classes: List[nn.Module]) -> Callable | None:
  """Get a wrap policy for a list of layer classes"""
  if not layer_classes:
    return None

  # scan the model for classes that have a name
  # that matches any of the layer classes
  layer_class_names = set(layer_classes)
  layer_class_set = set()

  for _, module in m.named_modules():
    if module.__class__.__name__ in layer_class_names:
      layer_class_set.add(module.__class__)

  return functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls=layer_class_set,
  )


def act_ckpt_wrapper(enabled: bool = True) -> Callable | None:
  """Enable activation checkpointing policy"""
  if not enabled:
    return None

  return functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
  )


def act_offload_wrapper(enabled: bool = True) -> Callable | None:
  """Enable activation offload policy"""
  if not enabled:
    return None

  return functools.partial(
    offload_wrapper,
  )


def apply_act_wrapper(module: nn.Module | FSDP,
                      wrapper: Callable, wrap_policy: Callable | None) -> None:
  """Apply activation wrapper to a module"""
  if wrap_policy is None or not wrapper:
    return

  if not isinstance(module, FSDP):
    print(
      f"Warning: module is not FSDP, type: {type(module)}. Skipping activation wrapper")

  apply_activation_checkpointing(
    model=module,
    auto_wrap_policy=wrap_policy,
    checkpoint_wrapper_fn=wrapper,
  )


def param_cpu_offload_policy(enabled: bool = True) -> CPUOffload:
  """Enable CPU offload policy"""
  if not enabled:
    return CPUOffload(offload_params=False)

  return CPUOffload(offload_params=True)


def precision_policy(
  param_dtype: str,
  reduce_dtype: str,
  buffer_dtype: str,
  low_precision_grad: bool
) -> MixedPrecision:
  """Get a precision policy for FSDP"""

  precision_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
  }

  param_dtype = precision_map.get(param_dtype, torch.float32)
  reduce_dtype = precision_map.get(reduce_dtype, torch.float32)
  buffer_dtype = precision_map.get(buffer_dtype, torch.float32)

  return MixedPrecision(
    param_dtype=param_dtype,
    reduce_dtype=reduce_dtype,
    buffer_dtype=buffer_dtype,
    keep_low_precision_grads=low_precision_grad,
  )


def sharding_policy(
  sharding_policy: str,
) -> ShardingStrategy:
  """Get a sharding policy for FSDP"""
  sharding_map = {
    "full": ShardingStrategy.FULL_SHARD,
    "optimizer": ShardingStrategy.SHARD_GRAD_OP,
  }

  return sharding_map.get(sharding_policy, ShardingStrategy.FULL_SHARD)


def fsdp_cpu_state(module: FSDP, optimizer: torch.optim.Optimizer, args: FineTuningConfig) -> Tuple[Dict[str, Any], ...]:
  """Get CPU state dict from FSDP"""
  if not isinstance(module, FSDP):
    print(f"Warning: module is not FSDP, type: {type(module)}")

  
  model_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

  with FSDP.state_dict_type(module,
                            StateDictType.FULL_STATE_DICT,
                            model_save_policy,
                            optim_save_policy):
    
    model_state_dict = module.state_dict()
    optim_state_dict = optimizer.state_dict()

  # don't return on non-zero ranks
  if torch.distributed.get_rank() != 0:
    model_state_dict = None
    optim_state_dict = None

  return model_state_dict, optim_state_dict


def base_state_dict(module: nn.Module | DDP,
                    optimizer: torch.optim.Optimizer) -> Tuple[Dict[str, Any], ...]:
  """Get state dict from DDP"""
  if not isinstance(module, (DDP, nn.Module)):
    print(f"Warning: module is not DDP, type: {type(module)}")

  if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
    model_state_dict = None
    optim_state_dict = None
  else:
    model_state_dict = module.state_dict()
    optim_state_dict = optimizer.state_dict()

  return model_state_dict, optim_state_dict


def get_state_dict(model: nn.Module | FSDP | DDP,
                   optimizer: torch.optim.Optimizer,
                   args: FineTuningConfig) -> Tuple[Dict[str, Any], ...]:
  if isinstance(model, FSDP):
    model_state_dict, optim_state_dict = fsdp_cpu_state(model, optimizer, args)
  else:
    model_state_dict, optim_state_dict = base_state_dict(model, optimizer)

  # keep only keys that are in the optimizer as well
    for k in list(model_state_dict.keys()):
      if k not in optim_state_dict["state"]:
        del model_state_dict[k]

  return model_state_dict, optim_state_dict


def get_run_name(args: FineTuningConfig) -> str:
  if torch.distributed.get_rank() != 0:
    return ""
  
  if args.save_path is None:
    if args.wandb:
      return wandb.run.name
    else:
      # return a random name
      return wandb.util.generate_id()

      # return whole config as a name # too long
      return "_".join(f"{k}={v}" for k, v in args_to_dict(args).items())

  return args.save_path


class CheckpointManager:
  def __init__(self, root: str) -> None:
    self.root = root

  def _run_name(self, args: FineTuningConfig) -> str:
    return get_run_name(args)
  
  def _checkpoint_dir(self, args: FineTuningConfig) -> str:
    run_name = self._run_name(args)
    run_dir  = f"{self.root}/{run_name}"
    if not os.path.exists(run_dir):
      os.makedirs(run_dir, exist_ok=True)
    return run_dir
  
  def _step_dir(self, args: FineTuningConfig, step: int) -> str:
    run_dir = self._checkpoint_dir(args)
    step_dir = f"{run_dir}/{step:06d}"
    if not os.path.exists(step_dir):
      os.makedirs(step_dir, exist_ok=True)

    # make a directory that will contain the weights
    step_weights_dir = f"{step_dir}/weights"
    if not os.path.exists(step_weights_dir):
      os.makedirs(step_weights_dir, exist_ok=True)

    return step_dir

  def _materialize_state_dict(self, model: nn.Module | FSDP | DDP,
                              optimizer: torch.optim.Optimizer,
                              lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                              scaler: ShardedGradScaler | GradScaler | None,
                              step: int, metrics: dict, args: FineTuningConfig) -> Dict[str, Any]:
    """Create a state dict for saving"""
    state_dict = {}
    # config related stuff
    state_dict["step"] = step
    state_dict["config"] = args_to_dict(args)
    state_dict["metrics"] = metrics

    # make paths in the weights dir for the model and optimizer
    state_dict["model_state_dict_path"] = f"{self._step_dir(args, step)}/weights/model.st"
    state_dict["optimizer_state_dict_path"] = f"{self._step_dir(args, step)}/weights/optimizer.pth.tar"

    # trivial stuff is alocated in the root dir
    state_dict["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if scaler is not None:
      state_dict["scaler_state_dict"] = scaler.state_dict()

    return state_dict

  def _save_state_dict(self,
                        step_dir: str,
                        model_state_dict: Dict[str, Any],
                        optimizer_state_dict: Dict[str, Any],
                        state_dict: Dict[str, Any], args: FineTuningConfig) -> None:
    """Save a state dict"""
    if torch.distributed.is_initialized():
      if torch.distributed.get_rank() != 0:
        return
    
    # save the model state dict
    torch.save(state_dict, f"{step_dir}/ckpt.pth")

    # save the optimizer state dict
    torch.save(
      optimizer_state_dict,
      state_dict["optimizer_state_dict_path"],
    )

    # save the state dict
    safetensors.torch.save_file(
      model_state_dict,
      state_dict["model_state_dict_path"],
    )


  def save(self, model: nn.Module | FSDP | DDP,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
            scaler: ShardedGradScaler | GradScaler | None,
            step: int, metrics: dict, args: FineTuningConfig) -> None:
  
    model_state_dict, optim_state_dict = get_state_dict(model, optimizer, args)
    
    # sometimes the LlmAdapter has some compute optimised parameters
    # i.e. we're padding embedding tables for CUDA cache performance
    # since this padding is not part of the standard model, we need to
    # remove it from the state dict
    if isinstance(model, (FSDP, DDP)):
      base_module = model.module
    else:
      base_module = model
    if isinstance(base_module, LlmAdapter):
      model_state_dict = base_module.process_state_dict(model_state_dict)

    step_dir = self._step_dir(args, step)
    
    # materialize the state dict
    state_dict = self._materialize_state_dict(
      model, optimizer, lr_scheduler, scaler, step, metrics, args)

    # save the state dict
    self._save_state_dict(
      step_dir, model_state_dict, optim_state_dict, state_dict, args)

  def load_tokenizer(self, path: str) -> LlamaTokenizer:
    """Load a tokenizer"""
    ckpt = torch.load(path, map_location="cpu")
    tokenizer_path = ckpt["config"]["tokenizer_path"]
    tokenizer = LlamaTokenizer(model_path=tokenizer_path)
    return tokenizer

  def init_model_from_ckpt(self, path: str, root_path: str = None) -> nn.Module:
    """Initialize a model from a checkpoint"""
    ckpt = torch.load(path, map_location="cpu")

    model_state_dict_path = ckpt["model_state_dict_path"]
    
    if root_path is not None:
      model_state_dict_path = f"{root_path}/{model_state_dict_path}"

    base_model, tokenizer = ModelFactory.from_json(ckpt["config"]["model_config_expanded"])
    adapter_config = AdapterConfigFactory.from_json(ckpt["config"]["adapter_config_expanded"]) 

    model = LlmAdapter(m=base_model, params=adapter_config)
    model.apply_adaptation()

    model_state_dict = safetensors.torch.load_file(model_state_dict_path)
    model.load_state_dict(model_state_dict, strict=False, assign=True)
    return model, tokenizer

  def reload_model_from_ckpt(self,
                             path: str,
                             model: nn.Module | LlmAdapter) -> nn.Module | LlmAdapter:
    """Reload a model from a checkpoint"""
    if path is None:
      print("Info: no checkpoint path provided. Skipping reload")
      return model
    
    if not os.path.exists(path):
      print(f"Warning: checkpoint path {path} does not exist. Skipping reload")
      return model
    
    rank = torch.distributed.get_rank()
    checkpoint = torch.load(path, map_location="cpu")
    model_state_path = checkpoint["model_state_dict_path"]

    if rank == 0:
      model_state = safetensors.torch.load_file(model_state_path)
      model.load_state_dict(model_state, strict=False, assign=True)

    return model

  def reload_optim_scaler_from_ckpt(self,
                                    path: str,
                                    is_fsdp: bool,
                                    optimizer: torch.optim.Optimizer,
                                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                                    scaler: GradScaler | ShardedGradScaler | None) -> None:
  
    if path is None:
      return

    if not os.path.exists(path):
      print(f"Warning: checkpoint path {path} does not exist. Skipping reload")
      return
    
    rank = torch.distributed.get_rank()

    ckpt = torch.load(path, map_location="cpu")
    opt_state_dict_path = ckpt["optimizer_state_dict_path"]
    
    if is_fsdp:
      if rank == 0:
        opt_state_dict = safetensors.torch.load_file(opt_state_dict_path)
        FSDP.shard_full_optim_state_dict(opt_state_dict, optimizer)
    else:
      opt_state_dict = safetensors.torch.load_file(opt_state_dict_path)
      optimizer.load_state_dict(opt_state_dict)

    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])

    if scaler is not None:
      scaler.load_state_dict(ckpt["scaler_state_dict"])    

  def reload_step_from_ckpt(self, path: str) -> int:
    """Reload a step from a checkpoint"""
    if path is None:
      print("Info: no checkpoint path provided. Skipping reload")
      return 1

    if not os.path.exists(path):
      print(f"Warning: checkpoint path {path} does not exist. Skipping reload")
      return 1

    checkpoint = torch.load(path, map_location="cpu")
    step = checkpoint["step"]
    return step


def wrap_ddp_model(model: nn.Module, args: FineTuningConfig) -> DDP:
  """Wrap a model with DDP"""
  model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  model = DDP(
    model,
    device_ids=torch.cuda.current_device(),
    output_device=torch.cuda.current_device(),
    broadcast_buffers=True,
  )

  return model

def wrap_fsdp_model(model: nn.Module, args: FineTuningConfig) -> FSDP:
  """Wrap a model with FSDP"""
  wrap_policy = get_layer_wrap_policy(model, args.layer_classes)
  param_offload = param_cpu_offload_policy(args.enabled_param_offload)
  precision = precision_policy(
    args.param_dtype, args.reduce_dtype, args.buffer_dtype, args.low_precision_grad)
  sharding = sharding_policy(args.sharding_policy)

  # safety measure
  model.to(precision.param_dtype)
  model.to(f"cuda:{torch.distributed.get_rank()}")

  model = FSDP(
    module=model,
    sharding_strategy=sharding,
    mixed_precision=precision,
    cpu_offload=param_offload,
    auto_wrap_policy=wrap_policy,
    device_id=torch.cuda.current_device(),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    use_orig_params=True,
  )

  # print(f"FSDP: {torch.distributed.get_rank()}, Model is None: {model is None}, current device: {torch.cuda.current_device()}")
  print_rank0(model)
  return model


def wrap_distributed_model(model: nn.Module, args: FineTuningConfig):
  """Wrap a model with DDP or FSDP"""
  if args.enabled_fsdp:
    model = wrap_fsdp_model(model, args)

  elif args.enabled_ddp:
    model = wrap_ddp_model(model, args)

  else:
    return model

  return model


def wrap_activation_saving(model: nn.Module, args: FineTuningConfig, layer_wrap_policy: Callable | None):
  """Wrap a model with activation checkpointing / offloading"""
  act_checkpointing = act_ckpt_wrapper(args.enabled_act_ckpt)
  act_offloading = act_offload_wrapper(args.enabled_act_offload)

  apply_act_wrapper(model, act_checkpointing, layer_wrap_policy)
  apply_act_wrapper(model, act_offloading, layer_wrap_policy)

  return model

#########################################
#            adapter wrappers           #
#########################################


def apply_adapters(model: nn.Module, args: FineTuningConfig) -> LlmAdapter:
  """Apply adapters to a model"""
  adapter_config = AdapterConfigFactory.from_file(args.adapter_config_path)
  model = LlmAdapter(m=model, params=adapter_config)
  model.apply_adaptation()
  return model

#########################################
#           Dataloader Init             #
#########################################


def get_collate_fn(dataset: Dataset) -> Callable | None:
  if not issubclass(dataset.__class__, Dataset):
    raise ValueError("Dataset must be a subclass of torch.utils.data.Dataset")
  
  if hasattr(dataset, "get_collate_fn"):
    return dataset.get_collate_fn()

  return None


#########################################
#          Optimizer / Scaler Setup     #
#########################################

def is_mixed_precision(args: FineTuningConfig) -> bool:
  precision_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
  }

  precision_values = [args.param_dtype, args.reduce_dtype, args.buffer_dtype]
  if any(precision_map.get(p, torch.float32) == torch.float16 for p in precision_values):
    return True

  return False


def setup_lr_scheduler(optimizer: torch.optim.Optimizer,
                       n_total_steps: int,
                       n_cycle_steps: int,
                       args: FineTuningConfig) -> torch.optim.lr_scheduler._LRScheduler:
  """Setup learning rate scheduler"""

  class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_fraction, cycle_length, max_lr, min_lr, last_epoch=-1):
      self.total_steps = total_steps
      self.warmup_steps = int(warmup_fraction * cycle_length)
      self.cycle_length = cycle_length
      self.max_lr = max_lr
      self.min_lr = min_lr
      self.last_epoch = last_epoch
      self.optimizer = optimizer
      
      lr_range = self.max_lr - self.min_lr
      self.cosine_schedule = self.min_lr + lr_range * 0.5 * (1 + np.cos(np.linspace(0, np.pi, self.total_steps)))
      
      super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
      cycle_position = self.last_epoch % self.cycle_length
      
      if cycle_position < self.warmup_steps:
          # Warm-up phase
          index = (self.last_epoch + self.warmup_steps - cycle_position) % self.total_steps
          warmup_target = self.cosine_schedule[index]
          warmup_lr = self.min_lr + (warmup_target - self.min_lr) * (cycle_position / self.warmup_steps)
          return [warmup_lr for _ in self.optimizer.param_groups]
      
      if self.last_epoch >= self.total_steps:
        return [self.min_lr for _ in self.optimizer.param_groups]
      
      return [self.cosine_schedule[self.last_epoch % self.total_steps] for _ in self.optimizer.param_groups]

  # set learning rates
  max_lr = optimizer.param_groups[0]["lr"]
  min_lr  = 1e-6  

  return CustomLRScheduler(
    optimizer=optimizer,
    total_steps=n_total_steps,
    warmup_fraction=args.warm_up,
    cycle_length=n_cycle_steps,
    max_lr=max_lr,
    min_lr=min_lr,
  )


def setup_scaler(args: FineTuningConfig) -> ShardedGradScaler | GradScaler | None:
  # skip if not using float16 anywhere
  if not is_mixed_precision(args):
    return None

  # FSDP case
  if args.enabled_fsdp:
    return ShardedGradScaler(
      init_scale=args.scaler_init_scale,
      backoff_factor=args.scaler_backoff_factor,
      growth_factor=args.scaler_growth_factor,
      growth_interval=args.scaler_growth_interval,
    )

  # other cases
  return GradScaler(
    init_scale=args.scaler_init_scale,
    backoff_factor=args.scaler_backoff_factor,
    growth_factor=args.scaler_growth_factor,
    growth_interval=args.scaler_growth_interval,
  )


def setup_optimizer(model: nn.Module, args: FineTuningConfig) -> torch.optim.Optimizer:
  return OptimizerFactory.from_file(model, args.optimizer_config_path)

#########################################
#           Training func               #
#########################################


def to_device(batch: Tuple[Tensor, ...], rank):
  return (b.to(f"cuda:{rank}") if isinstance(b, torch.Tensor) else b for b in batch)


def args_to_dict(args: FineTuningConfig) -> Dict[str, Any]:
  ret_d = vars(args)

  expand_keys = ["optimizer_config_path", "dataset_config_path",
                 "adapter_config_path", "model_config_path"]

  for k in expand_keys:
    ret_d[k] = hydra.utils.to_absolute_path(ret_d[k])
    expand_key_name = k.replace("_path", "_expanded")
    with open(ret_d[k], "r") as f:
      key_config = json.load(f)
      ret_d[expand_key_name] = key_config

  return ret_d

@torch.no_grad()
def evaluate(dataloader: DataLoader, model: nn.Module | FSDP | DDP,
             step: int, args: FineTuningConfig) -> dict[str, torch.Tensor]:
  
  losses = []
  precisions = []
  
  model.eval()

  proc_dataloader_len = len(dataloader) // torch.distributed.get_world_size()
  pbar = tqdm.tqdm(range(proc_dataloader_len), desc=f"Step: {step}, evaluating")

  for idx, batch in enumerate(dataloader):
    inputs_ids, output_ids, output_mask, position_ids, _ = to_device(
      batch, torch.cuda.current_device())
    
    # ignore index for entropy
    output_ids[output_mask == 0] = -100

    preds: torch.Tensor = model.forward(inputs_ids, 0, position_ids) # None, None, True) decoding=True

    bsz, seq_len, _ = preds.shape
    pred_tokens = torch.argmax(preds, dim=-1)
    pred_tokens = pred_tokens.view(bsz * seq_len)

    preds = preds.view(bsz * seq_len, -1)
    output_ids = output_ids.view(bsz * seq_len)

    output_mask = output_mask.view(bsz * seq_len)
    valid_preds = pred_tokens[output_mask == 1]
    valid_tokens = output_ids[output_mask == 1]

    loss = F.cross_entropy(preds, output_ids, ignore_index=-100)
    losses.append(loss.item())
    
    valid_preds = pred_tokens[output_mask == 1]   
    precision = torch.sum(valid_preds == valid_tokens) / valid_tokens.numel()

    precisions.append(precision.item())

    if torch.distributed.get_rank() == 0:
      pbar.update(1)

  # perplexity and entropy
  evaluate_entropy = torch.mean(
    torch.tensor(losses, device=f"cuda:{torch.distributed.get_rank()}"))
  evaluate_perplexity = 2 ** evaluate_entropy

  # precision
  precisions = torch.mean(
    torch.tensor(precisions, device=f"cuda:{torch.distributed.get_rank()}")
  )

  # delete tensors to free up memory
  del preds, output_ids, output_mask, valid_preds, valid_tokens
  torch.cuda.empty_cache()

  # move to train mode
  model.train()

  # keep in tensor form 
  metrics = {
    "entropy": evaluate_entropy,
    "perplexity": evaluate_perplexity,
    "precision": precisions,
  }

  # all reduce the metrics
  for k, v in metrics.items():
    torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
    metrics[k] /= torch.distributed.get_world_size()
    metrics[k] = metrics[k].item()

  return metrics


def wandb_init(args: FineTuningConfig):
  if args.wandb:
    wandb.login(key=args.wandb_key)
    name = args.wandb_experiment_name
    wandb.init(project="fsdp-llama-13b-instruct",
               name=name,
               config=args_to_dict(args))


def model_init(args: FineTuningConfig, manager: CheckpointManager) -> Tuple[nn.Module | FSDP | DDP, TokenizerBase]:
  
  # init on all ranks, this is super fast now
  if args.resume_ckpt:
    model, tokenizer = manager.init_model_from_ckpt(f"{args.resume_ckpt}/ckpt.pth", root_path=None)
    return model, tokenizer
  
  model, tokenizer = ModelFactory.from_config_file(args.model_config_path)
  model = apply_adapters(model, args)
  return model, tokenizer


def main(args: FineTuningConfig):

  torch.manual_seed(args.seed)

  setup_dist()
  rank = torch.distributed.get_rank()
  print(f"Rank: {rank}")

  setup_print(rank, debug=args.debug)
  torch.distributed.barrier()

  # initialise the checkpoint manager
  checkpoint_manager = CheckpointManager(args.ckpt_dir)

  # initialise model
  model: nn.Module
  model, tokenizer = model_init(args, checkpoint_manager)

  # get it here because FSDP might tamper with this later on
  layer_wrap_policy = get_layer_wrap_policy(model, args.layer_classes)
  
  # apply compute optimisation if need be
  if isinstance(model, LlmAdapter):
    model.apply_optimization()

  # print number of optimizable parameters
  print(f"Number of optimizable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

  # this comes only before fsdp
  torch.cuda.set_device(f"cuda:{rank}")

  # make the model distributed if need be
  model = wrap_distributed_model(model, args)

  # apply activation checkpointing / offloading
  model = wrap_activation_saving(model, args, layer_wrap_policy)

  print("=======================================")
  print(f"          Model {rank} wrapped         ")
  print("=======================================")

  train_dataset = DatasetFactory.gpt_dataset_from_json(
    tokenizer, args.dataset_config_path,
    process_rank=rank, world_size=torch.distributed.get_world_size()
  )

  train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=get_collate_fn(train_dataset),
    pin_memory=True,
  )

  train_dataloader_iter = iter(train_dataloader)

  test_dataset = DatasetFactory.gpt_datasets_from_json(
    tokenizer, args.test_dataset_config_path,
    process_rank=rank, world_size=torch.distributed.get_world_size(),
  )

  if not isinstance(test_dataset, list):
    test_dataset = [test_dataset]

  test_loaders = []
  for test_ds in test_dataset:
    
    # test_batch_size
    test_dataloader = DataLoader(
      test_ds, batch_size=args.test_batch_size, shuffle=False,
      num_workers=1, collate_fn=get_collate_fn(test_ds),
      pin_memory=True,
    )
    
    test_loaders.append(test_dataloader)

  ref_batch_size = args.ref_batch_size
  gpu_batch_size = args.gpu_batch_size
  steps_per_micro_batch = ref_batch_size // gpu_batch_size

  num_tokens_per_step = gpu_batch_size * \
      train_dataset.max_length * torch.distributed.get_world_size()
  num_training_steps = args.num_training_tokens / num_tokens_per_step
  num_training_steps = int(math.ceil(num_training_steps))

  # this is being used for he scheduler
  # because it defines the total number of optimization steps
  num_optim_steps = int(math.ceil(num_training_steps / steps_per_micro_batch))

  # split params into a group with weight decay and another without
  # this happens inside the factory
  optimizer = setup_optimizer(model=model, args=args)
  opt_steps = 0

  lr_scheduler = setup_lr_scheduler(optimizer, num_optim_steps, args.n_steps_apply_partial_lora, args)
  scaler = setup_scaler(args)

  # load optimizer and scaler from checkpoint
  if args.resume_training:
    checkpoint_manager.reload_optim_scaler_from_ckpt(
      args.resume_ckpt, args.enabled_fsdp, optimizer, lr_scheduler, scaler)

  if args.resume_training:
    start_step = checkpoint_manager.reload_step_from_ckpt(args.resume_ckpt)
  else:
    start_step = 1

  if torch.distributed.get_rank() == 0:
    bar_size = num_training_steps - start_step + 1
    pbar = tqdm.tqdm(range(bar_size))
    wandb_init(args)

  # for iteration purposes
  processed_tokens = (start_step - 1) * num_tokens_per_step
  micro_batch_loss = []
  log_batch_loss = []
  log_batch_token_losses = []
  save_per_token_sample_loss = []
  spike_softened_loss_hist = []

  # for saving purposes
  for step in range(start_step, num_training_steps + 1):

    # start with an eval step to benchmark frozen performance
    if step == 1 and args.eval_interval > 0:
      for idx, test_dataloader in enumerate(test_loaders):
        evaluate_metrics = evaluate(test_dataloader, model, step, args)

        # log the metric
        if torch.distributed.get_rank() == 0:
          for k, v in evaluate_metrics.items():
            if args.wandb:
              wandb.log({f"Test {idx} {k}": v}, step=step)
            print(f"Test {idx} {k}: {v}")

        best_metrics = {k: v for k, v in evaluate_metrics.items()}

    batch = next(train_dataloader_iter)
    inputs_ids, output_ids, output_mask, position_ids, seq_lens = to_device(
      batch, torch.cuda.current_device())

    if len(seq_lens) == 1 and isinstance(seq_lens[0], list):
      # make the list of lists into a singular list
      seq_lens = [item for sublist in seq_lens for item in sublist]

    with torch.no_grad():
      # convert output ids to -100 for padding
      output_ids[output_mask == 0] = -100

    # forward pass
    preds: torch.Tensor = model.forward(inputs_ids, 0, position_ids, seq_lens)

    bsz, seq_len, _ = preds.shape
    preds = preds.view(bsz * seq_len, -1)
    output_ids = output_ids.view(-1)
    
    # compute loss
    unmasked_loss_per_token = F.cross_entropy(preds, output_ids, reduction="none")

    with torch.no_grad():
      # split into samples
      ignore_mask = output_ids == -100
      split_ignore_mask = torch.split(ignore_mask, seq_lens)
      split_output_ids_unmasked = torch.split(output_ids, seq_lens)
      split_loss_per_token_unmasked = torch.split(unmasked_loss_per_token, seq_lens)

      # remove padding tokens for each sample
      split_output_ids = [out_ids[~mask].tolist()
                          for out_ids, mask in zip(split_output_ids_unmasked, split_ignore_mask)]
      split_loss_per_token = [loss[~mask].tolist()
                              for loss, mask in zip(split_loss_per_token_unmasked, split_ignore_mask)]
    
      # save the per token sample loss
      save_per_token_sample_loss = list(zip(split_output_ids, split_loss_per_token))

    # ignore padding tokens
    ignore_mask = output_ids == -100
    loss_per_token = unmasked_loss_per_token[~ignore_mask]

    # prep for logging per token entropy
    log_batch_token_losses.extend(tensor_to_numpy(loss_per_token))

    # soften down loss spikes
    if args.soften_loss_spikes:
      
      @torch.no_grad()
      def compute_soften_loss(x, a, b, c=1.0):
        result = (a - a * (torch.log(x - a + 1) / math.log(b - a + 1))**c) / x

        # lower is identity fn
        result[x < a] = 1
        # higher is 0
        result[x > b] = 0
        return result
      
      if args.apply_per_token_softening:
        loss_scalar = compute_soften_loss(loss_per_token,
                                          a=args.loss_spike_threshold,
                                          b=args.malformed_batch_loss_threshold,
                                          c=args.loss_spike_factor)
        spike_softened_loss = loss_scalar * loss_per_token
        spike_softened_loss = spike_softened_loss.mean()
        spike_softened_loss_hist.append(spike_softened_loss.item())
      else:
        loss = loss_per_token.mean()
        loss_scalar = compute_soften_loss(loss,
                                          a=args.loss_spike_threshold,
                                          b=args.malformed_batch_loss_threshold,
                                          c=args.loss_spike_factor)
        spike_softened_loss = loss_scalar * loss
    else:
      spike_softened_loss = loss

    scaled_loss = spike_softened_loss / steps_per_micro_batch

    # metric aggregation
    micro_batch_loss.append(spike_softened_loss)
    log_batch_loss.append(loss_per_token.mean())

    if scaler is not None:
      scaler.scale(scaled_loss).backward()
    else:
      scaled_loss.backward()

    if step % steps_per_micro_batch == 0:
      if args.clip_grad_norm > 0:
        if scaler is not None:
          scaler.unscale_(optimizer)

        if isinstance(model, FSDP):
          if sharding_policy(args.sharding_policy) == ShardingStrategy.NO_SHARD:
            torch.nn.utils.clip_grad_norm_(
              model.module.parameters(), args.clip_grad_norm)
          else:
            model.clip_grad_norm_(args.clip_grad_norm)

      # make sure we don't take into account malformed batches, with perplexity > 100
      if args.skip_malformed_batches:
        with torch.no_grad():
          step_mask = torch.stack(micro_batch_loss)
          step_mask = step_mask > args.malformed_batch_loss_threshold
          torch.distributed.all_reduce(step_mask, op=torch.distributed.ReduceOp.SUM)
          if step_mask.sum() > 0:
            malformed_batch = True
          else:
            malformed_batch = False

      malformed_batch = malformed_batch and args.skip_malformed_batches

      if not malformed_batch:
        if scaler is not None:
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_steps += 1
      
        # put this here because we want to make sure we don't do this
        # when staling on malformed batches
        if opt_steps % args.n_steps_apply_partial_lora == 0:
          if isinstance(model, FSDP):
            with FSDP.summon_full_params(model):
              model.module.partial_merge_adaptation(decay=args.partial_lora_decay)
              optimizer.state = collections.defaultdict(dict)

        # step scheduler only when micro batch is done
        lr_scheduler.step()
      
      else:
        # reset gradients when finding malformed batches
        optimizer.zero_grad(set_to_none=True)
        if torch.distributed.get_rank() == 0:
          print("Warning: malformed batch detected. Skipping step")

      micro_batch_loss = []

    processed_tokens += num_tokens_per_step

    with torch.no_grad():
      if step % args.log_interval == 0:
        logging_loss = torch.stack(log_batch_loss).mean().item()
        logging_ppl = 2 ** logging_loss

        if torch.distributed.get_rank() == 0:
          if args.wandb:
            wandb.log(
              {"Batch Perplexity": logging_ppl,
               "Batch Entropy": logging_loss,
               "Learning Rate": optimizer.param_groups[0]["lr"],
               "Tokens": processed_tokens}, step=step
            )
            wandb.log({"TokenEntropyHist": wandb.Histogram(log_batch_token_losses)}, step=step)
            wandb.log({"SpikeTokenEntropyHist": wandb.Histogram(spike_softened_loss_hist)}, step=step)
          
          if args.save_per_token_loss:
            # save token losses per sample
            per_sample_token_loss_df = pd.DataFrame(save_per_token_sample_loss, columns=["token_ids", "losses"])
            
            # make dir if it doesn't exist
            token_loss_dir_path = f"token_entr/batch_losses"
            if not os.path.exists("token_entr/batch_losses"):
              dir_path_tokens = token_loss_dir_path.split("/")
              for i in range(1, len(dir_path_tokens)):
                dir_path = "/".join(dir_path_tokens[:i])
                if not os.path.exists(dir_path):
                  os.mkdir(dir_path)
                   
            run_name = get_run_name(args)
            pickle_path = f"{token_loss_dir_path}/{run_name}_{step}.pkl"
            per_sample_token_loss_df.to_pickle(pickle_path)
          
          print(
            f"Step: {step}, Batch Entropy: {logging_loss:.4f},"
            f"Batch Perpplexity: {logging_ppl:.4f}, Tokens: {processed_tokens}"
          )

      # reinit buffer
      log_batch_loss = []
      log_batch_token_losses = []
      save_per_token_sample_loss = []
      spike_softened_loss_hist = []

    # checkpointing happens only at evaluation time
    if args.eval_interval > 0 and step % args.eval_interval == 0:
      for idx, test_dataloader in enumerate(test_loaders):
        evaluate_metrics = evaluate(test_dataloader, model, step, args)

        # log the metric
        if torch.distributed.get_rank() == 0:
          for k, v in evaluate_metrics.items():
            if args.wandb:
              wandb.log({f"Test {idx} {k}": v}, step=step)
            print(f"Test {idx} {k}: {v}")

        metric_op_map = {
          "entropy": ComparisonOp.LT,
          "perplexity": ComparisonOp.LT,
          "precision": ComparisonOp.GT,
        }

        has_improved_metrics = compare_metric_dicts(evaluate_metrics, best_metrics, metric_op_map)
        if has_improved_metrics:
          best_metrics = get_improved_metrics(evaluate_metrics, best_metrics, metric_op_map)
          checkpoint_manager.save(
            model, optimizer, lr_scheduler, scaler, step, evaluate_metrics, args)

    if torch.distributed.get_rank() == 0:
      pbar.update(1)

  # one final evaluation
  evaluate_metrics = evaluate(test_dataloader, model, step, args)

  # one final save
  checkpoint_manager.save(model, optimizer, lr_scheduler, scaler, step, evaluate_metrics, args)

  # cleanup
  cleanup_dist()


@hydra.main(version_base=None, config_path="hydra_configs", config_name="partial_lora_opt_finetuning")
def hydra_main(cfg: DictConfig):
  args = FineTuningConfig(**cfg)
  main(args)


if __name__ == "__main__":
  hydra_main()
