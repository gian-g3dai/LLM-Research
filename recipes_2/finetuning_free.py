from collections import defaultdict
import os
import pickle
import random
import string
import hydra
import torch
import math

import torch.distributed
from unakin.llm._data.datasources.train_sample import LMBatchMetadata, LMTrainingBatch
from unakin.llm.models.llama.model_compile import Transformer
from unakin.llm.kernels.fused_proj import FusedLossArgs, FusedCrossEntropyOutput
import wandb
import numpy as np
import xformers.ops as xops
import math

# TODO: DELETE imports when done

from torch import nn
from tqdm import tqdm
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from typing import Any, List, Optional, Tuple


from recipes_2.base_model_setup import load_base_model, parse_base_model_config, BaseModelConfig
from recipes_2.data_setup import load_data, DataConfig, Data
from recipes_2.adapter_setup import AdaptationStrategy, AdapterConfig 
from recipes_2.optimizer_setup import setup_optimizer, OptimizerConfig
from recipes_2.scheduler_setup import setup_lr_scheduler, setup_scaler
from recipes_2.parallelization_setups import __TYPE_FSDP_UNION__, ParallelStrategy, ParallelizationConfig, grad_sync_context_manager, parallelize_model
from recipes_2.checkpointing_setup import CheckpointingConfig, ResumeConfig
from recipes_2.logging_setup import init_wandb, log_stats
from recipes_2.finetuning_setup import FineTuningConfig
# from recipes_2.efficient_fce import FusedProjectionPlusCrossEntropyLoss


from torch.utils.data import DataLoader, Dataset
from unakin.llm._data.dataset_factory import DatasetFactory
from unakin.llm._data.code_dataset import GPTCodeDataset
from unakin.llm.tokenizers import LlamaTokenizer, Llama3Tokenizer, TokenizerBase
from unakin.llm.distributed.utils import setup_async_distributed_process_group, print_rank0, setup_print

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from flash_attn.losses.cross_entropy import CrossEntropyLoss



# from unakin.llm.adapters.llm_adapter_new import fuse_adaptations
import torch.distributed.checkpoint as dcp

class OnlineTrainState:
  def __init__(
    self,
    steps: int = 0,
    current_optimization_steps: int = 0,
    target_optimization_steps: int = 0,
    tokens_per_world_step: int = 0
  ):
    self.steps = steps
    self.optimization_steps = current_optimization_steps
    self.target_optimization_steps = target_optimization_steps
    self.tokens_per_world_step = tokens_per_world_step

    self.current_optimization_steps = current_optimization_steps
    self.current_tokens = 0
  
@dataclass
class ExperimentState:
  args: 'FineTuningConfig'
  saved_train_states: dict[str, OnlineTrainState]  

class CheckpointManager:
  def __init__(self, config: CheckpointingConfig = None, experiment_state: ExperimentState = None):
    self.config = config
    self.run_name = config.run_name if config is not None else None
    self.save_dir = config.save_dir if config is not None else None
    self.experiment_state = experiment_state

    # async only management
    self._m_future = None
    self._o_future = None

  def __DEFAULT_EXPERIMET_PATH__(self, prefix: str = None):
    prefix = prefix or self.base_path
    return f"{self.base_path}/experiment_state.pkl"

  def __DEFAULT_TRAIN_STATE_PATH__(self, checkpoint_name: str, prefix: str = None):
    prefix = prefix or self.base_path
    return f"{prefix}/train_state/{checkpoint_name}.pkl"

  def __DEFAULT_MODEL_PATH__(self, checkpoint_name: str, prefix: str = None):
    prefix = prefix or self.base_path
    return f"{prefix}/{checkpoint_name}"
  
  def __DEFAULT_OPTIMIZER_PATH__(self, checkpoint_name: str, prefix: str = None):
    prefix = prefix or self.base_path
    return f"{prefix}/{checkpoint_name}_optimizer"

  def _should_model_invoke_future(self):
    return self.config.async_save and self._m_future is not None
  
  def _should_optimizer_invoke_future(self):
    return self.config.async_save and self._o_future is not None    
    
  def save_experiment_state(self):
    with open(f"{self.base_path}/experiment_state.pkl", "wb") as f:
      pickle.dump(self.experiment_state, f)

  def save_train_state(self, train_state: OnlineTrainState, checkpoint_name: str):
    if not os.path.exists(f"{self.base_path}/train_state"):
      os.makedirs(f"{self.base_path}/train_state")
    
    with open(self.__DEFAULT_TRAIN_STATE_PATH__(checkpoint_name), "wb") as f:
      pickle.dump(train_state, f)

  def load_train_state(self, checkpoint_name: str, prefix: str = None) -> ExperimentState:
    if not prefix and  checkpoint_name not in self.experiment_state.saved_train_states.keys():
      raise ValueError(f"Checkpoint {checkpoint_name} not found in {self.experiment_state.saved_train_states.keys()}")

    with open(self.__DEFAULT_TRAIN_STATE_PATH__(checkpoint_name, prefix), "rb") as f:
      return pickle.load(f)
    

  @property
  def base_path(self):
    if self.run_name is None:
      # random alphanumeric string
      self.run_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
      if os.path.exists(f"{self.save_dir}/{self.run_name}"):
        raise ValueError(
          f"Random run-name {self.run_name} already exists in {self.save_dir}. \
            Please provide an explicit name in the configuration or run again."
        )

    base_path = f"{self.save_dir}/{self.run_name}" if self.run_name else self.save_dir
    
    if not os.path.exists(base_path):
      os.makedirs(base_path)
    
    return base_path
  
  def save_model_checkpoint(self,
                            checkpoint_name: str,
                            model: nn.Module,
                            optimizer: torch.optim.Optimizer,                            
                            train_state: OnlineTrainState = None,
                            train_args: 'FineTuningConfig' = None) -> bool:
    
    # set the base training args to this training state
    if self.experiment_state.args is None:
      self.experiment_state.args = train_args

    # update the available online states for this training state
    if train_state is not None:
      self.experiment_state.saved_train_states[checkpoint_name] = train_state

    # handle sync and async ops
    save_method = dcp.save if not self.config.async_save else dcp.async_save

    # save the model
    m_fs_storage_writer = dcp.FileSystemWriter(self.__DEFAULT_MODEL_PATH__(checkpoint_name))
    o_fs_storage_writer = dcp.FileSystemWriter(self.__DEFAULT_OPTIMIZER_PATH__(checkpoint_name))

    # this is cross-compatible with async and sync, because the return signature
    # of the non async `dcp.save` is a `None` object, therefore the if branch
    # will never be executed
    if self._should_model_invoke_future():
      self._m_future = self._m_future.result() 
    self._m_future   = save_method(state_dict=model.state_dict(), storage_writer=m_fs_storage_writer)

    if self._should_optimizer_invoke_future():
      self._o_future = self._o_future.result()
    self._o_future = save_method(state_dict=optimizer.state_dict(), storage_writer=o_fs_storage_writer)

    # save the scheduler
    # TODO: temporary solution
    self.save_train_state(self.experiment_state.saved_train_states, checkpoint_name)
    self.save_experiment_state()

    return True
  
  @staticmethod
  def _dcp_load_state_dict(obj_with_state_dict: nn.Module | torch.optim.Optimizer, path: str) -> nn.Module | torch.optim.Optimizer:
    state_dict = obj_with_state_dict.state_dict()
    fs_storage_reader = dcp.FileSystemReader(path)
    dcp.load(storage_reader=fs_storage_reader, state_dict=state_dict)
    obj_with_state_dict.load_state_dict(state_dict)
    return obj_with_state_dict
  
  @staticmethod
  def _load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
      return pickle.load(f)

  def load_model_checkpoint(self, checkpoint: str, model: nn.Module, prefix: str = None) -> nn.Module | None:
    if prefix and checkpoint not in self.experiment_state.saved_train_states.keys():
      raise ValueError(f"Checkpoint {checkpoint} not found in {self.experiment_state.saved_train_states.keys()}")
    
    # TODO: Meta magic code
    model = self._dcp_load_state_dict(model, self.__DEFAULT_MODEL_PATH__(checkpoint, prefix=prefix))
    return model

  def load_train_resume_checkpoint(self,
                        experiment_root: str,
                        checkpoint: str,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer) -> Tuple[nn.Module, torch.optim.Optimizer, OnlineTrainState]:

    # TODO: Meta magic code
    model = self._dcp_load_state_dict(model, self.__DEFAULT_MODEL_PATH__(checkpoint, prefix=experiment_root))
    optimizer = self._dcp_load_state_dict(optimizer, self.__DEFAULT_OPTIMIZER_PATH__(checkpoint, prefix=experiment_root))    

    # load the train state
    train_state = self.load_train_state(checkpoint, prefix=experiment_root)
    return model, optimizer, train_state
  
  def setup_model_optim_state_from_checkpoint(self,
                                  experiment_root: str,
                                  checkpoint: str,
                                  parallel_args: ParallelizationConfig | None = None,
                                  ) -> Tuple[nn.Module, torch.optim.Optimizer, OnlineTrainState]:
    base_experiment: ExperimentState = self._load_pkl(self.__DEFAULT_EXPERIMET_PATH__(experiment_root))
    model = load_and_adapt_model(base_experiment.args, parallelize=False)
    
    # apply parallelization at this stage
    # to ensure that the model is parallelized
    if parallel_args is not None:
      model = parallelize_model(model, parallel_args)

    optimizer = setup_optimizer(model, base_experiment.args.optimizer_args)
    model, optimizer, train_state = self.load_train_resume_checkpoint(
      experiment_root, checkpoint, model, optimizer,
    )
    return model, optimizer, train_state

  def setup_model_from_checkpoint(self,
                                  experiment_root: str,
                                  checkpoint: str,
                                  parallel_args: ParallelizationConfig | None = None,
                                  ) -> nn.Module:
    experiment: ExperimentState = self._load_pkl(self.__DEFAULT_EXPERIMET_PATH__(experiment_root))
    model = load_and_adapt_model(experiment.args, parallelize=False)
    if parallel_args is not None:
      model = parallelize_model(model, parallel_args)
    model = self.load_model_checkpoint(checkpoint, model, prefix=experiment_root)
    return model

def apply_adapter_to_model(model: nn.Module, args: FineTuningConfig):
  # rank0_only case
  if model is None:
    return None
  
  # if we have adaptation policies, freeze the model
  # as most likely we're not interested in performing a full-finetune
  if args.adapter_args.strategy != AdaptationStrategy.FULL_FT:

    for name, param in model.named_parameters():
      param.requires_grad_(False)
  
  # unfreeze policies will be applied here
  for policy in args.adapter_args.policies:
    model = policy.adapt(model)
  
  return model

# TODO: maybe change the look of this
def load_and_adapt_model(args: FineTuningConfig, parallelize: bool=True) -> nn.Module | __TYPE_FSDP_UNION__:
  # 1. Load the base model
  base_model = load_base_model(
    args.base_model_args.model.base,
    args.load_rank_0_only,
    args.base_model_args.base_path,
    args.data_args.train_config["dataset"]["max_length"])
    
  model_dtype = args.parallel_args.fsdp_settings.mixed_precision.value().param_dtype

  # force the base model to apply the fused loss
  base_model = base_model.apply_fused_loss()
  base_model = base_model.to(dtype=model_dtype)
  
  # 2. Setup the adapters
  adapted_model = apply_adapter_to_model(base_model, args)
  adapted_model = adapted_model.to(torch.cuda.current_device())

  # 3. Setup the parallelization
  if parallelize:
    adapted_parallel_model = parallelize_model(
      adapted_model,
      parallel_config=args.parallel_args
    )
    return adapted_parallel_model
  return adapted_model

def load_tokenizer(args: FineTuningConfig):
  if args.base_model_args.tokenizer.type == "llama":
    tokenizer = LlamaTokenizer(model_path=args.base_model_args.tokenizer.path)
  elif args.base_model_args.tokenizer.type == "llama3":
    tokenizer = Llama3Tokenizer(model_path=args.base_model_args.tokenizer.path)
  else:
    raise ValueError(f"Tokenizer type {args.base_model_args.tokenizer.type} not supported")
  return tokenizer

@torch.no_grad
def tensor_mask_sum_acc(
    values: torch.Tensor,
    mask: torch.Tensor,
):
  values = values.squeeze(0)
  mask = mask.squeeze(0)

  # Initialize tensors for sum and count in float32 to prevent overflow.
  # for 256K sequences, we need 8MB of memory per tensor
  # therefore a masked tensor requires around 16MB of memory for 256K sequences.
  # 6 such constructs will require around 100MBs of memory
  sum_vals = torch.zeros(mask.max() + 1, device=values.device, dtype=values.dtype)
  counts = torch.zeros(mask.max() + 1, device=values.device, dtype=values.dtype)

  # Use torch.scatter_add() to sum values based on mask positions
  sum_vals.scatter_add_(0, mask, values)
  # Count occurrences of each position in the mask
  counts.scatter_add_(0, mask, torch.ones_like(values, device=values.device, dtype=values.dtype))

  # print_rank0(sum_vals)
  # print_rank0(counts)

  return {
    "tensor": sum_vals.cpu(),
    "mask": counts.cpu()
  }

@torch.no_grad
def process_train_logging_metrics(
  batch: LMTrainingBatch,
  preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  batch_meta_data: LMBatchMetadata,
  metric_prefix: str = "train"
) -> dict:
  
  # preds = (bwd_loss, reporting_loss, precision)
  _, loss, precision = preds
  assert loss.ndim > 0, f"Loss tensor must have at least one dimension, got {loss.ndim}"
  assert precision.ndim > 0, f"Precision tensor must have at least one dimension, got {precision.ndim}"

  # truncate 
  batch_len = batch_meta_data.processed_tokens - batch_meta_data.padded_tokens

  # only trainable tokens
  mask = batch.training_mask[0].bool()
  mask = mask[:batch_len]

  # trim from predictions as well
  loss = loss[:batch_len][mask]
  precision = precision[:batch_len][mask]

  # meta-data for logging
  positional_ids = batch.positional_ids[0][:batch_len][mask]
  dataset_ids = batch.datasource_indices[0][:batch_len][mask]
  out_tokens = batch.output_tokens[0][:batch_len][mask]

  # edge case
  if out_tokens.numel() == 0:
    return {}

  # helper function
  def PREFIX_VAR(name: str) -> str:
    return f"{metric_prefix}_{name}"

  # perplexity and precision
  precision_per_token = precision

  # metric-per-position in sequence
  pos_ppl_per_token       = tensor_mask_sum_acc(loss, positional_ids)
  pos_precision_per_token = tensor_mask_sum_acc(precision_per_token, positional_ids)

  # metric-per-dataset source
  ds_ppl_per_token        = tensor_mask_sum_acc(loss, dataset_ids)
  ds_precision_per_token  = tensor_mask_sum_acc(precision_per_token, dataset_ids)

  # make everything a dict
  metrics = {
    PREFIX_VAR("perplexity"):  2 ** (loss.mean()),
    PREFIX_VAR("precision"): precision_per_token.mean(),

    PREFIX_VAR("pos_perplexity"): pos_ppl_per_token,
    PREFIX_VAR("pos_precision"): pos_precision_per_token,

    PREFIX_VAR("ds_perplexity"): ds_ppl_per_token,
    PREFIX_VAR("ds_precision"): ds_precision_per_token,
  }

  return metrics

# HELPER FUNCTIONS
def group_by_keys(dicts: List[dict]) -> dict:
  results = defaultdict(list)
  for d in dicts:
    for key, value in d.items():
      results[key].append(value)
  return results

# HELPER FUNCTIONS
def group_by_keys_recursive(dicts: List[dict], results = None) -> dict:
  if results is None:
    results = {}

  for d in dicts:
    for key, value in d.items():
      if isinstance(value, dict):
        results[key] = group_by_keys_recursive([value], results.get(key, None))
      else:
        if key in results:
          results[key].append(value)
        else:
          results[key] = [value]

  return results

# HELPER FUNCTIONS
def is_masked_tensor_dict(d: Any) -> bool:
  return "mask" in d and "tensor" in d

def fwd_bwd_block_chunk_ce(
  model: Transformer,
  batch: LMTrainingBatch,
  tokenizer: TokenizerBase,
  ds_names: List[str],
  block_size: int = 4096,
  training: bool = True,
  accumulation_steps: int = 1  
) -> dict:
  
  batch_metadata = LMBatchMetadata.from_training_batch(
    batch, tokenizer, ds_names
  )

  # batch = batch_metadata.trim_padding(batch)
  batch.output_tokens[batch.training_mask == 0] = -100
  labels = batch.output_tokens

  attn_mask = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
    batch.sequence_lengths[0])

  fused_loss_args = FusedLossArgs(
    labels=labels, ignore_index=-100, reduction="none", block_size=block_size)
  
  # fused_loss_outs = (bwd_loss, reporting_loss, precision)
  fused_loss_outs = model.forward(
    tokens=batch.input_tokens,
    positions=batch.positional_ids,
    mask=attn_mask,
    loss_args=fused_loss_args
  )

  # compute stats before backward because of inplace operations  
  reporting_stats = process_train_logging_metrics(
    batch=batch,
    preds=fused_loss_outs,
    batch_meta_data=batch_metadata,
    metric_prefix="train" if training else "test"
  )

  if training:
    bwd_loss = fused_loss_outs[0].mean()
    bwd_loss = bwd_loss / accumulation_steps
    bwd_loss.backward()

  return reporting_stats


@torch.no_grad
def reduce_multiple_stats_steps(stats: List[dict]) -> dict:

  device = torch.device("cpu")
  # this is going to be a dict with multiple keys
  flat_stats = group_by_keys_recursive(stats)
  for k, v in flat_stats.items():
      if is_masked_tensor_dict(v):
        v["mask"] = torch.nested.as_nested_tensor([vt.to(device) for vt in v["mask"]])
        v["tensor"] = torch.nested.as_nested_tensor([vt.to(device) for vt in v["tensor"]])

  # prepare everything for global-reduce
  for k, v in flat_stats.items():
    if is_masked_tensor_dict(v):
      v["mask"] = torch.nested.to_padded_tensor(v["mask"], padding=0)
      v["tensor"] = torch.nested.to_padded_tensor(v["tensor"], padding=0)
    
      v["mask"] = v["mask"].sum(dim=0)      
      v["tensor"] = v["tensor"].sum(dim=0)
      # v["mean"] = v["tensor"] / v["mask"]
    else:
      flat_stats[k] = torch.stack([vt.to(device) for vt in v]).mean()
  
  return flat_stats


def train_accumulate_n_steps(
  n_steps: int,
  model: Transformer,
  data: Data,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
  train_state: OnlineTrainState,  
  args: FineTuningConfig,  
):
  
  stats = []  
  no_sync_grad = (
    args.parallel_args.has_cpu_offload() and
    args.hyper_args.no_sync_grad
  )

  # forward and backward step
  def fwd_bwd_step():
    batch, sample_cnt = next(loader)
    batch = batch.to_device(f"cuda:{torch.distributed.get_rank()}")
    step_stats = fwd_bwd_block_chunk_ce(
      model=model, batch=batch, tokenizer=data.tokenizer,
      ds_names=data.datasources_names, training=True,
      accumulation_steps=n_steps
    )
    train_state.current_tokens += train_state.tokens_per_world_step
    train_state.steps += 1
    return step_stats, sample_cnt

  # compatibility with no_sync_grad
  # N-1 steps happen with no_sync_grad enabled
  # last steps happens with gradient sync to collect gradient information
  # across all ranks
  with grad_sync_context_manager(model, enabled=no_sync_grad):
    for step in range(n_steps - 1):
      step_stats, sample_cnt = fwd_bwd_step()
      stats.append(step_stats)

  step_stats, sample_cnt = fwd_bwd_step()    
  stats.append(step_stats)

  stats = reduce_multiple_stats_steps(stats)
  train_state.current_tokens += train_state.tokens_per_world_step * n_steps

  torch.distributed.barrier()

  # stats gather and reduce
  if torch.distributed.get_world_size() > 1:
    acc = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(acc, stats)
    stats = reduce_multiple_stats_steps(acc)

  # wandb logging  
  if args.logging_args.wandb and train_state.steps % args.logging_args.log_interval == 0:
    log_stats(stats, data, train_state, sample_cnt=sample_cnt)

  # gradient clipping
  if args.parallel_args.strategy.value == "FSDP_1":
    model.clip_grad_norm_(args.hyper_args.clip_grad_norm)
  else:
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=args.hyper_args.clip_grad_norm)

  optimizer.step()
  optimizer.zero_grad()
  lr_scheduler.step()
  train_state.optimization_steps += 1
  return step_stats
  
@torch.no_grad
def evaluate_loader(
  model: Transformer,
  data: Data,
  train_state: OnlineTrainState,
  args: FineTuningConfig
):
  
  RANK, WOLRD_SIZE = torch.distributed.get_rank(), torch.distributed.get_world_size()

  pbar = tqdm(
    range(len(data.dataloader) // WOLRD_SIZE), desc=f"Step: {train_state.steps}, evaluating")
  loader = iter(data.dataloader)

  model.eval()
  stats = []

  curr_idx = 0
  while True:
    try:
      batch, _ = next(loader)
      curr_idx += 1
      local_done  = False
    except StopIteration:
      local_done = True

    global_done = torch.tensor(local_done).to(f"cuda:{RANK}", dtype=torch.float)
    torch.distributed.all_reduce(global_done, op=torch.distributed.ReduceOp.SUM)
    if global_done == WOLRD_SIZE:
      break

    batch = batch.to_device(f"cuda:{torch.distributed.get_rank()}")
    step_stats = fwd_bwd_block_chunk_ce(
      model=model, batch=batch, tokenizer=data.tokenizer,
      ds_names=data.datasources_names, training=False
    )
    if not local_done:
      stats.append(step_stats)
      if torch.distributed.get_rank() == 0:
        pbar.update(len(batch.sequence_lengths[0]))

  del batch
  torch.distributed.barrier()
  model.train()

  # reduce stats from each batch
  stats = reduce_multiple_stats_steps(stats)

  # gather stats from all workers
  if torch.distributed.get_world_size() > 1:
    acc = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(acc, stats)
    stats = reduce_multiple_stats_steps(acc)

  # log the stats to wandb
  if args.logging_args.wandb:
    log_stats(stats, data, train_state)

  return stats

def compute_batch_optimizer_schedule(
  args: FineTuningConfig, data: Data
) -> Tuple[int, int, int]:
  """
  This function will return how many total optimizer steps we have to go
  through for this experiment, and how many gradient accumulation steps
  each optimizer step will have to go through. 
  
  This is much simpler for our API, as we can clearly separate the two
  and not rely on calculations. The actual OnlineTrainingState will keep
  track of how many dataset sampling steps we've went through for experiment
  resuming purposes.
  """

  WORLD_SIZE = torch.distributed.get_world_size()
  if args.hyper_args.batch_size_in_tokens:
    ref_batch_size = args.hyper_args.batch_size
    cur_batch_size = data.dataset.max_length * WORLD_SIZE
    tokens_per_batch = cur_batch_size
  else:
    # always batch-size 1 because of returning only
    # one sequence at a time from the dataset
    ref_batch_size = args.hyper_args.batch_size * WORLD_SIZE
    cur_batch_size = 1 * WORLD_SIZE
    # this is the upper bound of the tokens per batch
    # make sure you are taking this into account when 
    # when writting configurations
    # TODO: rewrite this to accept steps in batches as well
    #       for DPO / PPO
    tokens_per_batch = ref_batch_size * data.dataset.max_length

  # 1-extra step for the remainder
  total_optimization_steps = int(math.ceil(
    args.hyper_args.total_training_tokens / tokens_per_batch
  ))
  grad_accum_steps = ref_batch_size // cur_batch_size
  return total_optimization_steps, grad_accum_steps, tokens_per_batch

def main(args: FineTuningConfig):

  # move this to init_distributed(seed)
  torch.random.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  setup_async_distributed_process_group()

  rank = torch.distributed.get_rank()
  # setup_print(rank)
  torch.cuda.set_device(rank)
  # torch.set_default_device(torch.device(f"cuda:{rank}"))

  tokenizer = load_tokenizer(args)
  train_data, test_data = load_data(args=args, tokenizer=tokenizer)

  training_steps, accumulation_steps, tokens_per_step = compute_batch_optimizer_schedule(
    args, train_data)

  print_rank0(f"Training steps {training_steps}, Acc steps {accumulation_steps}, tokens per step {tokens_per_step}")

  # find closest multiple of accumulation steps for log_every_n_steps
  # we do this because we only reduce metrics every accumulation steps
  # because of 
  log_every_n_steps = args.logging_args.log_interval
  log_every_n_steps = 1 if (accumulation_steps == 0 or accumulation_steps > log_every_n_steps) else \
                      min(log_every_n_steps - (log_every_n_steps % accumulation_steps), 1)
  args.logging_args.log_interval = log_every_n_steps
  train_loader = iter(train_data.dataloader)

  # current_state = OnlineTrainState()
  experiment_state = ExperimentState(args=args, saved_train_states={})
  # TODO: The arguments should be modified check experiment state
  checkpoint_manager = CheckpointManager(args.checkpointing_args, experiment_state=experiment_state)

  if args.resume_args is None:
    fsdp_model = load_and_adapt_model(args, parallelize=True)
    optimizer = setup_optimizer(fsdp_model, args.optimizer_args)
    train_state = OnlineTrainState(
      target_optimization_steps=training_steps,
      tokens_per_world_step=tokens_per_step)
    
    # print_rank0(fsdp_model)
  else:
    # resume from the previous checkpoint definition
    if args.resume_args.weights_only:
      model = checkpoint_manager.setup_model_from_checkpoint(
        args.resume_args.resume_experiment_root,
        args.resume_args.resume_checkpoint,
        parallel_args=args.parallel_args
      )
    else:
      model, optimizer, train_state = checkpoint_manager.setup_model_optim_state_from_checkpoint(
        args.resume_args.resume_experiment_root,
        args.resume_args.resume_checkpoint,
        parallel_args=args.parallel_args
      )

  lr_scheduler = setup_lr_scheduler(optimizer, training_steps, args)

  # mambo-jambo for keeping track of when we should evaluate
  world_steps_eval = 0

  if torch.distributed.get_rank() == 0:
    init_wandb(args=args)

  # always kick off with an evaluation for reference purposes
  evaluate_loader(fsdp_model, test_data, train_state, args)

  world_pbar = tqdm(total=training_steps, desc=f"Step: {train_state.optimization_steps}, training")
  while train_state.optimization_steps < training_steps:
    # accumulation_steps = train_state.get(n_accumulation_steps)
    train_accumulate_n_steps(
      n_steps=accumulation_steps,
      model=fsdp_model,
      data=train_data,
      loader=train_loader,
      optimizer=optimizer,
      lr_scheduler=lr_scheduler,
      train_state=train_state,
      args=args
    )

    world_steps_eval += accumulation_steps # NOTE: too many eval 
    if rank == 0:
      world_pbar.update(1)

    should_evaluate, _ = divmod(world_steps_eval, args.hyper_args.eval_interval)

    if should_evaluate:
      evaluate_loader(fsdp_model, test_data, train_state, args)
      world_steps_eval = 0 

      # save the model checkpoint
      checkpoint_manager.save_model_checkpoint(
        checkpoint_name=f"checkpoint_{train_state.steps}",
        model=fsdp_model,
        optimizer=optimizer,
        train_state=train_state,
        train_args=args
      )

@hydra.main(version_base=None, config_path="configs", config_name="base")
def hydra_main(cfg: DictConfig):
  args = FineTuningConfig(**cfg)
  main(args)

if __name__ == "__main__":
  hydra_main()


# torchrun --nproc-per-node=8 finetuning.py +adapter_args=lora +base_model_args=llama3_8b +parallel_args=fsdp_1 +optimizer_args=adamw +data_args=sets +checkpointing_args=default +logging_args=wandb +hyper_args=old