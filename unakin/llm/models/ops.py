from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Tuple
import torch
import torch.distributed
from unakin.llm._data.datasources.train_sample import LMBatchMetadata
from unakin.llm.kernels.cross_entropy import (
  fast_cross_entropy_loss
)

from unakin.llm._data.code_dataset import (
  LMTrainingBatch,
)

from unakin.llm.models.llama.model_compile import (
  CODE_LLAMA_7B_WEIGHTS,
  Transformer,
)

from unakin.llm.metrics.utils import to_metric_value, aggregate_dictionaries, flatten_dict, partition_by_index_mask
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import torch.distributed as dist

EOS_ID = 0


@torch.inference_mode
def tensor_mask_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
):
  # Initialize tensors for sum and count
  sum_vals = torch.zeros(mask.max() + 1)
  counts = torch.zeros(mask.max() + 1)

  # Use torch.scatter_add() to sum values based on mask positions
  sum_vals.scatter_add_(0, mask, values)
  # Count occurrences of each position in the mask
  # TODO: remember there's a 1 here
  counts.scatter_add_(0, mask, torch.ones_like(values))

  return {
    "tensor": sum_vals,
    "mask": counts
  }

@torch.inference_mode
def process_train_logging_metrics(
  batch: LMTrainingBatch,
  preds: torch.Tensor,
  loss:  torch.Tensor,
  batch_meta_data: LMBatchMetadata,
  metric_prefix: str = "train"
) -> dict:
  
  # truncate 
  batch_len = batch_meta_data.processed_tokens - batch_meta_data.padded_tokens

  loss = loss[:batch_len]
  preds = loss[:batch_len]

  def PREFIX_VAR(name: str) -> str:
    return f"{metric_prefix}_{name}"

  ppl_per_token = 2 ** loss
  variance_per_token = loss.var(dim=1)
  precision_per_token = preds.argmax(dim=-1) == batch.output_tokens
  
  pos_ppl_per_token       = tensor_mask_mean(ppl_per_token, batch.positional_ids)
  pos_variance_per_token  = tensor_mask_mean(variance_per_token, batch.positional_ids)
  pos_precision_per_token = tensor_mask_mean(precision_per_token, batch.positional_ids)

  ds_ppl_per_token        = tensor_mask_mean(ppl_per_token, batch.datasource_indices)
  ds_variance_per_token   = tensor_mask_mean(variance_per_token, batch.datasource_indices)
  ds_precision_per_token  = tensor_mask_mean(precision_per_token, batch.datasource_indices)

  metrics = {
    PREFIX_VAR("perplexity"): ppl_per_token.mean(),
    PREFIX_VAR("variance"): variance_per_token.mean(),
    PREFIX_VAR("precision"): precision_per_token.mean(),

    PREFIX_VAR("pos_perplexity"): pos_ppl_per_token,
    PREFIX_VAR("pos_variance"): pos_variance_per_token,
    PREFIX_VAR("pos_precision"): pos_precision_per_token,

    PREFIX_VAR("ds_perplexity"): ds_ppl_per_token,
    PREFIX_VAR("ds_variance"): ds_variance_per_token,
    PREFIX_VAR("ds_precision"): ds_precision_per_token,
  }

  return metrics

  # # Example of dict aggregation
  # if torch.distributed.get_rank() == 0:
  #   tensor_a = torch.tensor([1, 2, 3], device=f"cuda:0")
  #   mydict = {
  #     "a": tensor_a
  #   }
  # if torch.distributed.get_rank() == 1:
  #   tensor_a = torch.tensor([3, 4, 5], device=f"cuda:0")
  #   mydict = {
  #     "a": tensor_a,
  #     "b": tensor_a
  #   }
  # torch.distributed.barrier()
  # acc = [None] * 2
  # torch.distributed.all_gather_object(acc, mydict)

def group_by_keys(dicts: List[dict]) -> dict:
  results = defaultdict(list)
  for d in dicts:
    for key, value in d.items():
      results[key].append(value)
  return results

def group_by_keys_recursive(dicts: List[dict], results = {}) -> dict:
  for d in dicts:
    for key, value in d.items():
      if isinstance(value, dict):
        results[key] = group_by_keys_recursive([value], results.get(key, {}))
      else:
        results[key] = results.get(key, []) + [value]
  return results

def is_masked_tensor_dict(d: Any) -> bool:
  return "mask" in d and "tensor" in d

def fwd_bwd_block_chunk_ce(
  model: Transformer,
  cross_entropy_loss: CrossEntropyLoss,
  batch: LMTrainingBatch,
  block_size: int = 4096,
  training: bool = True,  
) -> None:

  RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()

  labels = batch.output_tokens
  pre_logits = model.forward_features(batch.input_tokens, batch.positional_ids)
  metric_prefix = "test" if not training else "training"

  # TODO: Ted make sure this won't be an issue in the future
  assert cross_entropy_loss.inplace_backward is True, "Support is currently provided only for inplace backward"
  assert cross_entropy_loss.reduction == 'none', "Cross-Entropy reduction must be performed by hand!"

  N_BLOCKS, should_inc = divmod(pre_logits.shape[1], block_size)
  N_BLOCKS = N_BLOCKS + 1 if should_inc else N_BLOCKS
  
  grad_input_acc = []
  reporting_stats = []

  # fast path for single block
  if N_BLOCKS == 1:
    logit_block     = model.forward_logits(pre_logits)
    loss            = cross_entropy_loss.forward(logit_block, labels)
    batch_metadata  = LMBatchMetadata.from_training_batch(batch)
    block_reporting_stats = process_train_logging_metrics(
      batch, logit_block, loss, batch_metadata, metric_prefix=metric_prefix)
    reporting_stats.append(block_reporting_stats)
    if training:
      loss.mean()
      loss.backward()
    return

  COMPUTED_BLOCKS = 0

  # forward pass on each block
  for i in range(N_BLOCKS):
    start = i * block_size
    end = (i + 1) * block_size
    batch_slice = batch[start:end]
    batch_slice_metadata = LMBatchMetadata.from_training_batch(batch_slice)

    # early stop if we encounter 
    if batch_slice_metadata.padded_tokens == batch_slice_metadata.processed_tokens:
      break
    
    
    pre_logit_block = pre_logits[:, start:end].detach()
    
    if training:
      pre_logit_block.requires_grad = True

    logit_block = model.forward_logits(pre_logit_block)
    label_block = labels[:, start:end]
    logit_block     = model.forward_logits(pre_logits)
    loss            = cross_entropy_loss.forward(logit_block, label_block)
    batch_slice_metadata = LMBatchMetadata.from_training_batch(batch_slice)
    block_reporting_stats = process_train_logging_metrics(
      batch, logit_block, loss, batch_metadata, metric_prefix=metric_prefix)
    reporting_stats.append(block_reporting_stats)
    # we need to divide this loss by the number of blocks
    # this is very similar to gradient accumulation
    # think of how we virtually increase the batch_size, by not performing
    # the optimizer step which resets gradients

    # what we want to achieve here is to viartually "reduce" the computed batch_size
    # the .mean() / gradient computation now will account only for the computational
    # block size, which in our case is `block_size`. Since we will be summing up gradients
    # N_BLOCKS times, we need to divide the loss by N_BLOCKS to get the correct gradient
    if training:
      loss       /= N_BLOCKS
      loss.mean()
      loss.backward()
      grad_input_acc.append(pre_logit_block.grad.detach())
    
    COMPUTED_BLOCKS += 1

  with torch.inference_mode():
    # first aggregate local-level metrics
    local_reporting_stats = group_by_keys_recursive(reporting_stats)
    for k, v in local_reporting_stats.items():
      if is_masked_tensor_dict(v):
        v["mask"] = torch.nested.as_nested_tensor(v["mask"])
        v["tensor"] = torch.nested.as_nested_tensor(v["tensor"])

    # prepare everything for global-reduce
    for k, v in local_reporting_stats.items():
      if is_masked_tensor_dict(v):
        v["mask"] = torch.nested.to_padded_tensor(v["mask"], padding=0)
        v["tensor"] = torch.nested.to_padded_tensor(v["tensor"], padding=0)
        
        v["mask"].sum_(dim=0)      
        v["tensor"].sum_(dim=0)
        v["mean"] = v["tensor"] / v["mask"]
      else:
        local_reporting_stats[k] = torch.stack(v).mean()


  if training:
    # backward step on the rest of the network
    grad_pre_logits = torch.cat(grad_input_acc, dim = 1)

    # we might be skipping padding blocks
    # the deal with them is that we pre-normalized the loss
    # assuming that we would be computing the loss for all blocks
    # in order to get the correct grad-scaling   
    if COMPUTED_BLOCKS != N_BLOCKS:
      normalizing_factor = N_BLOCKS / COMPUTED_BLOCKS
      with torch.inference_mode():
        grad_pre_logits.mul_(normalizing_factor)
        model.output.weight.grad.mul_(normalizing_factor)

    pre_logits.backward(grad_pre_logits)

  return local_reporting_stats


def train():
  for step in range(...):
    batch_stats = fwd_bwd_block_chunk_ce(model, batch, block_size)
    batch_stats = reduce_training_stats(batch_stats)
    wandb.log(batch_stats)

if __name__ == "__main__":
  torch.random.manual_seed(0)
  torch.set_default_device("cuda")
  base_path = "/mnt/raid0/llamas/llama_base"
  transformer = Transformer.from_base_model(model_weights=CODE_LLAMA_7B_WEIGHTS, base_path=base_path).to(torch.bfloat16).cuda()
  input_ids = torch.randint(0, transformer.output.out_features, (1, 4096)).cuda()
  output_ids = torch.randint(0, transformer.output.out_features, (1, 4096)).cuda()
  input_pos = torch.arange(0, 4096).cuda()
  pre_logits = transformer.forward_features(input_ids, input_pos)
  # chunked_ce(transformer, 1024, pre_logits, input_ids) 
  print(transformer.output.weight.grad.norm())
  print(transformer.tok_embeddings.weight.grad.norm())