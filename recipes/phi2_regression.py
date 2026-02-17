import os
import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import random

from functools import partial
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from itertools import combinations
from phi2_dataset import (
  HelpSteerDataset,
  HelpSteerAttrDataset,
  pair_collate_fn,
  test_collate_fn,
  BatchIterator,
  make_distributed_dataset_shard
)
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

import torch.distributed

from transformers import LlamaForCausalLM, LlamaTokenizer

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-2-7b-chat-hf",
                    trust_remote_code=True, 
                    cache_dir="/mnt/raid0/hf_cache/llama2_7b_chat_cache",
                    padding_side="left",
                    add_eos_token=True,
                    add_bos_token=True,
                    use_fast=False
                    ) 
tokenizer.pad_token = tokenizer.eos_token
max_length = 2048
batch_size = 24


class Phi2Regressor(nn.Module):
  def __init__(self,
              attribute_count: int,
              pooling_type: str = "mean",
              dtype_: torch.dtype = torch.bfloat16):
    super(Phi2Regressor, self).__init__()
    self.phi_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                          torch_dtype=dtype_,
                                                          trust_remote_code=True,
                                                          attn_implementation="flash_attention_2",
                                                          cache_dir="/mnt/raid0/hf_cache/lama2_7b_chat",
                                                          use_cache=False,)
    self.phi_model._fsdp_wrap = True
    self.phi_model.lm_head = torch.nn.Identity(dtype=dtype_)
    self.phi_model.train()
    self.attr_count = attribute_count

    # self.attr_linear = nn.Sequential(
    #     nn.Linear(2560, 1024, dtype=dtype_),
    #     nn.ReLU(),
    #     nn.Linear(1024, 512, dtype=dtype_),
    #     nn.ReLU(),
    #     nn.Linear(512, attribute_count, dtype=dtype_),
    # )

    self.attr_linear = nn.Linear(4096, self.attr_count, dtype=dtype_)

    self.dtype_ = dtype_
    self.pooling_type = pooling_type

  def forward(self, inputs):    
    phi_out = self.phi_model.forward(**inputs)

    if self.pooling_type == "mean":
      phi_out = (phi_out.logits * inputs["attention_mask"].unsqueeze(-1)).sum(1)
      phi_out = phi_out / inputs["attention_mask"].sum(1).unsqueeze(-1)
    elif self.pooling_type == "last":
      phi_out = phi_out.logits[:, -1, :]
    else:
      raise ValueError("Invalid pooling type")

    phi_out = phi_out.to(self.dtype_)
    attr_out = self.attr_linear(phi_out)
    attr_out = attr_out.reshape(-1, self.attr_count)
    return attr_out
  
class BPRLoss(nn.Module):
  def __init__(self):
    super(BPRLoss, self).__init__()

  def forward(self, pos, neg):
    return -torch.log(torch.sigmoid(pos - neg)).mean()
  

def one_fwd_step(model, batch, criterion, device):
  """
  train loop
  """
  inputs, diffs, attr_vals = batch
  inputs = inputs.to(device)
  diffs = diffs.to(device)
  attr_vals = attr_vals.to(device)

  pred = model.forward(inputs)

  if criterion.__class__.__name__ == "BPRLoss":
    higher = pred[len(pred) // 2:]
    lower = pred[:len(pred) // 2]
    loss = criterion(higher, lower)
  else:
    pred = pred[:, 0]
    loss = criterion(pred, attr_vals)

  return loss


def reduce_across_ranks(tensor: torch.Tensor):
  torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
  tensor /= torch.distributed.get_world_size()
  return tensor

def print_rank0(*args, **kwargs):
  if torch.distributed.get_rank() == 0:
    print(*args, **kwargs)

def setup_distributed():
  torch.distributed.init_process_group(
      backend="nccl",
      init_method="env://",
  )
  torch.cuda.set_device(torch.distributed.get_rank())
  

def main():

  random.seed(42)

  setup_distributed()


  train_path = "/mnt/raid0/datasets/hf/helpsteer/helpsteer_train.parquet"
  test_path = "/mnt/raid0/datasets/hf/helpsteer/helpsteer_test.parquet"
  attributes = ["helpfulness"]
  train_template = "{prompt}\n{response}"
  
  log_interval = 10
  eval_interval = 16
  grad_accum = 2

  epochs = 10
  train_mode = "rank"
  lr = 3e-4
  weight_decay = 0.01
  use_wandb = True
  test_use_percentage = 0.4
  warmup_ratio = 0.1
  device = torch.device("cuda")
  print(device)

  
  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()

  pooling_type = "mean"

  if use_wandb and rank == 0:
    wandb.login(key="REMOVED")
    wandb.init(project="attr_regr_new",
                name="phi2_regression_with_pairs",
                config={
                    "log_interval": log_interval,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "attributes": attributes,
                    "train_template": train_template
                  }
                )

  train_dls = []
  test_dls = []

  if train_mode == "rank":
    for attr in attributes:
      train_dataset = HelpSteerAttrDataset(train_path, train_template, attr)
      test_dataset = HelpSteerDataset(test_path, attr, train_template, use_percentage=test_use_percentage)

      train_dataset = make_distributed_dataset_shard(train_dataset, rank=rank, world_size=world_size)
      test_dataset = make_distributed_dataset_shard(test_dataset, rank=rank, world_size=world_size)

      train_dl = DataLoader(train_dataset,
                            batch_size=batch_size,
                            collate_fn=partial(pair_collate_fn,
                                                tokenizer=tokenizer,
                                                max_length=max_length),
                            shuffle=True,
                            num_workers=4)

      test_dl = DataLoader(test_dataset,
                            batch_size=batch_size * 4,
                            collate_fn=partial(test_collate_fn,
                                               tokenizer=tokenizer,
                                               max_length=max_length),
                            shuffle=False,
                            num_workers=4)
      
      train_dls.append(train_dl)
      test_dls.append(test_dl)
  else:
    train_data = HelpSteerDataset(train_path, attributes, train_template)
    test_data = HelpSteerDataset(test_path, attributes, train_template)

  phi_regressor = Phi2Regressor(attribute_count=len(attributes),
                                pooling_type=pooling_type,
                                )
  
  precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
    keep_low_precision_grads=False,
  )

  wrap_policy = partial(
    transformer_auto_wrap_policy, 
    transformer_layer_cls={LlamaDecoderLayer}
  )  

  # transformer = torch.compile(transformer)

  fsdp_model = FSDP(
    phi_regressor.to(torch.bfloat16),
    auto_wrap_policy=wrap_policy,
    device_id=torch.distributed.get_rank(),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=precision,
    forward_prefetch=True,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    use_orig_params=False,
  )

  # fsdp_model = torch.compile(fsdp_model, mode="reduce-overhead")

  wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
  apply_activation_checkpointing(
    fsdp_model, auto_wrap_policy=wrap_policy,
    checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
  # apply_activation_checkpointing(
  #   fsdp_model, auto_wrap_policy=wrap_policy,
  #   checkpoint_wrapper_fn=partial(offload_wrapper,)
  # )


  # criterion = BPRLoss()
  # criterion = nn.MSELoss()
  # criterion = nn.L1Loss()

  total_steps = len(train_dls[0]) * epochs // grad_accum 
  warmup_steps = int(total_steps * warmup_ratio)

  params = [p for p in fsdp_model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

  lr_scheduler_1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
  lr_scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
  lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [lr_scheduler_1, lr_scheduler_2], [warmup_steps])

  fsdp_model.train()


  curr_step = 0
  for epoch in range(epochs):

    train_batch_iterator = BatchIterator(train_dls)

    torch.distributed.barrier()

    for grad_acc_idx, batches in enumerate(tqdm(train_batch_iterator, total=len(train_batch_iterator))):

      curr_losses = []
      
      # if multiple dataloaders, then we need to iterate over them
      batch = batches[0]

      loss = one_fwd_step(fsdp_model, batch, criterion, device)

      # print("Loss: ", loss.item())
      curr_losses.append(loss.item())
      
      loss = loss / grad_accum
      loss.backward()            

      if (grad_acc_idx + 1) % grad_accum == 0:
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

      if grad_acc_idx % log_interval == 0:

        curr_loss = reduce_across_ranks(torch.tensor(np.mean(curr_losses), device=device))

        if rank == 0:
          curr_loss = curr_loss.item()
          print("Train Loss: ", curr_loss)
          if use_wandb:
            wandb.log({"train_loss": curr_loss}, step=curr_step)
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=curr_step)

      if (grad_acc_idx + 1) % eval_interval == 0:

        fsdp_model.eval()
        with torch.no_grad():
          
          test_loss = 0
          higher_cnt = 0

          for dl_idx, test_dl in enumerate(test_dls):
            for batch in test_dl:

              inputs, pairs, attr_values = batch
              if len(pairs) == 0:
                continue

              inputs = inputs.to(device)
              attr_values = attr_values.to(device)

              pred = fsdp_model(inputs)

              pred = pred[:, dl_idx]
              # attr_values = attr_values[:, dl_idx]

              # # get the pairs
              higher = pairs[:, 0]
              lower = pairs[:, 1]

              pred_higher = pred[higher]
              pred_lower = pred[lower]


              loss = criterion(pred, attr_values)
              test_loss += loss.item()

              # acc
              higher_cnt += (pred_higher > pred_lower).sum().item() / len(pred_higher)

            # reduce the loss and acc
            test_loss = reduce_across_ranks(torch.tensor(test_loss, device=device))
            higher_cnt = reduce_across_ranks(torch.tensor(higher_cnt, device=device))

            if rank == 0:
              test_loss = test_loss.item()
              higher_cnt = higher_cnt.item()
              print("Test Loss: ", test_loss / len(test_dl))
              print("Acc :", higher_cnt / len(test_dl))
              if use_wandb:
                wandb.log({"test_loss_" + str(dl_idx): test_loss / len(test_dl)}, step=curr_step)
                wandb.log({"acc_" + str(dl_idx): higher_cnt / len(test_dl)}, step=curr_step)

        fsdp_model.train()  
      curr_step += 1





if __name__ == "__main__":
  main()



# /home/fsuser/miniconda3/envs/ncs-stable/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:149: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
#   warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 171/171 [1:15:33<00:00, 25.43s/it]Train Loss:  0.09783935546875
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 171/171 [1:15:33<00:00, 26.51s/it]
# [rank1]:[E ProcessGroupNCCL.cpp:463] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=17118, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) ran for 1800071 milliseconds before timing out.
#   0%|                                                                                                                                                                                                                                                                                                                                                               | 0/171 [00:00<?, ?it/s][rank1]:[E ProcessGroupNCCL.cpp:477] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
# [rank1]:[E ProcessGroupNCCL.cpp:483] To avoid data inconsistency, we are taking the entire process down.
# [rank1]:[E ProcessGroupNCCL.cpp:917] [Rank 1] NCCL watchdog thread terminated with exception: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=17118, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) ran for 1800071 milliseconds before timing out.
# [2024-03-19 02:09:57,550] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2651786 closing signal SIGTERM
# [2024-03-19 02:09:57,550] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2651788 closing signal SIGTERM
# [2024-03-19 02:09:57,551] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2651789 closing signal SIGTERM
# [2024-03-19 02:09:57,552] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2651790 closing signal SIGTERM
# [2024-03-19 02:09:57,553] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2651791 closing signal SIGTERM
# [2024-03-19 02:09:57,553] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2651792 closing signal SIGTERM
# [2024-03-19 02:09:57,554] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2651793 closing signal SIGTERM
# [2024-03-19 02:09:58,248] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: -6) local_rank: 1 (pid: 2651787) of binary: /home/fsuser/miniconda3/envs/ncs-stable/bin/python
# Traceback (most recent call last):
#   File "/home/fsuser/miniconda3/envs/ncs-stable/bin/torchrun", line 33, in <module>
#     sys.exit(load_entry_point('torch==2.2.0.dev20231018', 'console_scripts', 'torchrun')())
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/fsuser/miniconda3/envs/ncs-stable/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
#     return f(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^
#   File "/home/fsuser/miniconda3/envs/ncs-stable/lib/python3.11/site-packages/torch/distributed/run.py", line 806, in main
#     run(args)
#   File "/home/fsuser/miniconda3/envs/ncs-stable/lib/python3.11/site-packages/torch/distributed/run.py", line 797, in run
#     elastic_launch(
#   File "/home/fsuser/miniconda3/envs/ncs-stable/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
#     return launch_agent(self._config, self._entrypoint, list(args))
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/fsuser/miniconda3/envs/ncs-stable/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 264, in launch_agent
#     raise ChildFailedError(
# torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
# ========================================================
# phi2_regression.py FAILED
# --------------------------------------------------------
# Failures:
#   <NO_OTHER_FAILURES>
# --------------------------------------------------------
# Root Cause (first observed failure):
# [0]:
#   time      : 2024-03-19_02:09:57
#   host      : fluidstackg1801
#   rank      : 1 (local_rank: 1)
#   exitcode  : -6 (pid: 2651787)
#   error_file: <N/A>
#   traceback : Signal 6 (SIGABRT) received by PID 2651787