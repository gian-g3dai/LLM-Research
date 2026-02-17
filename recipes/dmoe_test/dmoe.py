
import os
import hydra
import torch
import functools

import torch.distributed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload, FullOptimStateDictConfig,
    FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType)
from torch.distributed.fsdp.wrap import (
    _or_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from pprint import pprint
from omegaconf import DictConfig, OmegaConf

from megablocks.layers import dmoe, moe
from megablocks.layers.arguments import Arguments

def setup():
    # initialize the process group
    dist.init_process_group("nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout, dmoe_args):
    super(TransformerBlock, self).__init__()
    
    self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
    # self.attn_linear = nn.Linear(embed_dim, embed_dim)

    # self.feedforward = nn.Sequential(
    #     nn.Linear(embed_dim, ff_hidden_dim),
    #     nn.ReLU(),
    #     nn.Linear(ff_hidden_dim, embed_dim),
    # )
    self.feedforward = dmoe.dMoE(dmoe_args)
    # self.feedforward = moe.MoE(dmoe_args)
    # self.feedforward = nn.Linear(embed_dim, embed_dim)

    
    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.layer_norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    # Multi-head attention
    x_att, _ = self.attention(x, x, x, attn_mask=mask)
    # x_att = self.attn_linear(x)

    x = x + self.dropout(self.layer_norm1(x_att))

    # Feedforward layer
    x_ff, _ = self.feedforward(x)
    # x_ff = self.feedforward(x)
    x = x + self.dropout(self.layer_norm2(x_ff))
    return x

class Transformer(nn.Module):
  def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_length, dropout, dmoe_args):
    super(Transformer, self).__init__()
    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
    self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)
    self.transformer_blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout, dmoe_args)
        for _ in range(num_layers)
    ])
    self.dropout = nn.Dropout(dropout)
    self.lm_head = nn.Linear(embed_dim, vocab_size)

  def forward(self, input_ids):
    positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    x = self.token_embedding(input_ids)  # + self.pos_embedding(positions)
    x = self.dropout(x)

    for transformer_block in self.transformer_blocks:
        x = transformer_block(x)

    x = self.lm_head(x)

    return x


def single_main(args_dict):
  
  dmoe_conf = args_dict["dmoe"]
  dmoe_args = Arguments(**dmoe_conf)

  trf_conf = args_dict["trf"]
  trf_conf["dmoe_args"] = dmoe_args

  model = Transformer(**trf_conf)
  model.cuda(torch.cuda.current_device()).to(torch.float16)

  vocab_size = trf_conf["vocab_size"]
  max_seq_length = trf_conf["max_seq_length"]
  
  # optimizer = A
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)


  input_ids = torch.randint(0, vocab_size, (1, max_seq_length + 1))  # Example input sequence

  labels = input_ids[:, 1:]
  input_ids = input_ids[:, :-1]

  input_ids = input_ids.to(torch.cuda.current_device())
  labels = labels.to(torch.cuda.current_device())
  
  logits = model(input_ids)

  print("Logits shape:", logits.shape)
  print("Labels shape:", labels.shape)

  loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
  print(loss.item())

  loss.backward()
  optimizer.step()
    
@hydra.main(version_base=None, config_path="config", config_name="dmoe_config")
def main(cfg: DictConfig) -> None:

  args_dict = OmegaConf.to_container(cfg)

  # single_main(args_dict=args_dict)
  # print("Single GPU werks")
  # exit()

  setup()

  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()

  transformer_auto_wrapper_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={dmoe.ParallelDroplessMLP}
            )


  torch.cuda.set_device(rank)

  print(args_dict)

  dmoe_conf = args_dict["dmoe"]
  dmoe_args = Arguments(**dmoe_conf)

  trf_conf = args_dict["trf"]
  trf_conf["dmoe_args"] = dmoe_args

  model = Transformer(**trf_conf).to(rank)

  # how to wrap the model
  # offloading
  # prefetching
  model = FSDP(
    module=model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=False),
    auto_wrap_policy=transformer_auto_wrapper_policy,
    device_id=torch.cuda.current_device(),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    use_orig_params=True,
    forward_prefetch=True,
  )
  
  vocab_size = trf_conf["vocab_size"]
  max_seq_length = trf_conf["max_seq_length"]
  input_ids = torch.randint(0, vocab_size, (1, max_seq_length + 1))  # Example input sequence

  input_ids = input_ids.to(torch.device(f"cuda:{rank}"))




  while True:
    out = model(input_ids)
    # loss = out.mean()
    # loss.backward()






  cleanup()


if __name__ == "__main__":
  main()