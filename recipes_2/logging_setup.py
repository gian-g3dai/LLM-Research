import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt


from dataclasses import dataclass
from checkpointing_setup import CheckpointingConfig

@dataclass
class WandbConfig:
  project: str
  experiment_name: str
  key: str

@dataclass
class LoggingConfig:
  log_interval: int
  wandb: WandbConfig

  def __post_init__(self):
    if self.wandb is not None:
      self.wandb = WandbConfig(**self.wandb)


from finetuning_setup import FineTuningConfig

def random_id():
  return wandb.util.generate_id()

def run_exists(args: CheckpointingConfig) -> bool:
  run_dir  = f"{args.save_dir}/{args.run_name}"
  return os.path.exists(run_dir)

def init_wandb(args: FineTuningConfig) -> None:
  if not args.logging_args.wandb:
    return
  
  if run_exists(args.checkpointing_args):
    run_name = f"{args.checkpointing_args.run_name}_{random_id()}"
    args.logging_args.wandb_experiment_name = run_name

  wandb.login(key=args.logging_args.wandb.key)
  wandb.init(project=args.logging_args.wandb.project,
              name=args.logging_args.wandb.experiment_name,
              config=args.full_config)

@torch.no_grad
def log_stats(stats,
              data,
              train_state,
              sample_cnt=None):
  
  if sample_cnt != None:
    curr_sample_cnt = sample_cnt
    torch.distributed.all_reduce(curr_sample_cnt, op=torch.distributed.ReduceOp.SUM)

  if torch.distributed.get_rank() == 0:
    logging_dict = {}

    if sample_cnt != None:
      curr_sample_prc = curr_sample_cnt / data.datasources_lengths * 100
      for smp_cnt, ds_n in zip(curr_sample_prc, data.datasources_names):
        logging_dict[ds_n + "_used_prc"] = smp_cnt

    for k, v in stats.items():
      if isinstance(v, dict):
        avg_tensor = v["tensor"] / v["mask"]
        if "perplexity" in k:
          avg_tensor = 2 ** avg_tensor
        if "_pos_" in k :
          data_ = avg_tensor.detach().float().cpu().numpy()
          positions = np.arange(len(data_))

          interval = 50
          cumulated_data = [np.mean(data_[i:i+interval]) for i in range(0, len(data_), interval)]
          positions = np.arange(0, len(data_), interval) + interval // 2 

          # Create bar plot
          plt.figure(figsize=(16, 9))
          plt.bar(positions, cumulated_data, width=interval)
          # Add labels and title
          plt.xlabel('Position')
          plt.ylabel('Value')
          plt.yscale("log")
          plt.title('Bar Plot ' + k)
          logging_dict[k] = wandb.Image(plt)

          if "perplexity" in k:
            plt.figure(figsize=(10, 5))
            n, bins, _ = plt.hist(data_, bins=150, cumulative=True, density=True, alpha=0.7, edgecolor='black', linewidth=2.2, color='blue')
            plt.grid()
            plt.gca().set_axisbelow(True)
            plt.xlabel("log(token ppl)")
            plt.ylabel("cumulative density")
            plt.title("Cumulative density of token ppls")
            logging_dict[k + "_hist"] = wandb.Image(plt)

          plt.close()
        elif "_ds_" in k:
          # iterate through the avg tensor 
          for ds_name, val in zip(data.datasources_names, avg_tensor.detach().float().cpu().numpy()):
            logging_dict[k + "_" + ds_name] = val
      else:
        # already agg vals
        logging_dict[k] = v.item()

    wandb.log(
      logging_dict, step=train_state.steps
    )

