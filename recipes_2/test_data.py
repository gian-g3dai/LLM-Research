import hydra
import torch
import numpy as np
import random

from pprint import pprint
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from finetuning import FineTuningConfig

from unakin.llm._data.datasources.pipeline_utils import set_pipeline_cache_path
from unakin.llm._data.dataset_factory import DatasetFactory
from unakin.llm.tokenizers import LlamaTokenizer, Llama3Tokenizer


def main(args: FineTuningConfig):
  torch.random.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  torch.distributed.init_process_group(
  backend="nccl",
  init_method="env://",
  )

  # tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  tokenizer = Llama3Tokenizer(model_path="/mnt/raid0/llamas/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model")

  set_pipeline_cache_path(args.data_args.base_cache_path)
  train_dataset, train_ds_names = DatasetFactory.dataset_from_yaml(args.data_args, tokenizer,
                                                    process_rank=0, world_size=torch.distributed.get_world_size())


  train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=train_dataset.get_collate_fn(),
    pin_memory=True,
  )

  train_dataloader_iter = iter(train_dataloader)




@hydra.main(version_base=None, config_path="configs", config_name="base")
def hydra_main(cfg: DictConfig):

  args = FineTuningConfig(**cfg)
  pprint(cfg.keys())
  main(args)

if __name__ == "__main__":
  hydra_main()
