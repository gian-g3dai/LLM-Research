
import torch
import yaml
import omegaconf
from typing import List
from dataclasses import dataclass
from unakin.llm._data.dataset_factory import DatasetFactory
from unakin.llm._data.code_dataset import GPTCodeDataset
from torch.utils.data import DataLoader
from unakin.llm._data.datasources.pipeline_utils import set_pipeline_cache_path
from unakin.llm.tokenizers import TokenizerBase



@dataclass
class DataConfig:
  base_cache_path: str
  train_config: str
  test_config: str

  def load_config(self, config_path):
    if isinstance(config_path, str):
      with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    elif isinstance(config_path, dict):
      config = config_path
    elif isinstance(config_path, omegaconf.DictConfig):
      config = omegaconf.OmegaConf.to_container(config_path)
    else:
      raise ValueError(f"config_path must be a str or dict but got {type(config_path)}")
    return config

  def __post_init__(self):
    self.train_config = self.load_config(self.train_config)
    self.test_config = self.load_config(self.test_config)

@dataclass
class Data:
  dataloader: DataLoader
  dataset: GPTCodeDataset
  datasources_names: List[str]
  datasources_lengths: List[int]
  tokenizer: TokenizerBase
  
def load_data(args: dict, tokenizer):
  set_pipeline_cache_path(args.data_args.base_cache_path)
  train_dataset, train_ds_names, train_ds_lengths = DatasetFactory.dataset_from_yaml(args.data_args.train_config, tokenizer,
                                                    process_rank=torch.distributed.get_rank(),
                                                    world_size=torch.distributed.get_world_size())
  test_dataset, test_ds_names, _ = DatasetFactory.dataset_from_yaml(args.data_args.test_config, tokenizer,
                                                    process_rank=torch.distributed.get_rank(),
                                                    world_size=torch.distributed.get_world_size())

  train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=train_dataset.get_collate_fn(),
    pin_memory=True,
  )

  test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=test_dataset.get_collate_fn(),
    pin_memory=True,
  )

  train_data = Data(dataloader=train_dataloader,
                       dataset=train_dataset,
                       datasources_names=train_ds_names,
                       datasources_lengths=train_ds_lengths,
                       tokenizer=tokenizer)
  test_data = Data(dataloader=test_dataloader,
                      dataset=test_dataset,
                      datasources_names=test_ds_names,
                      datasources_lengths=None,
                      tokenizer=tokenizer)
  
  return train_data, test_data