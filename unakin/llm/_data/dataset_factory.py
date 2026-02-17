import torch.distributed
import yaml
import torch
import numpy as np
import random
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader


from unakin.llm.tokenizers.base import TokenizerBase
from unakin.llm.tokenizers import LlamaTokenizer, Llama3Tokenizer

from unakin.llm._data.datasources.base_datasource import LMDataSource
from unakin.llm._data.datasources.datasource_loader import get_datasource
from unakin.llm._data.code_dataset import GPTCodeDataset
from unakin.llm._data.datasources.pipeline_utils import set_pipeline_cache_path
from unakin.llm.distributed.utils import print_rank0

class DatasetFactory:
  @staticmethod
  def dataset_from_yaml(config, tokenizer: TokenizerBase,
                        process_rank: int = 0, world_size: int = 1) -> LMDataSource:
    """
    Dataset from yaml or dict.
    """
    ds_weights_dict = config["datasource_weights"]

    if ds_weights_dict is None:
      raise ValueError("Datasource weights must be specified in the config")

    assert set(ds_weights_dict.keys()) - set(config['datasources'].keys()) == set(), \
      "Datasource weights must be a subset of the datasources"

    mod_sampling_config = {}    
    if "mod_sampling" in config:
      mod_sampling_list = config["mod_sampling"]
      for ds_names, sampling_config in mod_sampling_list:
        for ds_name in ds_names:
          mod_sampling_config[ds_name] = sampling_config["fim_rev_config"]
          mod_sampling_config[ds_name]["fim"]["prefix_token"] = tokenizer.prefix_token
          mod_sampling_config[ds_name]["fim"]["middle_token"] = tokenizer.middle_token
          mod_sampling_config[ds_name]["fim"]["suffix_token"] = tokenizer.suffix_token
          mod_sampling_config[ds_name]["fim"]["eot_token"] = tokenizer.eot_token
          mod_sampling_config[ds_name]["rev"]["rev_token"] = tokenizer.rev_token

    for ds_name in ds_weights_dict.keys():
      if ds_name not in mod_sampling_config:
        mod_sampling_config[ds_name] = None
        
    datasources = []
    datasources_weights = []
    datasources_names = []
    datasources_lengths = []
    for ds_name, ds_weight in ds_weights_dict.items():
      print_rank0("############### Loading Datasource", ds_name, " ###############")
      datasources_names.append(ds_name)
      datasources_weights.append(ds_weight)
      datasources.append(get_datasource(config["datasources"][ds_name],
                                        ds_name,
                                        tokenizer,
                                        data_root=config['data_root'],
                                        fim_rev_config=mod_sampling_config[ds_name]))
      datasources_lengths.append(len(datasources[-1]))
      
    return GPTCodeDataset(datasources=datasources,
                          datasources_weights=datasources_weights,
                          tokenizer=tokenizer,
                          process_rank=process_rank,
                          world_size=world_size,
                          **config['dataset']), datasources_names, torch.tensor(datasources_lengths)
  
  @staticmethod
  def datasets_from_yaml(config_path: str, tokenizer: TokenizerBase,
                        process_rank: int = 0, world_size: int = 1) -> LMDataSource:
    
    with open(config_path, 'r') as yaml_file:
      config = yaml.safe_load(yaml_file)

    datasets = []
    for ds_config in config:
      datasets.append(DatasetFactory.dataset_from_yaml(ds_config, tokenizer, process_rank, world_size))

    return datasets
  

if __name__ == '__main__':

  torch.random.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  torch.distributed.init_process_group(
  backend="cpu:gloo,cuda:nccl",
  init_method="env://",
  )
  torch.cuda.set_device(torch.distributed.get_rank())


  ds_config = "/home/fsuser/workdir/ncs/ml-research/recipes_2/configs/datasets/dataset_pretrain_fr.yaml"
  # tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  tokenizer = Llama3Tokenizer(model_path="/mnt/raid0/llamas/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model")

  set_pipeline_cache_path("/mnt/raid0/datasets/tmp_cache")
  with open(ds_config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
  dataset, datasource_names, datasource_lengths = DatasetFactory.dataset_from_yaml(config, tokenizer,
                                                process_rank=torch.distributed.get_rank(),
                                                world_size=torch.distributed.get_world_size())

  train_dl = DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=0, collate_fn=dataset.get_collate_fn(),
    pin_memory=True,
  )

  
  num_training_tokens = 0
  for idx, (batch, sample_cnt) in enumerate(train_dl):
    batch.decode_colorprint_samples(tokenizer)

    # print("SAMPL CNT: ", sample_cnt, torch.distributed.get_rank())

    # curr_sample_cnt = sample_cnt

    # torch.distributed.all_reduce(curr_sample_cnt, op=torch.distributed.ReduceOp.SUM)

    # print(curr_sample_cnt)
    # print(curr_sample_cnt / datasource_lengths)

    if idx > 2:
      break

    # num_training_tokens += batch.get_num_training_tokens()
      
