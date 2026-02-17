
import torch
import random
import pandas as pd

from functools import partial
from itertools import combinations
from typing import List
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from typing import Any, Dict, List
from copy import deepcopy

class HelpSteerDataset(Dataset):
  def __init__(self,
               data_path: str,
               attribute_col: str,
               template: str, 
               use_percentage: float = 0.1):
    self.data = pd.read_parquet(data_path)
    self.attribute = attribute_col
    self.template = template
    self.data[self.attribute] = self.data[self.attribute] / self.data[self.attribute].max()
    self.data.reset_index(drop=True, inplace=True)

    # calculate weights for sampling according to the attr distribution
    sampling_weights = self.data[self.attribute].value_counts(normalize=True)
    self.data["sampling_weights"] = self.data[self.attribute].map(sampling_weights)

    # sample fewer data points
    self.data = self.data.sample(frac=use_percentage, weights="sampling_weights")

  def __slice__(self, start, end) -> Dataset:
    clone = deepcopy(self)
    clone.data = clone.data.iloc[start:end]
    clone.data.reset_index(drop=True, inplace=True)
    return clone

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    prompt = row["prompt"]
    response = row["response"]
    attr_val = row[self.attribute]
    sample = self.template.format(prompt=prompt, response=response)
    return sample, attr_val

class HelpSteerAttrDataset(Dataset):
  def __init__(self, data_path: str, template: str, attribute: str):
    self.data = pd.read_parquet(data_path)

    # normalize the attribute
    self.data[attribute] = self.data[attribute] / self.data[attribute].max()
    self.attribute = attribute
    self.template = template

    # code that goes in here is subject to reinitialization in the __re_init_sampling_data
    # function that is used for slicing the dataset
    self.idx_attr_dict = self._make_idx_attr_dict()
    
    # we get the sampling weights, possible values and min value of the attribute
    stats = self._make_sampling_weights()
    self.attr_sampling_weights = stats[0]
    self.attr_possible_values  = stats[1]
    self.min_attr_val          = stats[2]

    self.samplig_data          = self.data[self.data[self.attribute] > self.min_attr_val]
    self.samplig_data.reset_index(drop=True, inplace=True)


  def __slice__(self, start, end) -> Dataset:
    clone = deepcopy(self)
    clone.samplig_data = clone.samplig_data.iloc[start:end]
    clone.samplig_data.reset_index(drop=True, inplace=True)
    # clone._re_init_sampling_data()
    return clone

  def _re_init_sampling_data(self):
    self.idx_attr_dict = self._make_idx_attr_dict()        
    # we get the sampling weights, possible values and min value of the attribute
    stats = self._make_sampling_weights()
    self.attr_sampling_weights = stats[0]
    self.attr_possible_values  = stats[1]
    self.min_attr_val          = stats[2]

  def _make_sampling_weights(self):
    sample_weights = dict(self.data[self.attribute].value_counts()) 
    attribute_values = sorted(list(self.idx_attr_dict.keys()))   
    min_attribute_value = attribute_values[0] # this is sorted in asceding order

    return sample_weights, attribute_values, min_attribute_value

  def _make_idx_attr_dict(self):
    d: Dict[Any, List] = {}
    for idx, row in self.data.iterrows():
      attr_val = row[self.attribute]
      if attr_val not in d:
        d[attr_val] = []
      d[attr_val].append(idx)    
    return d

  def __len__(self):
    return len(self.samplig_data)

  def __getitem__(self, idx):
    row = self.samplig_data.iloc[idx]
    
    prompt = row["prompt"]
    response = row["response"]
    attr_value = row[self.attribute]
    
    # sample a lower attribute value
    lower_attr_values = [val for val in self.attr_possible_values if val < attr_value]
    lower_attr_weights = [self.attr_sampling_weights[val] for val in lower_attr_values]

    lower_attr_value = random.choices(lower_attr_values, weights=lower_attr_weights, k=1)[0]
    lower_idx = random.choice(self.idx_attr_dict[lower_attr_value])

    if lower_idx >= len(self.data):
      print("Lower idx: ", lower_idx, "Data len: ", len(self.data), "Rank: ", torch.distributed.get_rank())

    lower_row = self.data.iloc[lower_idx]
    lower_prompt = lower_row["prompt"]
    lower_response = lower_row["response"]

    diff = attr_value - lower_attr_value

    sample = self.template.format(prompt=prompt, response=response)
    lower_sample = self.template.format(prompt=lower_prompt, response=lower_response)

    return sample, lower_sample, diff, attr_value, lower_attr_value


def make_distributed_dataset_shard(dataset: Dataset, rank: int, world_size: int) -> Dataset:
  dataset_size = len(dataset)
  # make the shard size a multiple of the world size
  dataset_size = dataset_size - (dataset_size % world_size)
  shard_size = dataset_size // world_size
  start = rank * shard_size
  end   = start + shard_size
  return dataset.__slice__(start, end)

def pair_collate_fn(batch, tokenizer=None, max_length=2048):
  samples, lower_samples, diffs, attr_values, lower_attr_values = zip(*batch)
  all_samples = samples + lower_samples
  all_values = attr_values + lower_attr_values
  inputs = tokenizer(all_samples,
                      return_tensors="pt",
                      return_attention_mask=True,
                      padding="max_length",
                      max_length=max_length,
                      truncation=True               
                      )
  diffs = torch.tensor(diffs, dtype=torch.bfloat16)
  all_values = torch.tensor(all_values, dtype=torch.bfloat16)
  return inputs, diffs, all_values

def test_collate_fn(batch, tokenizer=None, max_length=2048):
  samples, attr_vals = zip(*batch)
  inputs = tokenizer(samples,
                      return_tensors="pt",
                      return_attention_mask=True,
                      padding="max_length",
                      max_length=max_length,
                      truncation=True
                      )
  all_pairs = list(combinations(range(len(attr_vals)), 2))
  
  pairs = []
  for pair in all_pairs:
    if attr_vals[pair[0]] > attr_vals[pair[1]]:
      pairs.append(pair)
    if attr_vals[pair[0]] < attr_vals[pair[1]]:
      pairs.append(pair[::-1])

  attr_vals = torch.tensor(attr_vals, dtype=torch.bfloat16)
  pairs = torch.tensor(pairs, dtype=torch.long)
  return inputs, pairs, attr_vals

class BatchIterator():
  def __init__(self, dls: List[DataLoader]):
    self.dls = dls
    self.iterators = [iter(dl) for dl in dls]

  def __iter__(self):
    return self

  def __next__(self):
    try:
      batches = [next(it) for it in self.iterators]
      return batches
    except StopIteration:
      self.iterators = [iter(dl) for dl in self.dls]
      raise StopIteration
    
  def __len__(self):
    return len(self.dls[0])


def main():
  train_path = "/mnt/raid0/datasets/hf/helpsteer/helpsteer_train.parquet"
  test_path = "/mnt/raid0/datasets/hf/helpsteer/helpsteer_test.parquet"
  attributes = ["helpfulness"]
  train_template = "{prompt}\n{response}"
  tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/phi-2",
                    trust_remote_code=True, 
                    cache_dir="/mnt/raid0/hf_cache/phi2_cache",
                    padding_side="left",
                    add_eos_token=True,
                    add_bos_token=True,
                    use_fast=False
                    )
  tokenizer.pad_token = tokenizer.eos_token

  attr = "helpfulness"

  rank, world_size = 0, 8

  for i in range(10):


    train_dataset = HelpSteerAttrDataset(train_path, train_template, attr)    
    test_dataset = HelpSteerDataset(test_path, attr, train_template, use_percentage=0.1)
    
    train_dataset = make_distributed_dataset_shard(train_dataset, rank, world_size)
    test_dataset = make_distributed_dataset_shard(test_dataset, rank, world_size)


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  collate_fn=partial(pair_collate_fn,
                                                      tokenizer=tokenizer,
                                                      max_length=2048),
                                  shuffle=True,
                                  num_workers=1)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=4,
                                  collate_fn=partial(test_collate_fn,
                                                     tokenizer=tokenizer,
                                                     max_length=2048),
                                  shuffle=False,
                                  num_workers=4)

    # print("Train Dataloader")

    # for batch in train_dataloader:
    #   inputs, diffs = batch
    #   print(inputs)
    #   print(diffs)
    #   break

  print("Test Dataloader")


  for batch in test_dataloader:
    # all pairs are left has higher attribute value than right
    inputs, pairs, attr_vals = batch
    print(inputs)
    print(pairs)
    print(attr_vals)
    





if __name__ == "__main__":
  main()