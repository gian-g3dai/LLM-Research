from typing import List, Union, Optional
import torch
from enum import Enum
from unakin.llm.metrics.base import BaseMetric, MetricValue, MetricType



def aggregate_dictionaries(dicts: List[dict]):
  results = {}
  for d in dicts:
    for metric, datasource_split in d.items():
      if metric not in results:
        results[metric] = {}
      for datasource, value in datasource_split.items():
        if datasource not in results[metric]:
          results[metric][datasource] = []
        results[metric][datasource].append(value.item())

  for metric, datasource_split in results.items():
    for datasource, values in datasource_split.items():
      results[metric][datasource] = sum(values) / len(values)

  return results

def flatten_dict(d: dict):
  # take a 2 level nested dictionary and flatten it
  # by combining the keys
  results = {}
  for metric, datasource_split in d.items():
    for datasource, value in datasource_split.items():
      results[f"{metric}_{datasource}"] = value
  return results

def to_metric_value(name: str, value: Union[float, int]) -> MetricValue:
  if "perplexity" in name:
    return MetricValue(name, MetricType.PERPLEXITY, value)
  elif "loss" in name:
    return MetricValue(name, MetricType.LOSS, value)
  elif "precision" in name:
    return MetricValue(name, MetricType.PRECISION, value)
  else:
    raise ValueError(f"Unknown metric name: {name}")
  
def partition_by_index_mask(tensor_: torch.Tensor, mask: torch.Tensor, indices_set: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
  unique_indices = mask.unique() if indices_set is None else indices_set
  partition_masks = [mask == idx for idx in unique_indices]
  return {idx: tensor_[mask] for idx, mask in zip(unique_indices, partition_masks)}

def mean_with_empty_tensor(tensor: torch.Tensor) -> torch.Tensor:
  if tensor.numel() == 0:
    return torch.tensor(0.0, device=tensor.device)
  else:
    return tensor.mean()