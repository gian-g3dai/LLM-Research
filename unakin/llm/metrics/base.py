from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union, NamedTuple
import torch
import torch.utils.data
from enum import Enum

from unakin.llm._data.datasources.train_sample import LMTrainingBatch
from unakin.llm.tokenizers.base import TokenizerBase


class ComparisonOp(Enum):
  LT = 0
  LE = 1
  GT = 2
  GE = 3
  EQ = 4
  NE = 5

  def __call__(self, a: float, b: float) -> bool:
    return apply_op(self, a, b)

def apply_op(op: ComparisonOp, a: float, b: float) -> bool:
  if op == ComparisonOp.LT:
    return a < b
  elif op == ComparisonOp.LE:
    return a <= b
  elif op == ComparisonOp.GT:
    return a > b
  elif op == ComparisonOp.GE:
    return a >= b
  elif op == ComparisonOp.EQ:
    return a == b
  elif op == ComparisonOp.NE:
    return a != b
  else:
    raise ValueError(f"Invalid operator: {op}")

class MetricType(Enum):
  PERPLEXITY = (0, ComparisonOp.LT)
  PRECISION = (1, ComparisonOp.GT)
  LOSS = (2, ComparisonOp.LT)

class MetricValue:
  # this is not a NamedTuple because we want to be able to modify the value
  # in-place for the name
  name: str
  type: MetricType
  value: float

  def __init__(self, name: str, type: MetricType, value: float) -> None:
    self.name = name
    self.type = type
    self.value = value
  
def metric_is_better(a: MetricValue, b: MetricValue) -> bool:
  assert a.type == b.type
  return apply_op(a.type.value[1], a.value, b.value)

def metric_set_is_better(a: List[MetricValue], b: Dict[str, MetricValue]) -> bool:
  # check if only one metric has improved
  # assert len(a) == len(b)
  for i in range(len(a)):
    if metric_is_better(a[i], b[a[i].name]):
      return True
  return False

def get_better_metrics(a: List[MetricValue], b: List[MetricValue]) -> List[MetricValue]:
  assert len(a) == len(b)
  ret = []
  for i in range(len(a)):
    if metric_is_better(a[i], b[i]):
      ret.append(a[i])
    else:
      ret.append(b[i])
  return ret

def select_best_metric(a: MetricValue, b: MetricValue) -> float:
  assert a.type == b.type
  if metric_is_better(a, b):
    return a.value
  else:
    return b.value

def update_metric_dict(d: Dict[str, MetricValue], new_metrics: List[MetricValue]) -> None:
  for new_metric in new_metrics:
    if new_metric.name not in d:
      d[new_metric.name] = new_metric
    else:
      current = d[new_metric.name]
      assert current.type == new_metric.type
      current.value = select_best_metric(current, new_metric)

class BaseMetric:
  def __init__(self, *args, **kwargs) -> None:
    pass

  @abstractmethod
  def evaluate_step(self, model: torch.nn.Module, batch: LMTrainingBatch, tokenizer: Optional[TokenizerBase]) -> List[MetricValue]:
    raise NotImplementedError
  
  @abstractmethod
  def evaluate_loader(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader, tokenizer: Optional[TokenizerBase]) -> List[MetricValue]:
    raise NotImplementedError
  
  @abstractmethod
  def unpack_batch(self, batch: LMTrainingBatch) -> Any:
    raise NotImplementedError