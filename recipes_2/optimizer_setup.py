from dataclasses import dataclass, field
from typing import List, Tuple
import torch
import yaml
from functools import partial

CONST_ALLOWED_OPTIMIZERS = ["AdamW", "SGD"]
CONST_OPTIMIZER_CLS_MAP  = {
  "AdamW": partial(torch.optim.AdamW, fused=True, betas=(0.9, 0.95)),
  "SGD": torch.optim.SGD,
}
CONST_NON_ALLOWED_WD_TYPES = ["embedding", "bias", "norm"]

__TYPE_EMPTY_LIST_DATACLASS__ = field(default_factory=lambda: [])

@dataclass
class ParameterGroup:
  lr: float
  weight_decay: float = 0.0
  included_params: List[str] = __TYPE_EMPTY_LIST_DATACLASS__

  def __post_init__(self):
    if not self.included_params:
      self.included_params = []


@dataclass
class OptimizerConfig:
  algo: str = "AdamW"
  parameter_groups: List[ParameterGroup] = __TYPE_EMPTY_LIST_DATACLASS__

  # custom initialization
  def __post_init__(self):
    self.parameter_groups = [ParameterGroup(**param_group) for param_group in self.parameter_groups]
    assert self.algo in CONST_ALLOWED_OPTIMIZERS, f"algo must be one of {CONST_ALLOWED_OPTIMIZERS}"
    assert len(self.parameter_groups) > 0, "at least one group must be provided"


def param_group_filter(
  params: list[tuple], parameter_group: ParameterGroup
) -> Tuple[list[tuple], list[tuple]]:
  included_params = parameter_group.included_params 
  if not included_params:
    return params, []
  
  # return only allowed parameters
  valid_params = []
  for name, param in params:
    if any([included_param in name for included_param in included_params]):
      valid_params.append((name, param))

  # functional API
  remainder_params = list(filter(lambda x: x not in valid_params, params))
  return valid_params, remainder_params

def split_params_by_wd(params: list[tuple], parameter_group: ParameterGroup) -> list[dict]:
  weight_decay_value = parameter_group.weight_decay
  if not weight_decay_value:
    return [{"params": [p for n, p in params]}]

  wd_params = []
  non_wd_params = []

  for name, param in params:
    if any([t in name for t in CONST_NON_ALLOWED_WD_TYPES]):
      non_wd_params.append(param)
    else:
      wd_params.append(param)
  
  # split the parameters by weight decay
  split = [
    {"params": wd_params, "weight_decay": parameter_group.weight_decay},
    {"params": non_wd_params, "weight_decay": 0.0},
  ]

  # remove empty groups
  split = list(filter(lambda x: len(x["params"]) > 0, split))
  return split

# TODO: @NCS, find prettier names
def setup_optimizer(
  model: torch.nn.Module, optimizer_config: OptimizerConfig
) -> torch.optim.Optimizer:
  optimizer_cls = CONST_OPTIMIZER_CLS_MAP[optimizer_config.algo]
  trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
  numel_params = len(trainable_params)
  optimizer_groups = []

  # we want to apply the split on named parameters first
  ordered_parameter_groups = sorted(
    optimizer_config.parameter_groups, key=lambda x: len(x.included_params), reverse=True)

  # just a rename for clarity in the following loop
  unmatched_trainable_params = trainable_params

  for group in ordered_parameter_groups:
    matched_params, unmatched_trainable_params = param_group_filter(unmatched_trainable_params, group)
    parameter_splits = split_params_by_wd(matched_params, group)
    for split in parameter_splits:
      optimizer_groups.append(split)
  
  # check that all parameters are accounted for
  assert numel_params == sum([len(g["params"]) for g in optimizer_groups]), "all parameters must be accounted for"
  optimizer = optimizer_cls(optimizer_groups)
  return optimizer