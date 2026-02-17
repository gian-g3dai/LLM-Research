
import torch
from torch.optim import Optimizer

from unakin.llm.optimizers.any_precision_adam import AnyPrecisionAdamW
from unakin.llm.optimizers.shampoo import Shampoo


def create_decay_param_group(model: torch.nn.Module, weight_decay: float) -> list[dict]:
  no_decay = []
  decay    = []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    
    if "norm" in name or "embedding":
      no_decay.append(param)
    else:
      decay.append(param)

  return [
    {"params": no_decay, "weight_decay": 0.0},
    {"params": decay,    "weight_decay": weight_decay}
  ]

def setup_param_group(model: torch.nn.Module, config: dict) -> list:
  if "weight_decay" in config:
    return create_decay_param_group(model, config["weight_decay"])
  
  return [param for param in model.parameters() if param.requires_grad]
class OptimizerFactory:

  @staticmethod
  def from_file(model: torch.nn.Module, path: str, **model_param_kwargs) -> Optimizer:
    if path.endswith(".json"):
      return OptimizerFactory.from_json(model, path, **model_param_kwargs)
    
  @staticmethod
  def from_json(model: torch.nn.Module, json_path: str, **model_param_kwargs) -> Optimizer:
    import json
    
    with open(json_path, "r") as f:
      config = json.load(f)
    
    if config["type"] == "AdamW":
      params = setup_param_group(model, config)
      return torch.optim.AdamW(params=params, **config["params"], **model_param_kwargs)
    
    elif config["type"] == "AnyPrecisionAdamW":
      params = setup_param_group(model, config)
      return AnyPrecisionAdamW(params=params, **config["params"], **model_param_kwargs)
    
    elif config["type"] == "Shampoo":
      params = setup_param_group(model, config)
      return Shampoo(params, **config["params"], **model_param_kwargs)

    raise ValueError(f"Unknown optimizer type: {config['type']}")
