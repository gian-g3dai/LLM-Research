from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import List
import warnings
import yaml
from unakin.llm.adapters.llm_adapter_new import (
  ModuleFilterConfig,
  ModuleFilter,
  LoraConfig,
  LoraPolicy,
  LoraPlusPolicy,
  NefTunePolicy,
  AdaptationPolicy,
  NeftuneConfig,
  UnfreezeConfig,
  UnfreezePolicy
)

__TYPE_EMPTY_LIST_DATACLASS__ = field(default_factory=lambda: [])

# TODO: I don't like this aesthetically
__CONST_ADAPTATION_POLICY_MAP = {
  "lora": LoraPolicy,
  "neftune": NefTunePolicy,
  "lora_plus": LoraPlusPolicy,
  "unfreeze": UnfreezePolicy
}

# TODO: I don't like this aesthetically
__CONST_ADAPTATION_CONFIG_MAP = {
  "lora": LoraConfig,
  "neftune": NeftuneConfig,
  "lora_plus": LoraConfig,
  "unfreeze": UnfreezeConfig
}

def _strip_adapter_name(name: str) -> str:
  for adapter in __CONST_ADAPTATION_POLICY_MAP.keys():
    if name.endswith(adapter):
      return adapter
  raise ValueError(f"Unkown adapter name: {name}. Can only be one of {__CONST_ADAPTATION_POLICY_MAP.keys()}")

def GET_ADAPTER_SETUP(name: str) -> tuple[callable, callable]:
  adapter_type_name = _strip_adapter_name(name)
  constructor   = __CONST_ADAPTATION_POLICY_MAP[adapter_type_name] # TODO: conflict with the validator
  configuration = __CONST_ADAPTATION_CONFIG_MAP[adapter_type_name]
  return constructor, configuration

def MAKE_MODULE_FILTER(config: dict) -> ModuleFilter:
  return ModuleFilter(ModuleFilterConfig(**config)) 

def validate_adapter_name(name: str) -> bool:
  return any(name.endswith(adapter) for adapter in __CONST_ADAPTATION_POLICY_MAP.keys())

class AdaptationStrategy(Enum):
  FULL_FT = "FULL_FT"
  PEFT    = "PEFT"

@dataclass
class AdapterConfig:
  policies: List[AdaptationPolicy] = __TYPE_EMPTY_LIST_DATACLASS__
  strategy: AdaptationStrategy = AdaptationStrategy.FULL_FT

  def __post_init__(self):
    """
    Here's an example config in order for an easier time to wrap your head around this:
    
    ```yaml
    policies:
      w1_lora:
        module_filter:
          layer_names: ["w1"]
          layer_classes: ["Linear"]
        config:
          rank: 256
          dropout: 0.1
          alpha: 1.0
    ```
    We use both `layer_names` and `layer_classes` to filter the layers simply for safety reasons.
      Although not a perfect mechanism, we do not want to add extra-bloat for tracking which parameters 
      have, or have not been adapted. Should an already adapted module be adapted again, the recursive
      adaptation will throw an error to the user. We opt for this approach in order to enforce it on user
      level to specify the correct and consistent behaviour that they want.

    Below is the python equivalent of the above yaml config: 
    ```python
    adapt_module_filter = ModuleFilter(
      ModuleFilterConfig(
        layer_names=["w1"],
        layer_classes=["Linear"],
      )
    )

    lora_policy = LoraPolicy(
      LoraConfig(rank=256, dropout=0.1, alpha=0.1),
      adapt_module_filter,
    )
    """

    if len(self.policies) == 0:
      self.strategy = AdaptationStrategy.FULL_FT
      warnings.warn("No adapter policies found in the config file. This is equivalent to performing a full finetune.")
      return

    policy_module_filters = []
    instatiated_policies  = []
    
    for adapter_key in self.policies:

      # if you reached this error, the configuration files expect the following format
      # `my_weird_parameter_name_{ADAPTER_NAME}` i.e. `w1_lora`
      assert validate_adapter_name(adapter_key), f'Invalid adapter name: {adapter_key}'
      assert self.policies[adapter_key].get("module_filter", None), f"adapter {adapter_key} must have a module_filter"
      
      policy_filter = MAKE_MODULE_FILTER(self.policies[adapter_key]["module_filter"])
      policy_module_filters.append(policy_filter)
      
      # get a constructor and configurator
      constructor, configurator = GET_ADAPTER_SETUP(adapter_key)
      
      # instantiate the policy, remember that any AdapterPolicy has the same constructor signature
      # which is (config, target_filter)
      config = configurator(**self.policies[adapter_key]["config"])
      policy = constructor(config=config, target_filter=policy_filter)
      instatiated_policies.append(policy)

    self.policies = instatiated_policies