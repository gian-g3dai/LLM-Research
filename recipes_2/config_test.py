
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from dataclasses import dataclass, field
from typing import List

@dataclass
class Typed:
  def validate(self):
    ret = True
    for field_name, field_def in self.__dataclass_fields__.items():
      field_val = getattr(self, field_name)
      try:
        # transforming OmegaConf types to python types (ListConfig to list)
        actual_type = type(OmegaConf.to_object(field_val))
        if actual_type == list:
          # get the type of the elements
          elm_type = type(field_val[0])

          if all(isinstance(elm, elm_type) for elm in field_val):
            actual_type = List[elm_type]

      except:
        actual_type = type(field_val)

      if actual_type != field_def.type:
          print(f"\t{field_name}: '{actual_type}' instead of '{field_def.type}'")
          ret = False
      else:
        print(f"Key {field_name} OK ...")
    return ret


@dataclass
class ParameterGroup(Typed):
  lr: float
  weight_decay: float = 0.0
  keys: List[str] = field(default_factory=lambda: [])
  def __post_init__(self):
    if not self.validate():
      raise ValueError("Wrong Types")

  

@dataclass
class OptimizerConfig(Typed):
  algo: str = "AdamW"
  parameter_groups: List[ParameterGroup] = field(default_factory=lambda: [])
  def __post_init__(self):
    self.parameter_groups = [ParameterGroup(**param_group) for param_group in self.parameter_groups]
    if not self.validate():
      raise ValueError("Wrong Types")
    

@dataclass
class ModelConfig:
  model: dict
  tokenizer: dict

@dataclass
class AdapterConfig:
  lora: dict
  norm: dict
  bias_scale: dict
  lm_head: dict

@dataclass
class ParallelConfig:
  type: str

@dataclass
class FineTuningConfig:
  optimizer: OptimizerConfig
  model: ModelConfig
  adapter: AdapterConfig
  parallel: ParallelConfig
  # load_rank_0_only: bool = True
  # some other default stuff
  def __post_init__(self):
    self.optimizer = OptimizerConfig(**self.optimizer)
    self.model = ModelConfig(**self.model)
    self.adapter = AdapterConfig(**self.adapter)
    self.parallel = ParallelConfig(**self.parallel)


@hydra.main(version_base=None, config_path="configs")
def hydra_main(cfg: DictConfig):
  
  print(OmegaConf.to_yaml(cfg))
  print(cfg.optimizer)


  optimizer_config   = OptimizerConfig(**cfg.optimizer)

  print("Optimizer Config print: ")
  print(optimizer_config)

  # base_model_config = ModelConfig(**cfg.model)
  # adapter_config = AdapterConfig(**cfg.adapter)
  # parallel_config = ParallelConfig(**cfg.parallel)

  # stf_config = FineTuningConfig(
  #   optimizer_config=optimizer_config,
  #   base_model_config=base_model_config,
  #   adapter_config=adapter_config,
  #   parallel_config=parallel_config,
  #   **cfg.finetuning
  # )

  # stf_config = FineTuningConfig(**cfg)

  # print(stf_config)

  # print(stf_config.optimizer_config.params.betas)
  

if __name__ == "__main__":
  hydra_main()
