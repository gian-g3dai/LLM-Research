from omegaconf import OmegaConf
from dataclasses import dataclass, asdict
from pprint import pprint

from recipes_2.data_setup import DataConfig
from recipes_2.adapter_setup import AdapterConfig 
from recipes_2.optimizer_setup import OptimizerConfig
from recipes_2.checkpointing_setup import CheckpointingConfig, ResumeConfig
from recipes_2.parallelization_setups import ParallelizationConfig
from recipes_2.base_model_setup import BaseModelConfig
from recipes_2.logging_setup import LoggingConfig

@dataclass
class HyperArgs:
  total_training_tokens: int
  batch_size_in_tokens: int
  batch_size: int
  clip_grad_norm: float
  lr_warmup_ratio: float
  no_sync_grad: bool
  eval_interval: int

@dataclass
class FineTuningConfig:
  optimizer_args: OptimizerConfig
  base_model_args: BaseModelConfig
  data_args: DataConfig
  adapter_args: AdapterConfig
  parallel_args: ParallelizationConfig
  checkpointing_args: CheckpointingConfig
  logging_args: LoggingConfig
  hyper_args: HyperArgs
  resume_args: ResumeConfig = None
  load_rank_0_only: bool = True
  full_config = None

  def __post_init__(self):
    self.full_config = asdict(self)
    self.optimizer_args = OptimizerConfig(**self.optimizer_args)
    self.base_model_args = BaseModelConfig(**self.base_model_args)
    self.adapter_args = AdapterConfig(**self.adapter_args)
    self.parallel_args = ParallelizationConfig(**self.parallel_args)
    self.checkpointing_args = CheckpointingConfig(**self.checkpointing_args)
    self.logging_args = LoggingConfig(**self.logging_args)
    self.hyper_args = HyperArgs(**self.hyper_args)
    self.data_args = DataConfig(**self.data_args)

