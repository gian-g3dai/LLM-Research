from typing import Optional
from dataclasses import dataclass


@dataclass
class CheckpointingConfig:
  save_dir: str
  run_name: Optional[str] = None
  save_every_n_steps: int = 1000
  format: str = "pt"
  async_save: bool = False

@dataclass
class ResumeConfig:
  resume_checkpoint: str
  resume_experiment_root: str
  weights_only: bool = True