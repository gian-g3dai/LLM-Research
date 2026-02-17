from dataclasses import dataclass

import yaml
import unakin.llm.distributed.utils as unakin_dist_utils
from unakin.llm.models.llama.model_compile import (
  Transformer,
  CODE_LLAMA_13B_WEIGHTS,
  CODE_LLAMA_34B_WEIGHTS,
  CODE_LLAMA_7B_WEIGHTS,
  LLAMA3_70B_WEIGHTS,
  LLAMA3_8B_WEIGHTS,
  LLAMA3_8B_CKPT_WEIGHTS
)

BASE_PATH = "/mnt/raid0/llamas/llama_base"

CONST_ALLOWED_BASE_MODELS = {
  "code-llama-13b-instruct": CODE_LLAMA_13B_WEIGHTS,
  "code-llama-34b-instruct": CODE_LLAMA_34B_WEIGHTS,
  "code-llama-7b-instruct": CODE_LLAMA_7B_WEIGHTS,
  "llama-3-70b": LLAMA3_70B_WEIGHTS,
  "llama-3-8b": LLAMA3_8B_WEIGHTS,
  "llama-3-8b-ckpt": LLAMA3_8B_CKPT_WEIGHTS,

}

def load_base_model(base_model: str,
                    rank0_only: bool,
                    base_path: str = BASE_PATH,
                    max_seq_len: int = None) -> Transformer | None:
  weights = CONST_ALLOWED_BASE_MODELS.get(base_model)
  if max_seq_len:
    weights.max_seq_len = max_seq_len
  assert weights != None 

  if rank0_only:
    model = weights.load(base_path) if unakin_dist_utils.is_master() else None
    return model
  
  return weights.load(base_path)

@dataclass
class BaseModelConfig:
  model: str
  tokenizer: str
  base_path: str = BASE_PATH

def parse_base_model_config(file: str) -> BaseModelConfig:
  with open(file, "r") as f:
    # file is a yaml file
    config = yaml.safe_load(f)
  
  model_config = BaseModelConfig(**config)
  assert model_config.model in CONST_ALLOWED_BASE_MODELS, f"base_model must be one of {CONST_ALLOWED_BASE_MODELS.keys()}"

  return model_config
