import json
import os
import torch
from torch import nn
from dataclasses import dataclass

from unakin.llm.models.llama import LlamaTransformer
from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer


@dataclass
class ModelJsonTemplate:
  
  model: dict
  tokenizer: dict

  # optional resume arguments
  model_path: str | None = None
  ckpt: dict = None
  base: str | None = None

  @staticmethod
  def is_valid_json(json_val: dict) -> bool:
    try:
      ModelJsonTemplate(**json_val)
      return True
    except Exception:
      return False

class ModelFactory:

  base_tokenizer = {
    "code-llama-7b-instruct": "/mnt/raid0/llamas/codellama/CodeLlama-7b-Instruct/tokenizer.model",
    "code-llama-13b-instruct": "/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model",
    "code-llama-34b-instruct": "/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model",
  }

  @staticmethod
  def get_model_tokenizer_path(config: dict) -> str:
    tokenizer_path = config["tokenizer"].get("path", None)
    pretrained_base_model = config["model"]["args"].get("base", None)

    print(f"XxxXXXXXXXXXXX Tokenizer path: {tokenizer_path}")

    if tokenizer_path is not None:
      model_tokenizer_path = tokenizer_path
    elif pretrained_base_model is not None:
      model_tokenizer_path = ModelFactory.base_tokenizer[pretrained_base_model]
    else:
      raise ValueError("Missing tokenizer path or model base pretrained")
    
    return model_tokenizer_path

  @staticmethod
  def from_config_file(config_file: str):
    if config_file.endswith('.json'):
      return ModelFactory.from_json(config_file)

  @staticmethod
  def from_json(json_path_or_dict: str | dict):
    if isinstance(json_path_or_dict, str):
      with open(json_path_or_dict) as f:
        config = json.load(f)
    else:
      config = json_path_or_dict

    if not ModelJsonTemplate.is_valid_json(config):
      raise ValueError("Invalid JSON config file")

    if config["model"]["type"] == "llama":

      if config["model"].get("args") is None or config["model"]["args"] == {}:
        raise ValueError("Missing model args")

      model = LlamaTransformer.from_json_file(path_or_json=config["model"]["args"])

      model_tokenizer_path = ModelFactory.get_model_tokenizer_path(config=config)

      print(f"Loading tokenizer from {model_tokenizer_path}")

      tokenizer = LlamaTokenizer(model_path=model_tokenizer_path)

      if tokenizer.vocab_size - model.tok_embeddings.weight.shape[0] == 1:
        print("Adding reverse embedding")
        # 0 mean, 0.02 std
        rev_emb = nn.Embedding(1, 5120)
        nn.init.normal_(rev_emb.weight, mean=0, std=0.02)
        model.tok_embeddings.weight = nn.Parameter(torch.cat([model.tok_embeddings.weight, rev_emb.weight], dim=0))

        output_lin = nn.Linear(5120, 1)
        nn.init.normal_(output_lin.weight, mean=0, std=0.025)
        model.output.weight = nn.Parameter(torch.cat([model.output.weight, output_lin.weight], dim=0))

      assert model.tok_embeddings.weight.shape[0] == tokenizer.vocab_size
      assert model.output.weight.shape[0] == tokenizer.vocab_size
      
    else:
      raise ValueError(f'Unknown model type {config["model"]["type"]}')

    return model, tokenizer


if __name__ == "__main__":
  model, tokenizer = ModelFactory.from_config_file(config_file="/home/fsuser/workdir/ncs/ml-research/recipes/configs/models/default_conf.json")