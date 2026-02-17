import argparse
from pathlib import Path
from typing import Dict

import safetensors.torch
import torch

from unakin.llm.adapters.llm_adapter import LlmAdapter
from unakin.llm.adapters.utils import AdapterConfigFactory
from unakin.llm.dialogue.base import DialogueModel
from unakin.llm.models.utils import ModelFactory
from unakin.llm.tokenizers.base import TokenizerBase


def get_ckpt(path: str, skip_path: bool = True) -> Dict:
  if skip_path:
    return {}
    
  ckpt = torch.load(f"{path}/ckpt.pth", map_location="cpu")
  return ckpt

def get_dev_model() -> DialogueModel:
  model, tokenizer = ModelFactory.from_json({
    "model": {
      "type": "llama",
      "args": {
        "base": "code-llama-13b-instruct"
      }
    }, "tokenizer": {
      "type": "llama",
      "path": "/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model"
    }})

  model.enable_xops(enable=True)
  # model.register_cache_kv(max_bsz=1, max_seq_len=4096)

  model = model.cuda()

  return DialogueModel(model=model, tokenizer=tokenizer)

def get_dialogue_model_from_ckpt(path: str, dev_mode: bool = True) -> DialogueModel:
  if dev_mode:
    return get_dev_model()
  
  ckpt = get_ckpt(path, dev_mode)

  # unwrap the checkpoint
  ckpt_args = ckpt["config"]

  model_state_dict_path = ckpt["model_state_dict_path"]
  model_state_dict = safetensors.torch.load_file(model_state_dict_path)

  model_config_dict = ckpt_args["model_config_expanded"]
  adapter_config_dict = ckpt_args["adapter_config_expanded"]

  # load the model
  model, tokenizer = ModelFactory.from_json(model_config_dict)
  # apply the adapter
  adapter_config = AdapterConfigFactory.from_json(adapter_config_dict)
  model = LlmAdapter(m=model, params=adapter_config)
  model.apply_adaptation()
  # loading in strict mode because this might only containt adapter weights
  model.load_state_dict(model_state_dict, strict=False)

  model = model.cuda()
  
  # create the dialogue model
  dialogue_model = DialogueModel(model=model, tokenizer=tokenizer)
  
  return dialogue_model

def make_dialogue_model(model: torch.nn.Module, tokenizer: TokenizerBase) -> DialogueModel:
  return DialogueModel(model=model, tokenizer=tokenizer)

if __name__ == "__main__":
  torch.cuda.set_device("cuda:0")
  torch.set_default_dtype(torch.float16)

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_ckpt", type=str, default="./checkpoints/default_multi_dataset_bsz2/000032", help="")
  parser.add_argument("--skip_path", action="store_true", default=False, help="Skip loading the model checkpoint")
  args = parser.parse_args()

  dialogue_model = get_dialogue_model_from_ckpt(args.model_ckpt, args.skip_path)

  predictions = dialogue_model.text_completion(["In unity what is the C# code necessary to make a game object change color every 5 seconds?"], echo=True)
  print(predictions)
