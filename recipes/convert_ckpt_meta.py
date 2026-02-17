import argparse
import os
import json

import gc
import shutil
import warnings

import torch
import torch.distributed
from finetuning import CheckpointManager
from unakin.llm.adapters.llm_adapter import LlmAdapter
import argparse

def main():

  parser = argparse.ArgumentParser()
  # /mnt/raid0/checkpoints/bigmix_ft_gpeay8pc/002366/ckpt.pth
  parser.add_argument("--unakin_model_ckpt_path", type=str, default="/mnt/raid0/checkpoints/bigmix_ft_q0fowtum/000864/ckpt.pth")
  parser.add_argument("--base_model_params", type=str, default="/home/fsuser/raid/llamas/codellama/CodeLlama-13b-Python/")
  parser.add_argument("--output_root_dir", type=str, default="/mnt/raid0/prebuild_checkpoints/")
  parser.add_argument("--output_dir", type=str, default="bigmix_ft_q0fowtum")
  args = parser.parse_args()

  # make out dir if not exists
  args.output_dir = os.path.join(args.output_root_dir, args.output_dir)
  os.makedirs(args.output_dir, exist_ok=True)

  ckpt_manager = CheckpointManager(args.unakin_model_ckpt_path)
  
  # model
  model, tokenizer = ckpt_manager.init_model_from_ckpt(path=args.unakin_model_ckpt_path, root_path=None)
  model: LlmAdapter

  # move model to device
  model = model.to(torch.cuda.current_device())
  model = model.half()

  # merge adaptation
  model.merge_adaptation()

  loaded = model.base_model.state_dict()

  # drop layers that end in A and B
  loaded = {k: v for k, v in loaded.items() if not k.endswith("A") and not k.endswith("B")}
  # replace ".adapted_linear" with ""
  loaded = {k.replace(".adapted_linear", ""): v for k, v in loaded.items()}

  params_path = os.path.join(args.base_model_params, "params.json")
  params_out = os.path.join(args.output_dir, "params.json")

  params = json.load(open(params_path, "r"))
  json.dump(params, open(params_out, "w"))

  torch.save(loaded, os.path.join(args.output_dir, "consolidated.00.pth"))



if __name__ == "__main__":
  main()