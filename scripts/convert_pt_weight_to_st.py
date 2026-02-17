import argparse

import safetensors.torch
import torch

if __name__ == "__main__":
  base_path = "/mnt/raid0/llamas/llama_base/"
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--pt-weight", type=str, required=False, default=f"{base_path}/code-llama-13b-instruct.pth")
  parser.add_argument("--st-weight", type=str, required=False, default=f"{base_path}/code-llama-13b-instrcut.st")

  args = parser.parse_args()

  pt_weight = torch.load(args.pt_weight)
  st_weight = safetensors.torch.save_file(pt_weight, args.st_weight)