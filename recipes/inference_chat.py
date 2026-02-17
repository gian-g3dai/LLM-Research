import argparse
from pathlib import Path
from typing import Dict
import os
import json

import torch
import torch.distributed
from finetuning import CheckpointManager

from unakin.llm.adapters.llm_adapter import LlmAdapter
from unakin.llm.dialogue.base import DialogueModel
from unakin.llm.models.utils import ModelFactory
from unakin.llm.tokenizers import LlamaTokenizer
from unakin.llm.tokenizers.base import TokenizerBase

def get_ckpt(path: str, skip_path: bool = True) -> Dict:
  if skip_path:
    return {}

  ckpt = torch.load(f"{path}/ckpt.pt", map_location="cpu")
  return ckpt


def _get_dev_model() -> DialogueModel:
  model, tokenizer = ModelFactory.from_json({
    "model": {
      "type": "llama",
      "args": {
        "base": "code-llama-13b-instruct"
      }
    },
    "tokenizer": {
      "type": "llama",
      "path": "/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model"
    }})

  model.enable_xops(enable=True)
  model.to(torch.cuda.current_device())

  # model.to(torch.cuda.current_device())

  return DialogueModel(model=model, tokenizer=tokenizer)


def get_dialogue_model_from_ckpt(
    ckpt_root: str, ckpt_name: str, ckpt_step: str,
    use_refference: bool = False, batch_size: int = 1) -> DialogueModel:
  
  path = Path(ckpt_root) / ckpt_name / ckpt_step / "ckpt.pth"

  if use_refference:
    model = _get_dev_model()
    model.model.register_cache_kv(max_bsz=batch_size, max_seq_len=4096)
    return model

  ckpt_manager = CheckpointManager(path)
  
  # model
  model, tokenizer = ckpt_manager.init_model_from_ckpt(path=path, root_path=ckpt_root)
  model: LlmAdapter

  # move model to device
  model = model.to(torch.cuda.current_device())
  model = model.half()

  # merge adaptation
  model.merge_adaptation()
  model.register_cache_kv(max_bsz=batch_size, max_seq_len=4096)

  # create the dialogue model
  dialogue_model = DialogueModel(model=model, tokenizer=tokenizer)

  return dialogue_model

def read_json_lines(path: str):
  assert path.endswith(".jsonl")
  with open(path, "r") as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    lines = [line["task"] for line in lines]
    return lines

def read_system_prompt(path: str):
  if not path:
    return None

  with open(path, "r") as f:
    return f.read().strip()
  
def get_pad_dialogue(batch: list[list], target_size: int = 16, add_system_prompt: bool = True):
  pad = [{"role": "user", "content": "blank"}]
  if add_system_prompt:
    pad = [{"role": "system", "content": "blank"}] + pad
  
  for _ in range(target_size - len(batch)):
    batch.append(pad)
    
  return batch  


def init_torch(args):
  if args.world_size > 1:
    init_torch_distributed(args)
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
  else:
    device = torch.device(f"cuda:{args.device_id}")
  torch.cuda.set_device(device)
  torch.set_default_dtype(torch.float16)

def init_torch_distributed(args):
  init_torch(args)
  torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
  )

def make_dialogue_message(content: str, system_prompt: str | None):
  user_message = {"role": "user", "content": content}
  if system_prompt is None:
    return [user_message]
  
  system_message = {"role": "system", "content": system_prompt}
  return [system_message, user_message]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # basic arguments
  parser.add_argument("--world-size", type=int, default=1, help="number of nodes for distributed training")
  parser.add_argument("--device-id", type=int, default=0, help="GPU device id")
  parser.add_argument("--use-refference", action="store_true", default=False)

  # use openai model
  parser.add_argument("--openai-model", type=str, required=False)

  # cehckpoint arguments
  parser.add_argument("--ckpt-root", type=str, default="/home/ted/workdir/ted/ml-research/recipes")
  parser.add_argument("--ckpt-name", type=str, default="checkpoints/default_multi_dataset_bsz2")
  parser.add_argument("--ckpt-step", type=str, default="000032")

  # input data arguments
  parser.add_argument("--input-data-path", type=str, required=True)
  parser.add_argument("--system-prompt-path", type=str, required=False)
  parser.add_argument("--repeat-penaly", type=float, default=1.0, required=False)

  # batch size arguments
  parser.add_argument("--batch-size", type=int, default=16)
  
  args = parser.parse_args()

  init_torch(args)

  if args.openai_model:
    from unakin.llm.dialogue.dialogue_openai import DialogueModelOpenAI
    dialogue_model = DialogueModelOpenAI(
      api_key="TODO",
      model_name=args.openai_model
    )
  else:
    dialogue_model = get_dialogue_model_from_ckpt(
        ckpt_root=args.ckpt_root,
        ckpt_name=args.ckpt_name,
        ckpt_step=args.ckpt_step,
        use_refference=args.use_refference,
        batch_size=args.batch_size
    )

  predictions = []
  dialogs = []
  args.system_prompt = read_system_prompt(args.system_prompt_path)
  input_prompts = read_json_lines(args.input_data_path)

  if torch.distributed.is_initialized():
    world_size = torch.distributed.get_world_size()
    slice_size = len(input_prompts) // torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    start_index = local_rank * slice_size
    end_index = (local_rank + 1) * slice_size if local_rank < world_size - 1 else len(input_prompts)
    input_prompts = input_prompts[start_index:end_index]

  for i in range(0, len(input_prompts), args.batch_size):
    batch = input_prompts[i:i+args.batch_size]
    batch_size = len(batch)
    batch = [make_dialogue_message(line, args.system_prompt) for line in batch]
    batch = get_pad_dialogue(batch, target_size=args.batch_size)

    prompt_prediction = dialogue_model.chat_completion(
      batch, temperature=0.2, top_p=0.9, max_gen_len=2048, repeat_penalty=1.0)
    
    for p in prompt_prediction[:batch_size]:
      print(p["generation"]["content"])
      predictions.append(p["generation"]["content"])

  if torch.distributed.is_initialized():
    torch.distributed.barrier()
    with open(f"predictions/tmp.{local_rank}.predictions.jsonl", "w") as f:
      for p in predictions:
        f.write(json.dumps({"prediction": p})) + "\n"
    
    torch.distributed.barrier()
    if local_rank == 0:
      # concatenate all predictions
      os.system(f"cat predictions/tmp.*.predictions.jsonl > predictions/tmp.predictions.jsonl")
      with open(f"predictions/tmp.predictions.jsonl", "r") as f:
        predictions = [json.loads(line)["prediction"] for line in f.readlines()]

      # remove temporary files
      os.system(f"rm predictions/tmp.*.predictions.jsonl")
      os.system(f"rm predictions/tmp.predictions.jsonl")
  
  if torch.distributed.is_initialized() and local_rank != 0:
    exit(0)
  
  # save each independent request in text format for
  if args.openai_model:
    model_name = f"openai_{args.openai_model}"
  else:
    model_name = "ref" if args.use_refference else f"{args.ckpt_name}_{args.ckpt_step}"
    
  os.makedirs("predictions", exist_ok=True)
  os.makedirs(f"predictions/{model_name}", exist_ok=True)
  
  predictions_save_name = f"predictions/{model_name}.predictions.jsonl"
  
  with open(predictions_save_name, "w") as f:
    for p in predictions:
      f.write(json.dumps({"prediction": p}) + "\n")
  
  for i, p in enumerate(predictions):
    with open(f"predictions/{model_name}/{i}_predictions.txt", "w") as f:
      f.write(p)
  