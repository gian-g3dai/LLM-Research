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

def make_dialogue_message(content: str, system_prompt: str | None):
  user_message = {"role": "user", "content": content}
  if system_prompt is None:
    return [user_message]
  
  system_message = {"role": "system", "content": system_prompt}
  return [system_message, user_message]


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

  # cehckpoint arguments
  parser.add_argument("--ckpt-root", type=str, default="/home/ted/workdir/ncs/ml-research/recipes")
  parser.add_argument("--ckpt-name", type=str, default="checkpoints/l4_code_oasst_muffin_mix")
  parser.add_argument("--ckpt-step", type=str, default="000916")
  
  args = parser.parse_args()

  init_torch(args)

  dialogue_model = get_dialogue_model_from_ckpt(
      ckpt_root=args.ckpt_root,
      ckpt_name=args.ckpt_name,
      ckpt_step=args.ckpt_step,
      use_refference=args.use_refference,
  )

  code = """
  using UnityEngine;
using UnityEngine.UI;

public class RoundedButton : MonoBehaviour
{
    public float cornerRadius = 10.0f;
    public Color buttonColor = Color.white;
    public Color hoverColor = Color.gray;
    public Color pressedColor = Color.black;

    private Button button;

    void Start()
    {
        button = GetComponent<Button>();
        button.onClick.AddListener(OnClick);
    }

    void OnClick()
    {
        Debug.Log("Button clicked!");
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Vector2 mousePos = Input.mousePosition;
            if (button.Contains(mousePos))
            {
                button.GetComponent<Image>().color = pressedColor;
            }
        }
        else if (Input.GetMouseButtonUp(0))
        {
            button.GetComponent<Image>().color = buttonColor;
        }
    }

    void OnGUI()
    {
        GUI.Box(new Rect(0, 0, 100, 100), "Button", buttonStyle);
    }

    void OnDestroy()
    {
        button.onClick.RemoveListener(OnClick);
    }
}"""

  system_prompt = None # "You are a problem description creator. Engineers provide you solutions for which you have to reverse-engineer the problem description."
  instruction = f"Given the following code, generate the appropriate instructions for the problem:\n```\n{code}```"

  # /home/ted/workdir/ncs/ml-research/recipes/checkpoints/l4_code_oasst_mix/000256
  message = make_dialogue_message(
    content=instruction, system_prompt=system_prompt
  )

  prompt_prediction = dialogue_model.chat_completion(
    [message], temperature=0.2, top_p=0.9, max_gen_len=2048, repeat_penalty=1.2)
  
  print(prompt_prediction)