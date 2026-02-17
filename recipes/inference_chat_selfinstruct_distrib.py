import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict

import polars as pl
import torch
import torch.distributed
from finetuning import CheckpointManager
from tqdm import tqdm

from unakin.llm.adapters.llm_adapter import LlmAdapter
from unakin.llm.data.datasources import CSharpFilesChatsource
from unakin.llm.data.datasources.utils import GlobbingMethod
from unakin.llm.dialogue.base import DialogueModel
from unakin.llm.models.utils import ModelFactory
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
      "path": "/home/fsuser/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model"
    }})

  model.enable_xops(enable=True)
  model.register_cache_kv(max_bsz=1, max_seq_len=16384)

  # model.to(torch.cuda.current_device())

  return DialogueModel(model=model, tokenizer=tokenizer)


def get_dialogue_model_from_ckpt(path: str, get_dev_model: bool = True) -> DialogueModel:
  ckpt_path = f"{path}/ckpt.pth"

  if get_dev_model:
    return _get_dev_model()

  ckpt_manager = CheckpointManager(path)
  
  # model
  model, tokenizer = ckpt_manager.init_model_from_ckpt(
    ckpt_path, root_path="/home/fsuser/workdir/ncs/ml-research/recipes")
  
  # adapter
  model: LlmAdapter
  model.merge_adaptation()

  # create the dialogue model
  dialogue_model = DialogueModel(model=model, tokenizer=tokenizer)

  return dialogue_model


def make_dialogue_model(model: torch.nn.Module, tokenizer: TokenizerBase) -> DialogueModel:
  return DialogueModel(model=model, tokenizer=tokenizer)

#########################################
#       Distributed utils               #
#########################################

def setup_dist():
  """Setup distributed process group"""
  torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
  )


def cleanup_dist():
  """Cleanup distributed process group"""
  if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()


def setup_print(rank: int, debug: bool = False):
  """Setup stdout/stderr redirection"""
  if debug:
    return

  if rank != 0:
    f = open(os.devnull, "w")
    sys.stdout = f
    sys.stderr = f


def main(model_ckpt: str, skip_path: bool = False, do_rating: bool = False):

  setup_dist()
  rank = torch.distributed.get_rank()
  print(f"Rank: {rank}")

  torch.set_default_dtype(torch.bfloat16) # TODO here?
  torch.cuda.set_device(torch.device(f"cuda:{rank}"))

  setup_print(rank, debug=args.debug)
  torch.distributed.barrier()

  dialogue_model = get_dialogue_model_from_ckpt(
    model_ckpt, skip_path)

  dialogue_model.model = dialogue_model.model.to(torch.cuda.current_device())

  chatsource = CSharpFilesChatsource(
      root='/home/fsuser/datasets/games_scraped',
      globbing_method=GlobbingMethod.user_only,
  )

  # get current rank
  rank = torch.distributed.get_rank()
  total_len = len(chatsource)

  # slice chatsource to current rank
  chatsource = chatsource.slice(
    start_index=rank * (total_len // torch.distributed.get_world_size()),
    end_index=min(
      (rank + 1) * (total_len // torch.distributed.get_world_size()),
      total_len
    )
  )

  new_cache = chatsource.cache.clone()

  descriptions = []
  ratings = []

  for i in tqdm(range(len(chatsource)), position=rank + 1):
    raw_sample = chatsource[i]
    num_tokens = new_cache.row(i, named=True)['num_tokens']

    if num_tokens > 8192:
      descriptions.append("")
      continue

    dialogs = [
      [
        {"role": "system", "content": "You are a software architect. You provide very high level and functional descriptions of code. You talk about the code structure from a product feature perspective.\
          You never describe functions, classes or methods individually. Only provide a specification summary of what the script does as a singular workflow or process."},
        {"role": "user", "content": "Describe the functional features of this Unity C# script and describe the most important aspect of it:\n" + raw_sample},
      ],
    ]

    predictions = dialogue_model.chat_completion(
      dialogs, temperature=0.2, repeat_penalty=1.1, max_gen_len=1024)

    description = predictions[0]["generation"]["content"]
    # remove trailing spaces
    description = description.strip()
    print(description)

    descriptions.append(description)

    if do_rating:
      
      # paper_prompt = """Below is an instruction from an user and a candidate answer. Evaluate whether or
      # not the answer is a good example of how AI Assistant should respond to the user’s
      # instruction. Please assign a score using the following 5-point scale:
      # 1: It means the answer is incomplete, vague, off-topic, controversial, or not
      # exactly what the user asked for. For example, some content seems missing, numbered
      # list does not start from the beginning, the opening sentence repeats user’s question.
      # Or the response is from another person’s perspective with their personal experience
      # (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains
      # promotional text, navigation text, or other irrelevant information.
      # 2: It means the answer addresses most of the asks from the user. It does not
      # directly address the user’s question. For example, it only provides a high-level
      # methodology instead of the exact solution to user’s question.
      # 3: It means the answer is helpful but not written by an AI Assistant. It addresses
      # all the basic asks from the user. It is complete and self contained with the
      # drawback that the response is not written from an AI assistant’s perspective, but
      # from other people’s perspective. The content looks like an excerpt from a blog post,
      # web page, or web search results. For example, it contains personal experience or
      # opinion, mentions comments section, or share on social media, etc.
      # 4: It means the answer is written from an AI assistant’s perspective with a
      # clear focus of addressing the instruction. It provide a complete, clear, and
      # comprehensive response to user’s question or instruction without missing or
      # irrelevant information. It is well organized, self-contained, and written in a
      # helpful tone. It has minor room for improvement, e.g. more concise and focused.
      # 5: It means it is a perfect answer from an AI Assistant. It has a clear focus on
      # being a helpful AI Assistant, where the response looks like intentionally written
      # to address the user’s question or instruction without any irrelevant sentences. The
      # answer provides high quality content, demonstrating expert knowledge in the area, is
      # very well written, logical, easy-to-follow, engaging and insightful.\n\n
      # Please first provide a brief reasoning you used to derive the rating score, and
      # then write "Score: <rating>" in the last line.\n"""

      our_prompt = """Below is a code snippet and a candidate description for it. 
      Evaluate whether or not the description could be used to reconstruct the code snippet.\n
      Please assign a score using a scale of 1 to 5 defined as:\n
      1. There is not enough information to construct a working code snippet.\n
      2. The description is missing some information, but it is possible to construct a working code snippet.\n
      3. A working code snippet can be constructed from the description, 
      but we don't know if it will respect the functionality of the original code snippet.\n
      4. The code snippet can be reconstructed from the description and it will respect 
      the functionality of the original code snippet, however it is missing some details.\n
      5. The code snippet can be fully reconstructed using the description and it will respect
      the functionality of the original code snippet.\n\n

      Output only "Score: <rating>" on the last line.\n
      For example, the output should look like this:\n
      <user code>\n
      <user description>\n
      <your rating reasoning>\n
      Score: <your score from 1 to 5>\n
      """

      dialogs = [
        [
          {"role": "user", "content": our_prompt + '\n' + description + '\n' + raw_sample + 
            '\nPlease first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line.\n'},
        ],
      ]

      predictions = dialogue_model.chat_completion(
          dialogs, temperature=0.2, max_gen_len=1024)

      description = predictions[0]["generation"]["content"]
      # remove trailing spaces
      description = description.strip()
      print(description)

      # extract rating from last line: Score: <rating>
      pattern = r'Score: ([1-5])'

      # Find all occurrences of the pattern in the string
      matches = re.findall(pattern, description)

      # If there are matches, find the last one
      if matches:
          rating = int(matches[-1])
      else:
          print("Pattern not found in the string.")
          rating = 0

      ratings.append(rating)

  # add descriptions as a separate column to new_cache
  new_cache = new_cache.with_columns(pl.Series("metaprompt", descriptions))

  # remove rows where description is empty
  new_cache = new_cache.filter(pl.col("metaprompt") != "")

  if do_rating:
    # add scores as a separate column to new_cache
    new_cache = new_cache.with_columns(pl.Series("rating", ratings))
    # remove rows where scoring failed
    new_cache = new_cache.filter(pl.col("rating") != 0)

  # write new_cache to a parquet file
  if do_rating:
    new_cache.write_parquet(f"metaprompt_cache_tmp_{rank}_rating.parquet")
  else:
    new_cache.write_parquet(f"metaprompt_cache_tmp_{rank}.parquet")
  
  # wait for all processes to write their own parquet cache
  torch.distributed.barrier()
  print("Combining caches...")

  # recombine cache in the rank 0 worker
  if rank == 0:
    # build a large cache from all the smaller ones
    all_caches = []
    for i in range(torch.distributed.get_world_size()):
      if do_rating:
        cache_path = f"metaprompt_cache_tmp_{i}_rating.parquet"
      else:
        cache_path = f"metaprompt_cache_tmp_{i}.parquet"
      cache = pl.read_parquet(cache_path)
      all_caches.append(cache)

    large_cache = pl.concat(all_caches)

    # write the large cache to a parquet file
    # get home directory
    home = str(Path.home())
    if do_rating:
      path = f"{home}/datasets/metaprompts_rating.parquet"
    else:
       path = f"{home}/datasets/metaprompts.parquet"
    # if it exists create a new version numbered file
    if os.path.exists(path):
      # glob for all files with the same name
      if do_rating:
        pattern = f"{home}/datasets/metaprompts*_rating.parquet"
      else:
        pattern = f"{home}/datasets/metaprompts*.parquet"
      
      files = glob.glob(pattern)
      # get the highest version number
      new_version = len(files)
      # rename the old file
      if do_rating:
        path = f"{home}/datasets/metaprompts_{new_version}_rating.parquet"
      else:
        path = f"{home}/datasets/metaprompts_{new_version}.parquet"
      
    large_cache.write_parquet(path)

    for rank in range(torch.distributed.get_world_size()):
      if do_rating:
        cache_path = f"metaprompt_cache_tmp_{rank}_rating.parquet"
      else:
        cache_path = f"metaprompt_cache_tmp_{rank}.parquet"
      os.remove(cache_path)

  # cleanup temp files
  torch.distributed.barrier()
  cleanup_dist()

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_ckpt", type=str, default="/home/fsuser/workdir/ncs/ml-research/recipes/checkpoints/copper-water-252/000016",
                      help="Path to model config file")
  parser.add_argument("--debug", action="store_true", default=True,
                      help="Debug mode")
  parser.add_argument("--skip_path", action="store_true",
                      default=False, help="Skip loading the model checkpoint") 
  parser.add_argument("--do_rating", action="store_true", default=True, help="Rating as in the Self-Alignment with Instruction Backtranslation paper")
  args = parser.parse_args()

  main(args.model_ckpt, args.skip_path, args.do_rating)
