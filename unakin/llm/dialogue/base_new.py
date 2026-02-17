from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.distributed
import torch.nn.functional as F
from tqdm import tqdm

from unakin.llm.models.llama.model_compile import CODE_LLAMA_13B_WEIGHTS, Transformer, KVCache, LLAMA3_70B_WEIGHTS, LLAMA3_8B_WEIGHTS
from unakin.llm.tokenizers import TokenizerBase
from unakin.llm import BaseLLM

from unakin.llm.models.llama.model_compile import Transformer, TransformerBlock
from unakin.llm.tokenizers import LlamaTokenizer, Llama3Tokenizer

from unakin.llm.dialogue._generate import decode_tokens
from unakin.llm.distributed.utils import (
  PyObjShard,
  dist_partition_py_type,
  dist_gather_py_type,
  get_rank,
  print_rank0,
  setup_distributed_process_group,
  destroy_process_group,
  DistributedList,
  ParallelSpec
)
from unakin.llm.distributed.model_parallelism import apply_tensor_parallelism_to_model

from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
  role: Role
  content: str


class CompletionPrediction(TypedDict, total=False):
  generation: str
  tokens: List[str]  # not required
  logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
  generation: Message
  tokens: List[str]  # not required
  logprobs: List[float]  # not required


Dialog = List[Message]

class DialogueModelBase(ABC):
  def __init__(self) -> None:
      super().__init__()

  @abstractmethod
  def chat_completion(
      self,
      dialogs: List[Dialog],
      temperature: float = 0.6,
      top_p: float = 0.9,
      repeat_penalty: float = 1.0,
      max_gen_len: Optional[int] = None,
      logprobs: bool = False,
  ) -> List[ChatPrediction]:
      raise NotImplementedError
    
class DialogueModel(DialogueModelBase):
    
  def __init__(
    self,
    model: Transformer,
    tokenizer: TokenizerBase,
    parallel_spec: ParallelSpec = ParallelSpec.TENSOR_PARALLEL
  ) -> None:
    super().__init__()
    self.model = model
    self.tokenizer = tokenizer
    self.parallel_spec = parallel_spec

    if self.parallel_spec == ParallelSpec.TENSOR_PARALLEL:
      self.model = apply_tensor_parallelism_to_model(self.model)

    self.model.eval()
    self.model = self.model.cuda()

  def to(self,
          device: Optional[torch.device] = "cpu",
          dtype: Optional[torch.dtype] = torch.float) -> None:
    self.model.to(device, dtype)

  def text_completion(
    self,
    texts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = 4_096
  ) -> List[str]:
    
    dist_l = DistributedList(texts, self.parallel_spec)
    local_texts = dist_l.shard()

    # edge case: less than 1 text per rank
    if len(local_texts) != 0:
      prompt_tokens = self.tokenizer.tokenize(local_texts, bos=True, eos=False)
      local_result = decode_tokens(
        model=self.model,
        tokens=prompt_tokens,
        tokenizer=self.tokenizer,
        stop_tokens=self.tokenizer.stop_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
      )
    else:
      local_result = []

    global_result = dist_l.gather(local_result)
    return global_result

  def chat_completion(
    self,
    dialogs: List[Dialog],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = 4_096,
    cache: Optional[KVCache] = None,
  ) -> List[ChatPrediction]:
      
    dist_l = DistributedList(dialogs, self.parallel_spec)
    local_dialogs = dist_l.shard()
    
    if len(local_dialogs) != 0:
      dialogs_tokens = []
      unsafe_dialog_requests = []
      for dialog_prompt in local_dialogs:
        prompt_tokens, unsafe_request = self.tokenizer.chat_formatter.encode_dialog_prompt(dialog_prompt)
        dialogs_tokens.append(prompt_tokens)
        unsafe_dialog_requests.append(unsafe_request)

      local_result = decode_tokens(
        model=self.model,
        tokens=dialogs_tokens,
        tokenizer=self.tokenizer,
        stop_tokens=self.tokenizer.stop_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
      )
    else:
      local_result = []

    global_result = dist_l.gather(local_result)
    return global_result

def str_to_prompt(prompt: str) -> Dialog:
  return [{"role": "user", "content": prompt}]


if __name__ == "__main__":

  setup_distributed_process_group()

  base_path = "/mnt/raid0/llamas/llama_base"
  transformer = Transformer.from_base_model(
    LLAMA3_8B_WEIGHTS, # CODE_LLAMA_13B_WEIGHTS,
    base_path=base_path,
  )

  tokenizer = Llama3Tokenizer(model_path="/mnt/raid0/llamas/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model")
  dialogue_model = DialogueModel(transformer, tokenizer, parallel_spec=ParallelSpec.PROCESS)
  
  prompts = [
    "Do goats make good pets? ",
    "Complete this function:  def add(a, b) -> int:",
  ]

  dialogue_prompts = [str_to_prompt(p) for p in prompts]

  for t in dialogue_model.chat_completion(
    dialogue_prompts, max_gen_len=150, temperature=0.0, top_p=0.0):
    print_rank0(t, "\n")

  destroy_process_group()

# if __name__ == "__main__":

#   setup_distributed_process_group()

#   base_path = "/mnt/raid0/llamas/llama_base"
#   transformer = Transformer.from_base_model(
#     LLAMA3_70B_WEIGHTS, # CODE_LLAMA_13B_WEIGHTS,
#     base_path=base_path,
#   )

#   tokenizer = Llama3Tokenizer(model_path="/mnt/raid0/llamas/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model")
#   dialogue_model = DialogueModel(transformer, tokenizer, parallel_spec=ParallelSpec.TENSOR_PARALLEL)

#   import re
#   import pickle
#   import pandas as pd
#   pattern = r'<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)<\|eot_id\|>'

#   batch_size = 64
#   df = pd.read_parquet("/mnt/raid0/datasets/oasst_rw/oasst_subtree.parquet")

#   df = df.sort_values(by='last_asst_attr')

#   num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

#   platy_rw = []
#   for i in tqdm(range(num_batches)):
#     start_idx = i * batch_size
#     end_idx = start_idx + batch_size
#     df_chunk = df[start_idx:end_idx]

#     dialogs = []
#     for idx, row in df_chunk.iterrows():
#       curr_dialog = []
#       for user_m, asst_m in zip(row["user_messages"], row["assistant_messages"]):
#         curr_dialog.append({"role": "user", "content": user_m})
#         curr_dialog.append({"role": "assistant", "content": asst_m})
#       curr_dialog.append({"role": "user", "content": "Rewrite the previous reponse as a high quality AI assistant would respond like. Do not reference that the response is rewritten."})
#       dialogs.append(curr_dialog)

#     for t in dialogue_model.chat_completion(
#       dialogs, max_gen_len=4096, temperature=0.0, top_p=0.0):

#       if torch.distributed.get_rank() == 0:
#         matches = re.findall(pattern, t, re.DOTALL)
#         if len(matches) % 2 != 0:
#           print("NOT COMPLETE GENERATION")
#           continue
        
#         platy_rw.append(matches)

#       if torch.distributed.get_rank() == 0:
#         with open("/mnt/raid0/datasets/oasst_rw/rewrite.pkl", "wb") as f:
#           pickle.dump(platy_rw, f)

#   destroy_process_group()

################################## USED FOR PLATYPUS
# if __name__ == "__main__":

#   setup_distributed_process_group()

#   base_path = "/mnt/raid0/llamas/llama_base"
#   transformer = Transformer.from_base_model(
#     LLAMA3_70B_WEIGHTS, # CODE_LLAMA_13B_WEIGHTS,
#     base_path=base_path,
#   )

#   tokenizer = Llama3Tokenizer(model_path="/mnt/raid0/llamas/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model")
#   dialogue_model = DialogueModel(transformer, tokenizer, parallel_spec=ParallelSpec.TENSOR_PARALLEL)

#   import re
#   import pickle
#   import pandas as pd
#   pattern = r'<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)<\|eot_id\|>'

#   batch_size = 64
#   platy_df = pd.read_parquet("/mnt/raid0/datasets/platy/train.parquet")

#   platy_df = platy_df.assign(total_length=lambda x: x['instruction'].str.len() + x['output'].str.len()).sort_values(by='total_length', ascending=False).drop(columns=['total_length'])

#   num_batches = len(platy_df) // batch_size + (1 if len(platy_df) % batch_size != 0 else 0)

#   platy_rw = []
#   for i in tqdm(range(num_batches)):
#     start_idx = i * batch_size
#     end_idx = start_idx + batch_size
#     platy_df_chunk = platy_df[start_idx:end_idx]

#     platy_dialogs = []
#     for idx, row in platy_df_chunk.iterrows():
#       platy_dialogs.append([
#         {"role": "user", "content": row["instruction"]},
#         {"role": "assistant", "content": row["output"]},
#         {"role": "user", "content": "Rewrite the previous reponse as an academic would respond like. Retain the formatting that was used to respond such as LaTeX. Do not reference that the response is rewritten."},
#       ])

#     for t in dialogue_model.chat_completion(
#       platy_dialogs, max_gen_len=4096, temperature=0.0, top_p=0.0):

#       if torch.distributed.get_rank() == 0:
#         matches = re.findall(pattern, t, re.DOTALL)
#         if len(matches) % 2 != 0:
#           print("NOT COMPLETE GENERATION")
#           continue
        
#         platy_rw.append(matches)

#       if torch.distributed.get_rank() == 0:
#         with open("/mnt/raid0/datasets/platy/rewrite.pkl", "wb") as f:
#           pickle.dump(platy_rw, f)

#   destroy_process_group()
