from __future__ import annotations

import os
import torch
import random


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

from unakin.llm.tokenizers import TokenizerBase, LlamaTokenizer, Llama3Tokenizer

from unakin.llm.oss.llama3.llama import Dialog as DialogLlama3
from unakin.llm.oss.llama3.llama.tokenizer import ChatFormat as ChatFormatLlama3


from .atomic import RawData, Templater
from .train_sample import LMTrainingSample, PromptTokensSample, prompt_to_training_sample

B_INST, E_INST = "[INST]", "[/INST]"

class LMDataSource(ABC, torch.utils.data.Dataset):
  
  def __init__(self,
              raw_data: RawData,
              templaters: List[Templater], 
              tokenizer: TokenizerBase,
              data_root: str,
              data_path: str,
              percent_interval: Tuple[int, int],
              metric_list: List[str],
              fim_rev_config: Dict[str, Any] = None,
              *args, **kwargs) -> None:
    super().__init__()
    
    self.raw_data = raw_data
    self.templaters = templaters
    self.tokenizer = tokenizer

    if isinstance(tokenizer, LlamaTokenizer):
      self.type = "llama2"
      self.chat_formatter = None
    elif isinstance(tokenizer, Llama3Tokenizer):
      self.type = "llama3"
      self.chat_formatter = ChatFormatLlama3(tokenizer.tokenizer)
    else:
      raise NotImplementedError(f"Must provide supproted tokenizers (Llama2/3)")

    self.fim_rev_config = fim_rev_config

    if metric_list is None:
      self.metric_list = []
    else:
      self.metric_list = metric_list

    # get the columns that are needed from the raw data
    self._columns_needed = []
    for templater in templaters:
      self._columns_needed.append(templater.get_columns_needed())

    # check if the columns exist in the raw data
    for column_list in self._columns_needed:
      if set(column_list) - set(self.raw_data.columns) != set():
        raise ValueError(f"Columns {column_list} are not present in the raw data columns: {self.raw_data.columns}.")

    # slice the data
    if percent_interval[0] > 0 or percent_interval[1] < 100:
      skip_rows = int(len(self.raw_data.data) * percent_interval[0] / 100)
      use_rows = int(len(self.raw_data.data) * percent_interval[1] / 100)
      self.raw_data.data = self.raw_data.data.slice(skip_rows, use_rows)

  @abstractmethod
  def __getitem__(self, index: int) -> LMTrainingSample:
    # gets a row of data, and is responsible for converting it into a (input, output) pair

    # required steps:
    # 1. know what columns of the data are required
    # 2. template them into a string that can be tokenized
    raise NotImplementedError
  
  def __len__(self) -> int:
    return len(self.raw_data.data)
  
class SimpleLMDataSource(LMDataSource):
  
  def __init__(self, raw_data: RawData, templaters: List[Templater], *args, **kwargs) -> None:
    assert len(templaters) == 1, "`SimpleLMDataSource` requires 1 templater"
    super().__init__(raw_data, templaters, *args, **kwargs)

  def __getitem__(self, index: int) -> LMTrainingSample:

    row = self.raw_data.get_index(index)

    code_template = self.templaters[0]
    code_columns = self._columns_needed[0]

    # create mappings col: row_data
    code_mapping = {
      column: row[column][0] for column in code_columns
    }

    # format the row data
    code = code_template.format(code_mapping)

    # tokenize
    code_tok = self.tokenizer.tokenize(code, False, False)[0]

    pt_sample = PromptTokensSample([], code_tok)

    training_sample = prompt_to_training_sample(pt_sample,
                                                min_token_length=1,
                                                bos_token=self.tokenizer.bos_token,
                                                eos_token=self.tokenizer.eos_token,
                                                fim_rev_config=self.fim_rev_config
                                                )

    return training_sample
  
  def __len__(self) -> int:
    return len(self.raw_data.data)
  
  
class InstructionLMDataSource(LMDataSource):
  
  def __init__(self, raw_data: RawData, templaters: List[Templater], *args, **kwargs) -> None:
    assert len(templaters) == 2, "`InstructionLMDataSource` requires 2 templaters"
    super().__init__(raw_data, templaters, *args, **kwargs)

  def __getitem__(self, index: int) -> LMTrainingSample:

    row = self.raw_data.get_index(index)

    # assign the templaters
    instruction_template = self.templaters[0]
    output_template = self.templaters[1]

    # columns needed for the templaters
    instruction_columns = self._columns_needed[0]
    output_columns = self._columns_needed[1]

    # create mappings col: row_data
    instruction_mapping = {
      column: row[column][0] for column in instruction_columns
    }
    output_mapping = {
      column: row[column][0] for column in output_columns
    }

    # format the row data
    instruction = instruction_template.format(instruction_mapping)
    output = output_template.format(output_mapping)

    if self.type == "llama2":
      instruction_tok = self.tokenizer.tokenize(instruction, False, False)[0]
    elif self.type == "llama3":
      if "system_prompt" in row:
        instruction_tok = self.chat_formatter.encode_dialog_prompt([
          {"role": "system", "content": row["system_prompt"][0]},
          {"role": "user", "content": instruction}
          ])
      else:
        instruction_tok = self.chat_formatter.encode_dialog_prompt([{"role": "user", "content": instruction}])

    output_tok = self.tokenizer.tokenize(output, False, False)[0]

    pt_sample = PromptTokensSample(instruction_tok, output_tok)

    training_sample = prompt_to_training_sample(pt_sample,
                                                min_token_length=len(instruction_tok),
                                                bos_token=self.tokenizer.bos_token,
                                                eos_token=self.tokenizer.tokenizer.special_tokens["<|eot_id|>"],
                                                fim_rev_config=self.fim_rev_config
                                                )

    return training_sample
  
  def __len__(self) -> int:
    return len(self.raw_data.data)
  

class MultiTurnLMDataSource(LMDataSource):
  """
  Multi-turn LM data source
  assistant_messages: List[str]
  user_messages: List[str]
  
  assistant_messages: a1, a2, a3
  user_messages:      u1, u2, u3
  """
  def __init__(self, raw_data: RawData, templaters: List[Templater], *args, **kwargs) -> None:
    assert len(templaters) == 2, "`MultiTurnLMDataSource` requires 2 templaters"
    super().__init__(raw_data, templaters, *args, **kwargs)
    # just for this one  - transform to pandas
    self.raw_data.data = self.raw_data.data.to_pandas()

  def __getitem__(self, index: int) -> LMTrainingSample:

    row = self.raw_data.data.iloc[index]

    asst = row["assistant_messages"]
    user = row["user_messages"]

    # assign the templaters
    user_template = self.templaters[0]
    asst_template = self.templaters[1]

    dialogue_mask = []
    dialogue_tokens = []

    if self.type == "llama3":
      dialogue_mask = [0]
      dialogue_tokens = [self.tokenizer.bos_token]

    for user_msg, asst_msg in zip(user, asst):
      # format the row data

      # create mappings col: row_data
      user_mapping = {
        "user_messages": user_msg.strip(),
      }
      assistans_mapping = {
        "assistant_messages": asst_msg.strip(),
      }

      user_msg = user_template.format(user_mapping)
      asst_msg = asst_template.format(assistans_mapping)

      
      if self.type == "llama2":
        # tokenize
        user_msg_tok = self.tokenizer.tokenize(f"{B_INST} {(user_msg).strip()} {E_INST} ", True, False)[0]
        asst_msg_tok = self.tokenizer.tokenize(f"{(asst_msg).strip()} ", False, True)[0]
        
      elif self.type == "llama3":
        user_msg_tok = self.chat_formatter.encode_message({"role": "user", "content": user_msg}) + \
                                        self.chat_formatter.encode_header({"role": "assistant", "content": ""})
        asst_msg_tok = self.tokenizer.tokenize(asst_msg, False, False)[0]
        asst_msg_tok.append(self.tokenizer.tokenizer.special_tokens["<|eot_id|>"])

      dialogue_tokens.extend(user_msg_tok)
      dialogue_tokens.extend(asst_msg_tok)

      dialogue_mask.extend([0] * len(user_msg_tok))
      dialogue_mask.extend([1] * len(asst_msg_tok))


    input_tokens = dialogue_tokens[:-1]
    dialogue_mask = dialogue_mask[1:]
    output_tokens = dialogue_tokens[1:]
    min_token_length = len(input_tokens)

    multi_turn_sample = LMTrainingSample(input_tokens, output_tokens, dialogue_mask, min_token_length)

    return multi_turn_sample
  
  def __len__(self) -> int:
    return len(self.raw_data.data)
