
import torch
import random
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union

from unakin.llm.tokenizers import TokenizerBase
from itertools import groupby

from dataclasses import dataclass

import numpy as np

class LMTrainingSample(NamedTuple):
  input_tokens: List[int] = []
  output_tokens: List[int] = []
  training_mask: List[int] = []
  sequence_lengths: List[int] = [0,]
  positional_ids: List[int] = []
  min_token_length: int = 0     # minimum space to include the current sample
  datasource_index_mask: List[int] = []

  def __len__(self) -> int:
    assert len(self.input_tokens) == len(self.output_tokens)
    assert len(self.input_tokens) == len(self.training_mask)
    assert len(self.input_tokens) == len(self.datasource_index_mask)
    return len(self.input_tokens)
  
  def __eq__(self, __value: object) -> bool:
    if not isinstance(__value, LMTrainingSample):
      return False
    
    return self.input_tokens == __value.input_tokens and \
           self.output_tokens == __value.output_tokens and \
           self.training_mask == __value.training_mask and \
           self.sequence_lengths == __value.sequence_lengths and \
           self.positional_ids == __value.positional_ids and \
           self.min_token_length == __value.min_token_length and \
           self.datasource_index_mask == __value.datasource_index_mask

  def decode(self, tokenizer: TokenizerBase) -> Tuple[str | List[str], str | List[str]]:
    """
    Returns the decoded input and output
    """
    input_str = tokenizer.decode([a for a in self.input_tokens])
    output_str = tokenizer.decode([a for a in self.output_tokens])

    return input_str, output_str
  

  def inplace_extend_(self, other: 'LMTrainingSample') -> None:
    """Inplace op"""
    if len(self.input_tokens) == 0:
      # joining an empty sequence with another
      self.sequence_lengths.clear()
    
    self.sequence_lengths.extend([len(other.input_tokens)])

    self.input_tokens.extend(other.input_tokens)
    self.output_tokens.extend(other.output_tokens)
    self.training_mask.extend(other.training_mask)
    self.datasource_index_mask.extend(other.datasource_index_mask)
    self.positional_ids.clear()

  def inplace_datasource_mask_(self, datasource_index: int) -> None:
    """Inplace op"""
    self.datasource_index_mask.clear()
    self.datasource_index_mask.extend([datasource_index] * len(self.input_tokens))

  def truncate(self, size: int) -> Tuple['LMTrainingSample', 'LMTrainingSample']:
    if len(self) <= size:
      return self, LMTrainingSample()

    remainder = self._remainder(size)

    # we are assuming only the last sequence added is the overflow,
    # based on how the circular buffer is constructed
    cut_size = len(remainder)
    assert cut_size <= self.sequence_lengths[-1]

    new_seq_lengths = self.sequence_lengths[:-1] + [self.sequence_lengths[-1] - len(remainder)]

    truncated = LMTrainingSample(
      self.input_tokens[:size],
      self.output_tokens[:size],
      self.training_mask[:size],
      new_seq_lengths,
      [],
      self.min_token_length,
      self.datasource_index_mask[:size]
    )

    return truncated, remainder
    
  def inplace_fill_(self, size: int, value: int) -> None:
    """Inplace op"""
    if len(self) >= size:
      return
    
    fill_size = size - len(self)
    fill_tokens = [value] * (fill_size)
    fill_mask   = [0]     * (fill_size)

    self.input_tokens.extend(fill_tokens)
    self.output_tokens.extend(fill_tokens)
    self.training_mask.extend(fill_mask)

    # treat the last padding block as a new subsequence
    self.sequence_lengths.append(fill_size)
    self.positional_ids.clear()

    self.datasource_index_mask.extend([-1] * fill_size)

  def _remainder(self, size: int) -> 'LMTrainingSample':
    """Returns a new sample with the remainder of the current sample"""
    if len(self) <= size:
      return self
    
    return LMTrainingSample(
      self.input_tokens[size:],
      self.output_tokens[size:],
      self.training_mask[size:],
      [len(self.input_tokens[size:])], 
      [],
      self.min_token_length,
      self.datasource_index_mask[size:]
    )
  
  def resample_input_as_prompt(self, max_tokens_as_prompt: int):
    """Inplace op"""
    if not self.has_no_starting_prompt():
      return
    
    # sample up to half of the existing tokens as a prompt
    sample_size = random.randint(0, len(self) // 2)
    prompt_size = min(sample_size, max_tokens_as_prompt)

    # faster to traverse than to realocate
    for idx in range(prompt_size):
      self.training_mask[idx] = 0

  def has_no_starting_prompt(self):
    # ignore blank samples
    if len(self) == 0:
      return False
    
    # if we have a valid training token in the beggining
    # this means this buffer was cut off in the middle of a training token
    return self.training_mask[0] == 1
  
  def positional_ids_from_seq_lengths(self, sequence_lengths: List[int]) -> List[int]:
    """Returns a list of positional ids based on the sequence lengths"""
    positional_ids = []
    for seq_len in sequence_lengths:
      positional_ids.extend(list(range(seq_len)))
    return positional_ids

  def generate_positional_ids(self):
    """Inplace op, will generate a list of positional ids based on the sequence lengths"""
    self.positional_ids.clear()
    self.positional_ids.extend(self.positional_ids_from_seq_lengths(self.sequence_lengths))

@dataclass
class LMTrainingBatch:
  input_tokens: torch.Tensor
  output_tokens: torch.Tensor
  training_mask: torch.Tensor
  positional_ids: torch.Tensor
  sequence_lengths: List[List[int]]
  datasource_indices: torch.Tensor

  @torch.inference_mode
  def slice_seq_lengths(self,
                        seq_lengths: List[List[int]],
                        slice_: slice) -> List[List[int]]:
    cpu = torch.device("cpu")
    seq_lengths     = torch.tensor(sum(seq_lengths, []), device=cpu)
    seq_len_cumsum  = torch.cumsum(seq_lengths, dim=0).to(cpu)

    end_mask = (seq_len_cumsum >= slice_.stop).nonzero()

    if len(end_mask) > 0:
      end_idx = end_mask[0][0]
      end_seq_len =  seq_lengths[end_idx] - (seq_len_cumsum[end_idx] - slice_.stop)
      seq_lengths[end_idx] = end_seq_len
    else:
      end_idx = len(seq_len_cumsum) - 1

    start_idx = (seq_len_cumsum >= slice_.start).nonzero()[0][0]
    start_seq_len = seq_len_cumsum[start_idx] - slice_.start
    if start_seq_len == 0:
      start_idx += 1
    else:
      seq_lengths[start_idx] = start_seq_len
    
    return [seq_lengths[start_idx:end_idx + 1].tolist()]

  def __getitem__(self, item: slice):
    sliced_input_tokens = self.input_tokens[:, item]
    sliced_output_tokens = self.output_tokens[:, item]
    sliced_training_mask = self.training_mask[:, item]
    sliced_positional_ids = self.positional_ids[:, item]
    sliced_sequence_lengths = self.slice_seq_lengths(self.sequence_lengths, item)
    sliced_datasource_indices = self.datasource_indices[:, item]
    return LMTrainingBatch(
        input_tokens=sliced_input_tokens,
        output_tokens=sliced_output_tokens,
        training_mask=sliced_training_mask,
        positional_ids=sliced_positional_ids,
        sequence_lengths=sliced_sequence_lengths,
        datasource_indices=sliced_datasource_indices
    )
  
  def to_device(self, device):
    input_tokens = self.input_tokens.to(device)
    output_tokens = self.output_tokens.to(device)
    training_mask = self.training_mask.to(device)
    positional_ids = self.positional_ids.to(device)
    datasource_indices = self.datasource_indices.to(device)
    return LMTrainingBatch(input_tokens, output_tokens, training_mask, positional_ids, self.sequence_lengths, datasource_indices)

  def decode(self, tokenizer: TokenizerBase) -> Tuple[str | List[str], str | List[str]]:
    """
    Returns the decoded input and output
    """
    input_str = tokenizer.decode(self.input_tokens.to(torch.long).cpu().tolist())
    output_str = tokenizer.decode(self.output_tokens.to(torch.long).cpu().tolist())

    return input_str, output_str
  
  def decode_colorprint_samples(self, tokenizer: TokenizerBase, max_samples: int = 10) -> Tuple[List[str], List[str]]:
    """
    Decode and print a batch of samples with color according to the training mask
    """
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    slen_list = self.sequence_lengths[0]

    # split the input tokens by the sequence lengths 
    # TODO: modify all other occurances with output_tokens (OUTPUT TOKENS IS CORRECT)
    input_tokens_ = [*self.output_tokens[0].split(slen_list, dim=0)]
    training_mask_ = [*self.training_mask[0].split(slen_list, dim=0)]

    # if the last sequence is a padding sequence, remove it
    if sum(training_mask_[-1]) == 0:
      input_tokens_ = input_tokens_[:-1]
      training_mask_ = training_mask_[:-1]


    if len(input_tokens_) != len(training_mask_):
      print("ERROR: input tokens and training mask do not match")
      return
    
    for idx, (input_tokens, training_mask) in enumerate(zip(input_tokens_, training_mask_)):

      # split by mask to get masked tokens and unmasked tokens
      curr_mask_lens = [len(list(group)) for key, group in groupby(training_mask)]
      curr_mask_val = [key.item() for key, group in groupby(training_mask)]

      # split the input tokens by the mask
      input_tokens = [*input_tokens.split(curr_mask_lens, dim=0)]
      
      for input_tokens, mask_val in zip(input_tokens, curr_mask_val):
        if mask_val == 0:
          if input_tokens[0] == tokenizer.bos_token:
            print(RED + tokenizer.id_to_piece(tokenizer.bos_token) + RESET, end='')

          print(RED + tokenizer.decode(input_tokens.to(torch.long).cpu().tolist()) + RESET, end='')

          if input_tokens[-1] == tokenizer.eos_token:
            print(RED + tokenizer.id_to_piece(tokenizer.eos_token) + RESET, end='')

        elif mask_val == 1:
          if input_tokens[0] == tokenizer.bos_token:
            print(GREEN + tokenizer.id_to_piece(tokenizer.bos_token) + RESET, end='')

          print(GREEN + tokenizer.decode(input_tokens.to(torch.long).cpu().tolist()) + RESET, end='')

          if input_tokens[-1] == tokenizer.eos_token:
            print(GREEN + tokenizer.id_to_piece(tokenizer.eos_token) + RESET, end='')
    
  def get_num_training_tokens(self) -> int:
    """Returns the number of training tokens in the batch"""
    return torch.sum(self.training_mask).item()

class LMBatchMetadata(NamedTuple):
  backprop_tokens: torch.Tensor
  processed_tokens: torch.Tensor
  padded_tokens: torch.Tensor 
  dataset_indices: set

  @classmethod
  @torch.inference_mode
  def from_training_batch(
    cls,
    batch: LMTrainingBatch,
    tokenizer: TokenizerBase,
    dataset_names: List[str],
  ) -> 'LMBatchMetadata':
    
    """Returns a new instance from a training batch"""
    padded_tokens = (batch.output_tokens == tokenizer.eos_token) & (batch.training_mask == 0)
    padded_tokens = padded_tokens.sum()

    return cls(
      backprop_tokens = batch.training_mask.sum(),
      processed_tokens = batch.input_tokens.shape[1],
      padded_tokens = padded_tokens,
      dataset_indices = set(range(len(dataset_names)))
    )

  @classmethod
  @torch.inference_mode
  def from_training_batch(
    cls,
    batch: LMTrainingBatch,
    tokenizer: TokenizerBase,
    dataset_names: List[str],
  ) -> 'LMBatchMetadata':
    
    """Returns a new instance from a training batch"""
    padded_tokens = (batch.output_tokens == tokenizer.eos_token) & (batch.training_mask == 0)
    padded_tokens = padded_tokens.sum()

    return cls(
      backprop_tokens = batch.training_mask.sum(),
      processed_tokens = batch.input_tokens.shape[1],
      padded_tokens = padded_tokens,
      dataset_indices = set(range(len(dataset_names)))
    )
  
  @torch.inference_mode
  def trim_padding(
    self,
    batch: LMTrainingBatch,
  ) -> LMTrainingBatch:
    
    """Returns a new instance from a training batch"""
    valid_tokens = self.processed_tokens - self.padded_tokens
    # need 0 explicitely for slice
    return batch[0:valid_tokens.item()]

class PromptTokensSample(NamedTuple):
  prompt_tokens: List[int]
  sample_tokens: List[int]

  def merge(self) -> List[int]:
    return self.prompt_tokens + self.sample_tokens
  
def reverse_sample(
    sample: List[int],
    rev_config: dict
):
  rev_token = rev_config["rev_token"]
  max_len = rev_config["max_len"]
  min_len = rev_config["min_len"]

  len_sample = len(sample)
  split_sample = []
  curr_sum = 0
  for _ in range(len_sample):
    next_seg_size = random.randint(min_len, max_len)
    # if next size is lower than the min_len, just take the rest of the sample
    if len_sample - curr_sum - next_seg_size < min_len:
      split_sample.append(sample[curr_sum:])
      break
    else:
      split_sample.append(sample[curr_sum:curr_sum + next_seg_size])
    if curr_sum + next_seg_size >= len_sample:
      break
    curr_sum += next_seg_size

  rev_split_sample = []
  for curr_seg in split_sample[::-1]:
    rev_split_sample.append([rev_token])
    rev_split_sample.append(curr_seg)

  # flatten
  rev_split_sample = [item for sublist in rev_split_sample for item in sublist]
  return rev_split_sample

def fim_sample(
    sample: List[int],
    fim_config: dict
):
  suffix_lims = fim_config["suffix_lims"]
  prefix_lims = fim_config["prefix_lims"]
  pre_token = fim_config["prefix_token"]
  suf_token = fim_config["suffix_token"]
  mid_token = fim_config["middle_token"]
  eot_token = fim_config["eot_token"]

  suffix_percentage = random.uniform(suffix_lims[0], suffix_lims[1])
  prefix_percentage = random.uniform(prefix_lims[0], prefix_lims[1])

  suffix_len = int(len(sample) * suffix_percentage)
  prefix_len = int(len(sample) * prefix_percentage)

  pre_sample = sample[:prefix_len]
  suf_sample = sample[-suffix_len:]
  mid_sample = sample[prefix_len:-suffix_len]

  # psm or spm with equal probability
  if random.random() < 0.5:
    # psm
    return [pre_token] + pre_sample + [suf_token] + suf_sample + [mid_token], mid_sample + [eot_token]
  else:
    return [suf_token] + suf_sample + [pre_token] + pre_sample + [mid_token], mid_sample + [eot_token]


def prompt_to_training_sample(sample: PromptTokensSample,
                              min_token_length: int,
                              bos_token: int,
                              eos_token: int,
                              fim_rev_config: dict) -> LMTrainingSample:
  
  if fim_rev_config is not None and \
    len(sample.sample_tokens) > fim_rev_config["min_token_limit"]:

    prob_dict = fim_rev_config["prob"]
    prep_type = random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()), k=1)[0]

    if prep_type == "fim":
      fim_config = fim_rev_config["fim"]
      new_prompt_tokens, new_sample_tokens = fim_sample(sample.sample_tokens, 
                                        fim_config
                                      )
      prompt_tokens = sample.prompt_tokens + new_prompt_tokens
      new_sample = PromptTokensSample(prompt_tokens, new_sample_tokens)
    elif prep_type == "rev":
      rev_config = fim_rev_config["rev"]
      new_sample_tokens = reverse_sample(sample.sample_tokens, 
                                            rev_config)
      new_sample = PromptTokensSample(sample.prompt_tokens, new_sample_tokens)
    elif prep_type == "original":
      new_sample = PromptTokensSample(sample.prompt_tokens, sample.sample_tokens)
    else:
      raise ValueError("Invalid preparation type")
  else:
    new_sample = sample

  full_sample_tokens = new_sample.merge()

  # add special tokens
  gpt_sample = [bos_token] + full_sample_tokens + [eos_token]
  input_tokens = gpt_sample[:-1]
  output_tokens = gpt_sample[1:]

  if fim_rev_config is not None and fim_rev_config["rev"]["rev_token"] in new_sample.sample_tokens:
    training_mask = [0] * (len(new_sample.prompt_tokens) + 1) + \
                    [1] * (len(new_sample.sample_tokens))
  else:
    training_mask = [0] * (len(new_sample.prompt_tokens)) + \
                    [1] * (len(new_sample.sample_tokens) + 1)

  # make a new training sample
  new_lm_sample = LMTrainingSample(
    input_tokens = input_tokens,
    output_tokens = output_tokens,
    training_mask = training_mask,
    sequence_lengths = [len(input_tokens)],
    positional_ids = [],
    min_token_length = min_token_length,
    datasource_index_mask = []
  )


  return new_lm_sample