import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple, List, Union, Dict
from unakin.llm.tokenizers import TokenizerBase
from abc import ABC, abstractmethod

class LMObjective(nn.Module):
  def __init__(self, *args, requires_prompt: bool, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.requires_prompt = requires_prompt

  def forward(self, *args, **kwargs):
    raise NotImplementedError()

  def preprocess(self, *args, **kwargs):
    raise NotImplementedError()
  
class CLMObjective(LMObjective):
  def __init__(self, *args, requires_prompt: bool, **kwargs) -> None:
    super().__init__(*args, requires_prompt=requires_prompt, **kwargs)

  def forward(self, logits: Tensor, input_ids: Tensor) -> Tensor:
    """
    logits: (batch_size, seq_len, vocab_size)
    input_ids: (batch_size, seq_len)
    """
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]
    return F.cross_entropy(logits.flatten(0, 1), input_ids.flatten(0, 1))
  
  def preprocess(self, tokens: List[int], prompt_tokens: List[int], tokenizer: TokenizerBase, **kwargs):
    """
    tokens: (seq_len)
    """
    input_tokens = [tokenizer.bos_token] + tokens
    output_tokens = tokens + [tokenizer.eos_token]
    seq_len = len(input_tokens)
    return (input_tokens, output_tokens, seq_len)
  
class PLMObjective(LMObjective):
  def __init__(self, *args, requires_prompt: bool, **kwargs) -> None:
    super().__init__(*args, requires_prompt=requires_prompt, **kwargs)

  def forward(self,
              logits: Tensor,
              input_ids: Tensor,
              prefix_length: Tensor) -> Tensor:
    pass

  def preprocess(self, tokens: List[int], prompt_tokens: List[int], tokenizer: TokenizerBase, **kwargs):
    """
    tokens: (seq_len)
    """
    prompt_tokens = [tokenizer.bos_token] + prompt_tokens
    output_tokens = tokens + [tokenizer.eos_token]
    input_tokens  = prompt_tokens + tokens
    output_start = len(prompt_tokens)

class FLLMObjective(LMObjective):
  def __init__(self, *args, requires_prompt: bool, **kwargs) -> None:
    super().__init__(*args, requires_prompt=requires_prompt, **kwargs)

  def forward(self,
              logits: Tensor,
              input_ids: Tensor,
              prefix_length: Tensor) -> Tensor:
    pass

  def preprocess(self, tokens: List[int], prompt_tokens: List[int], tokenizer: TokenizerBase, **kwargs):
    
  

    

    