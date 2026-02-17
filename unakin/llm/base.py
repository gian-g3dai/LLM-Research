import torch

from typing import List, Optional
from abc import ABC, abstractmethod
from xformers.ops.fmha import AttentionBias


class BaseLLM(torch.nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__()

  @abstractmethod
  def enable_xops(self, enable: bool = True):
    raise NotImplementedError

  @abstractmethod
  def post_init_meta_buffers(self):
    raise NotImplementedError

  @abstractmethod
  def forward(
          self,
          tokens: torch.Tensor,
          start_pos: int,
          positional_ids: Optional[torch.Tensor] = None,
          q_seq_lens: List[int] = None,
          mask: Optional[torch.Tensor | AttentionBias] = None,
          is_decoding: bool = False,):
    raise NotImplementedError

  @abstractmethod
  def register_cache_kv(self, *args, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def register_soft_prompt(self, *args, **kwargs):
    raise NotImplementedError
