from abc import ABC, abstractmethod
from typing import List, NamedTuple

import torch


class TokenizerOuts(NamedTuple):
  ids: torch.Tensor
  attention_mask: torch.Tensor | None

class TokenizerBase(ABC):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__()

  @abstractmethod
  def encode(self, text: str | List[str], **kwargs) -> TokenizerOuts:
    raise NotImplementedError

  @abstractmethod
  def decode(self, token_ids: List[int] | List[List[int]], **kwargs) -> str | List[str]:
    raise NotImplementedError
 
  @property
  @abstractmethod
  def vocab_size(self) -> int:
    raise NotImplementedError

  @property
  @abstractmethod
  def bos_token(self) -> int:
    raise NotImplementedError

  @property
  @abstractmethod
  def eos_token(self) -> int:
    raise NotImplementedError

  @property
  @abstractmethod
  def pad_token(self) -> int:
    raise NotImplementedError
