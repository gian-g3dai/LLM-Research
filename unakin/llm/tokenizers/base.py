from typing import List
from abc import ABC, abstractmethod
from torch import Tensor

class TokenizerBase(ABC):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__()

  @abstractmethod
  def tokenize(self, text: str | List[str], **kwargs) -> List[int] | List[List[int]]:
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
