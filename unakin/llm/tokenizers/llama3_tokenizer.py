from typing import List, Sequence, TypedDict, Literal
import torch
from torch import Tensor
import torch.distributed
from .base import TokenizerBase # might add chatformatterbase here
from unakin.llm.oss.llama3.llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class Llama3ChatFormatter:
  def __init__(self, tokenizer: Tokenizer):
    self.tokenizer = tokenizer

  def encode_header(self, message: Message) -> List[int]:
    tokens = []
    tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
    tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
    tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
    tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
    return tokens

  def encode_message(self, message: Message) -> List[int]:
    tokens = self.encode_header(message)
    tokens.extend(
      self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
    )
    tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
    return tokens

  def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
    tokens = []
    tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
    for message in dialog:
      tokens.extend(self.encode_message(message))
    # Add the start of an assistant message for the model to complete.
    tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
    return tokens, []


class Llama3Tokenizer(TokenizerBase):
  def __init__(self, *args, model_path: str, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.tokenizer = Tokenizer(model_path=model_path)
    self.chat_formatter = Llama3ChatFormatter(self.tokenizer)

  # NOTE: Special tokens can be added via encode args
  def tokenize(self, text: str | List[str], bos: bool=True, eos: bool=True) -> List[List[int]]:
    if isinstance(text, str):
      text = [text]

    ids = []
    for t in text:
      ids.append(self.tokenizer.encode(t, bos=bos, eos=eos))

    return ids
  
  def decode(self, token_ids: List[List[int]], **kwargs) -> str | List[str]:
    if isinstance(token_ids[0], list):
      ret_batch = True
    else:
      ret_batch = False

    if ret_batch:
      ret = []
      for t in token_ids:
        ret.append(self.tokenizer.decode(t))
    else:
      ret = self.tokenizer.decode(token_ids)

    return ret
  
  def id_to_piece(self, token: int) -> str:
    return self.tokenizer.model.decode([token])
  
  def subword_to_word(self, token: str) -> str:
    raise NotImplementedError("Llama3Tokenizer - not implemented subword to word")
  
  @property
  def bos_token(self) -> int:
    return self.tokenizer.bos_id
  
  @property
  def eos_token(self) -> int:
    return self.tokenizer.eos_id
  
  @property
  def pad_token(self) -> int:
    return self.tokenizer.pad_id
    # return 0
  
  @property
  def vocab_size(self) -> int:
    return self.tokenizer.n_words
  
  @property
  def stop_tokens(self) -> set:
    return self.tokenizer.stop_tokens