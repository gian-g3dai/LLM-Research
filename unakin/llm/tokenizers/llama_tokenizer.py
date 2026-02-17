from typing import List
import torch
from torch import Tensor
from .base import TokenizerBase
from unakin.llm.oss.llama.llama.tokenizer import Tokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

class LlamaChatFormatter:

  def __init__(self, tokenizer: TokenizerBase):
    self.tokenizer = tokenizer

  def encode_dialog_prompt(self, local_dialogs):
    prompt_tokens = []
    unsafe_requests = []
    for dialog in local_dialogs:
      unsafe_requests.append(
        any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
      )
      if dialog[0]["role"] == "system":
        dialog = [
            {
              "role": dialog[1]["role"],
              "content": B_SYS
              + dialog[0]["content"]
              + E_SYS
              + dialog[1]["content"]
            }
        ] + dialog[2:]
      assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
      ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
      )
      
      dialog_tokens: List[int] =  sum(
        [
          self.tokenizer.tokenize(
            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            bos=True,
            eos=True,
          )[0]
          for prompt, answer  in zip(dialog[::2], dialog[1::2])
        ],
        []
      )

      assert (dialog[-1]["role"] == "user"), f"Last message must be from user, got {dialog[-1]['role']}"
      dialog_tokens += self.tokenizer.tokenize(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        bos=True,
        eos=False,
      )[0]
      prompt_tokens.append(dialog_tokens)

    return prompt_tokens, unsafe_requests


class LlamaTokenizer(TokenizerBase):
  def __init__(self, *args, model_path: str, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.tokenizer = Tokenizer(model_path)
    self.chat_formatter = LlamaChatFormatter(self.tokenizer)

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
    return self.tokenizer.sp_model.id_to_piece(token)
  
  def subword_to_word(self, token: str) -> str:
    return self.tokenizer.sp_model.decode(token)
  
  @property
  def bos_token(self) -> int:
    return self.tokenizer.bos_id
  
  @property
  def eos_token(self) -> int:
    return self.tokenizer.eos_id
  
  @property
  def pad_token(self) -> int:
    # return self.tokenizer.pad_id
    return 0
  
  @property
  def vocab_size(self) -> int:
    return self.tokenizer.vocab_size()
  
  # fim tokens
  @property
  def prefix_token(self) -> int:
    return self.tokenizer.prefix_id
  
  @property
  def middle_token(self) -> int:
    return self.tokenizer.middle_id
  
  @property
  def suffix_token(self) -> int:
    return self.tokenizer.suffix_id
  
  @property
  def eot_token(self) -> int:
    return self.tokenizer.eot_id
  
  # reverse token
  @property
  def rev_token(self) -> int:
    return self.tokenizer.rev_id
