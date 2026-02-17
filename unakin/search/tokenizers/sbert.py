from transformers import AutoTokenizer

from unakin.search.tokenizers.base import TokenizerBase, TokenizerOuts


class SbertTokenizer(TokenizerBase):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

  def vocab_size(self) -> int:
    return self._tokenizer.vocab_size
  
  def bos_token(self) -> int:
    return self._tokenizer.bos_token_id
  
  def eos_token(self) -> int:
    return self._tokenizer.eos_token_id
  
  def pad_token(self) -> int:
    return self._tokenizer.pad_token_id
  
  def encode(self, text: str | list[str], **kwargs) -> TokenizerOuts:
    return_tensors = kwargs.get("return_tensors", "np")
    
    if isinstance(text, str):
      text = [text]
    
    outs = self._tokenizer.batch_encode_plus(
      text, padding='max_length', truncation=True,
      max_length=512, return_tensors=return_tensors, return_attention_mask=True
    )
    
    return TokenizerOuts(ids=outs["input_ids"], attention_mask=outs["attention_mask"])
  
  def decode(self, token_ids: list[int] | list[list[int]], **kwargs) -> str | list[str]:
    if isinstance(token_ids[0], list):
      return self._tokenizer.batch_decode(token_ids, **kwargs)
    else:
      return self._tokenizer.decode(token_ids, **kwargs)