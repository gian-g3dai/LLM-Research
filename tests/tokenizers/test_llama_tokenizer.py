import os
import pytest

from tests.helpers import get_resource_path

def test_import():
  from unakin.llm.tokenizers import TokenizerBase, LlamaTokenizer
  assert issubclass(LlamaTokenizer, TokenizerBase)

def test_llama_tokenizer_single_string():
  from unakin.llm.tokenizers import LlamaTokenizer
  tokenizer_path = get_resource_path() + "/tokenizer.model"
  tokenizer = LlamaTokenizer(model_path=tokenizer_path)
  tokens = tokenizer.tokenize("Hello world")
  assert isinstance(tokens, list)
  assert len(tokens) == 1

def test_llama_tokenizer_list_of_strings():
  from unakin.llm.tokenizers import LlamaTokenizer
  tokenizer_path = get_resource_path() + "/tokenizer.model"
  tokenizer = LlamaTokenizer(model_path=tokenizer_path)
  tokens = tokenizer.tokenize(["Hello world", "Hello world"])
  assert isinstance(tokens, list)
  assert len(tokens) == 2

def test_llama_tokenizer_decode():
  from unakin.llm.tokenizers import LlamaTokenizer
  tokenizer_path = get_resource_path() + "/tokenizer.model"
  tokenizer = LlamaTokenizer(model_path=tokenizer_path)
  tokens = tokenizer.tokenize("Hello world")
  decoded = tokenizer.decode(tokens)
  assert len(decoded) == 1
  assert isinstance(decoded[0], str)
