import pytest
from abc import ABC

def test_import():
  import unakin.llm
  print(unakin.llm.BaseLLM.__annotations__)
  assert issubclass(unakin.llm.BaseLLM, ABC)
