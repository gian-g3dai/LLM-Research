import json
from abc import abstractmethod
from typing import Tuple

from unakin.search.models.base import BaseSearchModel
from unakin.search.tokenizers.base import TokenizerBase
from unakin.search.tokenizers.utils import TokenizerFactory


class SearchModelFactory:

  @abstractmethod
  def from_file(file_path: str) -> Tuple[BaseSearchModel, TokenizerBase]:
    if file_path.endswith(".json"):
      return SearchModelFactory.from_json(file_path)
    
  def from_json(file_path_or_config: str | dict) -> Tuple[BaseSearchModel, TokenizerBase]:
    if isinstance(file_path_or_config, str):
      with open(file_path_or_config, "r") as f:
        json_config = json.load(f)
    else:
      json_config = file_path_or_config

    if "tokenizer" in json_config:
      tokenizer = TokenizerFactory.from_json(json_config["tokenizer"])
    else:
      tokenizer = None
    
    model_config = json_config["model"]
    if model_config["type"] == "sbert":
      from unakin.search.models.sbert import SbertSearchModel
      model = SbertSearchModel()
    else:
      raise ValueError(f"Unknown model type: {model_config['type']}")

    return model, tokenizer

