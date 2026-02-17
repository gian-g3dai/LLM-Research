import json

from unakin.search.tokenizers.base import TokenizerBase


class TokenizerFactory:
  
  @staticmethod
  def from_file(file_path: str) -> TokenizerBase:
    if file_path.endswith(".json"):
      return TokenizerFactory.from_json(file_path)
    else:
      raise ValueError(f"Unknown tokenizer file type: {file_path}")
    
  @staticmethod
  def from_json(file_path_or_config: str | dict) -> TokenizerBase:
    if isinstance(file_path_or_config, str):
      with open(file_path_or_config, "r") as f:
        json_config = json.load(f)
    else:
      json_config = file_path_or_config

    if json_config["type"] == "sbert":
      from unakin.search.tokenizers.sbert import SbertTokenizer
      return SbertTokenizer()
    else:
      raise ValueError(f"Unknown tokenizer type: {json_config['type']}")