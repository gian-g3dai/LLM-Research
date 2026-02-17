
import os
import re
import random
import polars as pl


from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CacheReader:
  def __init__(self):
    """
    Initializes a cache reader. This a simple FileSystem manager that is
    responsible for reading and writing to the cache, which is a local
    key-value store.
    """

    self._BASE_CACHE_PATH = "~/.unakin/cache"
    if not os.path.exists(self._BASE_CACHE_PATH):
      os.makedirs(self._BASE_CACHE_PATH)

  def contains(self, key: str) -> bool:
    """
    Performs a check for a full key match in the cache folder.

    Args:
      key (str): The lookup key.
    Returns:
      bool: True if the key exists, False otherwise.
    """

    # scan all files & folders in the cache & check for a string match
    # on both file names or folder names
    for root, dirs, files in os.walk(self._BASE_CACHE_PATH):
      if key in root or key in files:
        return True
      
  def save(self, name: str, data: Any, object_type: Any) -> None:

    if object_type == pl.DataFrame:
      data: pl.DataFrame
      assert isinstance(data, pl.DataFrame), "Data must be a polars DataFrame"
            
      path = os.path.join(self._BASE_CACHE_PATH, name)
      data.write_parquet(path)
      return

  def read(self, name: str, object_type: Any) -> Any:

    path = os.path.join(self._BASE_CACHE_PATH, name)
    if not self.contains(path):
      # TODO: check remote cache
      raise ValueError(f"Cache does not contain {path}")

    if object_type == pl.DataFrame:
      return pl.read_parquet(path)

    raise ValueError(f"Unsupported object type {object_type}")
  
CACHE_READER = CacheReader()

class RawData:
  
  def __init__(self, data: Optional[pl.DataFrame], name: Optional[str] = None) -> None:
    assert data is not None or name is not None, "Either `data` or `name` must be provided"
    
    if name is not None:
      self.data = CACHE_READER.read(name, pl.DataFrame)

    self.data = data
    self.columns = self.data.columns

  def save(self, name: str) -> None:
    self.data.write_parquet(name)

  def get_index(self, idx: int) -> dict:
    # get the row at the given index
    return self.data[idx]

class ParquetReader(ABC):
  
  def __init__(self):
    pass

  def read(self, path: str) -> RawData:
    _parquet = pl.read_parquet(path)
    return self.process_parquet(_parquet)

  @abstractmethod
  def process_parquet(self, parquet: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError

class FolderCodeReader(ABC):

  def __init__(self):
    pass

  @abstractmethod
  def read(self, path: str) -> RawData:
    raise NotImplementedError

  @abstractmethod
  def process_parquet(self, parquet: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError
  

############## TEMLATERS ##############

class Templater(ABC):
  def __init__(self, *args, **kwargs):
    pass

  @abstractmethod
  def format(self, row_data: dict, **kwargs) -> str:
    raise NotImplementedError
  
class FstringTemplater(Templater):
  def __init__(
      self,
      fstring: List[str] | str,  
      mapping: Optional[dict] = None,
      *args, **kwargs
    ):    
    super().__init__(*args, **kwargs)

    if isinstance(fstring, str):
      fstring = [fstring]
    self._fstring = fstring
    self._mapping = mapping

  def get_columns_needed(self) -> List[str]:
    """
    Returns a list of columns that are needed for the templater to work.
    """

    columns = []
    for fstring in self._fstring:
      # find all the keys in the fstring
      keys = re.findall(r"{(\w+)}", fstring)
      columns.extend(keys)

    # eliminate the ones in _mapping (provided when initializing the templater)
    if self._mapping is not None:
      for key, value in self._mapping.items(): 
        if isinstance(value, list):
          for val in value:
            keys = re.findall(r"{(\w+)}", val)
            columns.extend(keys)
        else:
          keys = re.findall(r"{(\w+)}", value)
          columns.extend(keys)

      columns = list(set(columns) - set(self._mapping.keys()))
    else:
      columns = list(set(columns))

    return columns

  def format(self, row_data: dict, **kwargs) -> str:

    mapped_row = {}

    if self._mapping is not None:
      # map the hardcoded values
      for key, value in self._mapping.items():
        if isinstance(value, list):
          curr_value = random.choice(value)
        else:
          curr_value = value

        mapped_row[key] = curr_value.format(**row_data)
    
    for key, value in row_data.items():
      mapped_row[key] = value

    # choose a template from the provided list
    elected_fstring = random.choice(self._fstring)
    return elected_fstring.format(**mapped_row)


if __name__ == "__main__":
  prompt = FstringTemplater(
    fstring=["Some bullshit hardcoded {hardcoded_2} {input} {input_2} {hardcoded}",
             "Bullshit hardcoded {hardcoded_2} {input} {input_2} {hardcoded}"],
    mapping={"hardcoded": ["END_HARD_CODED", "MIDDLE_HARD_CODED"],
              "hardcoded_2": "START_HARD_CODED"}
  )

  print(prompt.get_columns_needed())
  exit()

  data = {
    "input": "data_from_raw_data",
    "input_2": "other_data_from_raw_data",
  }

  print(prompt.format(data))