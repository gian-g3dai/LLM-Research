import polars as pl

from typing import Optional
from functools import partial

from unakin.llm._data.datasources.atomic import RawData
from unakin.llm._data.datasources.base_processors import Processor, LambdaProcessor
from unakin.llm._data.datasources.code_utils import backtick_code, normalize_code
from unakin.llm._data.datasources.util_processors import check_columns

class Backticks(LambdaProcessor):
  """
  Replace backticks with spaces.
  """
  def __init__(self, source_col: str, language: str):
    super().__init__(func=partial(backtick_code, language=language), source_col=source_col)


class BackticksLangs(Processor):
  """
  Replace backticks with spaces.
  """
  def __init__(self, source_col: str, language_col: str):
    super().__init__()
    self._source_col = source_col
    self._language_col = language_col

  def process(self, data: RawData) -> RawData:
    data: pl.DataFrame = data.data
    source_col = self._source_col
    language_col = self._language_col

    # check if the source column exists
    check_columns(data, [source_col, language_col])

    data = data.with_columns([
      pl.struct([source_col, language_col]) \
        .map_elements(lambda x: backtick_code(x[source_col], x[language_col]) if x[source_col] != "" else "").alias(source_col)
    ])
    
    return RawData(data)


class NormalizeCode(LambdaProcessor):
  """
  Replace backticks with spaces.
  """
  def __init__(self, source_col: str):
    super().__init__(func=normalize_code, source_col=source_col)

