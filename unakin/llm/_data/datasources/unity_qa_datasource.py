import polars as pl
import numpy as np
from unakin.llm._data.datasources.atomic import RawData, ParquetReader
from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.util_processors import check_columns

class UnityQAProcessor(Processor):
  
    __data_cols__ = ["question", "answer"]

    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class UnityQAReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)