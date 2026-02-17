import polars as pl
import numpy as np

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from unakin.llm._data.datasources.atomic import RawData, FstringTemplater, ParquetReader

from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline

from unakin.llm._data.datasources.util_processors import (
  TokenCountInfo, RenameColumns, check_columns, SumTokenCountFilter, StripProcessor)
from unakin.llm._data.datasources.base_datasource import InstructionLMDataSource, SimpleLMDataSource
from unakin.llm._data.datasources.pipeline_utils import run_pipeline_from_path, CacheProcessor, set_pipeline_cache_path


class UnnaturalProcessor(Processor):
  
    __data_cols__ = [
    "fst_sol",
    "snd_sol",
    "trd_sol",
    "best_sol",
    "improved_sol",
    "best_sol_no",
    "best_sol_reason",
    "game",
    "script_request",
    "instruct",
    "response"
    ]

    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      # check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class UnnaturalReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)