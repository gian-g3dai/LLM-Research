import os
import glob
import codecs
import polars as pl

from enum import Enum
from tqdm import tqdm
from functools import partial
from typing import List, Optional, Any, Tuple, Dict
from multiprocessing import Pool, cpu_count, current_process
from tests.helpers import get_resource_path


from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer
from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline, run_pipeline_from_path, CacheProcessor, set_pipeline_cache_path
from unakin.llm._data.datasources.atomic import RawData, FstringTemplater, ParquetReader
from unakin.llm._data.datasources.util_processors import check_columns, TokenCountInfo, SumTokenCountFilter, StripProcessor
from unakin.llm._data.datasources.base_datasource import SimpleLMDataSource


class RedPJProcessor(Processor):
  
    __data_cols__ = ["raw_content", "label"]
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class RedPJReader(ParquetReader):
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)
   

if __name__ == "__main__":
  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  set_pipeline_cache_path("/mnt/raid0/datasets/tmp_cache/")

  pipeline = LoadPipeline(
              reader=RedPJReader(),
              processors=[
                RedPJProcessor(),
                TokenCountInfo(source_col="raw_content"),
                SumTokenCountFilter(source_cols=["raw_content"], token_interval=(1024, 16_384)),
                StripProcessor(source_cols=["raw_content"])
                # CacheProcessor(name="redpajama_dummy")
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "red_pj/train10B", "/mnt/raid0/datasets/")

  print("Length of raw data:", len(raw_data.data))

  datasource = SimpleLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{prompt}{raw_content}"],
        mapping={"prompt": ""}
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="red_pj/train10B",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )

  print(datasource[0])
  print(tokenizer.decode(datasource[0].input_tokens))