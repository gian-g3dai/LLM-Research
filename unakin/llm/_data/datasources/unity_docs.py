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


class UnityDocsProcessor(Processor):
  
    __data_cols__ = ["fname", "contents"]

    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      # filter 

      new_df = _df
      return RawData(new_df)

class UnityDocsReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":

  np.random.seed(42)

  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  set_pipeline_cache_path("/mnt/raid0/datasets/tmp_cache")

  pipeline = LoadPipeline(
              reader=UnityDocsReader(),
              processors=[
                UnityDocsProcessor(),
                StripProcessor(source_cols=["contents"]),
                TokenCountInfo(source_col="contents", multithreaded=False),
                SumTokenCountFilter(source_cols=["contents"], token_interval=(16, 16300)),
                # CacheProcessor(name="unity_docs")
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "unity_docs/scripting_md", "/mnt/raid0/datasets/")
  
  datasource = SimpleLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{contents} "],
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="unity_docs/scripting_md",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )
  
  # print a sample
  print(datasource[5])
  print(tokenizer.decode(datasource[5].output_tokens))


  print("Datasource Length", len(datasource))

# Token count info:
# contents: 0.072004M
# TOKEN INTERVAL: (16, 16300) FILTERING...
# Original number of rows: 64
# Filtered to 64 rows.