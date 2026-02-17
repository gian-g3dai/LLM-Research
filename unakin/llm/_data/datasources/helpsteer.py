import polars as pl
import numpy as np

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from unakin.llm._data.datasources.atomic import RawData, FstringTemplater, ParquetReader

from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline

from unakin.llm._data.datasources.util_processors import (
  TokenCountInfo, RenameColumns, check_columns, SumTokenCountFilter, StripProcessor)
from unakin.llm._data.datasources.base_datasource import InstructionLMDataSource
from unakin.llm._data.datasources.pipeline_utils import run_pipeline_from_path, CacheProcessor, set_pipeline_cache_path


class HelpSteerProcessor(Processor):
  
    __data_cols__ = ["prompt", "response"]

    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class HelpSteerReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":

  np.random.seed(42)

  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  set_pipeline_cache_path("/mnt/raid0/datasets/tmp_cache")

  pipeline = LoadPipeline(
              reader=HelpSteerReader(),
              processors=[
                HelpSteerProcessor(),
                StripProcessor(source_cols=["prompt", "response"]),
                TokenCountInfo(source_col="prompt"),
                TokenCountInfo(source_col="response"),
                SumTokenCountFilter(source_cols=["prompt", "response"], token_interval=(16, 16300)),
                # CacheProcessor(name="hotpot_qa_train")
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "helpsteer/train", "/mnt/raid0/datasets/")
  
  datasource = InstructionLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{prompt}\n"],
      ),
      FstringTemplater(
        fstring=["{response}"],
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="helpsteer/train",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )
  
  # print a sample
  print(datasource[0])
  print(tokenizer.decode(datasource[0].output_tokens))


  print("Datasource Length", len(datasource))