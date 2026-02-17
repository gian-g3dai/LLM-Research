import polars as pl
import numpy as np

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from unakin.llm._data.datasources.atomic import RawData, FstringTemplater, ParquetReader

from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline

from unakin.llm._data.datasources.util_processors import (
  TokenCountInfo, RenameColumns, check_columns, SumTokenCountFilter)
from unakin.llm._data.datasources.base_datasource import InstructionLMDataSource
from unakin.llm._data.datasources.pipeline_utils import run_pipeline_from_path, CacheProcessor, set_pipeline_cache_path


class MuffinProcessor(Processor):
  
    __data_cols__ = ["output", "instruction"]
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class MuffinReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":

  np.random.seed(42)

  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  set_pipeline_cache_path("/mnt/raid0/datasets/tmp_cache")

  pipeline = LoadPipeline(
              reader=MuffinReader(),
              processors=[
                MuffinProcessor(),
                TokenCountInfo(source_col="instruction"),
                TokenCountInfo(source_col="output"),
                SumTokenCountFilter(source_cols=["instruction", "output"], token_interval=(16, 16300)),
                CacheProcessor(name="muffin_test")
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "muffin/validation", "/mnt/raid0/datasets/")
  
  datasource = InstructionLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{instruction}"],
      ),
      FstringTemplater(
        fstring=["{output}"],
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="muffin/validation",
    percent_interval = (99.9, 100),
    metric_list=["perplexity"],
    )

  # # count the number of trainable tokens
  # num_train_tokens = 0
  # longest_sample = 0
  # seq_lens_all = 0
  # for sample in datasource:
  #   seq_lens_all += len(sample)
  #   num_train_tokens += sum(sample.training_mask)

  # print("Seqlen sum", seq_lens_all)
  # print("Num Train Tokens", num_train_tokens)

  print("Datasource Length", len(datasource))