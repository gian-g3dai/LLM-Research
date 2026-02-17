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


class HotpotProcessor(Processor):
  
    __data_cols__ = ["question", "answer", "context", "supporting_context", "level", "type"]

    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class HotpotReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":

  np.random.seed(42)

  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  set_pipeline_cache_path("/mnt/raid0/datasets/tmp_cache")

  pipeline = LoadPipeline(
              reader=HotpotReader(),
              processors=[
                HotpotProcessor(),
                TokenCountInfo(source_col="question"),
                TokenCountInfo(source_col="context"),
                TokenCountInfo(source_col="answer"),
                SumTokenCountFilter(source_cols=["question", "context", "answer"], token_interval=(16, 16300)),
                # CacheProcessor(name="hotpot_qa_train")
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "hotpot_qa/train", "/mnt/raid0/datasets/")
  
  datasource = InstructionLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{context}\n\n{question}"],
      ),
      FstringTemplater(
        fstring=["{answer}"],
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="hotpot_qa/train",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )
  
  # print a sample
  print(datasource[0])
  print(tokenizer.decode(datasource[0].output_tokens))

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