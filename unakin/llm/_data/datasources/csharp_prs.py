
import difflib
import polars as pl
import pandas as pd

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from unakin.llm._data.datasources.atomic import RawData, FstringTemplater

from unakin.llm._data.datasources.base_processors import Processor, LambdaProcessor
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline, ParquetReader
from unakin.llm._data.datasources.pipeline_utils import run_pipeline_from_path, CacheProcessor, set_pipeline_cache_path


from unakin.llm._data.datasources.util_processors import TokenCountInfo, RenameColumns, StripProcessor, check_columns
from unakin.llm._data.datasources.code_processors import Backticks, BackticksLangs, NormalizeCode

from unakin.llm._data.datasources.base_datasource import InstructionLMDataSource

class CsharpPRsProcessor(Processor):
  
    __data_cols__ = ["repos",	"patches",	"pr_title",	"pr_description"]

    # NOTE: there are old_content == "" => there is a new content added
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class CsharpPRsReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":
  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")
  
  pipeline = LoadPipeline(
              reader=CsharpPRsReader(),
              processors=[
                CsharpPRsProcessor(),
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "commitpackft/train", "/mnt/raid0/datasets/")
  
  datasource = InstructionLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{prompt_1}\n{before_content_files}\n{message}\nPLEASE GENERATE THE PATCH\n\n"],
        mapping={
          "prompt_1": "THIS IS THE CODE\n",
          "prompt_2": "THIS IS THE INSTRUCTION\n",
        }
      ),
      FstringTemplater(
        fstring=["{out_prompt}{patches}"],
        mapping={"out_prompt": "THIS IS THE PATCH"}
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="commitpackft/train",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
  )

  print(datasource[0])
  print(tokenizer.decode(datasource[0].input_tokens))
