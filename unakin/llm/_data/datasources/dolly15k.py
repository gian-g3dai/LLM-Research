import polars as pl
import random

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline, ParquetReader, run_pipeline_from_path
from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.atomic import RawData, FstringTemplater
from unakin.llm._data.datasources.util_processors import TokenCountInfo, RenameColumns, check_columns, SumTokenCountFilter
from unakin.llm._data.datasources.base_datasource import InstructionLMDataSource

class DollyProcessor(Processor):
  
    __data_cols__ = ["response", "instruction", "context", "category"]
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class DollyReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":
  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  pipeline = LoadPipeline(
              reader=DollyReader(),
              processors=[
                DollyProcessor(),
                TokenCountInfo(source_col="instruction"),
                TokenCountInfo(source_col="response"),
                SumTokenCountFilter(source_cols=["instruction", "response"], token_interval=(256, 16_000)),
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "databricks_dolly/dolly15k", "/mnt/raid0/datasets/")

  datasource = InstructionLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["[INST] \n[CONTEXT]\n{context}\n[/CONTEXT]\n {instruction} [/INST] "],
        mapping={"prompt": "THIS IS A DESCRIPTION\n"}
      ),
      FstringTemplater(
        fstring=["{response} "],
        mapping={"out_prompt": ""}
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets/",
    data_path="databricks_dolly/dolly15k",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )

  # print(datasource[0])

  some_random_indices = random.sample(range(len(datasource)), 10)

  for i in some_random_indices:
    print(tokenizer.decode(datasource[i].input_tokens))
    print("===============================================")
