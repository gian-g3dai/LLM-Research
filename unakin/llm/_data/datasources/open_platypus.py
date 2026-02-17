import polars as pl

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from unakin.llm._data.datasources.pipeline_utils import LoadPipeline, ParquetReader, run_pipeline_from_path
from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.atomic import RawData, FstringTemplater
from unakin.llm._data.datasources.util_processors import TokenCountInfo, RenameColumns, check_columns, SumTokenCountFilter
from unakin.llm._data.datasources.base_datasource import InstructionLMDataSource

class OpenPlatypusProcessor(Processor):
  
    __data_cols__ = ["output", "instruction"]
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class OpenPlatypusReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":
  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  pipeline = LoadPipeline(
              reader=OpenPlatypusReader(),
              processors=[
                OpenPlatypusProcessor(),
                TokenCountInfo(source_col="instruction"),
                TokenCountInfo(source_col="output"),
                SumTokenCountFilter(source_cols=["instruction", "output"], token_interval=(16, 16_000)),
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "platy/train", "/mnt/raid0/datasets/")

  datasource = InstructionLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{prompt} {instruction}"],
        mapping={"prompt": "THIS IS A DESCRIPTION\n"}
      ),
      FstringTemplater(
        fstring=["\n{out_prompt} {output}"],
        mapping={"out_prompt": "THIS IS THE SOLUTION"}
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets/",
    data_path="platy/train",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )

  print(datasource[0])
  print(tokenizer.decode(datasource[0].input_tokens))
