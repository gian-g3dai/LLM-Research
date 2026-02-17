import polars as pl

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from unakin.llm._data.datasources.pipeline_utils import LoadPipeline, ParquetReader
from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.atomic import RawData, FstringTemplater
from unakin.llm._data.datasources.util_processors import *
from unakin.llm._data.datasources.base_datasource import MultiTurnLMDataSource
from unakin.llm._data.datasources.code_processors import Backticks, BackticksLangs, NormalizeCode

class OAsstProcessor(Processor):
  
    __data_cols__ = ["user_messages", "assistant_messages"]
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:

      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      # NOTE: Might add the tree processing here

      return RawData(_df)

class OAsstReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":
  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")
  
  datasource = MultiTurnLMDataSource(
    raw_data=LoadPipeline(
              reader=OAsstReader(),
              processors=[
                OAsstProcessor(),
                TokenCountInfo(source_cols=["user_messages", "assistant_messages"]),
                StripProcessor(source_cols=["user_messages", "assistant_messages"]),
                SumTokenCountFilter(source_cols=["user_messages", "assistant_messages"], max_tokens=1024),
                ]
            ),
    templaters = [
      FstringTemplater(
        fstring=["{prompt} {user_messages}"],
        mapping={"prompt": "THIS IS A DESCRIPTION\n"}
      ),
      FstringTemplater(
        fstring=["\n{out_prompt} {assistant_messages}"],
        mapping={"out_prompt": "THIS IS THE SOLUTION"}
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="multi_oasst/validation",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )

  print(datasource[0])

  # print(raw_data.data.head(2))

  # print(raw_data.data[0]["solution"][0])