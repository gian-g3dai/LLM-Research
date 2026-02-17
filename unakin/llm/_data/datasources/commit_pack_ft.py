
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

class CommitPackProcessor(Processor):
  
    __data_cols__ = ["subject", "message", "old_contents", "new_contents", "split"]

    # NOTE: there are old_content == "" => there is a new content added
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)
    

# TODO: This will be done at the preprocessing stage.
class ToPatches(Processor):
    def __init__(self,
                 source_col: str,
                 target_col: str):
      super().__init__()
      self._source_col = source_col
      self._target_col = target_col

    # Function to remove first 2 lines from each string
    def add_filenames_to_patch(self, row):
        lines = row["patches"].split('\n')
        fst_line = f"diff --git a/{row.old_file} b/{row.new_file}" 
        old_line = f"--- a/{row.old_file}"
        new_line = f"+++ b/{row.new_file}"
        lines = [fst_line] + [old_line] + [new_line] + lines[2:]
        lines = '\n'.join(lines)
        return lines

    def process(self, data: RawData) -> RawData:

      data: pl.DataFrame = data.data

      # data to pandas
      data_pd = data.to_pandas()

      data_pd = data_pd[data_pd[self._source_col] != '']

      data_pd["patches"] = data_pd.apply(lambda row: '\n'.join(difflib.unified_diff(row[self._source_col].splitlines(), row[self._target_col].splitlines(), lineterm='')), axis=1)

      data_pd["patches"] = data_pd.apply(self.add_filenames_to_patch, axis=1)

      data_pd[self._source_col] = data_pd[self._source_col].apply(lambda cell: "\n".join([str(idx + 1) + " " + line for idx, line in enumerate(cell.splitlines())]))

      data = pl.from_pandas(data_pd)

      return RawData(data)


class CommitPackReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":
  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")
  
  pipeline = LoadPipeline(
              reader=CommitPackReader(),
              processors=[
                CommitPackProcessor(),
                TokenCountInfo(source_col="old_contents"),
                TokenCountInfo(source_col="new_contents"),
                StripProcessor(source_cols=["subject", "message"]),
                NormalizeCode(source_col="old_contents"),
                NormalizeCode(source_col="new_contents"),
                # BackticksLangs(source_col="old_contents", language_col="split"),
                # BackticksLangs(source_col="new_contents", language_col="split"),
                ToPatches(source_col="old_contents", target_col="new_contents")
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "commitpackft/train", "/mnt/raid0/datasets/")
  
  datasource = InstructionLMDataSource(
    raw_data=raw_data,
    templaters = [
      FstringTemplater(
        fstring=["{prompt_1}\n[start of {old_file}]\n{old_contents}\n[end of {old_file}]\n{message}\nPLEASE GENERATE THE PATCH\n\n"],
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
