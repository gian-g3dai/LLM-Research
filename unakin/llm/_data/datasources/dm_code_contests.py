import polars as pl

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from unakin.llm._data.datasources.pipeline_utils import LoadPipeline, ParquetReader
from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.atomic import RawData, FstringTemplater
from unakin.llm._data.datasources.util_processors import TokenCountInfo, RenameColumns, check_columns
from unakin.llm._data.datasources.base_datasource import InstructionLMDataSource
from unakin.llm._data.datasources.code_processors import Backticks, BackticksLangs, NormalizeCode

class CodeContestsProcessor(Processor):
  
    __data_cols__ = ["difficulty", "solutions", "cf_rating", "cf_tags", "public_tests", "private_tests", "generated_tests"]
    
    def __init__(self, sol_cnt: int = 10):
      self.sol_cnt = sol_cnt
      super().__init__()
  
    def process(self, data: RawData) -> RawData:

      # NOTE: Might need some preprocessing to the description - too many newlines+

      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_items = []
      for row in _df.iter_rows(named=True):

        # normal data
        description = row["description"]
        difficulty = row["difficulty"]  
        rating = row["cf_rating"]
        tags = row["cf_tags"]

        # tests
        public_tests = row["public_tests"]
        private_tests = row["private_tests"]
        generated_tests = row["generated_tests"]

        # solutions is a dict
        solutions_dict = row["solutions"] # {"solution": [N], "language": [N]}
        solutions = solutions_dict["solution"]
        languages = solutions_dict["language"]

        # sample different languages
        lang_set = set(languages)

        # make a dict 
        lang_dict = {lang: [] for lang in lang_set}
        for sol, lang in zip(solutions, languages):
          lang_dict[lang].append(sol)

        # even split
        for lang, sols in lang_dict.items():
          for sol in sols[:self.sol_cnt]:
            new_items.append({
              "description": description,
              # "difficulty": difficulty,
              "solution": sol,
              "language": lang,
              # "rating": rating,
              # "tags": tags,
              # "public_tests": public_tests,
              # "private_tests": private_tests,
              # "generated_tests": generated_tests
            })
      # create a new dataframe from the new items
      new_df = pl.DataFrame(new_items)
      return RawData(new_df)

class CodeContestsReader(ParquetReader):
  
  def __init__(self):
    super().__init__()

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)

if __name__ == "__main__":
  
  ds_path = "/mnt/raid0/datasets/code_contests/valid.parquet"
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")
  
  datasource = InstructionLMDataSource(
    raw_data=LoadPipeline(
              reader=CodeContestsReader(),
              processors=[
                CodeContestsProcessor(sol_cnt=10),
                TokenCountInfo(source_cols=["description", "solution"]),
                NormalizeCode(source_col="solution"),
                BackticksLangs(source_col="solution", language_col="language"),
                ]
            ),
    templaters = [
      FstringTemplater(
        fstring=["{prompt} {description}"],
        mapping={"prompt": "THIS IS A DESCRIPTION\n"}
      ),
      FstringTemplater(
        fstring=["\n{out_prompt} {solution}"],
        mapping={"out_prompt": "THIS IS THE SOLUTION"}
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="code_contests/valid",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )

  print(datasource[0])

  # print(raw_data.data.head(2))

  # print(raw_data.data[0]["solution"][0])