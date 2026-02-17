

import polars as pl

from typing import List, Optional, Any

from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

from .atomic import RawData
from .base_processors import Processor, LambdaProcessor

# this should be loaded from the dataset
TOKENIZER_PATH = "/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model"

def check_columns(data: pl.DataFrame, columns: List[str]):

  data: pl.DataFrame = data

  for col in columns:
    if col not in data.columns:
      raise ValueError(f"Column {col} does not exist.")
    
class TokenCountInfo(LambdaProcessor):
  def __init__(self, source_col: str, target_col: str = None, multithreaded: bool = True, num_cpus: int = None):    
    """
    Count the tokens for each source_col and print the total number of tokens / source_col
    For each row there will be source_col + "_token_count" column added
    """
    if target_col is None:
      target_col = source_col + "_token_count"

    if multithreaded:
      super().__init__(self.mt_partial_process, source_col, target_col, multithreaded, num_cpus)
    else:
      tokenizer = LlamaTokenizer(model_path=TOKENIZER_PATH)
      super().__init__(lambda x: len(tokenizer.tokenize(x, False, False)[0]), source_col, target_col)

  def mt_partial_process(self, partial_df: pl.DataFrame) -> List[Any]:

    tokenizer = LlamaTokenizer(model_path=TOKENIZER_PATH)

    all_rows = []
    for row in partial_df.iter_rows(named=True):
      tokenized = tokenizer.tokenize(row[self._source_col], False, False)
      if len(tokenized) == 0:
        continue
      row[self._target_col] = len(tokenized[0])
      all_rows.append(row)


    return all_rows

  def process(self, data: RawData) -> RawData:
    raw_data = super().process(data)

    # print token count info
    data: pl.DataFrame = raw_data.data

    print("Token count info:")
    col_tokens = data[self._target_col].sum()
    print(f"{self._source_col}: {col_tokens / 1_000_000}M")

    return RawData(data)

# add token count flter
class SumTokenCountFilter(Processor):
  """
  Filter rows based on the sum of the token count of the source columns
  """
  def __init__(self, source_cols: List[str], token_interval: (int, int)):
    super().__init__()
    self._source_cols = source_cols
    self._token_interval = token_interval

  def process(self, data: RawData) -> RawData:    
    # check if the source columns exists
    check_columns(data.data, self._source_cols)
      
    data: pl.DataFrame = data.data

    print(f"TOKEN INTERVAL: {self._token_interval} FILTERING...")
    print(f"Original number of rows: {len(data)}")

    token_count_cols = [x + "_token_count" for x in self._source_cols]

    # check if all token_count_cols exist
    check_columns(data, token_count_cols)

    # this will be done in TokenCountInfo which is faster
    # for col, token_count_col in zip(self._source_cols, token_count_cols):
    #   if data[col].dtype == pl.datatypes.List:
    #     data = data.with_columns([
    #       pl.col(col).map_elements(lambda x: sum([len(tokenizer.tokenize(x_i, False, False)[0]) for x_i in x])).alias(token_count_col)
    #     ])
    #   elif data[col].dtype == pl.datatypes.Utf8:
    #     data = data.with_columns([
    #       pl.col(col).map_elements(lambda x: len(tokenizer.tokenize(x, False, False)[0])).alias(token_count_col)
    #     ])
    #   else:
    #     raise ValueError(f"Column {col} is not a list or string column.")
      
    excude_cols = list(set(data.columns) - set(token_count_cols))

    # sum all the token counts
    data = data.with_columns([
      pl.fold(0, lambda acc, s: acc + s, pl.all().exclude(excude_cols)).alias("sum_token_count")
    ])

    # filter
    data = data.filter(pl.col("sum_token_count") < self._token_interval[1])
    data = data.filter(pl.col("sum_token_count") > self._token_interval[0])

    # drop the token count columns
    data = data.drop(token_count_cols + ["sum_token_count"])

    print(f"Filtered to {len(data)} rows.")

    return RawData(data)
  

class RenameColumns(Processor):
  """
  Rename columns
  """
  def __init__(self, rename_dict: dict):
    super().__init__()
    self._rename_dict = rename_dict

  def process(self, data: RawData) -> RawData:
    
    # check if the source column exists
    check_columns(data.data, self._rename_dict.keys())
    
    data: pl.DataFrame = data.data
    
    data = data.rename(self._rename_dict)

    return RawData(data)
  
class StripProcessor(Processor):
  """
  Strip whitespace from string / list of strings columns 
  """
  def __init__(self, source_cols: List[str]):
    super().__init__()
    self._source_cols = source_cols

  def process(self, data: RawData) -> RawData:
      
      # check if the source column exists
      check_columns(data.data, self._source_cols)
      
      data: pl.DataFrame = data.data

      # TODO: make a processing function

      for col in self._source_cols:
        
        if data[col].dtype == pl.datatypes.List:
          data = data.with_columns([
            pl.col(col).map_elements(lambda x: [x_i.strip() for x_i in x]).alias(col)
          ])
        elif data[col].dtype == pl.datatypes.Utf8:
          data = data.with_columns([
            pl.col(col).map_elements(lambda x: x.strip()).alias(col)
          ])
        else:
          raise ValueError(f"Column {col} is not a list or string column.")
  
      return RawData(data)
  

class ShuffleProcessor(Processor):
  """
  Shuffle the rows of the dataframe
  """
  def __init__(self):
    super().__init__()

  def process(self, data: RawData) -> RawData:
    data: pl.DataFrame = data.data
    # transform to pandas
    data = data.to_pandas()
    # shuffle the rows
    data = data.sample(frac=1)
    # transform back to polars
    try:
      data = pl.from_pandas(data)
    except:
      raw_data = []
      for idx, row in data.iterrows():
        raw_data.append(row)
      data = pl.DataFrame(raw_data)

    return RawData(data)