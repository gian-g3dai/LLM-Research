import os
import glob
import codecs
import polars as pl

from enum import Enum
from tqdm import tqdm
from functools import partial
from typing import List, Optional, Any, Tuple, Dict
from multiprocessing import Pool, cpu_count, current_process
from tests.helpers import get_resource_path


from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer
from unakin.llm._data.datasources.base_processors import Processor
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline, run_pipeline_from_path, CacheProcessor, set_pipeline_cache_path
from unakin.llm._data.datasources.atomic import RawData, FstringTemplater, FolderCodeReader
from unakin.llm._data.datasources.util_processors import check_columns, TokenCountInfo, SumTokenCountFilter
from unakin.llm._data.datasources.base_datasource import SimpleLMDataSource
from unakin.llm._data.datasources.code_utils import (
  read_file_lines_multiencoding,
  GlobbingMethod,
  ReformatMethod,
  _reformat_using_dotnet_formatter)


def _parse_csharp_worker(files_list: List[str], reformat_method: ReformatMethod) -> Tuple[Dict[str, Any], Dict[str, int]]:

  # create an empty list to hold the row dictionaries
  rows = []

  total_lines = 0
  empty_files = 0
  format_errors = 0

  worker_rank = current_process()._identity[0]
  
  for f in tqdm(files_list, desc=f"Cache process {worker_rank}", position=worker_rank):
  # for f in files_list:
    lines = read_file_lines_multiencoding(f)
    line_count = len(lines)
    total_lines += line_count
    # print(f"{f}: {line_count} lines")

    contents = ''.join(lines)
    contents = contents.strip()

    bom_character = '\ufeff'
    # remove the BOM character if it exists
    if bom_character in contents:
      contents = contents.replace(bom_character, '')

    # check if the file is empty
    if len(contents) == 0:
      empty_files += 1
      continue

    if reformat_method == ReformatMethod.reformat or reformat_method == ReformatMethod.strip_errors_only:
      try:
        if reformat_method == ReformatMethod.strip_errors_only:
          _ = _reformat_using_dotnet_formatter(contents)
        else:
          contents = _reformat_using_dotnet_formatter(contents)
      except Exception:
        # print(f"Skipping {f} due to format error: {e}")
        format_errors += 1
        continue

    if contents is None:
      empty_files += 1
      continue

    # add a new row dictionary to the list
    rows.append({
      'file_path': f,
      'line_count': line_count,
      'contents': contents
      }
    )
  
  stats = {
    "total_lines": total_lines,
    "empty_files": empty_files,
    "format_errors": format_errors
  }

  return rows, stats

class CsharpProcessor(Processor):
  
    __data_cols__ = ["file_path", "line_count", "contents"]
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class CsharpReader(FolderCodeReader):
  """
  This should be migrated from the old codebase
  MUST: function that reads multithreaded
  """

  def __init__(self, globbing_method: str, reformat_method: str):
    super().__init__()
    try:
      self.globbing_method = GlobbingMethod[globbing_method]
    except KeyError:
      raise Exception(f"Unknown globbing method: {globbing_method}")
    
    try:
      self.reformat_method = ReformatMethod[reformat_method]
    except KeyError:
      raise Exception(f"Unknown reformat method: {reformat_method}")


  def read(self, path: str) -> RawData:

    if self.globbing_method == GlobbingMethod.all:
      # get all .cs files in the dataset
      cs_files = glob.glob(os.path.join(path, '**/*.cs'), recursive=True)
    elif self.globbing_method == GlobbingMethod.user_only:
      # get assets/scripts folders
      cs_files = glob.glob(os.path.join(path, '**/Assets/Scripts/**/*.cs'), recursive=True)
    elif self.globbing_method == GlobbingMethod.libs_only:
      all_cs_files = glob.glob(os.path.join(path, '**/*.cs'), recursive=True)
      assets_scripts_files = glob.glob(os.path.join(path, '**/Assets/Scripts/**/*.cs'), recursive=True)
      cs_files = list(set(all_cs_files) - set(assets_scripts_files))
    else:
      raise Exception(f"Unknown globbing method: {self.globbing_method}")
    
    num_cpus = cpu_count() - 1

    pool = Pool(num_cpus)

    # split files list into num_cpus chunks and run the parse_files function on each chunk
    tmp = cs_files.copy()

    chunk_size = len(tmp) // num_cpus
    jobs = []
    while tmp:
        job, tmp = tmp[:chunk_size], tmp[chunk_size:]
        jobs.append(job)

    # run the _parse_files_worker on each chunk
    results = list(
        pool.map(partial(_parse_csharp_worker, reformat_method=self.reformat_method), jobs)
    )

    # recombine results
    all_rows = []
    total_lines = 0
    empty_files = 0
    format_errors = 0

    for rows, stats in results:
        all_rows += rows
        total_lines += stats["total_lines"]
        empty_files += stats["empty_files"]
        format_errors += stats["format_errors"]

    print(f"Processed {len(cs_files)} files")
    print(f"Found {empty_files} empty files")
    print(f"Total lines: {total_lines}")
    print(f"Format errors: {format_errors}")

    # create the dataframe from the list of row dictionaries
    df = pl.DataFrame(all_rows)

    return self.process_parquet(df)

  def process_parquet(self, parquet: pl.DataFrame) -> RawData:
    return RawData(parquet)


if __name__ == "__main__":
  
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model")

  set_pipeline_cache_path("/mnt/raid0/datasets/tmp_cache/")

  pipeline = LoadPipeline(
              reader=CsharpReader(
                globbing_method="all",
                reformat_method="reformat"
              ),
              processors=[
                CsharpProcessor(),
                TokenCountInfo(source_col="contents"),
                SumTokenCountFilter(source_cols=["contents"], token_interval=(16, 16_384)),
                CacheProcessor(name="recompose_unity_user_filtered")
                ]
            )
  
  raw_data = run_pipeline_from_path(pipeline, "recompose_unity_user", "/mnt/raid0/datasets/")

  print("Length of raw data:", len(raw_data.data))

  # datasource = SimpleLMDataSource(
  #   raw_data=raw_data,
  #   templaters = [
  #     FstringTemplater(
  #       fstring=["{prompt} {contents}"],
  #       mapping={"prompt": "THIS IS A DESCRIPTION\n"}
  #     )],
  #   tokenizer=tokenizer,
  #   data_root = "/mnt/raid0/datasets",
  #   data_path="recompose_unity_user",
  #   percent_interval = (0, 100),
  #   metric_list=["perplexity"],
  #   )

  # print(datasource[0])
  # print(tokenizer.decode(datasource[0].input_tokens))