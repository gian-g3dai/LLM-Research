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
from unakin.llm._data.datasources.pipeline_utils import LoadPipeline
from unakin.llm._data.datasources.atomic import RawData, FstringTemplater, FolderCodeReader
from unakin.llm._data.datasources.util_processors import check_columns
from unakin.llm._data.datasources.base_datasource import SimpleLMDataSource
from unakin.llm._data.datasources.code_utils import (
  read_file_lines_multiencoding,
  GlobbingMethod,
  ReformatMethod,
  _reformat_using_dotnet_formatter)

def _parse_godot_worker(files_list: List[str], reformat_method: ReformatMethod) -> Tuple[Dict[str, Any], Dict[str, int]]:

  # create an empty list to hold the row dictionaries
  rows = []

  total_lines = 0
  empty_files = 0
  format_errors = 0

  worker_rank = current_process()._identity[0]
  
  # for f in tqdm(files_list, desc=f"Cache process {worker_rank}", position=worker_rank):
  for f in files_list:
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

class GodotProcessor(Processor):
  
    __data_cols__ = ["file_path", "line_count", "contents"]
    
    def __init__(self):
      super().__init__()
  
    def process(self, data: RawData) -> RawData:
      _df = data.data

      # check if the source columns exists
      check_columns(data, self.__data_cols__)

      new_df = _df
      return RawData(new_df)

class GodotReader(FolderCodeReader):
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
      cs_files = glob.glob(os.path.join(path, '**/*.gd'), recursive=True)
    else:
      raise Exception(f"GodotReader only supports GlobbingMethod.all")
    
    if self.reformat_method != ReformatMethod.no_reformat:
      raise Exception(f"GodotReader only supports ReformatMethod.no_reformat")

    
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
        pool.map(partial(_parse_godot_worker, reformat_method=self.reformat_method), jobs)
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
  
  datasource = SimpleLMDataSource(
    raw_data=LoadPipeline(
              reader=GodotReader(
                globbing_method="all",
                reformat_method="no_reformat"
              ),
              processors=[
                GodotProcessor(),
                # TokenCountInfo(source_cols=["output", "instruction"]),
                ]
            ),
    templaters = [
      FstringTemplater(
        fstring=["{prompt} {contents}"],
        mapping={"prompt": "THIS IS A DESCRIPTION\n"}
      )],
    tokenizer=tokenizer,
    data_root = "/mnt/raid0/datasets",
    data_path="godot_scraped",
    percent_interval = (0, 100),
    metric_list=["perplexity"],
    )

  print(datasource[0])
  # print(tokenizer.decode(datasource[0].input_tokens))