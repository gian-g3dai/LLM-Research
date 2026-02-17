import polars as pl

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
from .atomic import RawData, Templater, ParquetReader
from multiprocessing import Pool, cpu_count, current_process



class Processor(ABC):
# this should be written multithreaded
# process multithreaded & join at the end of the process

  def __init__(self, safe_mode: bool = True):
    self._safe_mode = safe_mode

  @abstractmethod
  def process(self, data: RawData) -> RawData:
    raise NotImplementedError
  

class FilterLambdaProcessor(Processor):
  """
  Filter the data using a function.
  """
  def __init__(self, func, source_col: str):
    super().__init__()
    self._func = func
    self.source_col = source_col
    self.multithreaded = False

  def process(self, data: RawData) -> RawData:
    data: pl.DataFrame = data.data

    # check if the source column exists
    if self.source_col not in data.columns:
      raise ValueError(f"Column {self.source_col} does not exist.")
    
    data = data.filter(self._func(pl.col(self.source_col)))

    return RawData(data)


class LambdaProcessor(Processor):
  """
  Applies a lambda function to source column and saves it to target column.
  """
  def __init__(self,
               func,
               source_col: str,
               target_col: Optional[str] = None,
               multithreaded: bool = False,
               num_cpus: int = None):
    super().__init__()
    self._func = func
    self._source_col = source_col
    self._target_col = target_col
    self._multithreaded = multithreaded
    self._num_cpus = num_cpus

  def mt_process(self, data: pl.DataFrame) -> pl.DataFrame:

    tmp = data.clone()

    if self._num_cpus is None:
      num_cpus = cpu_count()
    else:
      num_cpus = self._num_cpus

    pool = Pool(num_cpus)

    chunk_size = len(tmp) // num_cpus
    jobs = []
    while not tmp.is_empty():
        job, tmp = tmp[:chunk_size], tmp[chunk_size:]
        if len(job) > 0:
          jobs.append(job)

    results = list(
        pool.map(partial(self._func), jobs)
    )

    # flatten the list
    results = [item for sublist in results for item in sublist]

    # recreate the parquet table
    results_data = pl.DataFrame(results)

    return results_data

  def process(self, data: RawData) -> RawData:
    data: pl.DataFrame = data.data
    source_col = self._source_col
    target_col = self._target_col
    if target_col is None:
      target_col = source_col

    # check if the source column exists
    if source_col not in data.columns:
      raise ValueError(f"Column {source_col} does not exist")
    
    if self._multithreaded:
      data = self.mt_process(data)
    else:
      data = data.with_columns([
        pl.col(source_col).map_elements(self._func).alias(target_col)
      ])    
    return RawData(data)
  
