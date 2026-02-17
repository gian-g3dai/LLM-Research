import os
import polars as pl
from typing import List

from .atomic import RawData, Templater, ParquetReader
from .base_processors import Processor
from .base_datasource import LMDataSource

JOIN_PATH = os.path.join

class BaseCachePath():
  """
  Base class for all cache paths
  """
  def __init__(self) -> None:
    self.default_location = os.path.expanduser("~")
    self._cache_location = ".unakin/cache"
    self._init_cache_dir()

  def set_cache_path(self, cache_path: str) -> None:
    self.default_location = cache_path
    self._init_cache_dir()

  def _init_cache_dir(self) -> None:
    if not os.path.exists(self.path):
      os.makedirs(self.path, exist_ok=True)

  @property
  def path(self) -> str:
    return os.path.join(self.default_location, self._cache_location)

BASE_CACHE_PATH = BaseCachePath()

def set_pipeline_cache_path(cache_path: str) -> None:
  BASE_CACHE_PATH.set_cache_path(cache_path)

class CacheProcessor(Processor):
  """
  Cache the data
  """
  def __init__(self, name: str) -> None:
    self._cache_path = JOIN_PATH(BASE_CACHE_PATH.path, name)
    self._processors: List[Processor] = []

  def process(self, raw_data: RawData) -> RawData:
    """
    Apply the processors to the raw data and save it to the cache
    """

    print("PROCESSING CACHE:", self._cache_path)
    
    if os.path.exists(self._cache_path):
      return RawData(pl.read_parquet(self._cache_path))

    for processor in self._processors:
      raw_data = processor.process(raw_data)

    raw_data.save(self._cache_path)

    print("CACHING ", self._cache_path)

    return raw_data

  def set_processors(self, processors: List[Processor]) -> None:
    self._processors = processors


class LoadPipeline:
  """
  Read data
  Apply processors (the application of the processors should be multithreaded)
  """
  def __init__(self,
               reader: ParquetReader,
               processors: List[Processor]) -> LMDataSource:
    self._reader = reader
    self._processors = processors

  def load(self, path: str) -> RawData:

    # # TODO: check if cache exists
    # # if cache exists for this specific pipeline, load it and skip the processors
    # pipeline_hash = hash(self)
    # if storage contains cache for pipeline_hash:
    #   return cache

    cache_processor = None
    cache_position = -1

    for idx, p in enumerate(self._processors):
      if isinstance(p, CacheProcessor):
        cache_processor = p
        cache_position = idx
        break

    raw_data = self._reader.read(path)
    
    cached_processors = self._processors[:cache_position]
    uncached_processors = self._processors[cache_position + 1:]

    print("Cached processors:   ", cached_processors)
    print("Uncached processors: ", uncached_processors)

    if cache_processor is not None:
      cache_processor.set_processors(cached_processors)
      raw_data = cache_processor.process(raw_data)

      # check if the raw data is empty
      if raw_data.data.is_empty():
        raise ValueError("The raw data is empty after applying the cache processor.")
    
    for processor in uncached_processors:
      raw_data = processor.process(raw_data)

    return raw_data
  
    
def run_pipeline_from_path(pipeline: LoadPipeline,
                           data_path: str,
                           data_root: str) -> RawData:
  if os.path.isfile(os.path.join(data_root, data_path + ".parquet")):  
    datasource_path = os.path.join(data_root, data_path + ".parquet")
    return pipeline.load(datasource_path)
  else:
    # read from folder (for repos)
    datasource_path = os.path.join(data_root, data_path)
    return pipeline.load(datasource_path)