import torch.distributed
import yaml
import torch
import torch.distributed

from typing import List, Dict, Any, Optional, Union, Tuple
from copy import deepcopy

from .atomic import RawData, FstringTemplater
from .pipeline_utils import LoadPipeline, CacheProcessor, BASE_CACHE_PATH, JOIN_PATH, run_pipeline_from_path
from .base_datasource import *
from .base_processors import *
from .util_processors import *
from .code_processors import *
from .open_platypus import OpenPlatypusReader, OpenPlatypusProcessor
from .commit_pack_ft import CommitPackReader, CommitPackProcessor, ToPatches
from .multi_turn_oasst import OAsstReader, OAsstProcessor
from .csharp_code import CsharpReader, CsharpProcessor
from .godot_code import GodotReader, GodotProcessor
from .dm_code_contests import CodeContestsReader, CodeContestsProcessor
from .cpp_code import CppReader, CppProcessor
from .muffin import MuffinReader, MuffinProcessor
from .github_code import GithubReader, GithubProcessor
from .dolly15k import DollyReader, DollyProcessor
from .redpajama import RedPJProcessor, RedPJReader
from .helpsteer import HelpSteerReader, HelpSteerProcessor
from .hotpot_qa import HotpotReader, HotpotProcessor
from .stack_exchange import StackExReader, StackExProcessor
from .unity_docs import UnityDocsReader, UnityDocsProcessor
from .csharp_prs import CsharpPRsProcessor, CsharpPRsReader
from .swe_bench import SWEProcessor, SWEReader
from .tess import TessProcessor, TessReader
from .mathinst import MathInstReader, MathInstProcessor
from .webinst import WebInstReader, WebInstProcessor
from .stackexchange_qa import StackexchangeQAProcessor, StackexchangeQAReader
from .github_reason import GitReasonProcessor, GitReasonReader
from .magpie_datasource import MagpieProcessor, MagpieReader
from .unnatural_datasource import UnnaturalProcessor, UnnaturalReader
from .unity_qa_datasource import UnityQAProcessor, UnityQAReader


def load_config(file_path):
  with open(file_path, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
  return config

def get_pipeline(config: Union[dict, str]) -> LoadPipeline:
  """
  Returns a LoadPipeline instance
  """
  if isinstance(config, str):
    config = load_config(config)

  if isinstance(config['reader'], dict):
    # Handling readers with parameters
    reader_class_name = config['reader']['type']
    reader_class = globals()[reader_class_name]
    # drop the type key
    del config['reader']['type']
    reader = reader_class(**config['reader'])
  else:
    # Handling readers without parameters
    reader_class = globals()[config['reader']]
    reader = reader_class()

  processors = []
  for processor_config in config['processors']:
    if isinstance(processor_config, dict):
      # Handling processors with parameters
      processor_class_name = list(processor_config.keys())[0]
      processor_class = globals()[processor_class_name]
      processor_instance = processor_class(**processor_config[processor_class_name])
    else:
      # Handling processors without parameters
      processor_class = globals()[processor_config]
      processor_instance = processor_class()
    processors.append(processor_instance)

  pipeline = LoadPipeline(reader=reader, processors=processors)
  return pipeline

# def run_pipeline_from_path(pipeline: LoadPipeline,
#                            data_path: str,
#                            data_root: str) -> RawData:
#   if os.path.isfile(os.path.join(data_root, data_path + ".parquet")):  
#     datasource_path = os.path.join(data_root, data_path + ".parquet")
#     return pipeline.load(datasource_path)
#   else:
#     # read from folder (for repos)
#     datasource_path = os.path.join(data_root, data_path)
#     return pipeline.load(datasource_path)
                           

def is_master() -> bool:
  if torch.distributed.is_initialized():
    return torch.distributed.get_rank() == 0
  else:
    return True

def is_distributed_context() -> bool:
  return torch.distributed.is_initialized()


def get_datasource(config_path, cache_file_name, tokenizer, data_root, fim_rev_config = None) -> LMDataSource:
    
    config = load_config(config_path) if isinstance(config_path, str) else config_path

    tmp_path = JOIN_PATH(BASE_CACHE_PATH.path, cache_file_name) if is_distributed_context() \
                else None

    # Extract relevant information
    datasource_config = config
    pipeline_config = datasource_config["pipeline"]
    templaters_config = datasource_config["templaters"]
    datasource_type = datasource_config["type"]
    fim_rev_config = fim_rev_config
    
    # Instantiate the pipeline
    load_pipeline = get_pipeline(pipeline_config)
    pipeline_raw_data = None

    # In distributed contexts only the master process should run the pipeline
    if is_master():
      pipeline_raw_data = run_pipeline_from_path(load_pipeline,
                                                datasource_config["data_path"],
                                                data_root)
      # is_master will return true for non-distributed contexts
      # therefore we need to ensure that writing is only done in distributed contexts
      if is_distributed_context():
        pipeline_raw_data.data.write_parquet(tmp_path)

    # Block until the master process has finished processing the data
    # Non-master processes will have pipeline_raw_data set to None
    # therefore reading will happen here for them
    torch.distributed.barrier()

    # TODO: Unknown err
    while pipeline_raw_data is None:
      try:
        pipeline_raw_data = RawData(pl.read_parquet(tmp_path))
      except:
        pass

    # Wait until all processes have read the data
    torch.distributed.barrier()
    
    # Remove the temporary file
    # In non-distributed contexts we do not write the raw_data to the disk
    # therefore tmp_path will be set to None
    if is_master() and is_distributed_context():
      os.remove(tmp_path)

    # Last barrier is to ensure that we minimize sync overhead induced by
    # waiting for the master process to remove a potentially large file
    torch.distributed.barrier()

    templaters = [eval(t["type"])(**t) for t in templaters_config]

    # Dynamically instantiate the specified data source type
    data_source = eval(datasource_type)(
        raw_data=pipeline_raw_data,
        templaters=templaters,
        tokenizer=tokenizer,
        data_root=data_root,
        data_path=datasource_config["data_path"],
        percent_interval=datasource_config["percent_interval"],
        metric_list=datasource_config["metric_list"],
        fim_rev_config=fim_rev_config
    )

    return data_source