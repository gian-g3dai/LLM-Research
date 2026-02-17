from enum import Enum
from typing import Any, List, NamedTuple
import torch.distributed as dist
import torch
import pickle
import os
import sys

class PyObjShard(NamedTuple):
  data: List
  full_size: int

  def __len__(self) -> int:
    return len(self.data)

def is_distributed() -> bool:
  return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
  return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
  return dist.get_world_size() if is_distributed() else 1

def get_dist_info() -> tuple:
  return get_rank(), get_world_size()

def is_master() -> bool:
  return get_rank() == 0

def setup_distributed_process_group():
  """Setup distributed process group"""
  dist.init_process_group(
    backend="nccl",
    init_method="env://",
  )

  # Set the GPU to use
  torch.cuda.set_device(get_rank())

def setup_async_distributed_process_group():
  dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method="env://")
  torch.cuda.set_device(get_rank())

def print_rank0(*args, **kwargs):
  if get_rank() == 0:
    print(*args, **kwargs)

def dist_partition_py_type(data: List) -> PyObjShard:
  if not is_distributed():
    return PyObjShard(data=data, full_size=len(data))

  rank, world_size = get_dist_info()
  return PyObjShard(
    data=data[rank::world_size],
    full_size=len(data)
  )

def dist_gather_py_type(data: PyObjShard) -> List:
  if not is_distributed():
    return data.data

  acc = [None] * get_world_size()
  dist.all_gather_object(acc, data.data)
  acc = sum(acc, [])
  return acc

def destroy_process_group():
  dist.destroy_process_group()


def setup_print(rank: int, debug: bool = False):
  """Setup stdout/stderr redirection"""
  if debug:
    return

  if rank != 0:
    f = open(os.devnull, "w")
    sys.stdout = f
    sys.stderr = f

class ParallelSpec(Enum):
  PROCESS = "process"
  TENSOR_PARALLEL = "tensor_parallel"

class DistributedList:
  def __init__(self, data: List[Any], parallel_spec: ParallelSpec):
    self.data = data
    self.parallel_spec = parallel_spec
  
  def shard(self) -> List[Any]:
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    SHARD_SIZE  = max(len(self.data) // WORLD_SIZE, 1)
    chunk_start = SHARD_SIZE * RANK
    chunk_end   = SHARD_SIZE + chunk_start
    if self.parallel_spec == ParallelSpec.PROCESS:
      return self.data[chunk_start:chunk_end]
    elif self.parallel_spec == ParallelSpec.TENSOR_PARALLEL:
      return self.data  
  
  def gather(self, data: List[Any]) -> List[Any]:
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    if self.parallel_spec == ParallelSpec.PROCESS:
      acc = [None] * WORLD_SIZE
      dist.all_gather_object(acc, data)
      acc = sum(acc, [])
      return acc
    elif self.parallel_spec == ParallelSpec.TENSOR_PARALLEL:
      return data