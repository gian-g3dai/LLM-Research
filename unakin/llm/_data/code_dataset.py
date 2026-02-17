import random
from collections import deque
from copy import deepcopy
from typing import Iterator, List, NamedTuple, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from unakin.llm.tokenizers import TokenizerBase

from .datasources import LMTrainingSample, LMDataSource, LMTrainingBatch


def collate_fn(samples_and_cnt) -> LMTrainingBatch:
  samples, sample_cnt = zip(*samples_and_cnt)
  return LMTrainingBatch(
    input_tokens=torch.tensor([s.input_tokens for s in samples]),
    output_tokens=torch.tensor([s.output_tokens for s in samples]),
    training_mask=torch.tensor([s.training_mask for s in samples]),
    positional_ids=torch.tensor([s.positional_ids for s in samples]),
    sequence_lengths=[s.sequence_lengths for s in samples],
    datasource_indices=torch.tensor([s.datasource_index_mask for s in samples]),
  ), sample_cnt[0]

def range_size(range: Tuple[int, int]) -> int:
  return range[1] - range[0] + 1

def shard_group_range(chunk: Tuple[int, int], shard_rank, num_shards):
  shard_size = range_size(chunk) // num_shards
  shard_start = chunk[0] + shard_rank * shard_size
  
  if shard_rank == num_shards - 1:
    shard_end = chunk[1]
  else:
    shard_end = shard_start + shard_size - 1
  
  return (shard_start, shard_end)


def make_empty_sample() -> LMTrainingSample:
  return LMTrainingSample(
    input_tokens=[],
    output_tokens=[],
    training_mask=[],
    positional_ids=[],
    sequence_lengths=[],
    min_token_length=0,
    datasource_index_mask=[],
  )

class GPTCodeDataset(IterableDataset):
  def __init__(
      self,
      datasources: List[LMDataSource],
      tokenizer: TokenizerBase,
      max_length: int,
      process_rank: int = 0,
      world_size: int = 1,
      multisample: bool = True,
      dynamic_seq_len: bool = False,
      infinite: bool = True,
      no_truncate_next: bool = False,
      datasources_weights: List[float] = None,
      ) -> None:
    super().__init__()

    self.ds_cursors = [0] * len(datasources)
    self.ds_ranges = [(0, len(ds) - 1) for ds in datasources]
    self.datasources = datasources

    self.available_datasources = list(range(len(datasources)))
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.process_rank = process_rank
    self.world_size = world_size
    self.multisample = multisample
    self.infinite = infinite
    self.no_truncate_next = no_truncate_next
    self.datasources_weights = datasources_weights
    self.dynamic_seq_len = dynamic_seq_len
    self.sample_queue = deque()
    
    self.sample_cnt = torch.tensor([0] * len(datasources))

    self.worker_id = 0
    self.num_workers = 1

    if self.dynamic_seq_len and self.no_truncate_next:
      raise ValueError("Dynamic seq len and no truncate next are incompatible")

    if self.dynamic_seq_len and self.multisample:
      raise ValueError("Dynamic seq len and multisample are incompatible")


  def increment_cursor(self,
                       cursor: int,
                       cursor_range: Tuple[int, int]) -> int:
    range_start = cursor_range[0]
    range_end   = cursor_range[1]

    if cursor <= range_end:
      cursor += 1

    if self.infinite:
      cursor = range_start if cursor > range_end else cursor

    return cursor


  def has_any_data(self, running_datasources: List[int]) -> bool:
    return len(running_datasources) > 0 or len(self.sample_queue) > 0

  def cursor_reached_end_of_range(self, cursor: int, cursor_range: Tuple[int, int]) -> bool:
    return cursor > cursor_range[1]

  def add_sample_to_buffer(self, current_buffer: LMTrainingSample, new_sample: LMTrainingSample) -> Tuple[LMTrainingSample, LMTrainingSample]:
    if len(current_buffer) + len(new_sample) > self.max_length:
      if len(current_buffer) == 0:
        current_buffer.inplace_extend_(new_sample)
        new_sample = make_empty_sample()
      else:
        if not self.no_truncate_next:
          current_buffer.inplace_extend_(new_sample)
          new_sample = make_empty_sample()
        else:
          current_buffer.inplace_fill_(self.max_length, 2)

      current_buffer, _ = current_buffer.truncate(self.max_length)
    else:
      current_buffer.inplace_extend_(new_sample)
      new_sample = make_empty_sample()
    return current_buffer, new_sample

  def buffer_is_saturated(self, current_buffer: LMTrainingSample) -> bool:
    if not self.multisample:
      return True
    
    if self.dynamic_seq_len:
      return True
    else:
      return len(current_buffer) >= self.max_length

  def __iter__(self) -> Iterator[LMTrainingSample]:
    
    # first split that we do is across all ranks
    for idx, ds in enumerate(self.datasources):
      if range_size(self.ds_ranges[idx]) < self.world_size:
        raise ValueError(f"Datasource {idx} has less samples than world size ({len(ds)} < {self.world_size})")
      
      self.ds_ranges[idx] = shard_group_range(self.ds_ranges[idx], self.process_rank, self.world_size)

    # second split that we do is thread workers
    worker_info = get_worker_info()
    if worker_info is not None:
      self.worker_id = worker_info.id
      self.num_workers = worker_info.num_workers

    for idx, ds in enumerate(self.datasources):
      # check that all ranges have a length of at least num_workers
      if range_size(self.ds_ranges[idx]) < self.num_workers:
        raise ValueError(f"After rank partitioning, Datasource {idx} has less samples than num workers ({len(ds)} < {self.num_workers})")
      
      self.ds_ranges[idx] = shard_group_range(self.ds_ranges[idx], self.worker_id, self.num_workers)

    # initialize cursors to their starting values
    for idx, ds in enumerate(self.datasources):
      self.ds_cursors[idx] = self.ds_ranges[idx][0]

    running_datasources = list(range(len(self.datasources)))

    current_buffer  = make_empty_sample()

    # start looping
    while True:
      if not self.has_any_data(running_datasources):
        if len(current_buffer) > 0 :
          if not self.dynamic_seq_len:
            current_buffer.inplace_fill_(self.max_length, self.tokenizer.eos_token)
            current_buffer, _ = current_buffer.truncate(self.max_length)

          current_buffer.generate_positional_ids()
          yield (current_buffer, self.sample_cnt)
        break

      while len(current_buffer) < self.max_length:
        if len(running_datasources) == 0 and len(self.sample_queue) == 0:
          break
        
        if len(self.sample_queue) == 0:
          running_ds_weights: List[float] = [self.datasources_weights[idx] for idx in running_datasources]
          sampled_ds_idx: int = random.choices(running_datasources, weights=running_ds_weights, k=1)[0]
          
          sampled_ds = self.datasources[sampled_ds_idx]
          sampled_ds_cursor = self.ds_cursors[sampled_ds_idx]
          sampled_ds_range = self.ds_ranges[sampled_ds_idx]
          self.sample_cnt[sampled_ds_idx] += 1

          # gets next cursor position
          # when the cursor passes the end of the range
          # the datasource is removed from the running list
          next_cursor = self.increment_cursor(sampled_ds_cursor, sampled_ds_range)

          if self.cursor_reached_end_of_range(next_cursor, sampled_ds_range):
            running_datasources.remove(sampled_ds_idx)

          self.ds_cursors[sampled_ds_idx] = next_cursor
      
          # we re-combine the current sample in the current buffer
          # when the current_buffer doesn't overflow, our next buffer will be a blank buffer
          # when the current buffer overflows, our next buffer might be a remaineder of the current sample
          sample = sampled_ds[sampled_ds_cursor]
          sample.inplace_datasource_mask_(sampled_ds_idx)
        else:
          sample = self.sample_queue.popleft()

        
        current_buffer, next_buffer = self.add_sample_to_buffer(current_buffer, sample)

        if next_buffer == sample and len(next_buffer) != 0:
          self.sample_queue.append(next_buffer)
          next_buffer = make_empty_sample()
        
        # if the current buffer is saturated, which might happen due to max size
        # or due to having dynamic seq_lens, we yield the current buffer
        if self.buffer_is_saturated(current_buffer):
          if not self.dynamic_seq_len:
            # we cannot guarantee there was no overflow prior to this point
            current_buffer.inplace_fill_(self.max_length, self.tokenizer)
            current_buffer, _ = current_buffer.truncate(self.max_length)

          current_buffer.generate_positional_ids()
          assert len(current_buffer) <= self.max_length
          assert len(current_buffer) != 0

          yield (current_buffer, self.sample_cnt)
          current_buffer = make_empty_sample()
        
  def __len__(self):
    total_len = 0
    for idx, ds in enumerate(self.datasources):
      total_len += len(ds)

    return total_len
  
  @staticmethod
  def get_collate_fn() -> callable:
    return collate_fn


















