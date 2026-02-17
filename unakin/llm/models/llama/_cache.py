import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass
from xformers.ops.fmha.attn_bias import (
  AttentionBias,
  BlockDiagonalCausalMask,
  BlockDiagonalCausalWithOffsetPaddedKeysMask
)

def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
  assert len(l1) == len(l2)
  return [
    v for pair in zip(l1, l2)
    for v in pair
  ]

@dataclass
class FixedSizeMetadata:
  max_capacity: int
  positions: torch.Tensor
  mask: AttentionBias

  @classmethod
  def from_capacity(cls, capacity: int, device: Optional[torch.device] = None):
    if device is None:
      device = torch.device("cpu")

    seq_lens = [1 for _ in range(capacity)]
    positions = torch.arange(capacity, dtype=torch.long, device=device)
    mask = BlockDiagonalCausalMask.from_seqlens(q_seqlen=seq_lens)
    
    mask.q_seqinfo.to(device=device)
    mask.k_seqinfo.to(device=device)

    mask.q_seqinfo.min_seqlen = 1
    mask.q_seqinfo.max_seqlen = capacity * 2

    return cls(
      max_capacity=capacity,
      positions=positions,
      mask=mask,
    )
  
  def set_seqlens(self, seqlens: List[int]):
    
    positions = torch.cat([
      torch.arange(
        seqlen, dtype=torch.long, device=self.positions.device)
      for seqlen in seqlens if seqlen > 0
    ])

    new_seqstart = torch.zeros_like(self.mask.q_seqinfo.seqstart)
    len_seqstart = len(self.mask.q_seqinfo.seqstart)
  
    seqlen_padded = seqlens + [0] * (len_seqstart - len(seqlens) - 1)  
    cum_sum_seqlens = torch.cumsum(torch.tensor(seqlens, dtype=torch.long), dim=0)
    
    seqlen_padded = seqlens + [0] * (len_seqstart - len(seqlens) - 1)
    cum_sum_seqlens = torch.cumsum(torch.tensor(seqlen_padded, dtype=torch.long), dim=0)

    new_seqstart[1:len(seqlen_padded)+1] = cum_sum_seqlens
    new_seqstart_py = new_seqstart.tolist()   

    self.mask.q_seqinfo.seqstart.copy_(new_seqstart)
    self.mask.q_seqinfo.seqstart_py = new_seqstart_py
    self.mask.q_seqinfo.seqlens_py = seqlens
    
    self.positions.copy_(positions)

@dataclass
class SimpleMetadda:
  seqlens: List[int]
  positions: torch.Tensor
  mask: AttentionBias
  
  @classmethod
  def from_seqlens(cls, seqlens: List[int], device: Optional[torch.device] = None):
    if device is None:
      device = torch.device("cpu")

    positions = torch.cat([
      torch.arange(seqlen, dtype=torch.long, device=device)
      for seqlen in seqlens
    ])

    mask = BlockDiagonalCausalMask.from_seqlens(q_seqlen=seqlens)
    
    return cls(seqlens=seqlens, positions=positions, mask=mask)

  def to(self, device: torch.device):
    self.positions = self.positions.to(device=device)
    return self
  
  def cuda(self):
    return self.to(device=torch.device("cuda"))

@dataclass
class CacheMetadata:
  positions: torch.Tensor
  cache_mask: torch.Tensor
  cache_numel: torch.Tensor
  cache_positions: torch.Tensor

  seqlens: List[int]
  prefill: bool
  mask: AttentionBias

class CacheView:
  def __init__(
    self,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    metadata: CacheMetadata,
    kv_seqlens: torch.Tensor,
  ) -> None:
    self.cache_k = cache_k
    self.cache_v = cache_v
    self.metadata = metadata
    self.kv_seqlens = kv_seqlens

  def update(self, keys: torch.Tensor, values: torch.Tensor):
    n_kv_heads, head_dim = keys.shape[-2:]
    
    flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
    flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

    # keys and values might come from upstream
    # with different batch sizes
    keys = keys.view(-1, n_kv_heads, head_dim)
    values = values.view(-1, n_kv_heads, head_dim)

    flat_cache_k.index_copy_(
      0,                              # index dim
      self.metadata.cache_positions,  # src indices
      keys[self.metadata.cache_mask]  # dst indices
    )

    flat_cache_v.index_copy_(
      0,
      self.metadata.cache_positions,
      values[self.metadata.cache_mask]
    )

  @property
  def keys(self):
    print(f"kv_seqlens: {self.kv_seqlens} len: {len(self.kv_seqlens)} cache_k: {self.cache_k.shape}")
    return self.cache_k[:len(self.kv_seqlens)]
  
  @property
  def values(self):
    print(f"kv_seqlens: {self.kv_seqlens} len: {len(self.kv_seqlens)} cache_v: {self.cache_v.shape}")
    return self.cache_v[:len(self.kv_seqlens)]

  def interleave_kv(self, keys: torch.Tensor, values: torch.Tensor):
    if keys.ndim == values.ndim == 4:
      keys = keys.view(-1, *keys.shape[-2:])
      values = values.view(-1, *values.shape[-2:])
    
    assert keys.ndim == values.ndim == 3
    assert keys.shape == values.shape

    if all([s == 0 for s in self.metadata.seqlens]):
      return keys, values

    # TODO: is this really needed to be performed this way?
    keys = torch.split(keys, self.metadata.seqlens)
    values = torch.split(values, self.metadata.seqlens)
    assert len(keys) == len(values) == len(self.metadata.seqlens)

    # TODO: this is equally inefficient, best way would be to have a gigantic tensor
    # with which we do a single index select
    cache_keys = [cache[:s] for cache, s in zip(self.cache_k, self.kv_seqlens)]
    cache_values = [cache[:s] for cache, s in zip(self.cache_v, self.kv_seqlens)]

    for idx, (k, v) in enumerate(zip(cache_keys, cache_values)):
      assert k.shape == v.shape, f"cache_keys: {k.shape} cache_values: {v.shape} at index {idx}"

    interleaved_keys = interleave_list(cache_keys, keys)
    interleaved_values = interleave_list(cache_values, values)

    keys = torch.cat(interleaved_keys, dim=0)
    values = torch.cat(interleaved_values, dim=0)
    return keys, values

class BufferCache:
  def __init__(
      self,
      n_layers: int,
      max_batch_size: int,
      max_seq_len: int,
      n_kv_heads: int,
      head_dim: int,
  ) -> None:
    
    # hyperparameters
    self.n_layers = n_layers
    self.max_batch_size = max_batch_size
    self.max_seq_len = max_seq_len
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim

    # buffers
    self.cache_k = torch.empty(
      size=(
        n_layers,
        max_batch_size,
        max_seq_len,
        n_kv_heads,
        head_dim,
      )
    )

    self.cache_v = torch.empty(
      size=(
        n_layers,
        max_batch_size,
        max_seq_len,
        n_kv_heads,
        head_dim,
      )
    )

    # kv_seqlens
    self.kv_seqlens = None

  def get_view(self, layer_idx: int, metadata: CacheMetadata):
    print(f"view kv_seqlens: {self.kv_seqlens}")
    return CacheView(
      cache_k=self.cache_k[layer_idx],
      cache_v=self.cache_v[layer_idx],
      metadata=metadata,
      kv_seqlens=self.kv_seqlens,
    )

  def init_kv_seqlens(self, batch_size: int):
    self.kv_seqlens = torch.zeros(
      size=(batch_size,),
      dtype=torch.long,
      device=self.device,
    )

  def update_seqlens(self, seqlens: List[int]):
    if self.kv_seqlens is None:
      self.init_kv_seqlens(batch_size=len(seqlens))
    
    assert self.kv_seqlens is not None, "kv_seqlens must be initialized first"
    assert len(seqlens) == len(self.kv_seqlens), "seqlens must match batch size"
    self.kv_seqlens += torch.tensor(data=seqlens, dtype=torch.long, device=self.device,)

  def get_input_metadata(self, seqlens: List[int]):
    if self.kv_seqlens is None:
      self.init_kv_seqlens(batch_size=len(seqlens))

    seqpos = self.kv_seqlens.tolist()
    seqmasks = [
      [True for _ in range(seqlen)]
      for seqlen in seqlens
    ]

    # important for static cahce
    assert all([pos+seqlen <= self.max_seq_len for pos, seqlen in zip(seqpos, seqlens)]), \
      "Exceeded maximum sequence length"

    cache_masks = torch.tensor(sum(seqmasks, []), dtype=torch.bool, device=self.device)
    cache_numel = torch.tensor([sum(mask) for mask in seqmasks], dtype=torch.long, device=self.device)
    positions = torch.cat([
      torch.arange(pos, pos+seqlen, dtype=torch.long, device=self.device)
      for pos, seqlen in zip(seqpos, seqlens)
    ])

    batch_idx = torch.tensor(
      sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
      device=self.device, dtype=torch.long,
    )

    cache_positions = positions + batch_idx * self.max_seq_len
    
    first_prefill = seqpos[0] == 0
    subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)

    if first_prefill:
      attn_mask = BlockDiagonalCausalMask.from_seqlens(
        q_seqlen=seqlens
      )
    else:
      attn_mask = BlockDiagonalCausalMask.from_seqlens(
        q_seqlen=seqlens,
        kv_seqlen=(self.kv_seqlens + cache_numel).tolist()
      ).make_causal_from_bottomright()
      # attn_mask = None

    return CacheMetadata(
      positions=positions,
      cache_mask=cache_masks,
      cache_numel=cache_numel,
      cache_positions=cache_positions[cache_masks],
      seqlens=seqlens,
      prefill=first_prefill or subsequent_prefill,
      mask=attn_mask,
    )
  
  def reset(self):
    self.cache_k.fill_(0)
    self.cache_v.fill_(0)
    if self.kv_seqlens is not None:
      self.kv_seqlens = None

  @property
  def device(self):
    return self.cache_k.device
  
  def to(self, device: torch.device, dtype: torch.dtype):
    self.cache_k = self.cache_k.to(device=device, dtype=dtype)
    self.cache_v = self.cache_v.to(device=device, dtype=dtype)

  def get_cache_view(self, layer_idx: int, metadata: CacheMetadata):
    return CacheView(
      cache_k=self.cache_k[layer_idx],
      cache_v=self.cache_v[layer_idx],
      kv_seqlens=self.kv_seqlens,
      metadata=metadata,
    )    