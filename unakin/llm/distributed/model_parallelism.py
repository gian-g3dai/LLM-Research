from enum import Enum
from typing import Dict, List, NamedTuple, Optional
from torch import distributed as dist
from torch import nn
import torch

from unakin.llm.distributed.utils import get_dist_info
from unakin.llm.models.llama.model_compile import FeedForward, Attention, Transformer, TransformerBlock
from torch.distributed import _functional_collectives as funcol
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

  
LLAMA_FFN_TP_PLAN = {
  "w1": ColwiseParallel(),
  "w2": RowwiseParallel(),
  "w3": ColwiseParallel()
}

LLAMA_ATTN_TP_PLAN = {
  "wq": ColwiseParallel(),
  "wk": ColwiseParallel(),
  "wv": ColwiseParallel(),
  "wo": RowwiseParallel()
}

def _parallelize_ffn(ffn: FeedForward, mesh: DeviceMesh, plan: Dict = LLAMA_FFN_TP_PLAN):
  return parallelize_module(module=ffn, device_mesh=mesh, parallelize_plan=plan)

def _parallelize_attn(attn: Attention, mesh: DeviceMesh, plan: Dict = LLAMA_ATTN_TP_PLAN):
  attn.n_local_heads       = attn.n_local_heads // mesh.size()
  attn.n_local_kv_heads = attn.n_local_kv_heads // mesh.size()
  return parallelize_module(module=attn, device_mesh=mesh, parallelize_plan=plan)


def apply_tensor_parallelism_to_model(
  model: Transformer,
  ffn_directions: Dict = LLAMA_FFN_TP_PLAN,
  attn_directions: Dict = LLAMA_ATTN_TP_PLAN
):
  rank, world_size = get_dist_info()
  # TODO: why not worldsize
  mesh = init_device_mesh("cuda", (world_size,))
  assert world_size >= 1, "Tensor parallelism requires more than one process"
  # model.n_kv_heads     = model.n_kv_heads // world_size # this is n_kv_heads
  # model.n_heads        = model.n_heads // world_size # this is n_heads
  # model.dim            = model.dim // world_size
  for block in model.layers:
    block: TransformerBlock
    _ = _parallelize_ffn(block.feed_forward, mesh, ffn_directions)
    _ = _parallelize_attn(block.attention, mesh, attn_directions)

  model.n_heads = model.n_heads // mesh.size()
  model.n_kv_heads = model.n_kv_heads // mesh.size()
  model.dim = model.dim // mesh.size()
  
  return model
  