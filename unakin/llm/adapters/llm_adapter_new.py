from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
import yaml
import tqdm
import xformers.ops as xops

from unakin.llm.models.llama.model_compile import (
  Transformer,
  CODE_LLAMA_7B_WEIGHTS,
  CODE_LLAMA_13B_WEIGHTS,
  TransformerBlock,
  Attention,
  FeedForward,
  RMSNorm
)

from unakin.llm.adapters.fused_lora import FusedLoraFn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
  CPUOffload,
  BackwardPrefetch,
  ShardingStrategy,
  MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
  CustomPolicy,
  transformer_auto_wrap_policy,
  lambda_auto_wrap_policy,
  _or_policy
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper)

import torch.distributed
from unakin.llm.adapters.triton_ce import Fast_CrossEntropyLoss, fast_cross_entropy_loss

class CrossEntropyModule(nn.Module):
  def __init__(self) -> None:
    super().__init__()
  
  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return fast_cross_entropy_loss(x, y)

@dataclass
class ModuleFilterConfig:
  layer_classes: Optional[List[str]] = None
  layer_names: Optional[List[str]] = None
  min_param_size: Optional[int] = None

class ModuleFilter:
  def __init__(self, config: ModuleFilterConfig) -> None:
    self.config = config

  def __call__(self, module: nn.Module, name: Optional[str]) -> bool:
    acc = True

    # name rule
    if self.config.layer_names is not None:
      acc = acc and (any(name.endswith(layer_name) for layer_name in self.config.layer_names))

    # class rule
    if self.config.layer_classes is not None:
      module_class = module.__class__.__name__
      acc = acc and (module_class in self.config.layer_classes)

    # param size rule
    if self.config.min_param_size is not None:
      param_size = sum(p.numel() for p in module.parameters())
      acc = acc and (param_size >= self.config.min_param_size)

    return acc

  def match_modules(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
    results = []
    for name, module in model.named_modules():
      if self(module, name):
        results.append((module, name))
    return results  
    
  @classmethod
  def from_yaml(cls, path: str) -> "ModuleFilter":
    with open(path, "r") as f:
      config = yaml.safe_load(f)

    return cls(config)

class AdapterLayer(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__()

  @abstractmethod
  def fuse(self, module: nn.Module) -> nn.Module:
    raise NotImplementedError

@dataclass
class LoraConfig:
  rank: int
  dropout: float
  alpha: float
  init_type: str = "normal"

def row_wise_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
  if not training:
    return x * (1 - p)
  
  # make sure that x.shape[0] * p is an integer
  assert x.shape[0] * p % 1 == 0, "p should be a multiple of x.shape[0]"
  n_rows = x.shape[0]
  n_rows_to_drop = int(n_rows * p)
  rows_to_drop = torch.randperm(n_rows)[:n_rows_to_drop]
  mask = torch.ones((n_rows, 1), device=x.device, dtype=torch.bfloat16)
  mask[rows_to_drop] = 0
  return x * mask

class LowRankFactor(nn.Module):
  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    rank: int,
    dropout: float,
    alpha: float,
    init_type: str = "normal",
    dropout_type: str = "rowwise",
    dtype: torch.dtype = torch.float32
  ) -> None:
    super().__init__()
    
    assert init_type in ["normal", "orthogonal"]
    assert dropout_type in ["rowwise", "standard"]

    self.A = nn.Parameter(torch.zeros(rank, in_dim, dtype=dtype))
    self.B = nn.Parameter(torch.randn(out_dim, rank, dtype=dtype))
    self.dropout_func = row_wise_dropout if dropout_type == "rowwise" else F.dropout
    self.dropout = dropout
    
    self.alpha = alpha / rank
    self.dropout_type = dropout_type
    self.init_type = init_type

  def reset_parameters(self):
    if self.init_type == "orthogonal":
      pass
    else:  
      self.A.data.zero_()
      self.B.data.normal_(0, 0.02)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.dropout_type == "rowwise":
      A     = self.dropout_func(self.A, p=self.dropout, training=self.training)
      delta = self.B @ A
    else:
      delta = self.dropout_func(self.B @ self.A, p=self.dropout, training=self.training)

    return delta * self.alpha
  
  def __repr__(self):
    return f"LowRankFactor(A={self.A.shape}, B={self.B.shape}, dropout={self.dropout.p}, alpha={self.alpha})"

# class LoraLayer(AdapterLayer):

#   ADAPTER = True

#   def __init__(self, base_linear: nn.Linear, low_rank: LowRankFactor) -> None:
#     super().__init__()
#     self.weight = base_linear.weight
#     self.low_rank = low_rank

#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     # custom function just prevents FSDP and autograd saving intermediate values
#     # which are not needed for the entire computation or distributed training
#     return FusedLoraFn.apply(
#       x,
#       self.weight.data.T,
#       self.low_rank.A.data.T,
#       self.low_rank.B.data.T,
#       self.low_rank.alpha,
#       self.low_rank.dropout.p,
#       self.training
#     )

#   def fuse(self, module: 'LoraLayer') -> nn.Module:
#     # dummy forward
#     module.low_rank.eval()
#     delta = module.forward(torch.zeros(1, 1))
#     delta_w = module.weight + delta
#     with torch.device("meta"):
#       new_linear = nn.Linear(module.weight.shape[1], module.weight.shape[0], bias=False)
#     new_linear = new_linear.to_empty(device=module.weight.device)
#     new_linear.weight.copy_(delta_w)
#     return new_linear
  
#   def __repr__(self):
#     return f"LoraLayer(low_rank={self.low_rank})"
  
#   def reset_parameters(self):
#     self.low_rank.A.data.zero_()
#     self.low_rank.B.data.normal_(0, 0.02)
#     self.weight.data.normal_(0, 0.02)

class LoraLayer(AdapterLayer):

  ADAPTER = True

  def __init__(self, base_linear: nn.Linear, low_rank: LowRankFactor) -> None:
    super().__init__()
    # self.weight = base_linear.weight
    self.weight = nn.Parameter(torch.empty_like(base_linear.weight).copy_(base_linear.weight))
    self.weight.requires_grad_(False)
    self.low_rank = low_rank

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    delta = self.low_rank(x)
    delta_w = self.weight + delta
    return F.linear(x, delta_w, bias=None)

  def fuse(self, module: 'LoraLayer') -> nn.Module:
    # dummy forward
    device = module.weight.device
    delta = module.low_rank.forward(torch.zeros(1, 1, device=device))
    delta_w = module.weight + delta
    with torch.device("meta"):
      new_linear = nn.Linear(module.weight.shape[1], module.weight.shape[0], bias=False,
                             device=device)
    new_linear.weight.copy_(delta_w)
    return new_linear
  
  def __repr__(self):
    return f"LoraLayer(low_rank={self.low_rank})"

@dataclass
class NeftuneConfig:
  noise_dist: str
  alpha: float

class NeftuneLayer(AdapterLayer):

  ADAPTER = True

  __noise_dist_funcs__ = {
    "normal": torch.randn,
    "uniform": torch.rand,
  }

  def __init__(self, base_embedding: nn.Embedding, alpha: float, noise_dist: str) -> None:
    super().__init__()
    self.base_embedding = base_embedding
    self._norm_factor   = base_embedding.weight.shape[1] ** 0.5
    self._norm_factor  *= alpha
    self.noise_dist     = noise_dist

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    embedding = self.base_embedding(x)
    if self.training:
      noise_fn  = self.__noise_dist_funcs__[self.noise_dist]
      noise     = noise_fn(embedding.shape, device=embedding.device)
      embedding = embedding + noise / self._norm_factor
    
    # inference is business as usual
    return embedding

  def fuse(self, module: 'NeftuneLayer') -> nn.Module:
    return module.base_embedding

class AdaptationPolicy:
  def __init__(self, *args, **kwargs) -> None:
    pass

  @abstractmethod
  def _adapt_layer(self, module: nn.Module, name: Optional[str]) -> AdapterLayer:
    raise NotImplementedError
  
  def _recurse_adapt(self, parent_module: nn.Module) -> List:
    results = []
    for child_name, child_module in parent_module.named_children():
      if len(list(child_module.children())) > 0:
        children_results = self._recurse_adapt(child_module)
        results.extend(children_results)
      # this shouldn't be an issue, but just in case the filter policy
      # does not have the best implementation
      else:
        # currently we don't support adapting the same module twice
        if self.target_filter(child_module, child_name):
          # TODO: this is just lazy, subclass everything at somepoint
          if hasattr(child_module, "ADAPTER"):
            raise ValueError(
              f"Module {child_name} has already been adapted into a new module of type: {type(child_module)}. \
                Pleasue double check your configuration values.")
          new_module = self._adapt_layer(child_module, child_name)
          results.append((parent_module, child_name, new_module))
    return results

  def adapt(self, model: nn.Module) -> nn.Module:
    targets = self._recurse_adapt(model)
    for (parent, child, module) in targets:
      # TODO: maybe this is stupid for unfreezing purposes
      if module is not None:
        setattr(parent, child, module)
    
    return model

class LoraPolicy(AdaptationPolicy):
  def __init__(self, config: LoraConfig, target_filter: ModuleFilter) -> None:
    self.config = config
    self.target_filter = target_filter

  def _adapt_layer(self, module: nn.Linear, name: Optional[str]) -> AdapterLayer:
    in_features, out_features = module.in_features, module.out_features
    low_rank_module = LowRankFactor(
      in_features, out_features, self.config.rank, self.config.dropout, self.config.alpha,
      dtype=module.weight.dtype
    )
    return LoraLayer(module, low_rank_module)

class NefTunePolicy(AdaptationPolicy):
  def __init__(self, config: NeftuneConfig, target_filter: ModuleFilter) -> None:
    self.noise_dist = config.noise_dist
    self.alpha = config.alpha
    self.target_filter = target_filter

  def _adapt_layer(self, module: nn.Embedding, name: Optional[str]) -> AdapterLayer:
    return NeftuneLayer(module, self.alpha, self.noise_dist)
  
class LoraPlusPolicy(AdaptationPolicy):
  def __init__(self, config: LoraConfig, target_filter: ModuleFilter) -> None:
    self.config = config
    self.target_filter = target_filter

  def _adapt_layer(self, module: nn.Linear, name: Optional[str]) -> AdapterLayer:
    in_features, out_features = module.in_features, module.out_features
    low_rank_module = LowRankFactor(
      in_features, out_features, self.config.rank, self.config.dropout, self.config.alpha,
      dtype=module.weight.dtype
    )

    # freeze the random projector
    low_rank_module.B.requires_grad_(False)
    return LoraLayer(module, low_rank_module)
  
class UnfreezeConfig:
  def __init__(self, *args, **kwargs) -> None:
    pass

class UnfreezePolicy(AdaptationPolicy):
  def __init__(self, config: UnfreezeConfig, target_filter: ModuleFilter) -> None:
    self.target_filter = target_filter

  def _adapt_layer(self, module: nn.Module, name: Optional[str]) -> nn.Module:
    module.requires_grad_(True)
    return None
  
  def fuse(self, module: nn.Module) -> nn.Module:
    return module
  
def fuse_adaptations(model: nn.Module) -> nn.Module:
  # recursively fuse the adaptations
  for name, child in model.named_children():
    if getattr(child, "ADAPTER", False):
      setattr(model, name, child.fuse(child))
    else:
      fuse_adaptations(child)
  return model

def setup_dist():
  """Setup distributed process group"""
  torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
  )

def dist_print_rank_0(*args, **kwargs):
  if torch.distributed.get_rank() == 0:
    print(*args, **kwargs)

def get_wrapping_policy(custom: bool = True):
  if custom:
    def lambda_policy_fn(module: nn.Module) -> bool:
      return isinstance(module, LowRankFactor)
  else:
    def lambda_policy_fn(module: nn.Module) -> bool:
      return (
        len(list(module.children())) == 0
        and getattr(module, 'weight', None) is not None
        and module.weight.requires_grad
      )
  
  def attn_policy_fn(module: nn.Module) -> bool:
    return isinstance(module, Attention)
  
  def ffn_policy_fn(module: nn.Module) -> bool:
    return isinstance(module, FeedForward)
  
  lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
  attn_policy = partial(lambda_auto_wrap_policy, lambda_fn=attn_policy_fn)
  ffn_policy = partial(lambda_auto_wrap_policy, lambda_fn=ffn_policy_fn)
  transformer_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
  return partial(_or_policy, policies=[lambda_policy, transformer_policy, attn_policy, ffn_policy])

if __name__ == "__main__":

  setup_dist()
  torch.cuda.set_device(torch.distributed.get_rank())

  transformer = Transformer.from_base_model(
    model_weights=CODE_LLAMA_7B_WEIGHTS,
    base_path="/mnt/raid0/llamas/llama_base",
  )

  adapt_module_filter = ModuleFilter(
    ModuleFilterConfig(
      layer_names=["w1", "w2", "w3", "wq", "wk", "wo", "wv"],
      layer_classes=["Linear"],
    )
  )

  lora_policy = LoraPolicy(
    LoraConfig(rank=256, dropout=0.1, alpha=0.1),
    adapt_module_filter,
  )

  linear_shard_module_filter = ModuleFilter(
    ModuleFilterConfig(
      layer_names=["base_linear"],
      layer_classes=["Linear"]
    )
  )

  lowrank_shard_module_filter = ModuleFilter(
    ModuleFilterConfig(
      layer_classes=["LowRankFactor"]
    )
  )

  
  transformer = lora_policy.adapt(transformer)

  cpu_offload = CPUOffload(offload_params=True)
  precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
    keep_low_precision_grads=True,
  )


  fsdp_model = FSDP(
    transformer.to(torch.bfloat16),
    auto_wrap_policy=get_wrapping_policy(custom=True),
    device_id=torch.distributed.get_rank(),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=precision,
    forward_prefetch=True,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    use_orig_params=False,
  )

  objective = CrossEntropyModule()

  wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})

  apply_activation_checkpointing(
    fsdp_model, auto_wrap_policy=wrap_policy,
    checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT))
  
  dist_print_rank_0(wrap_policy)


  # dist_print_rank_0(fsdp_model)f

  exit()

  optimizer = torch.optim.AdamW(
    params=list(p for p in fsdp_model.parameters() if p.requires_grad),
    lr=1e-4,
  )
   
  dist_print_rank_0(fsdp_model)

  MULT = 1
  SEQ_MULT = MULT * 1

  inputs = [
    (
      torch.randint(0, 100, (1, 8192 * MULT)).cpu(),
      torch.randint(0, 8192, (1, 8192 * MULT)).cpu(),
      torch.arange(8192 * MULT).cpu()
    )
    for _ in range(1_000)
  ]

  mask = xops.fmha.BlockDiagonalMask.from_seqlens([2048 * MULT * 4]).make_causal()

  for idx, inp in enumerate(tqdm.tqdm(inputs, total=len(inputs))):
    tokens, gt, positions = inp
    tokens, gt, positions = tokens.cuda(), gt.cuda(), positions.cuda()
    outs = fsdp_model(tokens, positions, mask=mask)
    dist_print_rank_0(outs.shape, outs.dtype) 
    loss = objective(outs, gt)
    torch.cuda.synchronize()
    loss.backward()
    # F.cross_entropy(outs.squeeze(0), gt.view(-1), reduction="mean").backward()
    if (idx + 1) % 2 == 0:
      optimizer.step()
      optimizer.zero_grad()