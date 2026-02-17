from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from unakin.llm.base import BaseLLM

import torch
import torch.nn.functional as F
from torch import nn

import torch.distributed
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp

class Adapter:
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def activate(self, *args, **kwargs):
    raise NotImplementedError()

  def deactivate(self, *args, **kwargs):
    raise NotImplementedError()

  def trainable_parameters(self):
    raise NotImplementedError()

  def remove_adaptation(self) -> nn.Module | None:
    raise NotImplementedError()


class AdaptedLinearLayer(nn.Module):
  def __init__(self, *args, m: Union['AdaptedLinearLayer', nn.Linear], **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.adapted_linear = m

  def forward(self, x: torch.Tensor):
    raise NotImplementedError()

  def trainable_parameters(self):
    raise NotImplementedError()

  @property
  def in_features(self):
    return self.adapted_linear.in_features

  @property
  def out_features(self):
    return self.adapted_linear.out_features

  @property
  def weight(self):
    return self.adapted_linear.weight

  @property
  def bias(self):
    return self.adapted_linear.bias


@dataclass
class BiasScaleAdaptation:
  layer_names: Tuple[str, ...]
  bias: bool = False
  scale: bool = False

  def is_adaptation_target(self, name: str, m: nn.Module) -> bool:
    if not isinstance(m, (nn.Linear, BiasScaleAdapterLinearLayer)):
      return False

    if not any(name.endswith(layer_name) for layer_name in self.layer_names):
      return False

    return True

  def adapt_layer(self, name: str, m: nn.Module) ->  \
          Union['BiasScaleAdapterLinearLayer', nn.Module]:

    return BiasScaleAdapterLinearLayer(m=m, params=self)


class BiasScaleAdapterLinearLayer(AdaptedLinearLayer, Adapter):

  def __init__(self, *args, m: Union[AdaptedLinearLayer, nn.Linear],
               params: BiasScaleAdaptation, **kwargs) -> None:
    AdaptedLinearLayer.__init__(self, *args, m=m, **kwargs)
    Adapter.__init__(self, *args, **kwargs)
    self.params = params

    if self.params.bias:
      self.additive_bias = nn.Parameter(torch.zeros(m.out_features))

    if self.params.scale:
      self.scale = nn.Parameter(torch.ones(m.out_features))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.adapted_linear(x)

    if self.params.bias:
      x = x + self.additive_bias

    if self.params.scale:
      x = x * self.scale

    return x

  def reset_parameters(self):
    if self.params.bias:
      nn.init.zeros_(self.additive_bias)
    if self.params.scale:
      nn.init.ones_(self.scale)

  def trainable_parameters(self):
    params = []
    if self.params.bias:
      params.append(self.additive_bias)
    if self.params.scale:
      params.append(self.scale)
    return params


@dataclass
class LoraAdaptation:
  rank: int
  alpha: float
  layer_names: Tuple[str, ...]
  dropout: float = 0.0
  split_ops: bool = True
  use_fourrier_features: bool = False

  def is_adaptation_target(self, name: str, m: nn.Module) -> bool:
    if not isinstance(m, (AdaptedLinearLayer, nn.Linear)):
      return False

    if not any(name.endswith(layer_name) for layer_name in self.layer_names):
      return False

    in_features = m.in_features
    out_features = m.out_features

    # can't adapt if rank is larger than the input or output dimension
    if in_features < self.rank or out_features < self.rank:
      return False

    return True

  def adapt_layer(self, name: str, m: nn.Module) -> \
          Union['LoraAdapterLinearLayer', nn.Module]:
    return LoraAdapterLinearLayer(m=m, params=self)


class FourierFeatureEncoder(nn.Module):
  def __init__(self, input_size, num_features, scale=1.0):
    super(FourierFeatureEncoder, self).__init__()
    scale = torch.randn(input_size, num_features) * scale
    self.register_buffer("scale", scale)
      
  def forward(self, x):
    rff_sin = torch.sin((2 * torch.pi * x) @ self.scale)
    rff_cos = torch.cos((2 * torch.pi * x) @ self.scale)
    rff = torch.cat([rff_sin, rff_cos], dim=-1)
    return rff

class LoraAdapterLinearLayer(AdaptedLinearLayer, Adapter):
  def __init__(self, *args, m: Union[AdaptedLinearLayer, nn.Linear],
               params: LoraAdaptation, **kwargs) -> None:
    AdaptedLinearLayer.__init__(self, *args, m=m, **kwargs)
    Adapter.__init__(self, *args, **kwargs)

    self.params = params
    self.B = nn.Parameter(torch.randn(self.out_features, self.params.rank))
    self.A = nn.Parameter(torch.zeros(self.params.rank, self.in_features))

    self._mult = params.alpha / params.rank
    self._active = True

    if self.params.use_fourrier_features:
      self._fourrier_features = self.params.rank // 2
      self._fourrier_encoder = FourierFeatureEncoder(
        input_size=self.params.rank,
        num_features=self._fourrier_features,
        scale=1.0
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._active:
      # branch for fourrier features
      if self.params.use_fourrier_features:
        wb = self.adapted_linear.weight @ self.B
        wb = self._fourrier_encoder(wb)
        w_hat = wb @ self.A
      else:
        w_hat = self.B @ self.A

      # common branch
      w_hat = F.dropout(w_hat, p=self.params.dropout, training=self.training) * self._mult
      dw = self.adapted_linear.weight - w_hat
      h = F.linear(x, weight=dw)
      return h
    
    else:
      h = self.adapted_linear(x)
      return h
    
  @torch.no_grad()
  def partial_merge(self, decay: float = 0.1):
    w_hat = F.dropout(self.B @ self.A, p=self.params.dropout, training=False) 
    self.adapted_linear.weight.data -= w_hat * decay * self._mult
    self.reset_parameters()
  

  def reset_parameters(self):
    # fan_in / fan_out
    nn.init.xavier_normal_(self.B)    
    nn.init.zeros_(self.A)

  def merge_adaptation(self) -> None:
    self._active = False
    dw = F.dropout(self.B @ self.A, p=self.params.dropout, training=False)
    self.adapted_linear.weight.data -= dw * self._mult

  def trainable_parameters(self):
    return [self.A, self.B]


@dataclass
class NormUnfreezeAdaptation:
  layer_names: Tuple[str, ...]

  def is_adaptation_target(self, name: str, m: nn.Module) -> bool:
    if not any(name.endswith(layer_name) for layer_name in self.layer_names):
      return False

    return True

  def adapt_layer(self, name: str, m: nn.Module) -> None:
    m.weight.requires_grad = True

@dataclass
class LMHeadUnfreezeAdaptation:
  layer_names: Tuple[str, ...]
  dropout: float = 0.0
  
  def is_adaptation_target(self, name: str, m: nn.Module) -> bool:
    if not any(name.endswith(layer_name) for layer_name in self.layer_names):
      return False

    if not isinstance(m, nn.Linear):
      return False

    return True

  def adapt_layer(self, name: str, m: nn.Module) -> None:
    m.weight.requires_grad = True
    return DropoutAdaptedLinearLayer(m=m, params=self)

class DropoutAdaptedLinearLayer(AdaptedLinearLayer, Adapter):
  def __init__(self, *args, m: Union[AdaptedLinearLayer, nn.Linear],
               params: LMHeadUnfreezeAdaptation, **kwargs) -> None:
    AdaptedLinearLayer.__init__(self, *args, m=m, **kwargs)
    Adapter.__init__(self, *args, **kwargs)

    self.params = params
    self._active = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.dropout(x, p=self.params.dropout, training=self.training)
    return self.adapted_linear(x)

  def reset_parameters(self):
    pass

  def merge_adaptation(self) -> None:
    self._active = False

  def trainable_parameters(self):
    return [self.adapted_linear.weight]

@dataclass
class SoftPromptAdaptation:
  n_layers: int
  n_tokens: int


@dataclass
class RopeContextAdaptation:
  base_ctx_len: int
  expand_ctx_len: int

@dataclass
class AdaptationConfig:
  lora: Optional[LoraAdaptation] = None
  norm: Optional[NormUnfreezeAdaptation] = None
  bias_scale: Optional[BiasScaleAdaptation] = None
  soft_prompt: Optional[SoftPromptAdaptation] = None
  rope_context: Optional[RopeContextAdaptation] = None
  lm_head: Optional[LMHeadUnfreezeAdaptation] = None

  def requires_adaptation(self) -> bool:
    return any([self.lora, self.norm, self.bias_scale, self.soft_prompt])

# m -> BaseLLM 
# LlmAdapter -> BaseLLM
# add abstract methods to LlmAdapter

class LlmAdapter(BaseLLM):
  def __init__(self, *args, m: BaseLLM, params: AdaptationConfig, **kwargs) -> None:
    super(LlmAdapter, self).__init__(*args, **kwargs)
    self.lora_params = params.lora
    self.bias_scale_params = params.bias_scale
    self.norm_params = params.norm
    self.soft_prompt_params = params.soft_prompt
    self.rope_context_params = params.rope_context
    self.base_model = m.base_model if isinstance(m, LlmAdapter) else m

    if params.requires_adaptation():
      for p in self.base_model.parameters():
        p.requires_grad = False

    self._compute_optimised_parameters = {
      "tok_embeddings_shape": None
    }

  def enable_xops(self, enable: bool = True):
    self.base_model.enable_xops(enable=enable)

  def post_init_meta_buffers(self):
    self.base_model.post_init_meta_buffers()

  def register_cache_kv(self, *args, **kwargs):
    self.base_model.register_cache_kv(*args, **kwargs)

  def register_soft_prompt(self, *args, **kwargs):
    self.base_model.register_soft_prompt(*args, **kwargs)

  def _apply_adaptation(self, m: BaseLLM,
                        adaptation: BiasScaleAdaptation | LoraAdaptation) -> None:
    queue: List[nn.Module] = []
    queue.append(m)
    while len(queue) > 0:
      parent = queue.pop(0)
      for child_name, child_module in parent.named_children():
        if adaptation.is_adaptation_target(name=child_name, m=child_module):
          new_module = adaptation.adapt_layer(name=child_name, m=child_module)
          # accept None for inplace
          if new_module is not None:
            setattr(parent, child_name, new_module)
        else:
          queue.append(child_module)

  def apply_adaptation(self) -> None:
    # TODO: clean-up task for someone else
    if self.bias_scale_params is not None:
      self._apply_adaptation(
        m=self.base_model, adaptation=self.bias_scale_params)
    if self.lora_params is not None:
      self._apply_adaptation(m=self.base_model, adaptation=self.lora_params)
    if self.norm_params is not None:
      self._apply_adaptation(m=self.base_model, adaptation=self.norm_params)

    # TODO: When we convert RoPE to a class, we can add it here
    if self.rope_context_params is not None:
      scale_factor = self.rope_context_params.expand_ctx_len / self.rope_context_params.base_ctx_len
      
      if not hasattr(self.base_model, "freqs_cis"):
        raise ValueError("RoPE context adaptation requires the model to have the `freqs_cis` attribute")
      
      self.base_model.freqs_cis /= scale_factor

    # this one works different
    # Soft-Prompt is a module that is added via model surgery / monkey patching
    if self.soft_prompt_params is not None:
      register_fn = "register_soft_prompt"
      # check if the base model has the register_soft_prompt method
      # TODO: this should require some interface
      if hasattr(self.base_model, register_fn):
        self.base_model.register_soft_prompt(
          soft_prompt_size=self.soft_prompt_params.n_tokens,
          n_layers=self.soft_prompt_params.n_layers
        )
      else:
        # just print a warning
        print(
          f"WARNING: {self.base_model.__class__.__name__} does not have the {register_fn} method")
        
  def apply_optimization(self) -> None:
    self._pad_embedding_table()

  def partial_merge_adaptation(self, decay: float = 0.1) -> None:
    if self.lora_params is not None:
      for name, module in self.base_model.named_modules():
        if isinstance(module, LoraAdapterLinearLayer):
          module.partial_merge(decay=decay)

  # def merge_adaptation(self, fused: bool = False) -> None:
  #   if self.lora_params is not None:
  #     for name, module in self.base_model.named_modules():
  #       module.merge_adaptation()
  #       if isinstance(module, LoraAdapterLinearLayer):
          
  #         new_linear = nn.Linear(
  #           in_features=module.adapted_linear.in_features,
  #           out_features=module.adapted_linear.out_features,
  #           bias=module.adapted_linear.bias is not None
  #         )

  #         new_linear.weight.data = module.adapted_linear.weight.data
  #         if module.adapted_linear.bias is not None:
  #           new_linear.bias.data = module.adapted_linear.bias.data

  #         setattr(self.base_model, name, new_linear)

  def merge_adaptation(self) -> None:
    if self.lora_params is not None:
      for name, module in self.base_model.named_modules():
        if isinstance(module, LoraAdapterLinearLayer):
          module.merge_adaptation()

  def parameters(self, recurse: bool = True):
    for name, module in self.base_model.named_modules():
      if isinstance(module, Adapter):
        for p in module.trainable_parameters():
          yield p
      else:
        for p in module.parameters(recurse=False):
          if p.requires_grad:
            yield p

  def active_param_keys(self):
    # TODO: maybe this isn't needed
    params = set()
    for name, param in self.named_parameters():
      if param.requires_grad:
        params.add(name)

    return params

  def _pad_embedding_table(self):
    table: torch.nn.Embedding = self.base_model.tok_embeddings
    table_requires_grad = table.weight.requires_grad
    n_rows, _ = table.weight.shape
    # make sure that the table is padded to be a multiple of 64
    if n_rows % 64 != 0:
      n_pad_rows = n_rows + (64 - (n_rows % 64))
      padded_table = torch.nn.Embedding(n_pad_rows, table.weight.shape[1])
      padded_table.requires_grad_(table_requires_grad)
      padded_table.weight.data[:n_rows, :] = table.weight.data
      self._compute_optimised_parameters["tok_embeddings_shape"] = (
        n_rows, table.weight.shape[1])
      setattr(self.base_model, "tok_embeddings", padded_table)
    else:
      pass  
  
  def process_state_dict(self, state_dict: dict) -> dict:
    """Function that removes any compute optimized padding from the state dict"""
    
    # this happens when using fsdp
    if not state_dict:
      return state_dict

    if self._compute_optimised_parameters["tok_embeddings_shape"] is not None:
      n_rows, _ = self._compute_optimised_parameters["tok_embeddings_shape"]
      key_name = "base_model.tok_embeddings.weight"
      if key_name in state_dict:
        state_dict[key_name] = state_dict[key_name][:n_rows, :]
    return state_dict


  def forward(self, *args, **kwargs):
    return self.base_model(*args, **kwargs)
  
  @property
  def params(self):
    return self.base_model.params
  

# class RegressionLLMAdapter(nn.Module):
#   """
#   Helpsteer adaptation to predict attributes values
#   """
#   def __init__(self, *args, m: BaseLLM, attribute_count: int, **kwargs) -> None:
#     super(RegressionLLMAdapter, self).__init__(*args, **kwargs)
#     self.base_model = m.base_model if isinstance(m, LlmAdapter) else m
#     # regression head for each attribute
#     self.trf_out_dim = self.base_model.params.dim

#     reg_head 

#     self.attr_reg_heads = nn.ModuleList(
#       [nn.Linear(self.trf_out_dim, 1) for _ in range(attribute_count)]
#     )

#   def forward(self, *args, **kwargs):
#     trf_out = self.base_model(*args, **kwargs)
#     return [head(trf_out) for head in self.attr_reg_heads]


