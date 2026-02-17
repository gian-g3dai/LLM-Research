from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import torch.distributed
from fairscale.nn.model_parallel.layers import (ColumnParallelLinear,
                                                ParallelEmbedding,
                                                RowParallelLinear,
                                                VocabParallelEmbedding)
from torch import Tensor

from unakin.llm.models.llama import LlamaTransformer
from unakin.llm.oss.llama.llama import Llama as MetaLlama
from unakin.llm.oss.llama.llama.model import RMSNorm

from unakin.llm.oss.llama3.llama import Llama as MetaLlama3
from unakin.llm.oss.llama3.llama.model import RMSNorm as RMSNorm3


def get_model_parallelism_requirement(checkpoint_dir: str) -> int:
  """
  Returns the model parallelism degree required to load the checkpoint
  """
  checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
  return len(checkpoints)

def _consolidate_checkpoint_dict(checkpoint_dir: str) -> dict[str, List[Tensor]]:
  """
  LLaMa checkpoints are saved with several checkpoints based on their
  model parallelism degree. This function consolidates the checkpoints
  into a single dictionary.
  """
  
  checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
  parameter_weights = defaultdict(list)
  for idx, ckpt_file in enumerate(checkpoints):
    ckpt = torch.load(ckpt_file, map_location="cpu")
    for name, weight in ckpt.items():
      if idx != 0:
        if name not in parameter_weights:
          raise ValueError(f"Parameter {name} not found in previous checkpoints")
        
        if weight.shape != parameter_weights[name][-1].shape:
          raise ValueError(f"Parameter {name} has different shape in previous checkpoints")
        
      parameter_weights[name].append(weight)

  return parameter_weights

def _consolidate_weight_dict(llama: MetaLlama,
                            weights: dict[str, List[Tensor]]) -> dict[str, Tensor]:
  """
  LLaMa models are defined with ColumnParallelLinear, RowParallelLinear, and
  ParallelEmbedding modules. This function consolidates the list of weights from
  the dictionary of weights into a single tensor that can be loaded into classic
  nn.Linear and nn.Embedding modules.
  """
  consolidated_weights = {}
  torch_module = llama.model
  idx = 0
  for module_name, module in torch_module.named_modules():
    for param_name, param in module.named_parameters():
      model_param_name = f"{module_name}.{param_name}"
      model_param_name = model_param_name.strip(".")
      idx += 1

      if model_param_name not in weights:
        raise ValueError(f"Parameter {model_param_name} not found in checkpoint")

      if isinstance(module, ColumnParallelLinear):
        consolidated_weights[model_param_name] = torch.cat(
          weights[model_param_name], dim=0)        
      elif isinstance(module, RowParallelLinear):
        consolidated_weights[model_param_name] = torch.cat(
          weights[model_param_name], dim=1)
      elif isinstance(module, ParallelEmbedding):
        consolidated_weights[model_param_name] = torch.cat(
          weights[model_param_name], dim=1)
      elif isinstance(module, VocabParallelEmbedding):
        consolidated_weights[model_param_name] = torch.cat(
          weights[model_param_name], dim=0)
      elif isinstance(module, RMSNorm) or isinstance(module, RMSNorm3):
        consolidated_weights[model_param_name] = weights[model_param_name][0]
      else:
        # very hacky way to skip some issues with double counting top level modules
        continue    

  return consolidated_weights

def _convert_llama_to_transformer(llama: MetaLlama) -> LlamaTransformer:
  """
  Converts a LLaMa model to a unakin-lib Transformer model
  """

  # TODO: rewrite with simply reading from CKPT_DIR params.json
  # TODO: Not crucial, at least for now, but simplies API

  torch.set_default_device(torch.device("cpu"))

  llama_model_args = llama.model.params
  llama_transformer = LlamaTransformer(llama_model_args)
  llama_transformer = llama_transformer.to(torch.device("cpu"))
  
  # sanity checks
  transformer_parameter_names = set(
    name for name, _ in llama_transformer.named_parameters())
  
  llama_parameter_names = set(
    name for name, _ in llama.model.named_parameters())

  # sort the parameter names to make sure they are in the same order
  transformer_parameter_names = set(sorted(transformer_parameter_names))
  llama_parameter_names = set(sorted(llama_parameter_names))

  if len(transformer_parameter_names) != len(llama_parameter_names):
    raise ValueError(
      f"LLaMa model has different number ({len(llama_parameter_names)})"
      f"of parameters than Transformer model ({len(transformer_parameter_names)})")  
  
  if transformer_parameter_names.union(llama_parameter_names) != transformer_parameter_names:
    raise ValueError("LLaMa model has parameters that are not in Transformer model")

  return llama_transformer

def _copy_llama_weights(llama: LlamaTransformer, weights: dict[str, Tensor]) -> None:
  """
  Copies the weights from the consolidated dictionary into the LLaMa model
  """

  idx = 0
  for name, parameter in llama.named_parameters():
    if name not in weights:
      raise ValueError(f"Parameter {name} not found in checkpoint")

    if parameter.device == torch.device("meta"):
      parameter = torch.zeros(parameter.shape, device=torch.device("cpu"))

    print(f"Copying weight {name} dtype {weights[name].dtype}")
    parameter.data.copy_(weights[name].cpu())
    idx += 1

  if idx != len(weights):
    raise ValueError("Not all parameters were copied")

def load_llama_checkpoint(
    checkpoint_dir: str, tokenizer_path: str, mode_parallel_size: int,
    max_seq_len: int = 2048, max_batch_size: int = 2048) -> LlamaTransformer | None:
  """
  Utility function to load a LLaMa checkpoint into a Transformer model that
  uses non-parallelized modules (i.e. nn.Linear and nn.Embedding) instead of
  the LLaMa modules (i.e. ColumnParallelLinear, RowParallelLinear, and ParallelEmbedding)

  Args:
    checkpoint_dir (str): directory where the LLaMa checkpoint is saved
    tokenizer_path (str): path to the tokenizer file
    mode_parallel_size (int): number of model parallelism shards
    max_seq_len (int): maximum sequence length used for kv cache. Defaults to 2048.
      Can be set to any arbitrary value later on by invoking `transformer.resize_cache` method.
    max_batch_size (int): maximum batch size used for kv cache. Defaults to 2048
      Can be set to any arbitrary value later on by invoking `transformer.resize_cache` method.

  Returns:
    LlamaTransformer: a Transformer model with LLaMa weights. On non-zero ranks,
      this function returns None in order to avoid RAM duplication.
  """
  
  llama = MetaLlama.build(
    ckpt_dir=checkpoint_dir,
    tokenizer_path=tokenizer_path,
    model_parallel_size=mode_parallel_size,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size)

  llama.model = llama.model.to(torch.device("cpu"))
  
  rank = torch.distributed.get_rank()
  
  if rank == 0:
    # merge all checkpoints into a single dictionary
    checkpoint_dict = _consolidate_checkpoint_dict(checkpoint_dir)
    # concatenate model parallelism dimension into a single tensor
    weight_dict = _consolidate_weight_dict(llama, checkpoint_dict)
    # create a transformer model
    llama_transformer = _convert_llama_to_transformer(llama)
    # copy the weights into the transformer model
    _copy_llama_weights(llama_transformer, weight_dict)
    return llama_transformer
  
  else:
    del llama
  
  return None

def load_llama3_checkpoint(checkpoint_dir: str, tokenizer_path: str, mode_parallel_size: int,
    max_seq_len: int = 2048, max_batch_size: int = 2048) -> LlamaTransformer | None:
  
  llama = MetaLlama3.build(
    ckpt_dir=checkpoint_dir,
    tokenizer_path=tokenizer_path,
    model_parallel_size=mode_parallel_size,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size)

  llama.model = llama.model.to(torch.device("cpu"))
  
  rank = torch.distributed.get_rank()
  
  if rank == 0:
    # merge all checkpoints into a single dictionary
    checkpoint_dict = _consolidate_checkpoint_dict(checkpoint_dir)
    # concatenate model parallelism dimension into a single tensor
    weight_dict = _consolidate_weight_dict(llama, checkpoint_dict)
    # create a transformer model
    llama_transformer = _convert_llama_to_transformer(llama)
    # copy the weights into the transformer model
    _copy_llama_weights(llama_transformer, weight_dict)
    return llama_transformer
  
  else:
    del llama
  
  return None
  