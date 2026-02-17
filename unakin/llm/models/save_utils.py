from typing import Callable, List, Tuple

import torch


def load_virtual_checkpoint(
  model_constructor: Callable, state_dict: dict,
  ignore_keys: list[str] = None, load_state: bool = False) -> Tuple[torch.nn.Module, dict]:

  import time

  import torch.nn.utils

  start_time = time.time()

  # DO NOT REMOVE THIS
  with torch.device("meta"):
    model: torch.nn.Module = model_constructor()

  if ignore_keys is not None and len(state_dict) > 0:
    for key in ignore_keys:
      state_dict.pop(key, None)

  print(f"Allocated virtual model in {time.time() - start_time}s")
  # load_device = torch.cuda.current_device()

  if load_state and len(state_dict) > 0:
    start = time.time()
    try:
      model.load_state_dict(state_dict, assign=True, strict=False)
    except:
      raise ValueError("Pytorch version is too old, please update to 2.1.0 or higher")

    ret_state_dict = {}
    print(f"Loaded virtual model in {time.time() - start}s")
  else:
    print("WARNING: Model parameter loading was deffered, please make sure to load on device before using")
    ret_state_dict = {}

  # sometthing might've escaped so we'll just make sure
  # to move everything to the device we copied to

  model = model.to(next(model.parameters()).device)
  return model, ret_state_dict

def copy_buffers_(
    named_params_or_buffs: List[Tuple[str, torch.Tensor]],
    state_dict: dict
):

  for name, buff in named_params_or_buffs:
    if name in state_dict:
      if buff.device == torch.device("meta"):
        new_buff = torch.empty_like(buff, device=torch.device("cpu"))
        new_buff.copy_(state_dict[name])
        buff.copy_(new_buff)
      else:
        buff.copy_(state_dict[name])
      del state_dict[name]
    
    else:
      if not buff.device == torch.device("meta"):
        new_buff = torch.empty_like(buff, device=torch.device("cpu"))
        buff.copy_(new_buff)
      else:
        buff.copy_(torch.zeros_like(buff, device=torch.device("cpu")))
      print(f"Warning: {name} not found in checkpoint")
      