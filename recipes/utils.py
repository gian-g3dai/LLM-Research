import torch
import numpy as np

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
  if tensor.requires_grad:
    return tensor.detach().cpu().numpy()
  else:
    return tensor.cpu().numpy()



