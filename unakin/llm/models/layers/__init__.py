import torch.nn as nn

class BaseAttention(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def forward(self, *args, **kwargs):
    raise NotImplementedError()
  
class BaseNorm(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def forward(self, *args, **kwargs):
    raise NotImplementedError()