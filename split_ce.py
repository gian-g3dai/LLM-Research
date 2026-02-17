import torch
import torch.nn as nn
from torch.nn import functional as F

if __name__ == "__main__":
  LIN_SIZE = 5120
  VOCAB_SIZE = 32000
  N_ELEMS = 8192 * 16

  linear_1 = nn.Linear(LIN_SIZE, LIN_SIZE, bias=False).cuda().to(dtype=torch.bfloat16)
  linear_2 = nn.Linear(LIN_SIZE, VOCAB_SIZE, bias=False).cuda().to(dtype=torch.bfloat16)
  ins = torch.randn(N_ELEMS, LIN_SIZE, requires_grad=False, dtype=torch.bfloat16, device="cuda")
  gt = torch.randint(0, VOCAB_SIZE, (N_ELEMS,), device="cuda")

  pre_logits = linear_1(ins)
  