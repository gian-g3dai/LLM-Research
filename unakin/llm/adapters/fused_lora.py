import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from unakin.llm.kernels.mem_efficient_dropout import seeded_dropout

class FusedLoraFn(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx,
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    dropout: float,
    train: bool=True,
  ):
    seed = random.randint(0, 2**32 - 1)
    dw = A @ B    
    if train: 
      dw = seeded_dropout(dw, dropout, seed)     
    else:
      dw = dw * (1 - dropout)
    
    # 0 multiplies are faster at hardware level
    # so do them after applying dropout

    dw = dw * alpha
    out = x @ (W + dw)
    
    ctx.hyper_params = (seed, alpha, dropout, train)
    ctx.save_for_backward(
      x, W, A, B, dw, 
    )

    return out
  
  @staticmethod
  def backward(ctx, grad_out):
    x, W, A, B, dw = ctx.saved_tensors
    seed, alpha, dropout, train = ctx.hyper_params
    
    assert ctx.needs_input_grad[1] is False, "W should not require grad in backward when using FusedLoraFn"

    grad_x = grad_A = grad_B = None
    if ctx.needs_input_grad[0]:    
      dw = seeded_dropout((A @ B), dropout, seed) * alpha
      grad_x = grad_out @ (W + dw).T
    
    precompute_masked_grad_out = ctx.needs_input_grad[2] or ctx.needs_input_grad[3]
    
    if precompute_masked_grad_out:
      grad_out = seeded_dropout(grad_out, dropout, seed)
      grad_out = grad_out * alpha
      xT = x.T 

    if ctx.needs_input_grad[2]:
      grad_A = xT @ grad_out @ B.T
    if ctx.needs_input_grad[3]:
      grad_B = A.T @ xT @ grad_out.T 

    return grad_x, None, grad_A, grad_B, None, None, None