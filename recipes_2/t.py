import torch
from torch import nn

if __name__ == "__main__":

  model = nn.Sequential(
    nn.Sequential(
      nn.Linear(20, 100),
      nn.ReLU(),
      nn.LayerNorm(100)
    ),
    nn.Linear(100, 1000)
  )
  model[0].requires_grad_(False)
  # model[0][0].requires_grad_(True)

  inputs = torch.randn(1, 10, 20)
  classes = torch.randint(0, 1000, (1, 10,))
  block_size = 5

  pre_logits = model[0].forward(inputs)
  print("LP RG: ", pre_logits.requires_grad)

  # # TODO: Ted make sure this won't be an issue in the future
  # assert ce_loss_fn.inplace_backward is True, "Support is currently provided only for inplace backward"

  N_BLOCKS, should_inc = divmod(pre_logits.shape[1], block_size)
  N_BLOCKS = N_BLOCKS + 1 if should_inc else N_BLOCKS
  
  grad_input_acc = []
  reporting_stats = []

  COMPUTED_BLOCKS = 0
  # forward pass on each block
  for i in range(N_BLOCKS):
    start = i * block_size
    end = (i + 1) * block_size
    pre_logit_block = pre_logits[:, start:end].detach()
    pre_logit_block.requires_grad = True

    logit_block = model[1].forward(pre_logit_block)
    label_block = classes[:, start:end]

    loss = torch.nn.functional.cross_entropy(logit_block.view(-1, 1000), label_block.view(-1), reduction='none')
    COMPUTED_BLOCKS += 1

    # we need to divide this loss by the number of blocks
    # this is very similar to gradient accumulation
    # think of how we virtually increase the batch_size, by not performing
    # the optimizer step which resets gradients

    # what we want to achieve here is to viartually "reduce" the computed batch_size
    # the .mean() / gradient computation now will account only for the computational
    # block size, which in our case is `block_size`. Since we will be summing up gradients
    # N_BLOCKS times, we need to divide the loss by N_BLOCKS to get the correct gradient
    loss = loss.mean()
    loss /= (N_BLOCKS * 1)
    loss.backward()
    grad_input_acc.append(pre_logit_block.grad.detach())

  
  # backward step on the rest of the network
  grad_pre_logits = torch.cat(grad_input_acc, dim=1)
  assert grad_pre_logits.ndim != 0, "grad_pre_logits should not be a scalar"

  # we might be skipping padding blocks
  # the deal with them is that we pre-normalized the loss
  # assuming that we would be computing the loss for all blocks
  # in order to get the correct grad-scaling   
  if COMPUTED_BLOCKS != N_BLOCKS:
    normalizing_factor = N_BLOCKS / COMPUTED_BLOCKS
    with torch.no_grad():
      grad_pre_logits.mul_(normalizing_factor)
      if model.output.weight.grad is not None: 
        model.output.weight.grad.mul_(normalizing_factor)

  pre_logits.requires_grad = True

  pre_logits.backward(grad_pre_logits)