import torch
import math
from recipes_2.finetuning_setup import FineTuningConfig
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from unakin.llm.distributed.fsdp import MP_POLICY


def setup_lr_scheduler(optimizer: torch.optim.Optimizer,
                       n_total_steps: int,
                       args: FineTuningConfig) -> torch.optim.lr_scheduler.LambdaLR:
  """Setup learning rate scheduler"""

  # fraction
  warm_up_fraction = args.hyper_args.lr_warmup_ratio

  # create a lambda scheduler (don't like concatenating lambdas)
  warm_up_steps = math.ceil(n_total_steps * warm_up_fraction)
  decay_steps = n_total_steps - warm_up_steps

  # base learning rate
  lr = optimizer.param_groups[0]["lr"]

  def _linear_warmup(
      step: int, max_step: int, peak_lr: float, start_lr: float) -> float:
    """Linear warmup"""
    return (peak_lr - start_lr) * step / max_step + start_lr

  def _cosine_anealing_decay(
      step: int, max_step: int, lr: float, min_lr: float) -> float:
    """Cosine annealing decay"""
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / max_step))

  def lambda_lr(step: int) -> float:
    """Lambda learning rate"""
    peak_lr = lr
    
    if step <= warm_up_steps and warm_up_fraction > 0.:
      curr_lr = _linear_warmup(step, warm_up_steps, peak_lr, start_lr=1e-10)
    else:
      curr_lr = _cosine_anealing_decay(
        step - warm_up_steps, decay_steps, peak_lr, min_lr=peak_lr/30)

    # hack, because lambda lr returns a multiplicative factor which is applied to the
    # original learning rate, we need to divide by the original learning rate to get
    # the actual learning rate
    return curr_lr / peak_lr

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1)
  return scheduler


def setup_scaler(args: FineTuningConfig) -> ShardedGradScaler | GradScaler | None:
  # skip if not using float16 anywhere
  if args.parallel_args.fsdp_settings.mixed_precision == MP_POLICY.FP_16:
    return None

  # FSDP case
  if args.parallel_args is not None:
    return ShardedGradScaler(
      init_scale=args.hyper_args.scaler_init_scale,
      backoff_factor=args.hyper_args.scaler_backoff_factor,
      growth_factor=args.hyper_args.scaler_growth_factor,
      growth_interval=args.hyper_args.scaler_growth_interval,
    )

  # other cases
  return GradScaler(
    init_scale=args.hyper_args.scaler_init_scale,
    backoff_factor=args.hyper_args.scaler_backoff_factor,
    growth_factor=args.hyper_args.scaler_growth_factor,
    growth_interval=args.hyper_args.scaler_growth_interval,
  )