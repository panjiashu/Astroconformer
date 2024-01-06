from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(
  optimizer: Optimizer,
  args
):
  
  def constant_lr(current_step):
    return 1
  
  def fish_lr(current_step):
    period = args.total_steps//3
    warmup_ratio = args.warmup_ratio
    i = current_step//period
    if i>=1:
      return 1e-3
    temp = (1/2)**i
    progress = current_step%period/period
    if 0<=progress<=warmup_ratio:# Remember bound points
      return max(1e-3, temp*progress/warmup_ratio)
    if warmup_ratio<progress:
      return max(1e-3, temp*(1-(progress-warmup_ratio)/(1-warmup_ratio)))
    
  def cyclic_lr(current_step):
    period = args.total_steps//args.num_cycles
    warmup_ratio = args.warmup_ratio
    lower_bound = args.lower_bound
    i = current_step//period
    temp = (1/2)**i
    progress = current_step%period/period
    if 0<=progress<=warmup_ratio:# Remember bound points
      return max(lower_bound, temp*progress/warmup_ratio)
    if warmup_ratio<progress:
      return max(lower_bound, temp*(1-(progress-warmup_ratio)/(1-warmup_ratio)))
    
  schedule_dict = {
    'constant': constant_lr,
    'fish': fish_lr,
    'cyclic': cyclic_lr
  }

  if args.scheduler == 'plateau':
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=15)
  
  return LambdaLR(optimizer, schedule_dict[args.scheduler])