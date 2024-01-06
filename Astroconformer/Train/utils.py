import copy
from itertools import product

import numpy as np
import torch
from torch.optim import AdamW, Adam
from torchinfo import summary
from nvitop import Device
import matplotlib.pyplot as plt

from ..utils import same_seeds
from ..Model.models import model_dict
from ..Model.utils import deepnorm_init
from .lr_scheduler import get_scheduler

def init_train(args):
  # model initialization
  same_seeds(args.randomseed)
  if args.model == 'Astroconformer':
    args.stride = int(20/args.sample_rates[0]**0.5)
  if args.deepnorm and args.num_layers >= 10:
    layer_coeff = args.num_layers/5.0
    args.alpha, args.beta = layer_coeff**(0.5), layer_coeff**(-0.5)
    
  model = model_dict[args.model](args).to(args.device)
  deepnorm_init(model, args)
  summary(model, args.input_shape, depth=10)
  # if args.fold == 0:
  #   with open(args.log_dir+'log.txt', 'a') as fp:
  #     fp.write(f"[Info]: Finish initializing {args.model}, summary of the model:\n{summary(model, args.input_shape, depth=10)}\n")
  
  # optimizer initialization
  # args.lr = args.batch_size/512*args.basic_lr if 'Kepseismic' not in args.dataset else args.basic_lr
  args.lr = args.basic_lr
  optimizer_dict = {'adamw': AdamW, 'adam': Adam}
  parameter = model.parameters()
  optimizer = optimizer_dict[args.optimizer](parameter, lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

  # scheduler initialization
  scheduler = get_scheduler(optimizer, args)

  # scaler initialization
  scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

  return model, optimizer, scheduler, scaler

def save_checkpoint(model, optimizer, scheduler, scaler, step, null_step, snapshot, args, name):
  torch.save({'net': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'scaler': scaler.state_dict(),
              'fold': args.fold,
              'step': step,
              'null_step': null_step,
              'snapshot': snapshot,
              }, args.log_dir + f'{name}.ckpt')
  
def check_gpu_use(args):

  devices = Device.all()  # or Device.cuda.all()

  with open(args.log_dir + 'log.txt', 'a') as fp:
    for device in devices:
      processes = device.processes()  # type: Dict[int, GpuProcess]
      sorted_pids = sorted(processes)

      fp.write(str(device) + '\n')
      fp.write(f'  - Fan speed:       {device.fan_speed()}%\n')
      fp.write(f'  - Temperature:     {device.temperature()}C\n')
      fp.write(f'  - GPU utilization: {device.gpu_utilization()}%\n')
      fp.write(f'  - Total memory:    {device.memory_total_human()}\n')
      fp.write(f'  - Used memory:     {device.memory_used_human()}\n')
      fp.write(f'  - Free memory:     {device.memory_free_human()}\n')
      fp.write(f'  - Processes ({len(processes)}): {sorted_pids}\n')
      for pid in sorted_pids:
          fp.write(f'    - {processes[pid]}\n')
      fp.write('-' * 120 + '\n')

def register_leaf_hooks(module, hook_fn):
    if not list(module.children()):
        # This is a leaf node
        print(f"Registering hook for {module}")
        module.register_forward_hook(hook_fn)
    else:
        # This is not a leaf node, so recurse on children
        for child in module.children():
            register_leaf_hooks(child, hook_fn)

def nan_detector(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in output of {module.__class__.__name__}")

def log_parameters_gradients_in_model(model, logger, step):
  for tag, value in model.named_parameters():
    logger.add_histogram(tag+"/param", value.data.cpu(), step)
    if value.grad is not None:
        logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

def uniform_sample(dataloader, args):
  # Get subset of data
  indices = dataloader.dataset.indices
  dataset = [dataloader.dataset.dataset.data[indice] for indice in indices]
  label = dataloader.dataset.dataset.label[indices]
  kids = dataloader.dataset.dataset.kids[indices]
  
  # Get sample indices
  sample_indices = []
  gap = (max(label)-min(label))/args.num_sample
  for i in range(0,args.num_sample):
    mask = (label>=min(label)+gap*i)&(label<=min(label)+gap*(i+1))
    sample_indice = np.where(mask)[0][0]
    sample_indices.append(sample_indice)
  sample_indices = torch.LongTensor(sample_indices)
  sample_labels = label[sample_indices]
  sample_kids = kids[sample_indices]
  if 'MQ' in args.dataset:
    sample_lcs = [dataset[indice][0][:4000] for indice in sample_indices]
  else:
    sample_lcs = [dataset[indice][:4000] for indice in sample_indices]
  sample_lcs = torch.stack(sample_lcs)
  sample_labels = sample_labels.float()

  return sample_lcs, sample_labels, sample_kids

def inspect_snapshot(step, snapshot, args):

  # Comparison scatter between train, val, test
  prefix = ['tr', 'val', 'test']
  plt.figure(figsize=(6,6))
  plt.xlabel('label')
  plt.ylabel('pred')
  for i, (loss, pred, label, _) in enumerate(snapshot):
    if i == 1:
      lim = label.min()-1e-2, label.max()+1e-2
    if loss is None:
      continue
    plt.scatter(label, pred, s=1.5, label=f'{prefix[i]}_loss={loss:.5f}')
  plt.plot(lim, lim, c='k', ls='--')
  plt.xlim(lim)
  plt.ylim(lim)
  plt.legend()
  plt.savefig(args.log_dir+f'figures/scatter/fold{args.fold}_step{str(step).zfill(5)}.png')
  plt.close()