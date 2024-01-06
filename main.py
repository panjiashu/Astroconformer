import argparse
import yaml
import os
from itertools import product

import numpy as np
import torch
from torch.utils.data import Subset

from Astroconformer.Train.forward import model_fn_nomask
from Astroconformer.Train.train_tookit_dev import train
from Astroconformer.Train.utils import init_train
from Astroconformer.Data.data import data_provider, get_dataloader
from Astroconformer.Data.utils import tr_val_test_split, inspect_data
from Astroconformer.utils import Container, same_seeds

def main():
  args = Container(**yaml.safe_load(open('./Astroconformer/experiment_config.yaml', 'r')))
  args.load_dict(yaml.safe_load(open('./Astroconformer/model_config.yaml', 'r'))[args.model])
  args.load_dict(yaml.safe_load(open('./Astroconformer/default_config.yaml', 'r')))

  args.log_dir = args.dir+args.comment+'/'
  os.system('mkdir '+args.log_dir)
  os.system('mkdir '+args.log_dir+'figures')
  os.system('mkdir '+args.log_dir+'figures/scatter')
  
  args.device_type = "cuda" if torch.cuda.is_available() else "cpu"
  args.device = torch.device(args.device_type)
  args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
  with open(args.log_dir+'log.txt', 'a') as fp:
    fp.write(f"[Info]: Use {args.device} now!\n")
  if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

  # data loading
  data_set, collate_fn = data_provider(args)
  with open(args.log_dir+'log.txt', 'a') as fp:
    fp.write(data_set.info)

  # model initialization
  model_fn = model_fn_nomask

  with open(args.log_dir+'log.txt', 'a') as fp:
    fp.write("Args in experiment:\n")
    fp.write("\n".join(f"{key}: {value}" for key, value in vars(args).items()) + '\n')
  
  same_seeds(args.randomseed)
  for fold, (tr_idx, val_idx, test_idx) in enumerate(tr_val_test_split(len(data_set), args.tr_val_test)):
    args.fold = fold
    args.batch_per_epoch = len(tr_idx) // args.batch_size

    tr_set, val_set, test_set = [Subset(data_set, idx) if idx is not None else None for idx in (tr_idx, val_idx, test_idx)]
    named_dataset = {'train': tr_set, 'valid': val_set, 'test': test_set}
    train_loader, valid_loader, test_loader = get_dataloader(named_dataset, collate_fn, args)

    model, optimizer, scheduler, scaler = init_train(args)

    with open(args.log_dir+'log.txt', 'a') as fp:
      fp.write(f"[Info]: Fold {fold}:\n")

    if fold == 0:
      inspect_data(train_loader, args.log_dir+'figures/')

    train(model, model_fn, optimizer, scheduler, scaler, train_loader, valid_loader, test_loader, args)

    same_seeds(args.randomseed)

    if not args.kfold:
      break
  return 

if __name__ == "__main__":
  main()
