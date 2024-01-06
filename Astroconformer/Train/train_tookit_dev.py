import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils import same_seeds
from .utils import save_checkpoint, check_gpu_use, log_parameters_gradients_in_model, uniform_sample, inspect_snapshot
from .forward import valid

def train(model, model_fn, optimizer, scheduler, scaler, train_loader, valid_loader, test_loader, args):
  '''Train the model for the full set of steps.'''

  init_step = 0
  epoch = 0
  null_step = 0
  best_loss = 500
  snapshot = []
  # samples = uniform_sample(train_loader, args)
  
  if args.use_checkpoint:
    # if there is a checkpoint, load checkpoint
    model_dir = args.checkpoint_dir
    state = torch.load(model_dir)

    if not args.from_pretrained:
      model_fold = state['fold']
      if model_fold > args.fold:
        with open(args.log_dir+'log.txt', 'a') as fp:
          stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
          fp.write(f'{stamp} The checkpoint is not in the same fold as the current fold. Skip loading checkpoint.\n')
        return 
    
    try:
      model.load_state_dict(state['net'])
      with open(args.log_dir+'log.txt', 'a') as fp:
        stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        fp.write(f'{stamp} Load checkpoint from {model_dir} successfully!\n')
      if not args.from_pretrained:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        init_step = state['step']+1
        epoch = init_step // args.batch_per_epoch
        null_step = state['null_step']
        snapshot = state['snapshot']
        best_loss = snapshot[1][0]
        
    except Exception as e:
      with open(args.log_dir+'log.txt', 'a') as fp:
        stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        fp.write(f'{stamp} Failed to strictly load the model. Error: {e}\n')
      model.load_state_dict(state['net'], strict=False)
      

  # training
  model.train()
  args.tb_dir = f'/g/data/y89/jp6476/Learning_curves/{args.target}/{args.comment}_{args.fold}/'
  writer = SummaryWriter(log_dir=args.tb_dir)
  same_seeds(args.randomseed)
  for step in range(init_step, args.total_steps):
    batch = next(iter(train_loader))
    tr_loss, tr_pred, tr_label, tr_kid = model_fn(batch, model, args)
    scaler.scale(tr_loss).backward()
    scaler.unscale_(optimizer)
    if args.grad_clip:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()
    if step == 0 and args.check_gpu:
      check_gpu_use(args)

    if (step + 1) % args.batch_per_epoch == 0:
      epoch += 1
    
    # validation
    if step % args.valid_steps == 0:
      if args.save_checkpoint:
        save_checkpoint(model, optimizer, scheduler, scaler, step, null_step, snapshot, args, 'model_now')
      if args.visualise_param:
        log_parameters_gradients_in_model(model, writer, step)

      batch_loss = tr_loss.cpu().item()
      writer.add_scalar('Loss/train', batch_loss, step)
        
      val_loss, val_pred, val_label, val_kid = valid(valid_loader, model, model_fn, args)
      writer.add_scalar('Loss/valid', val_loss, step)
        
      test_loss, test_pred, test_label, test_kid = valid(test_loader, model, model_fn, args)
      if test_loader is not None:
        writer.add_scalar('Loss/test', test_loss, step)
      
      with open(args.log_dir+'log.txt', 'a') as fp:
        stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        # kind of difficult to switch line in a string
        fp.write(stamp+f"epoch {step//args.batch_per_epoch}, step {step},tr_loss={batch_loss:.5f}, val_loss={val_loss:.5f}, test_loss={test_loss:.5f}, lr={optimizer.param_groups[0]['lr']}, bias={model.pred_layer[3].bias.item():.5f}\n")
 
      if val_loss < best_loss:
        null_step = 0
        best_loss = val_loss
        with open(args.log_dir+'log.txt', 'a') as fp:
          fp.write(f"step {step + 1}, best model. (loss={best_loss:.5f})\n")
        snapshot = [[batch_loss, tr_pred.detach().cpu(), tr_label, tr_kid], [val_loss, val_pred, val_label, val_kid], [test_loss, test_pred, test_label, test_kid]]
        if args.save_checkpoint:
          save_checkpoint(model, optimizer, scheduler, scaler, step, null_step, snapshot, args, f'modelbestloss_{args.fold}')

    # early stop
    null_step += 1
    if null_step == args.early_stop:
      break

    if step % args.visual_steps == 0 or step == args.total_steps - 1:
      # model_temp = copy.deepcopy(model)
      # model_temp.load_state_dict(torch.load(args.log_dir + f'modelbestloss_{args.fold}.ckpt')['net']).to(args.device)
      # outliers = outlier_detection(test_loader, test_pred, test_label, test_kid, args)
      inspect_snapshot(step, snapshot, args)
      # if args.visulize_attention_map:
      #   visulize_attention(step, model, samples, args)
  writer.close()

  return best_loss