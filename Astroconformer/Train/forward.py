import numpy as np
import torch
from torch.nn.functional import mse_loss
# from .utils import check_nan

def model_fn_nomask(batch, model, args):
  """Forward a batch through the model."""
  lcs, labels, kids = batch
  lcs = lcs.to(args.device)
  labelsondevice = labels.to(args.device)
  
  with torch.autocast(device_type=args.device_type, dtype=torch.float16, enabled=args.use_amp):
    outs = model(lcs)
    loss = mse_loss(outs, labelsondevice)

  return loss, outs, labels, kids

def valid(dataloader, model, model_fn, args): 
  """Validate on validation set."""
  if dataloader is None:
    return None, None, None, None
  
  model.eval()
  transform = dataloader.dataset.dataset.transform
  dataloader.dataset.dataset.transform = False
  preds = []
  labels = []
  kids = []
  
  with torch.no_grad():
    for i, batch in enumerate(dataloader):
      _, pred, label, kid = model_fn(batch, model, args)
      preds.append(pred.detach().cpu())
      labels.append(label)
      kids.append(kid)

  model.train()
  preds = torch.cat(preds)
  labels = torch.cat(labels)
  loss = mse_loss(preds, labels)
  dataloader.dataset.dataset.transform = transform
  return loss.numpy(), preds.numpy(), labels.numpy(), np.concatenate(kids)