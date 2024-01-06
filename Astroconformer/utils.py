import yaml
import random
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import torch

class Container(object):
  '''A container class that can be used to store any attributes.'''
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
  
  def load_dict(self, dict):
    for key, value in dict.items():
      if getattr(self, key, None) is None:
        setattr(self, key, value)

  def print_attributes(self):
    for key, value in vars(self).items():
      print(f"{key}: {value}")

def same_seeds(seed):
  '''Set random seed for torch, numpy and random.'''
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
  np.random.seed(seed)  
  random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def measure(bias):
  '''Measure the RMS and MAD of the bias.'''
  deviation = np.abs(bias)
  MAD = mad_std(bias)
  RMS = (np.sum(deviation**2)/len(deviation))**0.5
  return RMS, MAD

def see(predicts, labels):
  '''Plot the scatter plot of predicts and labels.'''
  plt.scatter(labels, predicts, s=0.5)
  lb, ub = min(labels.min(), predicts.mean())-1e-3, max(labels.max(), predicts.max())+1e-3
  plt.plot([lb, ub], [lb, ub], c='b')
  rmsfinal, madfinal = measure(predicts-labels)
  s1='RMS: {0:.3f}'.format(rmsfinal)
  s2='$\sigma_{MAD}$:'+'{0:.3f}'.format(madfinal)
  STR=s1+'\n'+s2+'\n'
  plt.text(lb+(ub-lb)*0.05,lb+(ub-lb)*0.95,s=STR,color='k',ha='left',va='top')

  plt.xlabel('Label')
  plt.ylabel('Predicts')

def getclosest(num,collection):
  '''Given a number and a list, get closest number in the list to number given.'''
  return min(collection, key=lambda x: abs(x-num))