import os

import math
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.signal import savgol_filter as savgol

import torch
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

from ..utils import getclosest

longcadence = pd.DataFrame()
longcadence.loc[:, "quarter"] = ["Q" + str(int(i)) for i in np.arange(18)]
longcadence.loc[:, "t_first_cadence"] = [54953.03815, 54964.01099, 55002.01748, 55092.72221, 55184.87774, 55275.99115,
                                        55371.94733, 55462.67251, 55567.86468, 55641.01696, 55739.34343, 55833.70579,
                                        55931.90966, 56015.23787, 56106.63736, 56205.98550, 56304.59804, 56391.72690]
longcadence.loc[:, "t_last_cadence"] = [54962.74411, 54997.48122, 55090.96492, 55181.99660, 55274.70384, 55370.66003,
                                        55461.79386, 55552.04909, 55634.84602, 55738.42395, 55832.76587, 55930.82669,
                                        56014.52273, 56105.55441, 56203.81957, 56303.63768, 56390.46005, 56423.50115]

def cp_dataset(args):
  jobfs = os.environ['PBS_JOBFS']
  os.chdir(jobfs)
  os.system('cp /g/data/y89/jp6476/Kepseismic_all.tar '+jobfs)
  os.system('tar -xf '+jobfs+f'/Kepseismic_all.tar > /dev/null 2>&1')
  os.chdir('/g/data/y89/jp6476/')

def tr_val_test_split(data_size, tr_val_test_ratio):
  """
  Args:
    data_size: int, number of data points
    tr_val_test_ratio: list of float, ratio of train, val, test set.
  Return:
    tr_idx, val_idx, test_idx: list of arrays, index of train, val, test set.
  For k-fold cross validation, ratio_train is recommended to be divisible by ratio_val, so that the size of 
  val set is the same for all folds.
  """
  ratio_train, ratio_val, ratio_test = tr_val_test_ratio

  tr_size, val_size = int(data_size*ratio_train), int(data_size*ratio_val)
  test_size = data_size-tr_size-val_size if ratio_test > 0 else 0

  perm = np.random.permutation(data_size)
  
  num_fold = round(ratio_train/ratio_val)

  remaining_idx, test_idx = perm[:-test_size], np.sort(perm[-test_size:]) if test_size > 0 else None # condition only applies to test_set
  for fold in range(num_fold):
    val_idx = remaining_idx[fold*val_size:(fold+1)*val_size]
    tr_idx = np.concatenate((remaining_idx[:fold*val_size], remaining_idx[(fold+1)*val_size:]))
    yield tr_idx, val_idx, test_idx

def inspect_data(data_loader, dir):
  sample_data = next(iter(data_loader))[0]
  B = sample_data.shape[0]
  if len(sample_data.shape) == 3:
    sample_data = sample_data.reshape(B, -1)
  plt.imshow(sample_data, aspect='auto')
  plt.colorbar()
  plt.savefig(dir+'data.png', dpi=300)
  plt.close()

def table_read(table_root):
  if 'gaia' in table_root:
    df = Table.read(table_root).to_pandas().sort_values(by='KIC_ID')
    df.rename(columns={'KIC_ID':'KIC'}, inplace=True)
    df.rename(columns={'logg':'_'}, inplace=True)
    df.rename(columns={'Radius':'R'}, inplace=True)
  else:
    df = pd.read_csv(table_root)
  return df

def width_radius(radius):
  width = 0.92+23.03*np.exp(-0.27*radius)
  return width
ref_radius = np.linspace(0,40,1000)
ref_width = np.round(1/(width_radius(ref_radius)/1e6)/60/29.4)

def get_boxsize(radius):
  closestrad = getclosest(radius,ref_radius)
  boxsize = ref_width[ref_radius == closestrad]
  return boxsize

def sigclip(x,y,subs,sig):
  keep = np.zeros_like(x)
  start=0
  end=subs
  nsubs=int((len(x)/subs)+1) if len(x)%subs!=0 else int(len(x)/subs)
  for i in range(0,nsubs):        
    me=np.mean(y[start:end])
    sd=np.std(y[start:end])
    good=(y[start:end] > me-sig*sd) & (y[start:end] < me+sig*sd)
    keep[start:end] = good
    start=start+subs
    end=end+subs
  return keep
  
def preprocess_norm(data, radius):
  flux0 = data['PDCSAP_FLUX']
  time0 = data['TIME']
  qual = data['SAP_QUALITY']

  # remove bad points
  qual[np.isnan(flux0)] = 1
  good=(qual == 0)
  if len(good) == 0:
    return None
  time=time0[good]
  flux=flux0[good]
  res=sigclip(time,flux,50,3)
  good=(res == 1)
  time=time[good]
  flux=flux[good]
  
  # smoothing
  closestrad = getclosest(radius,ref_radius)
  boxsize = ref_width[ref_radius == closestrad]
  if boxsize % 2 == 0:
      smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
  else:
      smoothed_flux = savgol(flux,int(boxsize),1,mode='mirror')
  flux = flux/smoothed_flux-1

  # interpolate
  time_interp = np.arange(time[0], time[-1], 30./(60.*24.))
  flux_interp = np.interp(time_interp, time, flux)
  return flux_interp

def preprocess_norm_mp(star_boxsize_quarter):
  # Unpack the tuple
  star, boxsize, quarter = star_boxsize_quarter # star: [Q, 2, length]
  
  data = []
  quarter_flag = np.zeros(len(star)).astype(int)
  for i, lc in enumerate(star):
    flux = lc[1]
    time = lc[0]

    # remove bad points
    res=sigclip(time,flux,50,3)
    good=(res == 1)
    time=time[good]
    flux=flux[good]
    
    # smoothing
    if boxsize % 2 == 0:
        smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
    else:
        smoothed_flux = savgol(flux,int(boxsize),1,mode='mirror')
    flux = (flux+1)/(smoothed_flux+1)-1 if np.abs(np.median(smoothed_flux)) < 0.1 else flux/smoothed_flux-1

    # interpolate
    time_interp = np.arange(time[0], time[-1], 30./(60.*24.))
    flux_interp = np.interp(time_interp, time, flux)
    if len(flux_interp) >= 4001:
      data.append(flux_interp)
      quarter_flag[i] = 1
  return data, np.concatenate(data).std(), quarter[quarter_flag.astype(bool)]