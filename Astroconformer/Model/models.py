import torch
import torch.nn as nn
from torch import Tensor

from .Modules.conformer import ConformerEncoder
from .Modules.mhsa_pro import RotaryEmbedding
from .Modules.ResNet18 import ResNet18

class Astroconformer(nn.Module):
  def __init__(self, args) -> None:
    super(Astroconformer, self).__init__()
    self.head_size = args.encoder_dim // args.num_heads
    self.rotary_ndims = int(self.head_size * 0.5)
    
    self.extractor = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
            kernel_size = args.stride, out_channels = args.encoder_dim, stride = args.stride, padding = 0, bias = True),
                    nn.BatchNorm1d(args.encoder_dim),
                    nn.SiLU(),
    )
    
    self.pe = RotaryEmbedding(self.rotary_ndims)
    
    self.encoder = ConformerEncoder(args)
    
    self.pred_layer = nn.Sequential(
        nn.Linear(args.encoder_dim, args.encoder_dim),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(args.encoder_dim,1),
    )
    if getattr(args, 'mean_label', False):
      self.pred_layer[3].bias.data.fill_(args.mean_label)
    
  def forward(self, inputs: Tensor) -> Tensor:
    x = inputs #initial input_size: [B, L]
    x = x.unsqueeze(1) # x: [B, 1, L]
    x = self.extractor(x) # x: [B, encoder_dim, L]
    x = x.permute(0,2,1) # x: [B, L, encoder_dim]
    RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
    x = self.encoder(x, RoPE) # x: [B, L, encoder_dim]
    x = x.mean(dim=1) # x: [B, encoder_dim]
    x = self.pred_layer(x) # x: [B, 1]
    return x
    
model_dict = {
          'Astroconformer': Astroconformer,
          'ResNet18': ResNet18,
      }