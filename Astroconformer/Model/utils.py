import torch.nn as nn
from .Modules.mhsa_pro import MHA_rotary

def deepnorm_init(model, args):

  def init_func(m):
    beta = getattr(args, 'beta', 1)
    if isinstance(m, MHA_rotary):  # adjust as necessary for your use case
      nn.init.xavier_normal_(m.query.weight, gain=1)
      nn.init.xavier_normal_(m.key.weight, gain=1)
      nn.init.xavier_normal_(m.value.weight, gain=beta)
      nn.init.xavier_normal_(m.output.weight, gain=beta)

      nn.init.zeros_(m.query.bias)
      nn.init.zeros_(m.key.bias)
      nn.init.zeros_(m.value.bias)
      nn.init.zeros_(m.output.bias)
      if getattr(m, 'ffn', None) is not None:
        nn.init.xavier_normal_(m.ffn.linear1.weight, gain=beta)
        nn.init.xavier_normal_(m.ffn.linear2.weight, gain=beta)
        nn.init.zeros_(m.ffn.linear1.bias)
        nn.init.zeros_(m.ffn.linear2.bias)

  model.apply(init_func)
