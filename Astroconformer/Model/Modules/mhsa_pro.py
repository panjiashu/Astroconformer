# Adapted from https://github.com/BlinkDL/RWKV-LM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from typing import Optional,Tuple
import math
import logging

logger = logging.getLogger(__name__)


rwkv_emb_scale = 0.4 # try 0.4 for char-level english. try 1.0 for chinese.
rwkv_layer_decay = 1.0 # decay weights in higher layers. try 0.5 ~ 1.0.

class AttentionConfig:
  def __init__(self, ctx_len=100, **kwargs):
    self.ctx_len = ctx_len
    for k,v in kwargs.items():
        setattr(self, k, v)


########################################################################################################
# MHA_rotary: Multi-head Attention + Rotary Encoding + GeGLU FFN
########################################################################################################

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return torch.stack([self.cos_cached, self.sin_cached])

class ContinuousRotaryEmbedding(torch.nn.Module):
    '''Continuous rotary position embedding'''
    def __init__(self, dim, sequence_scale):
        super().__init__()
        base=10000
        self.sequence_scale = sequence_scale
        self.register_buffer('inv_freq', 1. / (base ** (torch.arange(0, dim, 2))))
    
    def forward(self, t):
        t = (t + 0.5)* self.sequence_scale 
        freqs = torch.einsum('ij,k->ijk', t, self.inv_freq) # freqs: [B, L, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1) # emb: [B, 1, L, dim], 1 for broadcast in head_num dim
        return torch.stack([emb.cos(), emb.sin()])
    
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[...,:q.shape[2],:], sin[...,:q.shape[2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MHA_rotary(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.collect_attention_map = False
        self.attention_map = None
        assert args.encoder_dim % args.num_heads == 0
        self.num_heads = args.num_heads
        self.head_size = args.encoder_dim // args.num_heads

        if args.timeshift:
            self.time_shift = nn.ZeroPad2d((0,0,1,0))

        self.query = nn.Linear(args.encoder_dim, args.encoder_dim)
        self.key = nn.Linear(args.encoder_dim, args.encoder_dim)
        self.value = nn.Linear(args.encoder_dim, args.encoder_dim)

        # self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        
        self.rotary_ndims = int(self.head_size * 0.5)
        
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(args.encoder_dim, args.encoder_dim)

    def forward(self, x, RoPE, key_padding_mask=None):
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x)[:, :-1, :C//2], x[:, :, C//2:]], dim = -1)

        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        
        # cos, sin = self.rotary_emb(q, seq_len=T)
        cos, sin = RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]           # (B, T) -> (B, 1, 1, T)
            att = att.masked_fill(key_padding_mask == 0, float('-inf'))
        att = F.softmax(att, dim = -1)                                                  # softmax

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)

        if self.collect_attention_map:
            self.attention_map = att

        return x

class GeGLU(torch.nn.Module):
    def __init__(self, config, layer_id, time_shift = False):
        super().__init__()
        self.layer_id = layer_id

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0,0,1,0))

        hidden_sz = 3 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x)[:, :-1, :C//2], x[:, :, C//2:]], dim = -1)
        
        k = self.key(x)
        v = self.value(x)        
        y = self.weight(F.gelu(k) * v)
        return y