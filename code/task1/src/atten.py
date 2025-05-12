import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class Atten(nn.Module):
    def __init__(self, c_dim=128, n_token=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.token_dim = c_dim // n_heads
        
        # Learnable query
        self.Q = nn.Parameter(torch.randn(n_heads, n_token, self.token_dim) / self.token_dim**0.5)
        
        # kv
        self.kv_1 = nn.Linear(c_dim, 2 * c_dim)
        self.kv_2 = nn.Linear(self.token_dim, 2 * self.token_dim)
        self.q_3 = nn.Linear(c_dim, c_dim)
        
        self.out_proj = nn.Linear(c_dim, c_dim)
        

    def forward(self, x):
                
        # Step 1: First attention layer
        k1, v1 = self.kv_1(x).chunk(2, dim=-1)
        k1 = rearrange(k1, 'b n (h c) -> b h n c', h=self.n_heads)
        v1 = rearrange(v1, 'b n (h c) -> b h n c', h=self.n_heads)
        
        attn1 = torch.einsum('h m c, b h n c -> b h m n', self.Q, k1) / self.token_dim**0.5
        attn1 = self.dropout(F.softmax(attn1, dim=-1))
        out1 = torch.einsum('b h m n, b h n c -> b h m c', attn1, v1)
        
        # Step 2: Self-attention
        k2, v2 = self.kv_2(out1).chunk(2, dim=-1)
        
        # Step 3: Cross attention
        q3 = rearrange(self.q_3(x), 'b n (h c) -> b h n c', h=self.n_heads)
        attn2 = torch.einsum('b h n c, b h m c -> b h n m', q3, k2) / self.token_dim**0.5
        attn2 = self.dropout(F.softmax(attn2, dim=-1))
        out = torch.einsum('b h n m, b h m c -> b h n c', attn2, v2)
        
        # Reshape and apply gated output projection
        out = rearrange(out, 'b h n c -> b n (h c)')
        # gate = self.gate_net(torch.cat([x, out], dim=-1))
        # gate = self.gate_net(x)
        
        return self.out_proj(out)
