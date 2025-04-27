import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Atten(nn.Module):
    def __init__(self, c_dim=128, n_token=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.token_dim = c_dim // n_heads
        
        # Learnable query
        self.Q = nn.Parameter(torch.randn(n_heads, n_token, self.token_dim) / self.token_dim**0.5, requires_grad=True)
        
        # Projections
        self.qkv_1 = nn.Linear(c_dim, 3 * c_dim)
        self.kv_2 = nn.Linear(self.token_dim, 2 * self.token_dim)
        
        # Normalization and output
        self.norm = nn.LayerNorm(self.token_dim)
        self.out_proj = nn.Linear(c_dim, c_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
                
        # Step 1: First attention layer
        q1, k1, v1 = self.qkv_1(x).chunk(3, dim=-1)
        
        k1 = rearrange(k1, 'b n (h c) -> b h n c', h=self.n_heads)
        v1 = rearrange(v1, 'b n (h c) -> b h n c', h=self.n_heads)
        
        attn1 = torch.einsum('h m c, b h n c -> b h m n', self.Q, k1) / self.token_dim**0.5
        attn1 = self.dropout(F.softmax(attn1, dim=-1))
        out = torch.einsum('b h m n, b h n c -> b h m c', attn1, v1)
        
        # Step 2: Self-attention
        out = self.norm(out)
        k2, v2 = self.kv_2(out).chunk(2, dim=-1)
        
        # Step 3: Cross attention
        q3 = rearrange(q1, 'b n (h c) -> b h n c', h=self.n_heads)
        attn2 = torch.einsum('b h n c, b h m c -> b h n m', q3, k2) / self.token_dim**0.5
        attn2 = self.dropout(F.softmax(attn2, dim=-1))
        out = torch.einsum('b h n m, b h m c -> b h n c', attn2, v2)
        
        # Reshape and apply gated output projection
        out = self.out_proj(rearrange(out, 'b h n c -> b n (h c)'))
        
        return out