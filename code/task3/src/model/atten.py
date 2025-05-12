import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class Atten(nn.Module): 
    def __init__(self, 
                n_token=128,
                c_dim=128, 
                n_heads=4):
        super(Atten, self).__init__()
        
        self.c_dim = c_dim
        self.n_token = n_token
        self.n_heads = n_heads
        
        # Learnable query
        self.Q = nn.Parameter(torch.randn(self.n_token, self.c_dim), requires_grad=True)

        # Multihead attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)

    def forward(self, W0):   
        # Step 1: Initial attention with learned query
        
        batch = W0.shape[0]
        learned_Q = self.Q.unsqueeze(0).repeat(batch, 1, 1)
        W, _ = self.attention1(learned_Q, W0, W0)
    
        # Step 2: Self-attention on the transformed result
        W, _ = self.attention2(W, W, W)
        
        # Step 3: Position-aware attention
        W, _ = self.attention3(W0, W, W)
    
        return W
