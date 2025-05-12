import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GTO_Atten(nn.Module): 
    def __init__(self, 
                c_dim = 128, 
                n_token = 128,
                n_heads = 4,
                s_dim = 32
                ):
        super(GTO_Atten, self).__init__()
        
        self.n_heads = n_heads
        token_dim = c_dim // n_heads
        
        # Learnable query
        self.Q = nn.Parameter(torch.randn(n_heads, n_token, token_dim), requires_grad=True)
        
        # atten_1
        self.kv_1 = nn.Linear(c_dim, c_dim)
        
        # atten_2
        self.qkv_2 = nn.Linear(token_dim, token_dim * 2, bias = False)
        
        # atten_3
        self.q_3 = nn.Linear(c_dim, c_dim)
        
        # forward
        self.proj = nn.Linear(c_dim, c_dim)
        
        # 方法1: Xavier均匀初始化
        nn.init.xavier_uniform_(self.Q)

    def forward(self, W0):   
        # W0: [B, N, C]      
        # s_enc: [B, N, s_dim]
        # cond: [B, N, C]
        
        # Step 1: attention with learned query
        kv_1 = rearrange(self.kv_1(W0), 'b n (h c) -> b h n c', h=self.n_heads)
        proj_weights = torch.einsum('h m c, b h n c -> b h m n', self.Q, kv_1) / (kv_1.shape[-1]**0.5)
        proj_weights = F.softmax(proj_weights, dim=-1) # B, H, M, N
        
        proj_token = torch.einsum("bhmn, bhnc -> bhmc", proj_weights, kv_1)
        
        # Step 2: self-attention on the transformed result
        k, v = self.qkv_2(proj_token).chunk(2, dim=-1)
        
        # Step 3: cross attention
        # V_in = torch.cat([W0, s_enc, cond], dim=-1)
        # V_in = torch.cat([self.q_3_0(W0), self.q_3_1(s_enc), self.q_3_2(cond)], dim=-1)
        q_3 = rearrange(self.q_3(W0), 'b n (h c) -> b h n c', h=self.n_heads)
        re_proj_weights = F.softmax(q_3 @ k.transpose(-2,-1)/ q_3.shape[-1]**0.5, dim=-1)
        
        W = torch.einsum('bhnm, bhmc -> bhnc', re_proj_weights, v)
        W = rearrange(W, 'b h n c -> b n (h c)')
        
        # Output projection
        W = self.proj(W)  # 投影层
    
        return W