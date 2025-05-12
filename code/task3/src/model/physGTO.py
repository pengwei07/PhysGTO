import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .block import MLP, GNN, FourierEmbedding
from .atten import Atten

class Encoder(nn.Module): 
    def __init__(self, 
                 space_size = 2,
                 state_size = 4, 
                 enc_dim = 128,
                 ):
        super(Encoder, self).__init__()
        
        # for node embedding
        self.fv1 = MLP(input_size = state_size + space_size, output_size=enc_dim, act = 'SiLU', layer_norm = False)
        
        self.fv2 = MLP(input_size = 8, output_size=enc_dim, act = 'SiLU', layer_norm = False)
        
        self.fv3 = MLP(input_size = enc_dim, output_size=enc_dim, act = 'SiLU', layer_norm = False)
        
    def forward(self, node_pos, state_in, info):
        
        #  node embedding
        state_in = torch.cat((state_in, node_pos), dim = -1)
        
        V = self.fv1(state_in)
        cond = self.fv2(info)
        V = self.fv3(V + cond)
        
        return V, cond
    
class MixerBlock(nn.Module):
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        node_size = enc_dim + enc_s_dim
        
        # GNN
        self.gnn = GNN(node_size=node_size, 
                      edge_size=enc_dim, 
                      output_size=enc_dim, 
                      layer_norm=True)
        
        # Attention
        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.mha = Atten(n_token=n_token, 
                        c_dim=enc_dim, 
                        n_heads=n_head)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, 2 * enc_dim),
            nn.SiLU(),
            nn.Linear(2 * enc_dim, enc_dim)
        )
        
    def forward(self, V, E, edges, s_enc, cond):
        # 1. GNN
        V = V + cond
        V_in = torch.cat([V, s_enc], dim=-1)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        V = V + v

        # 2. Attention with Pre-norm
        V = V + self.mha(self.ln1(V))
        
        # 3. FFN with Pre-norm
        V = V + self.ffn(self.ln2(V))
        
        return V, E

class Mixer(nn.Module):
    def __init__(self, N, enc_dim, n_head, n_token, space_size, pos_enc_dim):
        super(Mixer, self).__init__()
        
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size

        self.fe = MLP(input_size= 2 * space_size + 1, output_size= enc_dim, n_hidden=1, act = 'SiLU', layer_norm=False)

        self.blocks = nn.ModuleList([
            MixerBlock(
                enc_dim=enc_dim, 
                n_head=n_head, 
                n_token=n_token, 
                enc_s_dim=enc_s_dim
                )
            for i in range(N)
        ])
    
    def get_edge_info(self, edges, node_pos):
        
        # mesh space
        senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        
        distance_1 = receivers - senders
        distance_2 = - distance_1
        
        norm = torch.sqrt((distance_1 ** 2).sum(-1, keepdims=True))
        
        E = torch.cat([distance_1, distance_2, norm], dim=-1)

        return E
    
    def forward(self, V, node_pos, edges, pos_enc, cond):
        
        # edge
        E = self.get_edge_info(edges, node_pos)
        E = self.fe(E)
        
        for block in self.blocks:
            V, E = block(V, E, edges, pos_enc, cond)

        return V
    
class Decoder(nn.Module):
    def __init__(self, enc_dim=128, state_size=1, if_pred_cd = False):
        super().__init__()
        
        self.if_pred_cd = if_pred_cd
        ###############
        # outputs
        self.delta_out = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim)
        )
        
        self.state_out = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )
        
        if if_pred_cd:
            self.const = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.act = nn.SiLU()
    
    def forward(self, V, cond):
        
        # 1. fusion
        V = V + self.delta_out(V + cond)

        # presss pred
        press_norm = self.state_out(V)
        
        if self.if_pred_cd:
            const = self.act(self.const)
            drag_norm = const * torch.sum(press_norm, dim=1, keepdim=False)
            return press_norm, drag_norm
        else:
            return press_norm

class Model(nn.Module): 
    def __init__(self, 
                space_size = 3,
                pos_enc_dim = 5,                
                in_dim = 1,
                out_dim = 1,
                N_block = 4,
                enc_dim = 128, 
                n_head = 4,
                n_token = 128,
                if_pred_cd = False
                ):
        super(Model, self).__init__()
                
        self.encoder = Encoder(
            space_size = space_size,
            state_size = in_dim,
            enc_dim = enc_dim,
            )

        self.pos_enc_dim = pos_enc_dim
        
        # edge_direction = "single"
        self.mixer = Mixer(
            N= N_block, 
            enc_dim = enc_dim,
            n_head = n_head,
            n_token = n_token,
            space_size = space_size,
            pos_enc_dim = pos_enc_dim
        )
        
        self.decoder = Decoder(
            enc_dim = enc_dim,
            state_size = out_dim,
            if_pred_cd = if_pred_cd
            )

    def forward(self, state_in, node_pos, edges, info): 
        
        # edges.dim = [B, L, 2]
        edges = edges.long()
        
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim) # 2 * 7 * 2 + 2 = 30
        
        # 1. encoder
        V, cond = self.encoder(node_pos, state_in, info)
        
        # 2. mixer
        V = self.mixer(V, node_pos, edges, pos_enc, cond)
        
        # decoder
        V = self.decoder(V, cond) 
        
        return V