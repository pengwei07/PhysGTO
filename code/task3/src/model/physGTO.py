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
        
    def forward(self, node_pos, state_in):
        
        #  node embedding
        state_in = torch.cat((state_in, node_pos), dim = -1)
        V = self.fv1(state_in)
        
        return V
    
class MixerBlock(nn.Module):
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        node_size = enc_dim + enc_s_dim
        
        # GNN部分保持不变
        self.gnn = GNN(node_size=node_size, 
                      edge_size=enc_dim, 
                      output_size=enc_dim, 
                      layer_norm=True)
        
        # Attention模块优化
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
        
    def forward(self, V, E, edges, s_enc):
        # 1. GNN部分
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
    # def get_edge_info(self, edges, node_pos):
    #     # 基本计算与之前相同
    #     senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
    #     receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        
    #     distance_1 = receivers - senders
    #     distance_2 = - distance_1
        
    #     norm = torch.sqrt((distance_1 ** 2).sum(-1, keepdims=True))
        
    #     # 为批次中的每条边创建随机交换掩码
    #     batch_size, num_edges = edges.shape[0], edges.shape[1]
    #     swap_mask = torch.rand(batch_size, num_edges, 1, device=edges.device) > 0.5
        
    #     # 使用掩码选择顺序
    #     distance_mix_1 = torch.where(swap_mask, distance_2, distance_1)
    #     distance_mix_2 = torch.where(swap_mask, distance_1, distance_2)
        
    #     E = torch.cat([distance_mix_1, distance_mix_2, norm], dim=-1)

    #     return E
    
    def forward(self, V, node_pos, edges, pos_enc):
        
        # edge
        E = self.get_edge_info(edges, node_pos)
        E = self.fe(E)
        
        for block in self.blocks:
            V, E = block(V, E, edges, pos_enc)

        return V
    
class Decoder(nn.Module):
    def __init__(self, enc_dim, state_size, if_pred_cd, space_size, pos_enc_dim):
        super().__init__()
        
        self.if_pred_cd = if_pred_cd
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        ###############
        # outputs
        self.delta_out = nn.Sequential(
            nn.Linear(enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim)
        )
        
        self.state_out = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )
        
        if if_pred_cd:
            # 将const的形状初始化为与state_size一致
            self.const = nn.Parameter(torch.zeros(state_size-1), requires_grad=True)
            self.act = nn.SiLU()
    
    def forward(self, V, s_enc):
        
        V = V + self.delta_out(torch.cat([V, s_enc], dim=-1))
        
        # presss pred
        if self.if_pred_cd:
            press_norm = self.state_out(V)
            const = self.act(self.const)
            drag_norm = const * torch.sum(press_norm[..., 1:], dim=1, keepdim=False)
            return press_norm[..., 0:1], drag_norm
        
        else:
            press_norm = self.state_out(V)
            return press_norm

class Model(nn.Module): 
    def __init__(self, 
                space_size = 3,
                pos_enc_dim = 5,                
                in_dim = 1,
                out_dim = 3,
                N_block = 4,
                enc_dim = 128, 
                n_head = 4,
                n_token = 128,
                if_pred_cd = False
                ):
        super(Model, self).__init__()
        
        self.if_pred_cd = if_pred_cd
        
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
            if_pred_cd = if_pred_cd,
            space_size = space_size,
            pos_enc_dim = pos_enc_dim
            )

    def forward(self, state_in, node_pos, edges): 
        
        # edges.dim = [B, L, 2]
        edges = edges.long()
        
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim) # 2 * 7 * 2 + 2 = 30
        
        # 1. encoder
        V = self.encoder(node_pos, state_in)
        
        # 2. mixer
        V = self.mixer(V, node_pos, edges, pos_enc)
        
        # decoder
        V = self.decoder(V, pos_enc) 
        
        return V