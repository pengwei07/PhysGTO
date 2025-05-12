import torch
import torch.nn as nn

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
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, edge_direction):
        super().__init__()
        node_size = enc_dim + enc_s_dim
        
        # GNN部分保持不变
        self.gnn = GNN(node_size=node_size, 
                      edge_size=enc_dim, 
                      output_size=enc_dim, 
                      layer_norm=True, 
                      edge_direction=edge_direction)
        
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
    def __init__(self, N, enc_dim, n_head, n_token, edge_direction, space_size, pos_enc_dim):
        super(Mixer, self).__init__()
        
        self.pos_enc_dim = pos_enc_dim
        self.edge_direction = edge_direction
        
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
                
        # for edge embedding    
        if edge_direction == "single":
            self.fe = MLP(input_size= 2 * space_size + 1, output_size=enc_dim, n_hidden=1, act = 'SiLU', layer_norm=False)
        elif edge_direction == "both":
            self.fe = MLP(input_size= space_size + 1, output_size= enc_dim, n_hidden=1, act = 'SiLU', layer_norm=False)
        
        self.blocks = nn.ModuleList([
            MixerBlock(
                enc_dim=enc_dim, 
                n_head=n_head, 
                n_token=n_token, 
                enc_s_dim=enc_s_dim,
                edge_direction = edge_direction
                )
            for i in range(N)
        ])
    
    def get_edge_info(self, edges, node_pos):
        
        # mesh space
        senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        
        if self.edge_direction == "single":
            distance_1 = receivers - senders
            distance_2 = senders - receivers
            norm = torch.sqrt((distance_1 ** 2).sum(-1, keepdims=True))
            
            E = torch.cat([distance_1, distance_2, norm], dim=-1)
        
        elif self.edge_direction == "both":
            distance_1 = receivers - senders
            norm = torch.sqrt((distance_1 ** 2).sum(-1, keepdims=True))
            
            E = torch.cat([distance_1, norm], dim=-1)
        
        return E
    
    def forward(self, V, node_pos, edges):
        
        # space
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim) # 2 * 7 * 2 + 2 = 30
        # edge
        E = self.get_edge_info(edges, node_pos)
        E = self.fe(E)
        
        for block in self.blocks:
            V, E = block(V, E, edges, pos_enc)
            
        return V
    
class Decoder(nn.Module):
    def __init__(self, enc_dim=128, state_size=4, dt=0.01):
        super().__init__()
        # 分离delta预测和状态更新
        self.delta_net = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )
        
    def forward(self, V):
        # 预测状态变化
        delta = self.delta_net(V)
        
        
        return delta

class PhysGTO(nn.Module): 
    def __init__(self, 
                space_size = 2,
                pos_enc_dim = 7,                
                in_dim = 1,
                out_dim = 1,
                N_block = 4,
                enc_dim = 64, 
                n_head = 4,
                n_token = 64,
                edge_direction = "single"
                ):
        super(PhysGTO, self).__init__()
                
        self.encoder = Encoder(
            space_size = space_size,
            state_size = in_dim,
            enc_dim = enc_dim,
            )

        self.mixer = Mixer(
            N= N_block, 
            enc_dim = enc_dim,
            n_head = n_head,
            n_token = n_token,
            edge_direction = edge_direction,
            space_size = space_size,
            pos_enc_dim = pos_enc_dim,
        )
        
        self.decoder = Decoder(
            enc_dim = enc_dim,
            state_size = out_dim
            )

    def forward(self, state_in, node_pos, edges):

        batch_size = state_in.shape[0]
        edges = edges.unsqueeze(0).repeat(batch_size, 1, 1).long()
        node_pos = node_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        
        edges = edges.long()
        # 1. Encoder
        V = self.encoder(node_pos, state_in)
            
        # 2. mixer
        V = self.mixer(V, node_pos, edges)
    
        # 3. decoder
        next_state = self.decoder(V)
        
        return next_state