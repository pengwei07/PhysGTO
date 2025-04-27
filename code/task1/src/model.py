import torch
import torch.nn as nn

from .block import MLP
from .atten import Atten
from .gnn import GNN, FourierEmbedding

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
        
        self.gnn = GNN(
            node_size=node_size, 
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
        
        # 1. GNN for local
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
        
        self.pos_enc_dim = pos_enc_dim
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        
        # for edge embedding     
        self.fe = MLP(input_size= space_size * 2 + 1, output_size=enc_dim, n_hidden=1, act = 'SiLU', layer_norm=False)
        
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
        distance_2 = senders - receivers
        norm = torch.sqrt((distance_1 ** 2).sum(-1, keepdims=True))
                        
        E = torch.cat([distance_1, distance_2, norm], dim=-1)
        
        E = self.fe(E)
        
        return E
    
    def forward(self, V, node_pos, edges):
        
        # space
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim) # 2 * 7 * 2 + 2 = 30
        # edge
        E = self.get_edge_info(edges, node_pos)
        
        for block in self.blocks:
            V, E = block(V, E, edges, pos_enc)
            
        return V
    
class Decoder(nn.Module):
    def __init__(self, N, enc_dim, state_size):
        super().__init__()
        
        self.delta_net = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )
        
    def forward(self, V):
        
        # 预测状态变化        
        delta = self.delta_net(V)

        return delta
    
class physGTO(nn.Module): 
    def __init__(self, 
                space_size = 2,
                pos_enc_dim = 7,                
                in_dim = 1,
                out_dim = 1,
                N_block = 4,
                enc_dim = 64, 
                n_head = 4,
                n_token = 64
                ):
        super(physGTO, self).__init__()
                
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
            space_size = space_size,
            pos_enc_dim = pos_enc_dim,
        )
        
        self.decoder = Decoder(
            N= N_block, 
            enc_dim = enc_dim,
            state_size = out_dim
            )
        
    def sample_edges(self, edges, sample_ratio):
        
        batch_size, N_edges = edges.shape[0], edges.shape[1]
        N_sample = int(N_edges * sample_ratio)
        
        # 为每个批次创建不同的随机索引
        # 生成形状为 [batch_size, N_edges] 的随机索引矩阵
        rand_indices = torch.stack([torch.randperm(N_edges, device=edges.device) for _ in range(batch_size)])
        
        # 只保留前 N_sample 个索引
        sample_indices = rand_indices[:, :N_sample]
        
        # 创建批次索引
        batch_indices = torch.arange(batch_size, device=edges.device).unsqueeze(1).expand(-1, N_sample)
        
        # 使用 batch_indices 和 sample_indices 进行索引
        sampled_edges = edges[batch_indices, sample_indices]
        
        return sampled_edges

    def forward(self, state_in, node_pos, edges, sample_ratio=0.1):
            
        batch_size = state_in.shape[0]
        node_pos = node_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        edges = edges.unsqueeze(0).repeat(batch_size, 1, 1).long()
        
        if sample_ratio<1.0:
            edges = self.sample_edges(edges, sample_ratio)
        
        # 1. Encoder
        V = self.encoder(node_pos, state_in)
            
        # 2. mixer
        V = self.mixer(V, node_pos, edges)
    
        # 3. decoder
        next_state = self.decoder(V)
        
        return next_state