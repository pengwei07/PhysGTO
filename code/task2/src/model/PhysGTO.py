import numpy as np
import torch
import torch.nn as nn

from .block import MLP, GNN
from .atten import GTO_Atten

class Encoder(nn.Module): 
    def __init__(self, 
                 space_size = 2,
                 pos_enc_dim = 7,
                 cond_dim = 7,
                 state_size = 4, 
                 enc_dim = 128,
                 ):
        super(Encoder, self).__init__()
        
        self.pos_enc_dim = pos_enc_dim
        # for node embedding
        self.fv1 = MLP(input_size = state_size, output_size=enc_dim, act = 'SiLU', layer_norm = False)
        
        self.fv_type = MLP(input_size = cond_dim, output_size=enc_dim, act = 'SiLU', layer_norm = False)
        self.enc_t = MLP(input_size = 1, output_size = enc_dim, act = 'SiLU', layer_norm = False)
        # self.fv_type = nn.Linear(cond_dim, enc_dim)
        # self.enc_t = nn.Linear(1, enc_dim)
        
        self.fv2 = MLP(input_size = enc_dim * 2, output_size=enc_dim, act = 'SiLU', layer_norm = False)
        
        # for edge embedding    
        self.fe = MLP(input_size= 2 * space_size + 1, output_size=enc_dim, n_hidden=1, act = 'SiLU', layer_norm=False)
        
    def FourierEmbedding(self, pos, pos_start, pos_length):
        # F(x) = [cos(2^i * pi * x), sin(2^i * pi * x)]
        
        original_shape = pos.shape
        new_pos = pos.reshape(-1, original_shape[-1])
        index = torch.arange(pos_start, pos_start + pos_length, device=pos.device)
        index = index.float()
        freq = 2 ** index * torch.pi
        cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
        sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
        embedding = torch.cat([cos_feat, sin_feat], dim=-1)
        embedding = embedding.view(*original_shape[:-1], -1)
        all_embeddings = torch.cat([embedding, pos], dim=-1)
        
        return all_embeddings

    def get_edge_info(self, edges, node_pos):
        
        # mesh space
        senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        
        distance_1 = receivers - senders
        distance_2 = senders - receivers
        
        norm = torch.sqrt((distance_1 ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance_1, distance_2, norm], dim=-1)
        
        return E
    
    def forward(self, node_pos, state_in, t_now, node_type, edges):
        
        #  1. node embedding
        V = self.fv1(state_in) # dim = [B, N, C]
        V_cond = self.fv_type(node_type) + self.enc_t(t_now).unsqueeze(1) # dim = [B, N, C]
        V = self.fv2(torch.cat((V,V_cond), dim=-1))
        
        # space
        pos_enc = self.FourierEmbedding(node_pos, 0, self.pos_enc_dim) # 2 * 7 * 2 + 2 = 30
        
        #  2. edges embedding
        edges = edges.long()
        E = self.get_edge_info(edges, node_pos)
        E = self.fe(E)
        
        return V, E, pos_enc, V_cond

class MixerBlock(nn.Module):
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super(MixerBlock, self).__init__()
        
        # 1. GNN
        node_size = enc_dim * 2 + enc_s_dim
        self.gnn = GNN(node_size=node_size, edge_size=enc_dim, output_size=enc_dim, layer_norm=True)

        # 2. Attention
        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.MHA = GTO_Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head, s_dim = enc_s_dim)

        self.linear1 = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim)
        )        
        
    def forward(self, V, E, edges, s_enc, cond):
        
        # 1. GNN
        V_in = torch.cat([V, s_enc, cond], dim=-1)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        V = V + v
        
        # 2. Attention with Pre-norm
        V = self.MHA(self.ln1(V)) + V
        V = self.linear1(self.ln2(V)) + V
        
        return V, E

class Mixer(nn.Module):
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim):
        super(Mixer, self).__init__()

        self.blocks = nn.ModuleList([
            MixerBlock(
                enc_dim=enc_dim, 
                n_head=n_head, 
                n_token=n_token, 
                enc_s_dim=enc_s_dim)
            for i in range(N)
        ])
    
    def forward(self, V, E, edges, s_enc, cond):
        
        for block in self.blocks:
            V, E = block(V, E, edges, s_enc, cond)
            
        return V
    
class Decoder(nn.Module):
    def __init__(self,
                 enc_dim = 128, 
                 state_size = 4,
                 ):
        super(Decoder, self).__init__()
        
        # outputs
        # self.delta_out = nn.Sequential(
        #     nn.Linear(enc_dim * 2, enc_dim),
        #     nn.LayerNorm(enc_dim),
        #     nn.SiLU(),
        #     nn.Linear(enc_dim, enc_dim)
        # )
        
        self.state_out = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )
        
        dt = 1/30
        self.alpha = nn.Parameter(torch.tensor(dt), requires_grad=True)
        self.act = nn.SiLU()
        
    def forward(self, V, state_in, cond):
        
        # 1. fusion
        # v_in = torch.cat([V, cond], dim=-1)
        # V = V + self.delta_out(v_in)
        
        # 2. next state
        v_in = torch.cat([V, cond], dim=-1)
        scale = self.act(self.alpha)
        delta_state = self.state_out(v_in) * scale
        next_state = state_in + delta_state
        
        return next_state, delta_state

class Model(nn.Module):
    def __init__(self, 
                space_size = 2,
                pos_enc_dim = 7,
                cond_dim = 7,
                N_block = 4,
                in_dim = 4,
                out_dim = 4,
                enc_dim = 128, 
                n_head = 4,
                n_token = 128
                ):
        super(Model, self).__init__()
                
        self.encoder = Encoder(
            space_size = space_size,
            pos_enc_dim = pos_enc_dim,
            cond_dim = cond_dim,
            state_size = in_dim,
            enc_dim = enc_dim,
            )

        self.mixer = Mixer(
            N= N_block, 
            enc_dim = enc_dim,
            n_head = n_head,
            n_token = n_token,
            enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        )
        
        self.decoder = Decoder(
            enc_dim = enc_dim,
            state_size = out_dim
            )

    def sample_edges(self, edges, sample_ratio):
        
        batch_size, N_edges = edges.shape[0], edges.shape[1]
        N_sample = int(N_edges * sample_ratio)
        
        rand_indices = torch.stack([torch.randperm(N_edges, device=edges.device) for _ in range(batch_size)])

        sample_indices = rand_indices[:, :N_sample]
        
        batch_indices = torch.arange(batch_size, device=edges.device).unsqueeze(1).expand(-1, N_sample)
        
        sampled_edges = edges[batch_indices, sample_indices]
        
        return sampled_edges

    def forward(self, node_pos, edges, t_all, state_in, node_type, sample_ratio=1.0):
        
        if sample_ratio<1.0:
            edges = self.sample_edges(edges, sample_ratio)
        
        edges = edges.long()
        # 1. Encoder + time_aggregator
        V, E, s_enc, cond = self.encoder(node_pos, state_in, t_all, node_type, edges)
        
        # 2. mixer
        V = self.mixer(V, E, edges, s_enc, cond)
        
        # 3. decoder
        next_state, delta_state = self.decoder(V, state_in, cond)
        
        return next_state, delta_state
    

# model = Model()

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
# params = int(sum([np.prod(p.size()) for p in model_parameters]))

# print(params)
# # 2379269

# node_pos = torch.randn(2, 1000, 2)
# edges = torch.randint(0, 1000, (2, 2000, 2))
# t_all = torch.randn(2, 1)
# state_in = torch.randn(2, 1000, 4)
# node_type = torch.randn(2, 1000, 7)

# output = model(node_pos, edges, t_all, state_in, node_type)

# print(output.shape)
# torch.Size([2, 1000, 4])
