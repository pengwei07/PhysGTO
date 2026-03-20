import torch
import torch.nn as nn
import torch.nn.functional as F
     
from torch_scatter import scatter_mean, scatter_sum

from atten import AttentionBlock
    
class MLP(nn.Module): 
    def __init__(self, 
                input_size=256, 
                output_size=256, 
                layer_norm=True, 
                n_hidden=2, 
                hidden_size=256, 
                act = 'PReLU',
                ):
        super(MLP, self).__init__()
        if act == 'GELU':
            self.act = nn.GELU()
        elif act == 'SiLU':
            self.act = nn.SiLU()
        elif act == 'PReLU':
            self.act = nn.PReLU()
            
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), self.act]
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)
    
class Encoder(nn.Module): 
    def __init__(self, 
                 space_size = 2,
                 pos_enc_dim = 7,
                 state_size = 1, 
                 state_embedding_dim = 128
                 ):
        super(Encoder, self).__init__()
        
        self.pos_enc_dim = pos_enc_dim
        # for node embedding
        self.fv = MLP(input_size = state_size + space_size, output_size=state_embedding_dim, act = 'SiLU', layer_norm = False)
    
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
    
    def forward(self, state_in, node_pos):
        
        #  1. node embedding
        V = self.fv(state_in)
 
        pos_enc = self.FourierEmbedding(node_pos, 0, self.pos_enc_dim) # 2 * 7 * 2 + 2 = 30

        return V, pos_enc
    
class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
        
        self.node_size = node_size
        self.output_size = output_size
        
        output_size = output_size or node_size
        
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm, act='GELU', output_size=edge_size)
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm, act='GELU', output_size=output_size)

    def get_edges_info(self, V, E, edges):

        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        
        return edge_inpt
    
    def forward(self, V, E, edges):
        """
        V: (batch_size, num_nodes, node_size)
        E: (batch_size, num_edges, edge_size)
        edges: (batch_size, num_edges, 2)
        """
        edge_inpt = self.get_edges_info(V, E, edges)
        edge_embeddings = self.f_edge(edge_inpt)

        edge_embeddings_0 = edge_embeddings[..., :edge_embeddings.shape[-1] // 2]
        edge_embeddings_1 = edge_embeddings[..., edge_embeddings.shape[-1] // 2:]
        
        col_0 = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings_0.shape[-1])
        col_1 = edges[..., 1].unsqueeze(-1).repeat(1, 1, edge_embeddings_1.shape[-1])
                
        edge_mean_0 = scatter_mean(edge_embeddings_0, col_0, dim=-2, dim_size=V.shape[1])
        edge_mean_1 = scatter_mean(edge_embeddings_1, col_1, dim=-2, dim_size=V.shape[1])
        
        edge_mean = torch.cat([edge_mean_0, edge_mean_1], dim=-1)
        
        node_inpt = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.f_node(node_inpt)
        
        return node_embeddings, edge_embeddings
    
class MixerBlock(nn.Module):
    def __init__(self, state_embedding_dim, n_head, n_token, enc_s_dim):
        super(MixerBlock, self).__init__()
        
        node_size = state_embedding_dim + enc_s_dim
        self.gnn = GNN(node_size=node_size, edge_size=state_embedding_dim, output_size=state_embedding_dim, layer_norm=True)
        
        self.ln1 = nn.LayerNorm(state_embedding_dim)
        self.ln2 = nn.LayerNorm(state_embedding_dim)
        
        self.linear1 = nn.Linear(state_embedding_dim, state_embedding_dim)
        
        self.MHA = AttentionBlock(
            n_token = n_token,
            c_dim = state_embedding_dim,
            n_heads = n_head,
        )

    def forward(self, V, E, edges, s_enc):
        
        # 1. GNN
        V_in = torch.cat([V, s_enc], dim=-1)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        V = V + v
        
        # Attention
        V = self.MHA(self.ln1(V)) + V
        V = self.linear1(self.ln2(V)) + V
        
        return V, E

class Mixer(nn.Module):
    def __init__(self, space_size, N, state_embedding_dim, n_head, n_token, enc_s_dim):
        super(Mixer, self).__init__()

        # for edge embedding    
        self.fe = MLP(input_size= 2 * space_size + 1, output_size=state_embedding_dim, n_hidden=1, act = 'SiLU', layer_norm=False)

        self.blocks = nn.ModuleList([
            MixerBlock(
                state_embedding_dim=state_embedding_dim, 
                n_head=n_head, 
                n_token=n_token, 
                enc_s_dim=enc_s_dim)
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
        
        return E
    
    def forward(self, V, node_pos, edges,  pos_enc):
                
        edges = edges.long()
        E = self.get_edge_info(edges, node_pos)
        E = self.fe(E)
                
        for block in self.blocks:
            V, E = block(V, E, edges, pos_enc)
            
        return V

class Decoder(nn.Module):
    def __init__(self,
                 state_embedding_dim = 128, 
                 state_size = 1,
                 ):
        super(Decoder, self).__init__()
        
        self.final_mlp_node = nn.Sequential(
            nn.Linear(state_embedding_dim, state_embedding_dim), nn.SiLU(),
            nn.Linear(state_embedding_dim, state_embedding_dim), nn.SiLU(),
            nn.Linear(state_embedding_dim, state_size)
        )
        
        # # dt = 1/30
        # dt = 0.0
        # self.alpha = nn.Parameter(dt * torch.tensor(1.0), requires_grad=True)
        # self.act = nn.SiLU()
        
    def forward(self, V):

        next_state = self.final_mlp_node(V)
        
        return next_state
    
class Model(nn.Module):
    def __init__(self, 
                space_size = 2,
                pos_enc_dim = 7,
                N_block = 4,
                state_size = 1,
                state_embedding_dim = 128, 
                n_head = 4,
                n_token = 128
                ):
        super(Model, self).__init__()
                
        self.encoder = Encoder(
            space_size = space_size,
            pos_enc_dim = pos_enc_dim,
            state_size = state_size,
            state_embedding_dim = state_embedding_dim,
            )

        self.mixer = Mixer(
            space_size = space_size,
            N= N_block, 
            state_embedding_dim = state_embedding_dim,
            n_head = n_head,
            n_token = n_token, 
            enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        )
        
        self.decoder = Decoder(
            state_embedding_dim = state_embedding_dim,
            state_size = state_size
            )

    def forward(self, state_in, node_pos, edges):
        
        batch_size = state_in.shape[0]
        edges = edges.unsqueeze(0).repeat(batch_size, 1, 1).long()
        node_pos = node_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        
        state_in = torch.cat([state_in, node_pos], dim=-1)
        
        # 1. Encoder + time_aggregator
        V, pos_enc = self.encoder(state_in, node_pos)
            
        # 2. mixer
        V = self.mixer(V, node_pos, edges, pos_enc)
    
        # 3. decoder
        next_state = self.decoder(V)
        
        return next_state