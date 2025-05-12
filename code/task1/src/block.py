import torch
import torch.nn as nn

from torch_scatter import scatter_mean

class MLP(nn.Module): 
    def __init__(self, 
                input_size = 128, 
                output_size = 128, 
                layer_norm = True, 
                n_hidden=1, 
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

class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False, edge_direction="single"):
        super(GNN, self).__init__()
        
        self.node_size = node_size
        self.output_size = output_size
        self.edge_direction = edge_direction
        
        output_size = output_size or node_size
        
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU', output_size=edge_size)
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU', output_size=output_size)

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
         
        if self.edge_direction == "single":
            edge_embeddings_0, edge_embeddings_1 = edge_embeddings.chunk(2, dim=-1)
            
            col_0 = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings_0.shape[-1])
            col_1 = edges[..., 1].unsqueeze(-1).repeat(1, 1, edge_embeddings_1.shape[-1])
                    
            edge_mean_0 = scatter_mean(edge_embeddings_0, col_0, dim=-2, dim_size=V.shape[1])
            edge_mean_1 = scatter_mean(edge_embeddings_1, col_1, dim=-2, dim_size=V.shape[1])
            edge_mean = torch.cat([edge_mean_0, edge_mean_1], dim=-1)
            
        elif self.edge_direction == "both":
            col_0 = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
            edge_mean = scatter_mean(edge_embeddings, col_0, dim=-2, dim_size=V.shape[1])
        
        node_inpt = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.f_node(node_inpt)
        
        return node_embeddings, edge_embeddings
    
    
def FourierEmbedding(pos, pos_start, pos_length):
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