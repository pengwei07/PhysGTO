import torch
import torch.nn as nn

from torch_scatter import scatter_mean
from .block import MLP

class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
        
        self.node_size = node_size        
        self.output_size = output_size or node_size
        
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
        
        edge_embeddings_0, edge_embeddings_1 = edge_embeddings.chunk(2, dim=-1)
        
        col_0 = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings_0.shape[-1])
        col_1 = edges[..., 1].unsqueeze(-1).repeat(1, 1, edge_embeddings_1.shape[-1])
                
        edge_mean_0 = scatter_mean(edge_embeddings_0, col_0, dim=-2, dim_size=V.shape[1])
        edge_mean_1 = scatter_mean(edge_embeddings_1, col_1, dim=-2, dim_size=V.shape[1])
        edge_mean = torch.cat([edge_mean_0, edge_mean_1], dim=-1)
    
        
        node_inpt = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.f_node(node_inpt)
        
        return node_embeddings, edge_embeddings
    

class flux_GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super().__init__()
        
        self.node_size = node_size
        self.output_size = output_size or node_size
        
        # 单元特征更新
        self.cell_encoder = MLP(
            input_size=node_size,
            n_hidden=n_hidden,
            output_size=node_size,
            act='SiLU',
            layer_norm=layer_norm
        )
        
        # 边特征更新
        self.edge_update = MLP(
            input_size = edge_size + node_size * 2,
            n_hidden = n_hidden,
            output_size = edge_size,
            act='SiLU',
            layer_norm = layer_norm
        )
        
        # 节点特征更新
        self.node_update = MLP(
            input_size=edge_size + node_size, 
            n_hidden=n_hidden,
            output_size=self.output_size,
            act='SiLU',
            layer_norm=layer_norm
        )
    
    def node2cell(self, V, cells):
        
        B, M, C = cells.shape
        
        batch_idx = torch.arange(B, device=V.device)[:, None, None].expand(-1, M, C)
        cell_nodes_feat = V[batch_idx, cells]        
        cell_features = self.cell_encoder(cell_nodes_feat.mean(dim=-2))
        
        return cell_features
        
    def flux_aware_update(self, cell_features, edge_to_cells):
        
        B = cell_features.shape[0]
        
        # Boundary handling
        left_idx = edge_to_cells[..., 0]   
        right_idx = edge_to_cells[..., 1]  
        
        # Mirror mapping at boundaries
        left_idx = torch.where(left_idx >= 0, left_idx, right_idx)
        right_idx = torch.where(right_idx >= 0, right_idx, left_idx)
        
        batch_idx = torch.arange(B, device=cell_features.device)[:, None].expand(-1, left_idx.size(1))
        cell_feat_left = cell_features[batch_idx, left_idx]
        cell_feat_right = cell_features[batch_idx, right_idx]
        
        return cell_feat_left, cell_feat_right
    
    def forward(self, V, E, edges, cells, edge_to_cells):
    
        N = V.shape[-2]
        
        # 1. node -> cell
        cell_features = self.node2cell(V, cells)  # [B, M, C]
        
        # 2. cell -> edge
        edge_flux_left, edge_flux_right = self.flux_aware_update(cell_features, edge_to_cells)
        edge_input = torch.cat([E, edge_flux_left, edge_flux_right], dim=-1)
        edge_embeddings = self.edge_update(edge_input)
        
        edge_size = edge_embeddings.shape[-1] // 2
        edge_embeddings_0 = edge_embeddings[..., :edge_size]
        edge_embeddings_1 = edge_embeddings[..., edge_size:]
              
        col_0 = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_size)
        col_1 = edges[..., 1].unsqueeze(-1).repeat(1, 1, edge_size)
                  
        # 3. edge -> node
        edge_mean_0 = scatter_mean(
            edge_embeddings_0,
            col_0,
            dim=-2,
            dim_size=N
        )
        
        edge_mean_1 = scatter_mean(
            edge_embeddings_1,
            col_1,
            dim=-2,
            dim_size=N
        )
        edge_mean = torch.cat([edge_mean_0, edge_mean_1], dim=-1)
                
        node_input = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.node_update(node_input)
        
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