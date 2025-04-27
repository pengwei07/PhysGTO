import torch

def get_mesh_edges(faces):
        
    edges = torch.cat([
        faces[..., :2],
        faces[..., 1:],
        faces[..., ::2]
    ], dim=0)
    
    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)
    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    
    # unique_edges, inverse_indices = torch.unique(packed_edges, dim=0, return_inverse=True)
    unique_edges = torch.unique(packed_edges, dim=0)
    
    return unique_edges
    
    
# def edges_and_mappings(faces):
#     """生成边的连接关系和映射
#     Args:
#         faces: [M, 3] 三角形单元的顶点索引
#     Returns:
#         edges: [E, 2] 唯一边的顶点索引对
#         edge_to_cells: [E, 2] 每条边相邻的单元索引 (-1表示边界)
#     """
#     M = faces.shape[0]
#     device = faces.device
    
#     # 提取所有边
#     edges = torch.cat([
#         faces[:, [0, 1]],  # 边: 顶点0->1
#         faces[:, [1, 2]],  # 边: 顶点1->2
#         faces[:, [2, 0]]   # 边: 顶点2->0
#     ], dim=0)  # [3M, 2]
    
#     # 标准化边方向（小索引在前）
#     # edges_sorted, _ = torch.sort(edges, dim=1)
#     receivers, _ = torch.min(edges, dim=-1)
#     senders, _ = torch.max(edges, dim=-1)
#     packed_edges = torch.stack([senders, receivers], dim=-1).int()
    
#     # 获取唯一边和反向映射
#     unique_edges, inverse_indices = torch.unique(packed_edges, dim=0, return_inverse=True)
    
#     # 建立边到单元的映射
#     edge_to_cells = torch.full((unique_edges.shape[0], 2), -1, dtype=torch.long, device=device)
#     cell_indices = torch.arange(M, device=device).repeat_interleave(3)
    
#     # 记录每条边相邻的单元
#     for i in range(len(inverse_indices)):
#         edge_idx = inverse_indices[i]
#         cell_idx = cell_indices[i]
#         if edge_to_cells[edge_idx, 0] == -1:
#             edge_to_cells[edge_idx, 0] = cell_idx
#         elif edge_to_cells[edge_idx, 1] == -1 and edge_to_cells[edge_idx, 0] != cell_idx:
#             edge_to_cells[edge_idx, 1] = cell_idx
    
#     return unique_edges, edge_to_cells

def edges_and_mappings(faces):
    
    device = faces.device
    M, _ = faces.shape
    
    edges = torch.cat([
        faces[..., :2],
        faces[..., 1:],
        faces[..., ::2]
    ], dim=0)
    
    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)
    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    
    unique_edges, inverse_indices = torch.unique(packed_edges, dim=0, return_inverse=True)
    edge_to_cells = torch.full((unique_edges.shape[0], 2), -1, dtype=torch.long, device=device)
    cell_indices = torch.arange(M, device=device).repeat_interleave(3)
    
    # 修正掩码维度
    mask_first = (edge_to_cells[inverse_indices, 0] == -1)
    mask_second = (edge_to_cells[inverse_indices, 1] == -1) & (edge_to_cells[inverse_indices, 0] != cell_indices)
    
    # 使用布尔索引更新
    edge_to_cells[inverse_indices[mask_first], 0] = cell_indices[mask_first]
    edge_to_cells[inverse_indices[mask_second], 1] = cell_indices[mask_second]
    
    return unique_edges, edge_to_cells

def precompute_geometry_info(Elements):
    
    # 如果是四边形网格
    if Elements.shape[-1] == 4:
        # 将每个四边形分解为两个三角形
        Elements = torch.cat([
            # 第一个三角形 (v0, v1, v2)
            torch.stack([Elements[:, 0], Elements[:, 1], Elements[:, 2]], dim=1),
            # 第二个三角形 (v0, v2, v3)
            torch.stack([Elements[:, 0], Elements[:, 2], Elements[:, 3]], dim=1)
        ], dim=0)
        
    
    # 生成边的连接关系
    edges, edge_to_cells = edges_and_mappings(Elements)
    
    
    return Elements, edges, edge_to_cells


