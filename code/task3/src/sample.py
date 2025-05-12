import torch
import h5py
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def reconstruct_graph_cuda(edges, sample_rate):
    """使用GPU加速的图重构过程"""
    # 确保在GPU上操作
    if not edges.is_cuda:
        edges = edges.to(device)
    
    # 1. 采样边
    num_edges = edges.shape[0]
    num_sample = int(num_edges * sample_rate)  # 直接使用float值
    
    # 使用GPU生成随机序列
    edge_ids = torch.randperm(num_edges, device=edges.device)[:num_sample]
    sampled_edges = edges[edge_ids]
    
    # 2. 使用GPU进行顶点去重和映射
    unique_vertices, inverse_indices = torch.unique(
        sampled_edges.reshape(-1), 
        sorted=True, 
        return_inverse=True
    ) 
    
    # 3. 直接在GPU上重构边索引
    new_edges = inverse_indices.reshape(-1, 2)
    
    return unique_vertices.cpu(), new_edges.cpu()

def split_edges_for_test(edges, num_splits):
    """在测试模式下将边分成多个部分,并保持原始索引映射"""
    num_edges = edges.shape[0]
    edge_ids = torch.randperm(num_edges, device=edges.device)
    edges = edges[edge_ids]
    
    # print("edges.max(), edges.min():", edges.max(), edges.min())
    
    edges_per_split = num_edges // num_splits
    splits_data = {}
    
    for i in range(num_splits):
        start_idx = i * edges_per_split
        end_idx = start_idx + edges_per_split if i < num_splits - 1 else num_edges
        
        # 获取当前分割的边
        current_edges = edges[start_idx:end_idx]
        
        # 获取当前分割中的唯一顶点
        unique_vertices = torch.unique(current_edges.reshape(-1))
        
        # 创建双向映射
        new_to_old = unique_vertices  # 保存原始索引
        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        
        # 重新索引边
        new_edges = torch.tensor([[old_to_new[v.item()] for v in edge] for edge in current_edges])
        
        splits_data[i] = {
            'new_to_old': new_to_old,  # 新索引到原始索引的映射
            'edges': new_edges,         # 重新编号的边
        }
        
    return splits_data    

def edge_sampling(edges, sample_rate):
    
    num_splits = math.ceil(1 / sample_rate)
    
    """在测试模式下将边分成多个部分,并保持原始索引映射"""
    num_edges = edges.shape[0]
    edge_ids = torch.randperm(num_edges, device=edges.device)
    edges = edges[edge_ids]
    
    # print("edges.max(), edges.min():", edges.max(), edges.min())
    
    edges_per_split = num_edges // num_splits
    splits_data = {}
    
    for i in range(num_splits):
        start_idx = i * edges_per_split
        end_idx = start_idx + edges_per_split if i < num_splits - 1 else num_edges
        
        # 获取当前分割的边
        current_edges = edges[start_idx:end_idx]
        
        # 获取当前分割中的唯一顶点
        unique_vertices = torch.unique(current_edges.reshape(-1))
        
        # 创建双向映射
        new_to_old = unique_vertices  # 保存原始索引
        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        
        # 重新索引边
        new_edges = torch.tensor([[old_to_new[v.item()] for v in edge] for edge in current_edges])
        
        splits_data[i] = {
            'new_to_old': new_to_old,  # 新索引到原始索引的映射
            'edges': new_edges,         # 重新编号的边
        }
    
    return splits_data  
    
def process_for_inference(pressure, centroids, areas, edges, info, sample_rate):
    
    # 获取分割数据，使用混合采样
    splits_data = edge_sampling(
        edges, 
        sample_rate
    )
    
    # 为每个分割准备数据
    processed_splits = {}
    for split_id, split_info in splits_data.items():
        
        new_to_old = split_info['new_to_old']
        
        # 使用原始索引获取对应的数据
        split_pressure = pressure[new_to_old]
        split_centroids = centroids[new_to_old]
        split_areas = areas[new_to_old]
        split_edges = split_info['edges']
        
        processed_splits[split_id] = {
            'pressure': split_pressure,
            'centroids': split_centroids,
            'areas': split_areas,
            'edges': split_edges,
            'new_to_old': new_to_old  # 保存映射关系
        }
    
    return {
        'splits': processed_splits,
        'info': info.unsqueeze(0),
        "ori_pressre": pressure
    }   
    
# def merge_split_results(original_size, splits_results):
    
#     """合并多个分割的结果,使用原始索引映射回去"""
#     # 初始化结果数组和计数数组
#     merged_results = torch.zeros(original_size, 1)  # 假设结果是一维的
#     counts = torch.zeros(original_size, 1)
    
#     # 对每个分割的结果进行合并
#     for split_result in splits_results:
#         values = split_result['values']          # 预测值
#         new_to_old = split_result['new_to_old']  # 索引映射
        
#         # 使用原始索引进行赋值
#         merged_results[new_to_old] += values
#         counts[new_to_old] += 1
    
#     # 处理重叠（取平均）
#     merged_results = merged_results / counts.clamp(min=1)
    
#     num_zeros = torch.sum(merged_results == 0).item()
    
#     return merged_results, num_zeros

def merge_split_results(original_size, splits_results):
    """合并多个分割的结果,使用原始索引映射回去"""
    # 确定device
    device = None
    for split_result in splits_results:
        if device is None and hasattr(split_result['values'], 'device'):
            device = split_result['values'].device
            break
    
    # 如果没有找到device或全是CPU，则默认使用CPU
    if device is None:
        device = torch.device('cpu')
    
    # 初始化结果数组和计数数组，确保在正确的设备上
    merged_results = torch.zeros(original_size, 1, device=device)
    counts = torch.zeros(original_size, 1, device=device)
    
    # 对每个分割的结果进行合并
    for split_result in splits_results:
        values = split_result['values'].to(device)      # 确保预测值在正确的设备上
        new_to_old = split_result['new_to_old'].to(device)  # 确保索引在正确的设备上
        
        # 使用原始索引进行赋值
        merged_results[new_to_old] += values
        counts[new_to_old] += 1
    
    # 处理重叠（取平均）
    merged_results = merged_results / counts.clamp(min=1)
    
    # 检查未预测的节点数量
    num_zeros = torch.sum(counts == 0).item()
    
    return merged_results, num_zeros

'''
# test
idx = 1
data_ = h5py.File("/home/code/car_task/data/ahmed_body_new/ahmed3d_test.hdf5", 'r')
coef = data_['{}/coef'.format(idx)][:]

centroids = coef[:,:3]
edges = data_['{}/edges'.format(idx)][:]

print(centroids.shape, edges.shape)
# (145236, 3) (217503, 2)

centroids = torch.from_numpy(centroids).float()
edges = torch.from_numpy(edges).long()

sample_rate = 0.1
spatial_weight = 0.4
top_percent = 0.2

sampled_vertices_idx_in, new_edges_in = reconstruct_graph_cuda(edges, sample_rate)

print(len(sampled_vertices_idx_in), new_edges_in.shape)
# 39220 21750

'''