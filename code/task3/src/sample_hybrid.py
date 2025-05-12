import torch
import math
import h5py

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def hybrid_edge_sampling(edges, centroids, sample_rate, spatial_weight, top_percent):
    
    device = edges.device
    num_edges = edges.shape[0]
    
    # 计算需要的分割数
    num_splits = math.ceil(1 / sample_rate)
    
    # 计算每个分割需要的总边数
    edges_per_split = int(num_edges * sample_rate)
    target_short_per_split = int(edges_per_split * spatial_weight)
    
    # 计算边距离 - 向量化操作
    edge_distances = torch.norm(centroids[edges[:,0]] - centroids[edges[:,1]], dim=1)
    
    # 分离短边和长边
    pool_size = int(num_edges * top_percent)
    _, shortest_indices = torch.topk(edge_distances, pool_size, largest=False)
    
    # 创建短边掩码并获取短边和长边索引 - 向量化操作
    short_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    short_mask.scatter_(0, shortest_indices, True)
    
    # 释放不再需要的变量
    del edge_distances, shortest_indices
    torch.cuda.empty_cache()
    
    # 获取短边和长边索引 - 向量化操作
    short_edge_ids = torch.nonzero(short_mask).squeeze()
    long_edge_ids = torch.nonzero(~short_mask).squeeze()
    
    # 释放掩码
    del short_mask
    torch.cuda.empty_cache()
    
    short_edges_total = len(short_edge_ids)
    long_edges_total = len(long_edge_ids)
    
    # 计算短边的基础划分和额外采样
    base_short_per_split = short_edges_total // num_splits
    extra_short_per_split = max(0, target_short_per_split - base_short_per_split)
    
    # 随机打乱短边和长边 - 向量化操作
    # torch.manual_seed(42)
    short_edge_ids = short_edge_ids[torch.randperm(short_edges_total, device=device)]
    
    # torch.manual_seed(142)
    long_edge_ids = long_edge_ids[torch.randperm(long_edges_total, device=device)]
    
    # 优化: 使用切片操作代替循环创建基础分割
    # 1. 短边基础划分
    base_short_splits = []
    for i in range(num_splits):
        start_idx = i * base_short_per_split
        end_idx = (i + 1) * base_short_per_split if i < num_splits - 1 else short_edges_total
        base_short_splits.append(short_edge_ids[start_idx:end_idx])
    
    # 2. 长边均等划分
    long_per_split = long_edges_total // num_splits
    long_splits = []
    for i in range(num_splits):
        start_idx = i * long_per_split
        end_idx = (i + 1) * long_per_split if i < num_splits - 1 else long_edges_total
        long_splits.append(long_edge_ids[start_idx:end_idx])
    
    # 3. 为短边额外采样 - 优化掩码创建过程
    if extra_short_per_split > 0:
        # 创建已使用短边的掩码 - 优化为批量操作
        used_short_ids = torch.cat(base_short_splits)
        used_short_mask = torch.zeros(short_edges_total, dtype=torch.bool, device=device)
        
        # 找出每个短边ID在原始short_edge_ids中的位置
        for used_id in used_short_ids:
            indices = torch.where(short_edge_ids == used_id)[0]
            used_short_mask[indices] = True
        
        # 获取未使用的短边
        unused_short_ids = short_edge_ids[~used_short_mask]
        
        # 释放不再需要的掩码
        del used_short_mask, used_short_ids
        torch.cuda.empty_cache()
        
        # 随机打乱未使用的短边
        # torch.manual_seed(100)
        unused_short_ids = unused_short_ids[torch.randperm(len(unused_short_ids), device=device)]
        
        # 计算每个分割的额外短边起止索引
        extra_start_indices = torch.arange(0, num_splits * extra_short_per_split, extra_short_per_split, device=device)
        extra_end_indices = torch.minimum(
            extra_start_indices + extra_short_per_split,
            torch.tensor(len(unused_short_ids), device=device)
        )
        
        # 为每个分割提取额外短边
        for i in range(num_splits):
            if extra_start_indices[i] < extra_end_indices[i]:
                extra_ids = unused_short_ids[extra_start_indices[i]:extra_end_indices[i]]
                base_short_splits[i] = torch.cat([base_short_splits[i], extra_ids])
    
    # 4. 随机组合短边和长边分割
    # torch.manual_seed(242)
    long_indices = torch.randperm(num_splits, device=device)
    
    # 创建最终的splits字典 - 向量化操作创建边组合
    splits_data = {}
    
    for i in range(num_splits):
        # 随机选择一个长边分割与当前短边分割组合
        long_idx = long_indices[i].item()
        
        # 合并边 - 向量化操作
        split_edge_ids = torch.cat([base_short_splits[i], long_splits[long_idx]])
        split_edges = edges[split_edge_ids]
        
        # 重新映射边索引 - 使用torch.unique的向量化操作
        unique_vertices, inverse_indices = torch.unique(
            split_edges.reshape(-1), 
            sorted=True, 
            return_inverse=True
        )
        
        # 重新映射边的索引
        new_edges = inverse_indices.reshape(-1, 2)
        
        # 存储分割信息
        splits_data[i] = {
            'edges': new_edges.cpu(),
            'new_to_old': unique_vertices.cpu()
        }
    
    # 释放内存
    del short_edge_ids, long_edge_ids
    del base_short_splits, long_splits, long_indices
    torch.cuda.empty_cache()
    
    return splits_data

def process_for_inference(pressure, centroids, areas, edges, info, sample_rate, spatial_weight, top_percent):
    """使用混合采样策略进行推理前的数据处理"""
    
    # 获取分割数据，使用混合采样
    splits_data = hybrid_edge_sampling(
        edges, 
        centroids,
        sample_rate=sample_rate,
        spatial_weight=spatial_weight,
        top_percent=top_percent
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
    
#     # 检查未预测的节点数量
#     num_zeros = torch.sum(counts == 0).item()
    
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


# # test
# idx = 1
# data_ = h5py.File("/home/code/car_task/data/ahmed_body_new/ahmed3d_test.hdf5", 'r')

# coef = data_['{}/coef'.format(idx)][:]

# centroids = coef[:,:3]
# areas = coef[:,3:]

# edges = data_['{}/edges'.format(idx)][:]
# info = data_['{}/param'.format(idx)][:]
# pressure = data_['{}/data'.format(idx)][:]

# print(centroids.shape, edges.shape)
# # (145236, 3) (217503, 2)

# centroids = torch.from_numpy(centroids).float()
# areas = torch.from_numpy(areas).float()
# edges = torch.from_numpy(edges).long()
# info = torch.from_numpy(info).float()
# pressure = torch.from_numpy(pressure).float()

# sample_rate = 0.1
# spatial_weight = 0.4
# top_percent = 0.2

# infer_data = process_for_inference(pressure, centroids, areas, edges, info, sample_rate, spatial_weight, top_percent)
# # edges, centroids, sample_rate, spatial_weight, top_percent
# # pressure, centroids, areas, edges, info, sample_rate, spatial_weight, top_percent
# processed_splits = infer_data['splits']
# info = infer_data['info']
# ori_pressre = infer_data['ori_pressre']

# print(len(processed_splits))

# processed_results = []
# for i in range(len(processed_splits)):
    
#     split_pressure = processed_splits[i]['pressure']
#     new_to_old = processed_splits[i]['new_to_old']
    
#     processed_results.append({
#         'values': split_pressure,
#         'new_to_old': new_to_old
#     })
        
# original_size = pressure.size(0)
# merged_result, num_zeros = merge_split_results(original_size, processed_results)

# print(merged_result.shape, num_zeros)


# error_1 = merged_result - ori_pressre
# error_2 = merged_result - pressure


# norm_error_1 = torch.mean(torch.norm(error_1, dim=-2) / (torch.norm(ori_pressre, dim=-2)))
# norm_error_2 = torch.mean(torch.norm(error_2, dim=-2) / (torch.norm(pressure, dim=-2)))

# print(f"{norm_error_1:.4e}, {norm_error_2:.4e}")
