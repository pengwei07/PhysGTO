import torch
import math
import h5py

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def hybrid_edge_sampling(edges, centroids, sample_rate, spatial_weight, top_percent):
    
    device = edges.device
    num_edges = edges.shape[0]
    
    num_splits = math.ceil(1 / sample_rate)
    
    edges_per_split = int(num_edges * sample_rate)
    target_short_per_split = int(edges_per_split * spatial_weight)
    
    edge_distances = torch.norm(centroids[edges[:,0]] - centroids[edges[:,1]], dim=1)
    
    pool_size = int(num_edges * top_percent)
    _, shortest_indices = torch.topk(edge_distances, pool_size, largest=False)
    
    short_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    short_mask.scatter_(0, shortest_indices, True)
    
    del edge_distances, shortest_indices
    torch.cuda.empty_cache()
    
    short_edge_ids = torch.nonzero(short_mask).squeeze()
    long_edge_ids = torch.nonzero(~short_mask).squeeze()
    
    del short_mask
    torch.cuda.empty_cache()
    
    short_edges_total = len(short_edge_ids)
    long_edges_total = len(long_edge_ids)
    
    base_short_per_split = short_edges_total // num_splits
    extra_short_per_split = max(0, target_short_per_split - base_short_per_split)
    
    short_edge_ids = short_edge_ids[torch.randperm(short_edges_total, device=device)]
    
    long_edge_ids = long_edge_ids[torch.randperm(long_edges_total, device=device)]
    
    base_short_splits = []
    for i in range(num_splits):
        start_idx = i * base_short_per_split
        end_idx = (i + 1) * base_short_per_split if i < num_splits - 1 else short_edges_total
        base_short_splits.append(short_edge_ids[start_idx:end_idx])
    
    long_per_split = long_edges_total // num_splits
    long_splits = []
    for i in range(num_splits):
        start_idx = i * long_per_split
        end_idx = (i + 1) * long_per_split if i < num_splits - 1 else long_edges_total
        long_splits.append(long_edge_ids[start_idx:end_idx])
    
    if extra_short_per_split > 0:
        used_short_ids = torch.cat(base_short_splits)
        used_short_mask = torch.zeros(short_edges_total, dtype=torch.bool, device=device)
        
        for used_id in used_short_ids:
            indices = torch.where(short_edge_ids == used_id)[0]
            used_short_mask[indices] = True
        
        unused_short_ids = short_edge_ids[~used_short_mask]
        
        del used_short_mask, used_short_ids
        torch.cuda.empty_cache()
        
        unused_short_ids = unused_short_ids[torch.randperm(len(unused_short_ids), device=device)]
        
        extra_start_indices = torch.arange(0, num_splits * extra_short_per_split, extra_short_per_split, device=device)
        extra_end_indices = torch.minimum(
            extra_start_indices + extra_short_per_split,
            torch.tensor(len(unused_short_ids), device=device)
        )
        
        for i in range(num_splits):
            if extra_start_indices[i] < extra_end_indices[i]:
                extra_ids = unused_short_ids[extra_start_indices[i]:extra_end_indices[i]]
                base_short_splits[i] = torch.cat([base_short_splits[i], extra_ids])
    
    long_indices = torch.randperm(num_splits, device=device)
    
    splits_data = {}
    
    for i in range(num_splits):
        long_idx = long_indices[i].item()
        
        split_edge_ids = torch.cat([base_short_splits[i], long_splits[long_idx]])
        split_edges = edges[split_edge_ids]
        
        unique_vertices, inverse_indices = torch.unique(
            split_edges.reshape(-1), 
            sorted=True, 
            return_inverse=True
        )
        
        new_edges = inverse_indices.reshape(-1, 2)
        
        splits_data[i] = {
            'edges': new_edges.cpu(),
            'new_to_old': unique_vertices.cpu()
        }
    
    del short_edge_ids, long_edge_ids
    del base_short_splits, long_splits, long_indices
    torch.cuda.empty_cache()
    
    return splits_data

def process_for_inference(pressure, centroids, areas, edges, info, sample_rate, spatial_weight, top_percent):
    
    splits_data = hybrid_edge_sampling(
        edges, 
        centroids,
        sample_rate=sample_rate,
        spatial_weight=spatial_weight,
        top_percent=top_percent
    )
    
    processed_splits = {}
    for split_id, split_info in splits_data.items():
        
        new_to_old = split_info['new_to_old']
        
        split_pressure = pressure[new_to_old]
        split_centroids = centroids[new_to_old]
        split_areas = areas[new_to_old]
        split_edges = split_info['edges']
        
        processed_splits[split_id] = {
            'pressure': split_pressure,
            'centroids': split_centroids,
            'areas': split_areas,
            'edges': split_edges,
            'new_to_old': new_to_old
        }
    
    return {
        'splits': processed_splits,
        'info': info.unsqueeze(0),
        "ori_pressre": pressure
    }


def merge_split_results(original_size, splits_results):
    device = None
    for split_result in splits_results:
        if device is None and hasattr(split_result['values'], 'device'):
            device = split_result['values'].device
            break
    
    if device is None:
        device = torch.device('cpu')
    
    merged_results = torch.zeros(original_size, 1, device=device)
    counts = torch.zeros(original_size, 1, device=device)
    
    for split_result in splits_results:
        values = split_result['values'].to(device)
        new_to_old = split_result['new_to_old'].to(device)
        
        merged_results[new_to_old] += values
        counts[new_to_old] += 1
    
    merged_results = merged_results / counts.clamp(min=1)
    
    num_zeros = torch.sum(counts == 0).item()
    
    return merged_results, num_zeros

