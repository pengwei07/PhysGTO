import torch
import h5py
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def reconstruct_graph_cuda(edges, sample_rate):
    if not edges.is_cuda:
        edges = edges.to(device)
    
    num_edges = edges.shape[0]
    num_sample = int(num_edges * sample_rate)
    
    edge_ids = torch.randperm(num_edges, device=edges.device)[:num_sample]
    sampled_edges = edges[edge_ids]
    
    unique_vertices, inverse_indices = torch.unique(
        sampled_edges.reshape(-1), 
        sorted=True, 
        return_inverse=True
    ) 
    
    new_edges = inverse_indices.reshape(-1, 2)
    
    return unique_vertices.cpu(), new_edges.cpu()

def split_edges_for_test(edges, num_splits):
    num_edges = edges.shape[0]
    edge_ids = torch.randperm(num_edges, device=edges.device)
    edges = edges[edge_ids]
    
    
    edges_per_split = num_edges // num_splits
    splits_data = {}
    
    for i in range(num_splits):
        start_idx = i * edges_per_split
        end_idx = start_idx + edges_per_split if i < num_splits - 1 else num_edges
        
        current_edges = edges[start_idx:end_idx]
        
        unique_vertices = torch.unique(current_edges.reshape(-1))
        
        new_to_old = unique_vertices
        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        
        new_edges = torch.tensor([[old_to_new[v.item()] for v in edge] for edge in current_edges])
        
        splits_data[i] = {
            'new_to_old': new_to_old,
            'edges': new_edges,
        }
        
    return splits_data    

def edge_sampling(edges, sample_rate):
    
    num_splits = math.ceil(1 / sample_rate)
    
    num_edges = edges.shape[0]
    edge_ids = torch.randperm(num_edges, device=edges.device)
    edges = edges[edge_ids]
    
    
    edges_per_split = num_edges // num_splits
    splits_data = {}
    
    for i in range(num_splits):
        start_idx = i * edges_per_split
        end_idx = start_idx + edges_per_split if i < num_splits - 1 else num_edges
        
        current_edges = edges[start_idx:end_idx]
        
        unique_vertices = torch.unique(current_edges.reshape(-1))
        
        new_to_old = unique_vertices
        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        
        new_edges = torch.tensor([[old_to_new[v.item()] for v in edge] for edge in current_edges])
        
        splits_data[i] = {
            'new_to_old': new_to_old,
            'edges': new_edges,
        }
    
    return splits_data  
    
def process_for_inference(pressure, centroids, areas, edges, info, sample_rate):
    
    splits_data = edge_sampling(
        edges, 
        sample_rate
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