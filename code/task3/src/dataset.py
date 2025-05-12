from torch.utils.data import Dataset
from functools import partial
import h5py
import torch
import os
import numpy as np
import pandas as pd

# from .sample_hybrid import hybrid_edge_sampling, hybrid_process_for_inference
from .sample import reconstruct_graph_cuda, process_for_inference

class Car_Dataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 mode, 
                 normalize, 
                 if_sample, 
                 sample_rate = 0.1,
                 spatial_weight = 0.3,
                 top_percent = 0.2,
                 normalize_way = "z_score"
                 ):
                
        self.data_dir = data_dir + '/ahmed3d_' + mode + '.hdf5'
        self.mode = mode
        self.data_loc = []
        
        length = len(h5py.File(self.data_dir, 'r').keys())

        if self.mode == "train":
            for i in range(legnth):
                self.data_loc.append(i) 
        elif self.mode == "test":
            for i in range(length):
                self.data_loc.append(i) 
                        
        self.normalize = normalize
        self.if_sample = if_sample
        
        self.sample_rate = sample_rate
        self.spatial_weight = spatial_weight
        self.top_percent = top_percent
        
        self.normalize_way = normalize_way
        self.num_samples = len(self.data_loc)
        
        # 计算分割数量
        self.num_splits = int(1 / sample_rate) if sample_rate > 0 else 1

        def open_hdf5_file_irregular(path, idx):

            data_ = h5py.File(path, 'r')
            coef = data_['{}/coef'.format(idx)][:]
            
            centroids = coef[:,:3]
            areas = coef[:,3:]
            
            edges = data_['{}/edges'.format(idx)][:]
            info = data_['{}/param'.format(idx)][:]
            pressure = data_['{}/data'.format(idx)][:]
            
            return pressure, centroids, areas, edges, info

        self.data_files = partial(open_hdf5_file_irregular, self.data_dir)

    def __len__(self):
        
        return self.num_samples

    def input_normalizer(self, centroids, areas):
        # 转换numpy数组为CUDA tensor
        centroids_mean = torch.tensor([0.2445, 0.2445, 0.2445], 
                                    device=centroids.device).reshape(1, 3)
        centroids_std = torch.tensor([0.4087, 0.3790, 0.3318], 
                                device=centroids.device).reshape(1, 3)
        
        areas_mean = torch.tensor(0.1230, device=areas.device)
        areas_std = torch.tensor(0.1544, device=areas.device)
        
        if self.normalize_way == "z_score":
            centroids = (centroids - centroids_mean) / centroids_std
            areas = (areas - areas_mean) / areas_std
        
        return centroids, areas

    def sample(self, info, pressure, centroids, areas, edges):
        
        # sample for edges & nodes
        sampled_vertices_idx, new_edges = reconstruct_graph_cuda(edges, self.sample_rate)

        pressure_in = pressure[sampled_vertices_idx]
        centroids_in = centroids[sampled_vertices_idx]
        areas_in = areas[sampled_vertices_idx]
        edges_in = new_edges
        
        processed_split = {
            'pressure': pressure_in,
            'centroids': centroids_in,
            'areas': areas_in,
            'edges': edges_in
        }
        
        return processed_split, info.unsqueeze(0)
        
    def __getitem__(self, idx):
        # get data for each sample
        pressure, centroids, areas, edges, info = self.data_files(idx)
        
        info = torch.from_numpy(info).float()
        pressure = torch.from_numpy(pressure).float()
        centroids = torch.from_numpy(centroids).float()
        areas = torch.from_numpy(areas).float()
        edges = torch.from_numpy(edges).long()
        
        if self.normalize:
            centroids, areas = self.input_normalizer(centroids, areas)
        
        if self.mode == "train":
            
            if self.if_sample:
                
                # for inputs
                return self.sample(info, pressure, centroids, areas, edges)
            
            else:
                
                data = {
                    'pressure': pressure,
                    'centroids': centroids,
                    'areas': areas,
                    'edges': edges,
                }
                
                return data, info.unsqueeze(0)
        
        elif self.mode == "test":

            if self.if_sample:
                # for inputs
                return self.sample(info, pressure, centroids, areas, edges)
            else:
                return process_for_inference(pressure, centroids, areas, edges, info, self.sample_rate), idx
'''
# test
data_dir = '../data/ahmed_body_new'

dataset = Car_Dataset(data_dir, 'test', normalize=True, if_sample=True, sample_rate = 0.2)

print(len(dataset))

# 获取数据
data = dataset[0]

if isinstance(data, dict):
    original_size = data['original_size']
    splits = data['splits']
    
    pressure_ori = data['ori_pressre']
    
    # 存储每个分割的处理结果
    processed_results = []
    
    # 处理每个分割
    for split_id, split_data in splits.items():
        # 获取这个分割的数据
        pressure = split_data['pressure']
        centroids = split_data['centroids']
        areas = split_data['areas']
        edges = split_data['edges']
        new_to_old = split_data['new_to_old']
        
        # print(pressure.shape, edges.shape)
        
        # 使用模型进行预测
        # predictions = model(pressure, centroids, areas, edges)
        
        # 保存结果和映射关系
        processed_results.append({
            'values': pressure,
            'new_to_old': new_to_old
        })
    
    # 合并所有分割的结果
    final_result, num_zeros = merge_split_results(original_size, processed_results)
    
print("original_size", original_size, "num_zeros", num_zeros)
print("final_result", final_result.shape, "pressure_ori", pressure_ori.shape)

error = final_result - pressure_ori
l2_loss = torch.norm(error, dim=0) / (torch.norm(pressure_ori, dim=0) + 1e-6)

print(l2_loss)
'''