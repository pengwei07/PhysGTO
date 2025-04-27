from torch.utils.data import Dataset
from functools import partial
import h5py
import torch
import os
import numpy as np
import pandas as pd 
     
from .sample import reconstruct_graph_cuda, process_for_inference

class Car_Dataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 mode, 
                 normalize, 
                 if_sample, 
                 sample_rate = 0.5, 
                 normalize_way = "z_score"):
                
        self.data_dir = data_dir
        self.mode = mode
        self.data_loc = []
        with open(f"/home/code/car_task_v1/drivaernet/data/split_ori/{mode}_design_ids.txt", "r") as f:
            for line in f.readlines():
                self.data_loc.append(line.strip()) 
                        
        self.normalize = normalize
        self.if_sample = if_sample
        self.sample_rate = sample_rate
        self.normalize_way = normalize_way
        self.num_samples = len(self.data_loc)
        
        # 计算分割数量
        self.num_splits = int(1 / sample_rate) if sample_rate > 0 else 1

        cd = pd.read_csv("/home/code/car_task_v1/drivaernet/data/AeroCoefficients_DrivAerNet_FilteredCorrected.csv")
        # cd_dict = dict(zip(cd['Design'], cd['Average Cd']))
        cd_dict = dict(zip(cd['Design'], zip(cd['Average Cd'], cd['Average Cl'], cd['Average Cl_f'], cd['Average Cl_r'])))
        
        def open_hdf5_file_irregular(path, data_loc, idx):

            item = data_loc[idx]
            
            data_ = h5py.File(path, 'r')
            data_i = data_[item]
            
            centroids = data_i['centroids'][:]
            areas = data_i['areas'][:]
            edges = data_i['edges'][:]
            pressure = data_i['pressure'][:]
                        
            cd_tensor = torch.tensor(np.array([cd_dict[item]])).float().reshape(-1)
            
            return pressure, centroids, areas, edges, cd_tensor, item

        self.data_files = partial(open_hdf5_file_irregular, self.data_dir, self.data_loc)

    def __len__(self):
        
        return self.num_samples

    def input_normalizer(self, centroids, areas):
        # 转换numpy数组为CUDA tensor
        centroids_mean = torch.tensor([1.5687, -0.0130, 0.4774], 
                                    device=centroids.device).reshape(1, 3)
        centroids_std = torch.tensor([1.3295, 0.6432, 0.3770], 
                                device=centroids.device).reshape(1, 3)
        
        areas_mean = torch.tensor(0.0001, device=areas.device)
        areas_std = torch.tensor(0.000115, device=areas.device)
        
        if self.normalize_way == "z_score":
            centroids = (centroids - centroids_mean) / centroids_std
            areas = (areas - areas_mean) / areas_std
        
        elif self.normalize_way == "min_max":
            centroids_minn = torch.tensor([-1.1514, -1.0217, 0.0000], 
                                        device=centroids.device).reshape(1, 3)
            centroids_maxx = torch.tensor([4.0932, 1.0216, 1.7608], 
                                        device=centroids.device).reshape(1, 3)
            
            areas_minn = torch.tensor(0.0000, device=areas.device)
            areas_maxx = torch.tensor(0.0019, device=areas.device)
            
            centroids = (centroids - centroids_minn) / (centroids_maxx - centroids_minn)
            areas = (areas - areas_minn) / (areas_maxx - areas_minn)
        
        return centroids, areas
        
    def sample(self, pressure, centroids, areas, edges):
        
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
        
        return processed_split
    
    def __getitem__(self, idx):
        pressure, centroids, areas, edges, cd, item = self.data_files(idx)
        
        # 转移到GPU
        pressure = torch.from_numpy(pressure).float().unsqueeze(-1)
        centroids = torch.from_numpy(centroids).float()
        areas = torch.from_numpy(areas).float().unsqueeze(-1)
        edges = torch.from_numpy(edges).long()
        
        if self.normalize:
            centroids, areas = self.input_normalizer(centroids, areas)
        
        if self.mode == "train":
            
            if self.if_sample:
                return self.sample(pressure, centroids, areas, edges), cd.unsqueeze(0)
            else:
                data = {
                    'pressure': pressure,
                    'centroids': centroids,
                    'areas': areas,
                    'edges': edges,
                }
            
            return data, cd.unsqueeze(0)
        
        elif self.mode == "test":
            
            if self.if_sample:

                return self.sample(pressure, centroids, areas, edges), cd.unsqueeze(0)
                
            else:
                
                return process_for_inference(pressure, centroids, areas, edges, self.sample_rate, cd), item




# # test
# data_dir = '../data/DrivAerNet_dataset.h5'

# dataset = Car_Dataset(data_dir, 'test', normalize=True, if_sample=True, sample_rate = 0.2)

# print(len(dataset))

# # 获取数据
# data = dataset[0]

# # if isinstance(data, dict):

# original_size = data['original_size']
# splits = data['splits']

# pressure_ori = data['ori_pressre']

# # 存储每个分割的处理结果
# processed_results = []

# # 处理每个分割
# for split_id, split_data in splits.items():
#     # 获取这个分割的数据
#     pressure = split_data['pressure']
#     centroids = split_data['centroids']
#     areas = split_data['areas']
#     edges = split_data['edges']
#     new_to_old = split_data['new_to_old']

#     print(pressure.shape, edges.shape, new_to_old.shape)

#     # 使用模型进行预测
#     # predictions = model(pressure, centroids, areas, edges)

#     # 保存结果和映射关系
#     processed_results.append({
#         'values': pressure.unsqueeze(-1),
#         'new_to_old': new_to_old
#     })
    
# # 合并所有分割的结果
# final_result, num_zeros = merge_split_results(original_size, processed_results)
    
# print("original_size", original_size, "num_zeros", num_zeros)
# print("final_result", final_result.shape, "pressure_ori", pressure_ori.shape)

# error = final_result - pressure_ori
# l2_loss = torch.norm(error, dim=0) / (torch.norm(pressure_ori, dim=0) + 1e-6)
# print(l2_loss)