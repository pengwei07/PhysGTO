import random
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import math
import scipy.io as sio

import numpy as np
 

def get_data(fn, path, window_length, all_length, dt, delta_t, input_step):

    if window_length == (all_length//delta_t - input_step):
        t = 1
    else:
        t = random.randint(1, all_length//delta_t - window_length - input_step)
        
    Length = t + (window_length + input_step) * delta_t
    
    key = path[1]
    key1 = path[2]
    
    cells = np.load(f"{fn}/data_new_other/{key}_{key1}_cells.npy")[t:Length:delta_t]
    node_pos = np.load(f"{fn}/data_new_other/{key}_{key1}_node_pos.npy")[t:Length:delta_t]
    node_type = np.load(f"{fn}/data_new_other/{key}_{key1}_node_type.npy")[t:Length:delta_t]
    clusters = np.load(f"{fn}/data_new_other/{key}_{key1}_clusters.npy")
    
    state = np.load(f"{fn}/data_new/{key}_{key1}_state.npy")

    Vx = state[t:Length:delta_t,:,0]
    Vy = state[t:Length:delta_t,:,1]
    Ps = state[t:Length:delta_t,:,2]
    Pg = state[t:Length:delta_t,:,3]
    velocity = np.stack([Vx, Vy], axis=-1)
    pressure = np.stack([Ps, Pg], axis=-1)
    
    all_time = np.arange(t, Length, delta_t)
    t = (all_time + 1) * dt
    
    return node_pos, node_type, t, velocity, pressure, cells, clusters

class UAV_Dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 mode="test",
                 all_length = 990,
                 delta_t = 1,
                 input_step = 5,
                 window_length = 10,
                 normalize = True,
                 model_name = None,
                 seed_path = None
                 ):

        super(UAV_Dataset, self).__init__()
        assert mode in ["train", "test"]

        self.model_name = model_name
        
        self.all_length = all_length # <= 600
        self.delta_t = delta_t
        self.input_step = input_step
        
        self.dt = 1/30
                
        if  window_length > (self.all_length//self.delta_t - self.input_step):
            self.window_length = self.all_length//self.delta_t - self.input_step
        else:
            self.window_length = window_length        
        
        self.dataloc = []
        self.mode = mode
        self.do_normalization = normalize

        self.fn = data_path
        
        if seed_path is None:
            seed_path = 'Splits_seed_two'
  
        with open(f"{data_path}/{seed_path}/{mode}.txt", "r") as f:
            for line in f.readlines():
                str_i = line.strip()
                parts = str_i.split("/")
                file_name = parts[0] + '_'  + parts[1] + '.h5'
                self.dataloc.append((file_name, parts[0] + '_'  + parts[1], parts[2])) 
        
    def __len__(self):
        return len(self.dataloc)
    
    def get_singel(self, item):
        
        node_pos, node_type, t, velocity, pressure, cells, clusters = get_data(
            self.fn,
            self.dataloc[item], 
            self.window_length,  
            self.all_length,
            self.dt,
            self.delta_t,
            self.input_step)
            
        cells = torch.from_numpy(cells).long()
        node_type = torch.from_numpy(node_type).long()
        
        node_pos = torch.from_numpy(node_pos).float()
        velocity = torch.from_numpy(velocity).float()
        pressure = torch.from_numpy(pressure).float()
        clusters = torch.from_numpy(clusters).long()
        
        if self.do_normalization:
            state = self.normalize(velocity, pressure)
            node_pos = self.scale_pos(node_pos)
        else:
            state = torch.cat([velocity, pressure], dim=-1)

        t_all = torch.from_numpy(t).float()        
        t_embed = t_all.unsqueeze(-1)
        
        #  for edges
        mesh_edges = self.make_edges(cells)
        
        # for all
        input = {
                'node_pos': node_pos,
                'edges': mesh_edges,
                'state': state,
                'node_type': node_type.unsqueeze(-1),
                't_all': t_embed,
                'cluster': clusters.unsqueeze(0),
                'cells': cells
                }
        
        return input
    
    def make_edges(self, faces):

        unique_edges_all = []
        
        for i in range(faces.shape[0]):
            edges = torch.cat([faces[i, :, :2], faces[i, :, 1:], faces[i, :, ::2]], dim=0)
            
            receivers, _ = torch.min(edges, dim=-1)
            senders, _ = torch.max(edges, dim=-1)
            
            packed_edges = torch.stack([senders, receivers], dim=-1).int() # dim = [edges, 2]
            unique_edges = torch.unique(packed_edges, dim=0)
            
            if self.model_name == "MGN" or self.model_name == "Graphvit":        
                unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0) 
                # dim = [edges, 2]
            
            unique_edges_all.append(unique_edges)
        
        return torch.stack(unique_edges_all, dim=0)
    
    def scale_pos(self, pos):
        
        x_min = -2.5
        x_max = 2.5
        
        y_min = -1.6743
        y_max = 1.5
        
        pos_x_01 = (pos[...,0] - x_min) / (x_max - x_min)
        pos_y_01 = (pos[...,1] - y_min) / (y_max - y_min)
        
        pos_01 = torch.stack((pos_x_01, pos_y_01), dim=-1)
        
        return pos_01
    
    def normalize(self, velocity=None, pressure=None):
        
        u = velocity[...,0]
        v = velocity[...,1]
        
        ps = pressure[...,0]
        pg = pressure[...,1]
        
        u_mean = 0.001260
        u_std =  1.775321
        u_norm = (u - u_mean) / u_std
        
        v_mean = 0.204374
        v_std = 2.046235
        v_norm = (v - v_mean) / v_std
        
        ps_mean = -0.561852
        ps_std = 7.510783
        ps_norm = (ps - ps_mean) / ps_std
        
        pg_mean = 4.866986
        pg_std = 9.844210
        pg_norm = (pg - pg_mean) / pg_std
        
        state = torch.stack((u_norm, v_norm, ps_norm, pg_norm), dim=-1)
        
        return state
    
    def __getitem__(self, item):
        
        input = self.get_singel(item)
        
        return input
    
if __name__ == "__main__":
    dataset = UAV_Dataset(data_path='../../data/UAV_flow/')
    for data in dataset:
        for key in data.keys():
            print(key)
            print(data[key].shape)
        break