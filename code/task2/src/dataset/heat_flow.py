import random
from torch.utils.data import Dataset
import torch
import numpy as np 
import h5py
import os

class Heat_Flow_Dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 mode="test",
                 all_length = 501,
                 delta_t = 1,
                 input_step = 5,
                 window_length = 50,
                 normalize = True,
                 model_name = None,
                 ):

        super(Heat_Flow_Dataset, self).__init__()
        assert mode in ["train", "test"]

        self.all_length = all_length 
        self.delta_t = delta_t
        self.input_step = input_step
        self.dt = 0.02
            
        if  window_length > (self.all_length//self.delta_t - self.input_step):
            self.window_length = self.all_length//self.delta_t - self.input_step
        else:
            self.window_length = window_length      
        
        self.dataloc = []
        self.mode = mode
        self.do_normalization = normalize
        self.model_name = model_name

        self.fn = data_path
        
        with open(f"{data_path}/{mode}.txt", "r") as f:
            for line in f.readlines():
                self.dataloc.append(line.strip()) 
        
    def __len__(self):
        return len(self.dataloc)
    
    def get_data(self, path, window_length, all_length, delta_t, input_step, dt):
        if window_length == (all_length//delta_t - input_step):
            t = 0
        else:
            t = random.randint(0, all_length//delta_t - window_length - input_step)
        t = t + 25
        
        Length = t + (window_length + input_step) * delta_t
        
        with h5py.File(path, 'r') as f:
            
            cells = f["cells"][:].copy() # (8569, 3)
            node_pos = f["node_pos"][:].copy()
            node_type = f["node_type"][:].copy()
            
            u = f['u'][t:Length:delta_t].copy()
            v = f['v'][t:Length:delta_t].copy()
            p = f['p'][t:Length:delta_t].copy()
            T = f['T'][t:Length:delta_t].copy() - 293.15
            time = f['t'][t:Length:delta_t,0].copy()
            
            cond = np.array(f['conditions']) # conditions (5,)
        
    
        # node_pos, node_type, time, u, v, p, T, cond
        return node_pos, node_type, time, u, v, p, T, cond, cells

    def make_edges(self, faces):

        edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0)
        
        receivers, _ = torch.min(edges, dim=-1)
        senders, _ = torch.max(edges, dim=-1)
        
        packed_edges = torch.stack([senders, receivers], dim=-1).int() # dim = [edges, 2]
        unique_edges = torch.unique(packed_edges, dim=0)

        if self.model_name == "MGN" or self.model_name == "Graphvit":        
            unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0)
            
        return unique_edges
    

    def get_singel(self, item):
        # get data from the h5 file
        node_pos, node_type, time, u, v, p, T, cond, cells = self.get_data(
            self.dataloc[item], 
            self.window_length,  
            self.all_length,
            self.delta_t,
            self.input_step,
            self.dt
            )
        
        # 1.
        cells = torch.from_numpy(cells).long()
        edges = self.make_edges(cells)
        
        # 2.
        t_all = torch.from_numpy(time).float()
        t_embed = t_all.unsqueeze(-1)
            
        # 3.    
        cond = torch.from_numpy(cond).float()
        cond = cond.reshape(1, -1)
        cond = self.cond_scale(cond)

        # 4        
        node_type = torch.from_numpy(node_type).long()
        node_pos = torch.from_numpy(node_pos).float()
        
        # 5.        
        u = torch.from_numpy(u).float()
        v = torch.from_numpy(v).float()
        p = torch.from_numpy(p).float()
        T = torch.from_numpy(T).float()
        
        state = torch.stack((u,v,p,T),dim=-1)

        if self.do_normalization:
            node_pos = self.scale_pos(node_pos)
            state = self.normalize(state)
            
        input = {'node_pos': node_pos,
                  'edges': edges,
                  'state': state,
                  'node_type': node_type.unsqueeze(-1),
                  't_all': t_embed,
                  'cells': cells
                }
        
        return input

    def normalize(self, state):

        u_mean = 0.249682
        u_std = 0.256363
        
        v_mean = -1.7705e-05
        v_std = 0.107197
        
        p_mean = 0.031665
        p_std = 0.095736
        
        T_mean = 13.689210
        T_std = 20.861255
        
        u_norm = (state[...,0] - u_mean) / u_std
        v_norm = (state[...,1] - v_mean) / v_std
        p_norm = (state[...,2] - p_mean) / p_std
        T_norm = (state[...,3] - T_mean) / T_std
        
        state_norm = torch.stack((u_norm, v_norm, p_norm, T_norm),dim=-1)
        
        return state_norm
    
    def cond_scale(self, cond):
        
        # cond.shape = [1, 5]

        maxx = torch.tensor([0.5992, 89.9287, 0.6998, 0.0800, 89.8434]).to(cond.device)
        minn = torch.tensor([0.1008, 40.0213, 0.3502, 0.0400, -44.9771]).to(cond.device)
        
        new_cond = (cond - minn.reshape(-1,5)) / (maxx.reshape(-1,5) - minn.reshape(-1,5))
        
        return new_cond
    
    def scale_pos(self, node_pos):
        
        # for max-min
        x_max = 3.0
        x_min = 0.0
        
        y_max = 1.0
        y_min = 0.0
        
        x_norm = (node_pos[...,0] - x_min) / (x_max - x_min)
        y_norm = (node_pos[...,1] - y_min) / (y_max - y_min)
        
        node_pos_new = torch.stack((x_norm, y_norm), dim=-1)
        
        return node_pos_new
    
    def __getitem__(self, item):
        
        input = self.get_singel(item)
        
        return input
    
# 740
# u (1903566122,)
# v (1903566122,)
# p (1903566122,)
# T (1903566122,)
# Overall statistics:
# 'u': {'mean': 0.24936437526477268, 'std': 0.25447829465490124}, 
# 'v': {'mean': -6.931294564716289e-05, 'std': 0.10622165020989414}, 
# 'p': {'mean': 0.10124105147008897, 'std': 0.7780049913243584}, 
# 'T': {'mean': 13.402454989872387, 'std': 20.76997677678825}}

# in_vel avg: 0.33651228456009474, std: 0.1445642226208701, max: 0.5992310065459722, min: 0.10075508639008543
# delta_T avg: 65.63247622931654, std: 14.32337403752817, max: 89.92875216167825, min: 40.021296789068785
# delta_L avg: 0.5275004147139768, std: 0.10212364914152541, max: 0.6998424436414363, min: 0.3501883203778883
# delta_R avg: 0.05962060909697847, std: 0.01180567350327386, max: 0.07999064775189044, min: 0.040045787210197414
# delta_A avg: 16.05954889714331, std: 29.632809142847595, max: 89.84343632171823, min: -44.97714049512132

# in_vel.append(conditions[0])
# delta_T.append(conditions[1])
# delta_L.append(conditions[2])
# delta_R.append(conditions[3])
# delta_A.append(conditions[4])
if __name__ == "__main__":
    dataset = Heat_Flow_Dataset(data_path = "../../data/Heat_Transfer/", mode="test")
    for data in dataset:
        for key in data.keys():
            print(key)
            print(data[key].shape)
        print(torch.min(data['node_type']), torch.max(data['node_type']))
        break