
import torch
import torch.nn as nn

import numpy as np
import random
import json
import argparse
from types import SimpleNamespace
from torch.nn.modules.loss import _WeightedLoss

def set_seed(seed: int = 0):    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保PyTorch使用相同的初始化权重
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # 禁用 cudnn 自动优化器，保证确定性



def collate(X):
   """
   对不同长度的feature进行维度对齐和batch处理
   X: list of tuples (centroids, areas, edges, cd)
   """
   # 找到最大长度
   N_max = max([centroids.shape[0] for centroids, areas, edges, cd in X]) 
   E_max = max([edges.shape[0] for centroids, areas, edges, cd in X])
   
   # 用于收集batch数据
   batch_centroids = []
   batch_areas = []
   batch_edges = []
   batch_cds = []
   batch_masks = []
   
   for centroids, areas, edges, cd in X:
       # 获取当前样本的维度
       N, Nc = centroids.shape  # 节点数, 节点特征维度
       Na = areas.shape[-1]     # 面积特征维度
       E, Ne = edges.shape      # 边数, 边特征维度(2)
       
       # Pad节点特征
       padded_centroids = torch.cat([
           centroids,
           torch.zeros(N_max - N + 1, Nc, device=centroids.device)
       ], dim=0)
       
       # Pad面积特征
       padded_areas = torch.cat([
           areas,
           torch.zeros(N_max - N + 1, Na, device=areas.device)
       ], dim=0)
       
       # Pad边特征，用N_max填充无效边
       padded_edges = torch.cat([
           edges,
           N_max * torch.ones(E_max - E + 1, Ne, device=edges.device)
       ], dim=0)
       
       # 创建mask标记有效节点
       node_mask = torch.cat([
           torch.ones(N, device=centroids.device),
           torch.zeros(N_max - N + 1, device=centroids.device)
       ], dim=0)
       
       # 收集batch数据
       batch_centroids.append(padded_centroids)
       batch_areas.append(padded_areas)
       batch_edges.append(padded_edges)
       batch_cds.append(cd)
       batch_masks.append(node_mask)
   
   # Stack成batch
   centroids_batch = torch.stack(batch_centroids, dim=0)  # [B, N_max, Nc]
   areas_batch = torch.stack(batch_areas, dim=0)          # [B, N_max, Na]
   edges_batch = torch.stack(batch_edges, dim=0)          # [B, E_max, 2]
   cds_batch = torch.tensor(batch_cds)                    # [B]
   masks_batch = torch.stack(batch_masks, dim=0)          # [B, N_max]
   
   return centroids_batch, areas_batch, edges_batch, cds_batch.unsqueeze(-1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # 初始化 Linear 层的权重
        if m.weight.numel() > 0:  # 确保张量非空
            torch.nn.init.xavier_uniform_(m.weight)
        
        # 检查是否有 bias 并进行初始化
        if m.bias is not None and m.bias.numel() > 0:  # 确保 bias 非空
            m.bias.data.fill_(0.01)
    
    elif isinstance(m, nn.MultiheadAttention):
        # 初始化 MultiheadAttention 的投影权重
        if m.in_proj_weight.numel() > 0:  # 确保张量非空
            torch.nn.init.xavier_uniform_(m.in_proj_weight)
        
        # 检查并初始化投影偏置
        if m.in_proj_bias is not None and m.in_proj_bias.numel() > 0:  # 确保 bias 非空
            m.in_proj_bias.data.fill_(0.01)

        # out_proj 是一个内部的 nn.Linear，初始化其权重
        if m.out_proj.weight.numel() > 0:  # 确保张量非空
            torch.nn.init.xavier_uniform_(m.out_proj.weight)
        
        # 检查并初始化偏置
        if m.out_proj.bias is not None and m.out_proj.bias.numel() > 0:  # 确保 bias 非空
            m.out_proj.bias.data.fill_(0.01)

            
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')  # Change the default config file name if needed

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)  # Load JSON instead of YAML
    
    args = SimpleNamespace(**config)
    
    return args





class RelLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(RelLpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def forward(self, x, y, normalizer=None, component='all'):
        num_examples = x.size()[0]

        if normalizer is not None:
            x = normalizer.decode(x,component=component)
            y = normalizer.decode(y,component=component)

        if component != 'all':
            x = x[..., component]
            y = y[..., component]


        diff_norms = torch.norm(x.reshape(num_examples, -1, x.shape[-1]) - y.reshape(num_examples, -1, x.shape[-1]),
                                self.p, dim=1)  ##N, C
        y_norms = torch.norm(y.reshape(num_examples, -1, y.shape[-1]), self.p, dim=1) + 1e-8

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)  ## deprecated
            else:
                return torch.sum(torch.mean(diff_norms / y_norms, dim=-1))  #### go this branch
        else:
            return torch.sum(diff_norms / y_norms, dim=-1)
        
        
def get_l2_loss(output, target):
    # output.dim = (batch, N, c)
    # target.dim = (batch, N, c)   
    # output = output.squeeze(-1) 
    # target = target.squeeze(-1) 

    error = output - target
    
    norm_error_sample = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-6)
    if norm_error_sample.shape[-1] == 1:
        norm_error_channnel = norm_error_sample.squeeze(-1) 
    else:
        norm_error_channnel = torch.mean(norm_error_sample, dim=-1)
    
    norm_error_batch = torch.mean(norm_error_channnel)
    
    return norm_error_batch