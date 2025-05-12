
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
    


def collate(X):
   """
   X: list of tuples (centroids, areas, edges, cd)
   """
   N_max = max([centroids.shape[0] for centroids, areas, edges, cd in X]) 
   E_max = max([edges.shape[0] for centroids, areas, edges, cd in X])
   
   batch_centroids = []
   batch_areas = []
   batch_edges = []
   batch_cds = []
   batch_masks = []
   
   for centroids, areas, edges, cd in X:
       N, Nc = centroids.shape
       Na = areas.shape[-1]
       E, Ne = edges.shape
       
       padded_centroids = torch.cat([
           centroids,
           torch.zeros(N_max - N + 1, Nc, device=centroids.device)
       ], dim=0)
       
       padded_areas = torch.cat([
           areas,
           torch.zeros(N_max - N + 1, Na, device=areas.device)
       ], dim=0)
       
       padded_edges = torch.cat([
           edges,
           N_max * torch.ones(E_max - E + 1, Ne, device=edges.device)
       ], dim=0)
       
       node_mask = torch.cat([
           torch.ones(N, device=centroids.device),
           torch.zeros(N_max - N + 1, device=centroids.device)
       ], dim=0)
       
       batch_centroids.append(padded_centroids)
       batch_areas.append(padded_areas)
       batch_edges.append(padded_edges)
       batch_cds.append(cd)
       batch_masks.append(node_mask)
   
   centroids_batch = torch.stack(batch_centroids, dim=0)  # [B, N_max, Nc]
   areas_batch = torch.stack(batch_areas, dim=0)          # [B, N_max, Na]
   edges_batch = torch.stack(batch_edges, dim=0)          # [B, E_max, 2]
   cds_batch = torch.tensor(batch_cds)                    # [B]
   masks_batch = torch.stack(batch_masks, dim=0)          # [B, N_max]
   
   return centroids_batch, areas_batch, edges_batch, cds_batch.unsqueeze(-1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None and m.bias.numel() > 0:
            m.bias.data.fill_(0.01)
    
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.in_proj_weight)
        
        if m.in_proj_bias is not None and m.in_proj_bias.numel() > 0:
            m.in_proj_bias.data.fill_(0.01)

        if m.out_proj.weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.out_proj.weight)
        
        if m.out_proj.bias is not None and m.out_proj.bias.numel() > 0:
            m.out_proj.bias.data.fill_(0.01)

            
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    args = SimpleNamespace(**config)
    
    return args





class RelLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(RelLpLoss, self).__init__()

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