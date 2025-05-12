
import torch
import torch.nn as nn

import numpy as np
import random
import json
import argparse
from types import SimpleNamespace


def set_seed(seed: int = 0):    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def collate(X):
    """ Convoluted function to stack simulations together in a batch. Basically, we add ghost nodes
    and ghost edges so that each sim has the same dim. This is useless when batchsize=1 though..."""
    
    # node_pos torch.Size([1, 1954, 2])
    # edges torch.Size([1, 5628, 2])
    # state torch.Size([1, 6, 1954, 3])
    # node_type torch.Size([1, 1954, 1])
    # t_all torch.Size([1, 6, 1])
        
    N_max = max([x["node_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])
    cells_max = max([x["cells"].shape[-2] for x in X])
    for batch, x in enumerate(X):
        # This step add fantom nodes to reach N_max + 1 nodes
        
        key = "state"
        tensor = x[key]
        T, N, S = tensor.shape
        x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)
                
        key = "node_pos"
        tensor = x[key]
        if len(tensor.shape)==2:
            N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(N_max - N + 1, S)], dim=0)
        elif len(tensor.shape)==3:
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)
            
        key = "cells"
        tensor = x[key]
        if len(tensor.shape)==2:
            N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(cells_max - N + 1, S)], dim=0)
        elif len(tensor.shape)==3:
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, cells_max - N + 1, S)], dim=1)
        
        key = "node_type"
        tensor = x[key]
        if len(tensor.shape)==2:
            N, S = tensor.shape
            x[key] = torch.cat([tensor, 2 * torch.ones(N_max - N + 1, S)], dim=0)
        elif len(tensor.shape)==3:
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, 2 * torch.ones(T, N_max - N + 1, S)], dim=1)

        key = "edges"
        edges = x[key]
        if len(tensor.shape)==2:
            E, S = edges.shape
            x[key] = torch.cat([edges, N_max * torch.ones(E_max - E + 1, S)], dim=0)
        elif len(tensor.shape)==3:
            T, E, S = edges.shape
            x[key] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)

        x['mask'] = torch.cat([torch.ones(N), torch.zeros(N_max - N + 1)], dim=0)


    output = {key: None for key in X[0].keys()}
    for key in output.keys():
        output[key] = torch.stack([x[key] for x in X], dim=0)

    return output


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




