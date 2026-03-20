import torch
from tqdm import tqdm
from torch.nn.functional import one_hot
import random
import torch.nn as nn
import numpy as np
import h5py
import os

from .all_loss import get_train_loss_heat_flow, get_val_loss_heat_flow

def rescale_data_heat(data, info, if_rescale):
    if if_rescale:
        data[...,0] = data[...,0] * info['u_std'] + info['u_mean']
        data[...,1] = data[...,1] * info['v_std'] + info['v_mean'] 
        data[...,2] = data[...,2] * info['p_std'] + info['p_mean']
        data[...,3] = data[...,3] * info['T_std'] + info['T_mean']
        
    return data

def save_result(args, node_pos, cells, state, predict_hat, if_rescale, info, device, idx):
    
    x_max = 3.0
    x_min = 0.0
    
    y_max = 1.0
    y_min = 0.0
    
    node_pos[:,:,0] = node_pos[:,:,0] * (x_max - x_min) + x_min
    node_pos[:,:,1] = node_pos[:,:,1] * (y_max - y_min) + y_min
    
    state = rescale_data_heat(state, info, if_rescale)
    # predict_hat = rescale_data_heat(predict_hat, info, if_rescale)
    
    node_pos = node_pos.cpu().numpy()
    cells = cells.cpu().numpy()
    state = state.cpu().numpy()
    predict_hat = predict_hat.cpu().numpy()
    
    if not os.path.exists(f"{args.save_path}/record/infer_result"):
        os.makedirs(f"{args.save_path}/record/infer_result")
        
    with h5py.File(f"{args.save_path}/record/infer_result/heat_result_{idx}.h5", "w") as f:
        f.create_dataset("node_pos", data=node_pos)
        f.create_dataset("cells", data=cells)
        f.create_dataset("state", data=state[:,::10,:,:])
        f.create_dataset("predict_hat", data=predict_hat[:,::10,:,:])
        
        
def forward(args, model, state, node_pos, edges, node_type, t_all, device, mode="train"):
    num_classes = int(torch.max(node_type+1))
    node_type = one_hot(node_type.long(), num_classes=num_classes).squeeze(-2)
    predict_hat = []
    delta_hat = []
    
    state = state.to(device)
    node_pos = node_pos.to(device)
    node_type = node_type.to(device)
    t_all = t_all.to(device)
    
    state_in = state[:,0]
    
    # state_in.dim = [B, N, 4]
    mask = node_type[:, :, 0] == 0
    mask_noise = node_type[:, :, 0] != 0
        
    if args.train["if_add_noise"] and mode == "train":
        # Following MGN, this add noise to the input. Better results are obtained with longer windows and no noise
        noise_std = 2e-2
        noise = torch.randn_like(state_in) * noise_std
        if len(node_type.shape)==3:
            state_in[mask_noise] = state_in[mask_noise] + noise[mask_noise].to(device)
        else:
            state_in[mask_noise[:,0]] = state_in[mask_noise[:,0]] + noise[mask_noise[:,0]].to(device)
    
    predict_hat.append(state_in) 
    
    for t in range(state.shape[1] - 1):
        if args.model["name"] == "Graphvit":
            next_state, delta_state = model(
                node_pos,
                edges,
                t_all[:,t], # t_all[:,t-input_step:t+1], 
                predict_hat[t].detach(),
                node_type.float()
                )
        else:
            next_state, delta_state = model(
                node_pos,
                edges,
                t_all[:,t], # t_all[:,t-input_step:t+1], 
                predict_hat[t].detach(),
                node_type.float()
                )
        # force BCs
        #############
        next_state[mask, :] = state[:, t + 1][mask, :]
        delta_state[mask, :] = (state[:, t + 1]-state[:, t])[mask, :]
        
        predict_hat.append(next_state)  
        delta_hat.append(delta_state)

    predict_hat = torch.stack(predict_hat[1:], dim=1)
    delta_hat = torch.stack(delta_hat, dim=1)
    return predict_hat, delta_hat
    
def train(args, model, train_dataloader, optim, device):

    #######################################
    h_train = args.dataset["horizon_train"]
    
    loss = 0
    
    L2_u = 0
    L2_v = 0
    L2_p = 0
    L2_T = 0
    
    RMSE_u = 0
    RMSE_p = 0
    RMSE_T = 0
    
    L2_mean = 0
    num = 0
    
    each_l2 = torch.zeros(h_train)
    
    model.train()
    # forward
    for i, input in enumerate(train_dataloader):
        
        state = input['state']        
        node_pos = input['node_pos']
        edges = input['edges']
        node_type = input['node_type']
        t_all = input['t_all']
        
        node_mask = input['mask']
        # dim = [b, t, n]
        
        batch_num = state.shape[0]
        
        predict_hat, delta_hat = forward(args, model, 
                            state,
                            node_pos, edges, 
                            node_type, t_all,
                            device)
        
        costs = get_train_loss_heat_flow(
            predict_hat,
            delta_hat,
            state,
            args.train["loss_flag"], 
            args.train["if_rescale"], 
            args.train["info"],
            node_mask
            )
        # print(f"predict_hat: {predict_hat.shape}, label_gt: {label_gt.shape}")
        # print(f"pred_edges: {pred_edges.shape}, edges_mask: {edges_mask.shape}")
                
        costs['loss'].backward()
        optim.step()
        optim.zero_grad()
        
        # for loss
        loss = loss + costs['loss'].item() * batch_num
        
        L2_u = L2_u + costs['L2_u'] * batch_num
        L2_v = L2_v + costs['L2_v'] * batch_num
        L2_p = L2_p + costs['L2_p'] * batch_num
        L2_T = L2_T + costs['L2_T'] * batch_num
        
        L2_mean = L2_mean + costs['mean_l2'] * batch_num
        
        RMSE_u = RMSE_u + costs['RMSE_u'] * batch_num
        RMSE_p = RMSE_p + costs['RMSE_u'] * batch_num
        RMSE_T = RMSE_T + costs['RMSE_u'] * batch_num
        
        each_l2 = each_l2 + costs["each_l2"] * batch_num
        #########################################
        
        num = num + batch_num
        
        # print(f"loss: {costs['loss'].item():.4e}, mean_l2: {costs['mean_l2']:.4e}")
        
        # print(f"loss: {costs['loss'].item():.4e}, mean_l2: {costs['mean_l2']:.4e}")
        # print(f"each time step loss: {losses_each_t}")
        
        # break 

    batch_error = {}
    batch_error['loss'] = loss / num
    
    batch_error['L2_u'] = L2_u / num
    batch_error['L2_v'] = L2_v / num
    batch_error['L2_p'] = L2_p / num
    batch_error['L2_T'] = L2_T / num
    
    batch_error['mean_l2'] = L2_mean / num
    
    batch_error['RMSE_u'] = RMSE_u / num
    batch_error['RMSE_p'] = RMSE_p / num
    batch_error['RMSE_T'] = RMSE_T / num
    
    batch_error['each_l2'] = each_l2 / num
            
    return batch_error

def validate(args, model, val_dataloader, device, if_save=False):
    
    h_test = args.dataset["horizon_test"]
    
    model.eval()
    
    L2_u = 0
    L2_v = 0
    L2_p = 0
    L2_T = 0
    L2_mean = 0
    
    RMSE_u = 0
    RMSE_p = 0
    RMSE_T = 0
    
    each_l2 = torch.zeros(h_test)
        
    num = 0
    
    with torch.no_grad():
        # inference
        # for i, [input, t] in enumerate(tqdm(val_dataloader, desc="Validation")):
        for i, input in enumerate(val_dataloader):    
            # if i not in [95, 131, 96, 107, 103, 17, 37, 1, 60, 16]: 
            #     continue
            
            state = input['state']
            node_pos = input['node_pos']
            edges = input['edges']
            
            node_type = input['node_type']
            t_all = input['t_all']
            
            node_mask = input['mask']
            # dim = [b, n]
            
            batch_num = state.shape[0]

            predict_hat, _ = forward(args, model, state,
                                    node_pos, edges, 
                                    node_type, t_all,
                                    device,
                                    "test"
                                    )
            
            costs = get_val_loss_heat_flow(
                predict_hat,
                state[:,1:1+h_test],
                args.train["if_rescale"], 
                args.train["info"],
                node_mask
                )
            
            if i == 0:
                with open(f"{args.save_path}/record/val_loss.txt", "a") as f:
                    f.write(f"h_test: {h_test}\n")
                    f.write(f"batch {i} loss: {costs['mean_l2']:.4e}\n")
            else:
                with open(f"{args.save_path}/record/val_loss.txt", "a") as f:
                    f.write(f"batch {i} loss: {costs['mean_l2']:.4e}\n")
                        
            if if_save:
                # with open(f"{args.save_path}/record/val_loss.txt", "a") as f:
                #     f.write(f"batch {i} loss: {costs['mean_l2']:.4e}\n")
                save_result(args,
                            node_pos = input['node_pos'],
                            cells = input['cells'],
                            state = state[:,1:1+h_test],
                            predict_hat = predict_hat,
                            if_rescale = args.train["if_rescale"],
                            info = args.train["info"],
                            device = device,
                            idx = i)
            
            #########################################
            L2_u = L2_u + costs['L2_u'] * batch_num
            L2_v = L2_v + costs['L2_v'] * batch_num
            L2_p = L2_p + costs['L2_p'] * batch_num
            L2_T = L2_T + costs['L2_T'] * batch_num
            
            L2_mean = L2_mean + costs['mean_l2'] * batch_num
            
            RMSE_u = RMSE_u + costs['RMSE_u'] * batch_num
            RMSE_p = RMSE_p + costs['RMSE_u'] * batch_num
            RMSE_T = RMSE_T + costs['RMSE_u'] * batch_num
            
            each_l2 = each_l2 + costs["each_l2"] * batch_num
            #########################################
            num = num + batch_num
            
            # break
            if i == 20 and if_save:
                break

    batch_error = {}

    batch_error['L2_u'] = L2_u / num
    batch_error['L2_v'] = L2_v / num
    batch_error['L2_p'] = L2_p / num
    batch_error['L2_T'] = L2_T / num
    
    batch_error['mean_l2'] = L2_mean / num
    
    batch_error['RMSE_u'] = RMSE_u / num
    batch_error['RMSE_p'] = RMSE_p / num
    batch_error['RMSE_T'] = RMSE_T / num
    
    batch_error['each_l2'] = each_l2 / num
    
    return batch_error