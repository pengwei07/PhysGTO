import torch
import torch.nn as nn

def compute_rmse_cylinder(predict, target, node_mask):
    """
    Args:
        predict: [batch, time_steps, nodes, channels]
        target: [batch, time_steps, nodes, channels]
        node_mask: [batch, 1, nodes, 1]
    """
    # 应用mask
    masked_pred = predict * node_mask
    masked_target = target * node_mask
    
    # 计算每个点的平方误差
    squared_error = (masked_pred - masked_target) ** 2
    
    # 对所有维度求平均（空间坐标、节点、时间步长、轨迹）
    mask_sum = node_mask.sum()
    
    # For velocity (u,v)
    rmse_u = torch.sqrt((squared_error[...,:2].sum()) / (mask_sum * 2 * predict.shape[1]))
    
    # For pressure (p)
    rmse_p = torch.sqrt((squared_error[...,2:].sum()) / (mask_sum * predict.shape[1]))
    
    return rmse_u.item(), rmse_p.item()

def compute_rmse_heat(predict, target, node_mask):
    """
    Args:
        predict: [batch, time_steps, nodes, channels]
        target: [batch, time_steps, nodes, channels]
        node_mask: [batch, 1, nodes, 1]
    """
    # 应用mask
    masked_pred = predict * node_mask
    masked_target = target * node_mask
    
    # 计算每个点的平方误差
    squared_error = (masked_pred - masked_target) ** 2
    
    # 对所有维度求平均（空间坐标、节点、时间步长、轨迹）
    mask_sum = node_mask.sum()
    
    # For velocity (u,v)
    rmse_u = torch.sqrt((squared_error[...,:2].sum()) / (mask_sum * 2 * predict.shape[1]))
    
    # For pressure (p)
    rmse_p = torch.sqrt((squared_error[...,2:3].sum()) / (mask_sum * predict.shape[1]))
    
    # For temperature (T)
    rmse_T = torch.sqrt((squared_error[...,3:].sum()) / (mask_sum * predict.shape[1]))
    
    return rmse_u.item(), rmse_p.item(), rmse_T.item()

def compute_rmse_uav(predict, target, node_mask):
    """
    Args:
        predict: [batch, time_steps, nodes, channels]
        target: [batch, time_steps, nodes, channels]
        node_mask: [batch, 1, nodes, 1]
    """
    # 应用mask
    masked_pred = predict * node_mask
    masked_target = target * node_mask
    
    # 计算每个点的平方误差
    squared_error = (masked_pred - masked_target) ** 2
    
    # 对所有维度求平均（空间坐标、节点、时间步长、轨迹）
    mask_sum = node_mask.sum()
    
    # For velocity (u,v)
    rmse_u = torch.sqrt((squared_error[...,:2].sum()) / (mask_sum * 2 * predict.shape[1]))
    
    # For pressure (p)
    rmse_p = torch.sqrt((squared_error[...,2:].sum()) / (mask_sum * 2 * predict.shape[1]))
    
    return rmse_u.item(), rmse_p.item()

def compute_rmse_plasma(predict, target, node_mask):
    """
    Args:
        predict: [batch, time_steps, nodes, channels]
        target: [batch, time_steps, nodes, channels]
        node_mask: [batch, 1, nodes, 1]
    """
    # 应用mask
    masked_pred = predict * node_mask
    masked_target = target * node_mask
    
    # 计算每个点的平方误差
    squared_error = (masked_pred - masked_target) ** 2
    
    # 对所有维度求平均（空间坐标、节点、时间步长、轨迹）
    mask_sum = node_mask.sum()
    
    # For velocity (u,v)
    rmse = torch.sqrt((squared_error.sum()) / (mask_sum * 4 * predict.shape[1]))
    
    return rmse.item()

def get_l2_loss(output, target):
    
    # output.dim = (batch, seq, N, c) or (batch, seq, N)
    # target.dim = (batch, seq, N, c) or (batch, seq, N)
    
    if output.dim() == 4:
        
        if output.shape[-1] == 1:
            output = output.squeeze(-1) 
            target = target.squeeze(-1) 
            
            error = (output - target)
            norm_error = torch.norm(error, dim=-1) / (torch.norm(target, dim=-1) + 1e-8)
            norm_error_time = torch.mean(norm_error, dim=-1)
            norm_error_batch = torch.mean(norm_error_time, dim=0)
        else:
            error = (output - target)
            norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
            norm_error_channel = torch.mean(norm_error, dim=-1)
            norm_error_time = torch.mean(norm_error_channel, dim=-1)
            norm_error_batch = torch.mean(norm_error_time, dim=0)
            
    elif output.dim() == 3:
        error = (output - target)
        norm_error = torch.norm(error, dim=-1) / (torch.norm(target, dim=-1) + 1e-8)
        norm_error_time = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_time, dim=0)
    
    return norm_error_batch

def rescale_data_cylinder(data, info, if_rescale):
    
    if if_rescale:
        data[...,0] = data[...,0] * info['u_std'] + info['u_mean']
        data[...,1] = data[...,1] * info['v_std'] + info['v_mean'] 
        data[...,2] = data[...,2] * info['p_std'] + info['p_mean']
        
    return data

def rescale_data_uav(data, info, if_rescale):
    if if_rescale:
        data[...,0] = data[...,0] * info['u_std'] + info['u_mean']
        data[...,1] = data[...,1] * info['v_std'] + info['v_mean'] 
        data[...,2] = data[...,2] * info['ps_std'] + info['ps_mean']
        data[...,3] = data[...,3] * info['pg_std'] + info['pg_mean']
        
    return data

def rescale_data_heat(data, info, if_rescale):
    if if_rescale:
        data[...,0] = data[...,0] * info['u_std'] + info['u_mean']
        data[...,1] = data[...,1] * info['v_std'] + info['v_mean'] 
        data[...,2] = data[...,2] * info['p_std'] + info['p_mean']
        data[...,3] = data[...,3] * info['T_std'] + info['T_mean']
        
    return data

def get_val_loss_plasma(predict_hat, state, if_rescale, info, node_mask):
    
    device = predict_hat.device
    #################################
    state = state.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    losses = {}
    ################
    
    losses['L2_ne'] = get_l2_loss(predict_hat[...,0] * node_mask[...,0], state[...,0] * node_mask[...,0]).item()
    losses['L2_te'] = get_l2_loss(predict_hat[...,1] * node_mask[...,0], state[...,1] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat[...,2] * node_mask[...,0], state[...,2] * node_mask[...,0]).item()
    losses['L2_T'] = get_l2_loss(predict_hat[...,3] * node_mask[...,0], state[...,3] * node_mask[...,0]).item()

    losses['mean_l2'] = get_l2_loss(predict_hat * node_mask, state * node_mask).item()
    
    losses['RMSE'] = compute_rmse_plasma(predict_hat, state, node_mask)    
    
    losses["each_l2"] = get_each_l2(predict_hat * node_mask, state * node_mask)
    
    return losses

def get_train_loss_plasma(predict_hat, delta_hat, label_gt, 
                   loss_flag, if_rescale, info, node_mask):
    assert loss_flag == 'MGN_norm_loss' or loss_flag == 'L2_loss'
    # predict_hat.dim = [b,t,n,c]
    # label_gt.dim = [b,t,n,c]
    
    MSE = nn.MSELoss()

    losses = {}
    device = predict_hat.device
    label_gt = label_gt.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    # data_1 = get_l2_loss(predict_hat * node_mask, label_gt * node_mask)
    # print(f"data_1: {data_1}")

    if loss_flag == 'MGN_norm_loss':
        
        losses['loss'] = MSE(delta_hat * node_mask, (label_gt[:,1:]-label_gt[:,:-1]) * node_mask)
    
    if loss_flag == 'L2_loss':
        
        # print(predict_hat_.shape, node_mask.shape, label_gt_.shape)
        
        losses['loss'] = get_l2_loss(predict_hat * node_mask, label_gt[:, 1:] * node_mask)
    
    label_gt = label_gt[:, 1:]
    losses['L2_ne'] = get_l2_loss(predict_hat[...,0] * node_mask[...,0], label_gt[...,0] * node_mask[...,0]).item()
    losses['L2_te'] = get_l2_loss(predict_hat[...,1] * node_mask[...,0], label_gt[...,1] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat[...,2] * node_mask[...,0], label_gt[...,2] * node_mask[...,0]).item()
    losses['L2_T'] = get_l2_loss(predict_hat[...,3] * node_mask[...,0], label_gt[...,3] * node_mask[...,0]).item()
    
    # losses['mean_l2'] = losses['loss'].item()
    losses['mean_l2'] = (losses['L2_ne'] + losses['L2_te'] + losses['L2_v'] + losses['L2_T']) / 4
    losses['RMSE'] = compute_rmse_plasma(predict_hat, label_gt, node_mask)    
    losses["each_l2"] = get_each_l2(predict_hat * node_mask, label_gt * node_mask)
    
    return losses

def get_val_loss_cylinder(predict_hat, state, if_rescale, info, node_mask):
    
    device = predict_hat.device
    #################################
    state = state.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    losses = {}
    ################
    state_ = rescale_data_cylinder(state, info, if_rescale)
    predict_hat_ = rescale_data_cylinder(predict_hat, info, if_rescale)
    
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], state_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], state_[...,1] * node_mask[...,0]).item()
    losses['L2_p'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], state_[...,2] * node_mask[...,0]).item()

    losses['mean_l2'] = get_l2_loss(predict_hat_ * node_mask, state_ * node_mask).item()
    
    losses['RMSE_u'], losses['RMSE_p'] = compute_rmse_cylinder(predict_hat_, state_, node_mask)    
    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, state_ * node_mask)
    
    return losses

def get_train_loss_cylinder(predict_hat, delta_hat, label_gt, 
                   loss_flag, if_rescale, info, node_mask):
    assert loss_flag == 'MGN_norm_loss' or loss_flag == 'L2_loss'
    # predict_hat.dim = [b,t,n,c]
    # label_gt.dim = [b,t,n,c]
    
    MSE = nn.MSELoss()

    losses = {}
    device = predict_hat.device
    label_gt = label_gt.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    # data_1 = get_l2_loss(predict_hat * node_mask, label_gt * node_mask)
    # print(f"data_1: {data_1}")

    if loss_flag == 'MGN_norm_loss':
        
        losses['loss'] = MSE(delta_hat[...,:2] * node_mask, (label_gt[:,1:,:,:2]-label_gt[:,:-1,:,:2]) * node_mask) + 0.1 * MSE(delta_hat[...,2:] * node_mask, (label_gt[:,1:,:,2:]-label_gt[:,:-1,:,2:]) * node_mask)
        
        
    predict_hat_ = rescale_data_cylinder(predict_hat, info, if_rescale)
    label_gt_ = rescale_data_cylinder(label_gt, info, if_rescale)
    
    if loss_flag == 'L2_loss':
        
        # print(predict_hat_.shape, node_mask.shape, label_gt_.shape)
        
        losses['loss'] = get_l2_loss(predict_hat_ * node_mask, label_gt_[:, 1:] * node_mask)
    
    label_gt_ = label_gt_[:, 1:]
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], label_gt_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], label_gt_[...,1] * node_mask[...,0]).item()
    losses['L2_p'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], label_gt_[...,2] * node_mask[...,0]).item()
    
    # losses['mean_l2'] = losses['loss'].item()
    losses['mean_l2'] = (losses['L2_u'] + losses['L2_v'] + losses['L2_p']) / 3
    losses['RMSE_u'], losses['RMSE_p'] = compute_rmse_cylinder(predict_hat_, label_gt_, node_mask)    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, label_gt_ * node_mask)
    
    return losses

def get_each_l2(predict_hat, label_gt):
    
    t_step = label_gt.shape[1]
    losses_each_t = torch.zeros(t_step)
    
    for t in range(t_step):
        
        error = predict_hat[:,t] - label_gt[:,t]
        
        norm_error = torch.norm(error, dim=-2) / (torch.norm(label_gt[:,t], dim=-2) + 1e-6)
        norm_error_channel = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_channel, dim=0)
        
        losses_each_t[t] = norm_error_batch.item()
    
    return losses_each_t


def get_train_l2(output, target):
    
    # output.dim = (batch, seq, N, c)
    # target.dim = (batch, seq, N, c)
    
    b,t,n,c = output.shape
    
    error = (output - target)
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
    norm_error_channel = torch.mean(norm_error, dim=-1)
    norm_error_time = torch.mean(norm_error_channel, dim=-1)
    norm_error_batch = torch.mean(norm_error_time, dim=0)
    
    return norm_error_batch

def get_val_loss_uav(predict_hat, state, if_rescale, info, node_mask):
    
    device = predict_hat.device
    #################################
    state = state.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    losses = {}
    ################
    state_ = rescale_data_uav(state, info, if_rescale)
    predict_hat_ = rescale_data_uav(predict_hat, info, if_rescale)
    
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], state_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], state_[...,1] * node_mask[...,0]).item()
    losses['L2_ps'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], state_[...,2] * node_mask[...,0]).item()
    losses['L2_pg'] = get_l2_loss(predict_hat_[...,3] * node_mask[...,0], state_[...,3] * node_mask[...,0]).item()

    losses['mean_l2'] = get_l2_loss(predict_hat_ * node_mask, state_ * node_mask).item()
    
    losses['RMSE_u'], losses['RMSE_p'] = compute_rmse_uav(predict_hat_, state_, node_mask)    
    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, state_ * node_mask)
    
    return losses

def get_train_loss_uav(predict_hat, delta_hat, label_gt, 
                   loss_flag, if_rescale, info, node_mask):
    
    # predict_hat.dim = [b,t,n,c]
    # label_gt.dim = [b,t,n,c]
    assert loss_flag == 'MGN_norm_loss' or loss_flag == 'L2_loss'
    
    MSE = nn.MSELoss()

    losses = {}
    device = predict_hat.device
    label_gt = label_gt.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    # data_1 = get_l2_loss(predict_hat * node_mask, label_gt * node_mask)
    # print(f"data_1: {data_1}")

    if loss_flag == 'MGN_norm_loss':
        
        losses['loss'] = MSE(delta_hat[...,:2] * node_mask, (label_gt[:,1:,:,:2]-label_gt[:,:-1,:,:2]) * node_mask) + 0.1 * MSE(delta_hat[...,2:] * node_mask, (label_gt[:,1:,:,2:]-label_gt[:,:-1,:,2:]) * node_mask)
        
    predict_hat_ = rescale_data_uav(predict_hat, info, if_rescale)
    label_gt_ = rescale_data_uav(label_gt, info, if_rescale)
    
    if loss_flag == 'L2_loss':
        losses['loss'] = get_l2_loss(predict_hat_ * node_mask, label_gt_[:, 1:] * node_mask)
        
    if loss_flag == 'MSE_loss':
        # for Graphvit
        losses['loss'] = MSE(label_gt_[..., :2] * node_mask, predict_hat_[..., :2] * node_mask) + 0.1 * MSE(
            label_gt_[..., 2:] * node_mask, predict_hat_[..., 2:] * node_mask)
    
    label_gt_ = label_gt_[:, 1:]
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], label_gt_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], label_gt_[...,1] * node_mask[...,0]).item()
    losses['L2_ps'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], label_gt_[...,2] * node_mask[...,0]).item()
    losses['L2_pg'] = get_l2_loss(predict_hat_[...,3] * node_mask[...,0], label_gt_[...,3] * node_mask[...,0]).item()
    
    # losses['mean_l2'] = losses['loss'].item()
    losses['mean_l2'] = (losses['L2_u'] + losses['L2_v'] + losses['L2_ps'] + losses['L2_pg']) / 4
    losses['RMSE_u'], losses['RMSE_p'] = compute_rmse_uav(predict_hat_, label_gt_, node_mask)    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, label_gt_ * node_mask)
    
    return losses

def get_val_loss_heat_flow(predict_hat, state, if_rescale, info, node_mask):
    
    device = predict_hat.device
    #################################
    state = state.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    losses = {}
    ################
    state_ = rescale_data_heat(state, info, if_rescale)
    predict_hat_ = rescale_data_heat(predict_hat, info, if_rescale)
    
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], state_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], state_[...,1] * node_mask[...,0]).item()
    losses['L2_p'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], state_[...,2] * node_mask[...,0]).item()
    losses['L2_T'] = get_l2_loss(predict_hat_[...,3] * node_mask[...,0], state_[...,3] * node_mask[...,0]).item()

    losses['mean_l2'] = get_l2_loss(predict_hat_ * node_mask, state_ * node_mask).item()
    
    losses['RMSE_u'], losses['RMSE_p'], losses['RMSE_T'] = compute_rmse_heat(predict_hat_, state_, node_mask)    
    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, state_ * node_mask)
    
    return losses

def get_train_loss_heat_flow(predict_hat, delta_hat, label_gt, 
                   loss_flag, if_rescale, info, node_mask):
    
    # predict_hat.dim = [b,t,n,c]
    # label_gt.dim = [b,t,n,c]
    assert loss_flag == 'MGN_norm_loss' or loss_flag == 'L2_loss'
    
    MSE = nn.MSELoss()

    losses = {}
    device = predict_hat.device
    label_gt = label_gt.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    # data_1 = get_l2_loss(predict_hat * node_mask, label_gt * node_mask)
    # print(f"data_1: {data_1}")

    if loss_flag == 'MGN_norm_loss':
        
        losses['loss'] = MSE(delta_hat * node_mask, (label_gt[:,1:]-label_gt[:,:-1]) * node_mask)
        
    predict_hat_ = rescale_data_heat(predict_hat, info, if_rescale)
    label_gt_ = rescale_data_heat(label_gt, info, if_rescale)
    
    if loss_flag == 'L2_loss':
        
        # print(predict_hat_.shape, node_mask.shape, label_gt_.shape)
        
        losses['loss'] = get_l2_loss(predict_hat_ * node_mask, label_gt_[:, 1:] * node_mask)
    
    label_gt_ = label_gt_[:, 1:]
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], label_gt_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], label_gt_[...,1] * node_mask[...,0]).item()
    losses['L2_p'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], label_gt_[...,2] * node_mask[...,0]).item()
    losses['L2_T'] = get_l2_loss(predict_hat_[...,3] * node_mask[...,0], label_gt_[...,3] * node_mask[...,0]).item()
    
    # losses['mean_l2'] = losses['loss'].item()
    losses['mean_l2'] = (losses['L2_u'] + losses['L2_v'] + losses['L2_p'] + losses['L2_T']) / 4
    losses['RMSE_u'], losses['RMSE_p'], losses['RMSE_T'] = compute_rmse_heat(predict_hat_, label_gt_, node_mask)    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, label_gt_ * node_mask)
    
    return losses