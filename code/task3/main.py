import os
import time
import numpy as np
import random

import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

# load
from src.dataset import Car_Dataset
from src.sample import merge_split_results

from src.model.physGTO import Model
from src.utils import set_seed, collate, init_weights, parse_args,RelLpLoss,get_l2_loss

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#print(device)

def setup():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return rank, world_size

def gather_tensor(tensor, world_size):
    """
    Gathers tensors from all processes and reduces them by summing up.
    """
    # Ensure the tensor is on the same device as specified for the operation
    tensor = tensor.to(device)
    # All-reduce: Sum the tensors from all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # Only on rank 0, we scale the tensor to find the average
    if dist.get_rank() == 0:
        tensor /= world_size
    return tensor

def coef_norm_funs(coef, args, flag):
    
    # coef.shape = [b, 4]
    
    if flag == "norm":
        # print(coef.shape)
        
        cd_norm = (coef[..., 0] - args.dataset["cd_mean"]) / args.dataset["cd_std"]
        cl_f_norm = (coef[..., 2] - args.dataset["cl_f_mean"]) / args.dataset["cl_f_std"]
        cl_r_norm = (coef[..., 3] - args.dataset["cl_r_mean"]) / args.dataset["cl_r_std"]

        coef_norm = torch.stack([cd_norm, cl_f_norm, cl_r_norm], dim=-1)
        
        return coef_norm
        
    elif flag == "denorm":
        
        cd_denorm = coef[..., 0] * args.dataset["cd_std"] + args.dataset["cd_mean"]
        cl_f_denorm = coef[..., 1] * args.dataset["cl_f_std"] + args.dataset["cl_f_mean"]
        cl_r_denorm = coef[..., 2] * args.dataset["cl_r_std"] + args.dataset["cl_r_mean"]
        
        coef_denorm = torch.stack([cd_denorm, cl_f_denorm, cl_r_denorm], dim=-1)
        
        return coef_denorm
    
def get_coef_l2_loss(pred_coef_norm, coef_norm):
    
    # pred_coef_norm.shape = [b, 3]
    # coef_norm.shape = [b, 3]
    
    errors = pred_coef_norm - coef_norm
    l2_loss = torch.mean(torch.norm(errors, dim=-1) / torch.norm(coef_norm, dim=-1))
    
    return l2_loss
    

def train(args, model, train_dataloader, optim, local_rank):
    
    model.train()
    train_loss = 0
    train_press_l2 = 0        
    
    train_cd_mse = 0
    train_cd_mae = 0
    train_cl_mse = 0
    train_cl_mae = 0
    train_cl_f_mse = 0
    train_cl_f_mae = 0
    train_cl_r_mse = 0
    train_cl_r_mae = 0
    
    num = 0
    
    for i, [splits_data, coef] in enumerate(train_dataloader):
        
        # 清除梯度
        optim.zero_grad()

        sampled_pressure = splits_data['pressure'].to(local_rank)
        sampled_pressure_norm = (sampled_pressure - args.dataset["press_mean"]) / args.dataset["press_std"]
        
        coef = coef.to(local_rank)
        coef_norm = coef_norm_funs(coef, args, flag="norm")
        
        sampled_centroids = splits_data['centroids'].to(local_rank)
        sampled_areas = splits_data['areas'].to(local_rank)
        sampled_edges = splits_data['edges']
        
        # forward
        pred_pressure_norm, pred_coef_norm = model(sampled_areas, sampled_centroids, sampled_edges)
                
        loss_l2_norm = get_l2_loss(pred_pressure_norm, sampled_pressure_norm) + get_coef_l2_loss(pred_coef_norm, coef_norm)
        
        pred_pressure = pred_pressure_norm * args.dataset["press_std"] + args.dataset["press_mean"]

        loss_l2 = get_l2_loss(pred_pressure, sampled_pressure).item()
            
        # 反向传播
        loss_l2_norm.backward()
        optim.step()

        train_loss += loss_l2_norm.item()
        train_press_l2 += loss_l2
        num = num + 1
        
        # for each coef
        pred_coef = coef_norm_funs(pred_coef_norm, args, flag="denorm")
        train_cd_mse += ((pred_coef[...,0] - coef[...,0])**2).mean().item()
        train_cd_mae += torch.mean(torch.abs(pred_coef[...,0] - coef[...,0])).item()

        train_cl_f_mse += ((pred_coef[...,1] - coef[...,2])**2).mean().item()
        train_cl_f_mae += torch.mean(torch.abs(pred_coef[...,1] - coef[...,2])).item()
        
        train_cl_r_mse += ((pred_coef[...,2] - coef[...,3])**2).mean().item()
        train_cl_r_mae += torch.mean(torch.abs(pred_coef[...,2] - coef[...,3])).item()
        
        pred_cl = pred_coef[...,1] + pred_coef[...,2]
        train_cl_mse += ((pred_cl - coef[...,1])**2).mean().item()
        train_cl_mae += torch.mean(torch.abs(pred_cl - coef[...,1])).item()

        # break
    
    train_coef_mse = {
        "cd_mse": train_cd_mse / num,
        "cl_mse": train_cl_mse / num,
        "cl_f_mse": train_cl_f_mse / num,
        "cl_r_mse": train_cl_r_mse / num
    }
    
    train_coef_mae = {
        "cd_mae": train_cd_mae / num,
        "cl_mae": train_cl_mae / num,
        "cl_f_mae": train_cl_f_mae / num,
        "cl_r_mae": train_cl_r_mae / num
    }
    
    return train_loss / num, train_press_l2 / num, train_coef_mse, train_coef_mae

def infer(args, model, test_dataloader_sample, test_dataloader, local_rank):
    
    model.eval()    
    with torch.no_grad():
        
        # for sample
        num = 0 
        test_press_l2_sample = 0
        test_cd_mse_0 = 0
        test_cd_mae_0 = 0
        test_cl_mse_0 = 0
        test_cl_mae_0 = 0
        test_cl_f_mse_0 = 0
        test_cl_f_mae_0 = 0
        test_cl_r_mse_0 = 0
        test_cl_r_mae_0 = 0
        for i, [splits_data, coef] in enumerate(test_dataloader_sample):
            
            coef = coef.to(local_rank)
            sampled_pressure = splits_data['pressure'].to(local_rank)
            sampled_centroids = splits_data['centroids'].to(local_rank)
            sampled_areas = splits_data['areas'].to(local_rank)
            sampled_edges = splits_data['edges']
            
            # forward
            pred_pressure_norm, pred_coef_norm = model(sampled_areas, sampled_centroids, sampled_edges)            
            pred_pressure = pred_pressure_norm * args.dataset["press_std"] + args.dataset["press_mean"]

            loss_l2 = get_l2_loss(pred_pressure, sampled_pressure).item()
            test_press_l2_sample += loss_l2
            
            pred_coef = coef_norm_funs(pred_coef_norm, args, flag="denorm")
            test_cd_mse_0 += ((pred_coef[...,0] - coef[...,0])**2).mean().item()
            test_cd_mae_0 += torch.mean(torch.abs(pred_coef[...,0] - coef[...,0])).item()

            test_cl_f_mse_0 += ((pred_coef[...,1] - coef[...,2])**2).mean().item()
            test_cl_f_mae_0 += torch.mean(torch.abs(pred_coef[...,1] - coef[...,2])).item()
            
            test_cl_r_mse_0 += ((pred_coef[...,2] - coef[...,3])**2).mean().item()
            test_cl_r_mae_0 += torch.mean(torch.abs(pred_coef[...,2] - coef[...,3])).item()
            
            pred_cl = pred_coef[...,1] + pred_coef[...,2]
            test_cl_mse_0 += ((pred_cl - coef[...,1])**2).mean().item()
            test_cl_mae_0 += torch.mean(torch.abs(pred_cl - coef[...,1])).item()
        
            num = num + 1
            
            # break
        
        test_press_l2_sample = test_press_l2_sample / num
        test_coef_mse_0 = {
            "cd_mse": test_cd_mse_0 / num,
            "cl_mse": test_cl_mse_0 / num,
            "cl_f_mse": test_cl_f_mse_0 / num,
            "cl_r_mse": test_cl_r_mse_0 / num
        }
        
        test_coef_mae_0 = {
            "cd_mae": test_cd_mae_0 / num,
            "cl_mae": test_cl_mae_0 / num,
            "cl_f_mae": test_cl_f_mae_0 / num,
            "cl_r_mae": test_cl_r_mae_0 / num
        }
        
        # for all
        test_press_l2_all = 0
        test_cd_mse_1 = 0
        test_cd_mae_1 = 0
        test_cl_mse_1 = 0
        test_cl_mae_1 = 0
        test_cl_f_mse_1 = 0
        test_cl_f_mae_1 = 0
        test_cl_r_mse_1 = 0
        test_cl_r_mae_1 = 0
        
        num = 0 
        for i, batch_data in enumerate(test_dataloader):
            
            splits_data = batch_data['splits']
            coef = batch_data['coef'].to(local_rank)
            ori_pressre = batch_data['ori_pressre'].to(local_rank) 

            processed_results = []
            
            pred_cd = torch.zeros(ori_pressre.shape[0], 1).to(local_rank)
            pred_cl = torch.zeros(ori_pressre.shape[0], 1).to(local_rank)
            pred_cl_f = torch.zeros(ori_pressre.shape[0], 1).to(local_rank)
            pred_cl_r = torch.zeros(ori_pressre.shape[0], 1).to(local_rank)
            
            for split_id, split_data in splits_data.items():
                
                # 获取这个分割的数据
                sampled_centroids = split_data['centroids'].to(local_rank) 
                sampled_areas = split_data['areas'].to(local_rank) 
                sampled_edges = split_data['edges']
                new_to_old = split_data['new_to_old']
                
                pred_pressure_norm, pred_coef_norm = model(sampled_areas, sampled_centroids, sampled_edges)
                pred_pressure = pred_pressure_norm * args.dataset["press_std"] + args.dataset["press_mean"]

                pred_coef_norm = coef_norm_funs(pred_coef_norm, args, flag="denorm")
                
                pred_cd = pred_cd + pred_coef_norm[...,0]
                pred_cl_f = pred_cl_f + pred_coef_norm[...,1]
                pred_cl_r = pred_cl_r + pred_coef_norm[...,2]

                pred_cl = pred_cl + pred_coef_norm[...,1] + pred_coef_norm[...,2]
                
                processed_results.append({
                    'values': pred_pressure[0],
                    'new_to_old': new_to_old[0]
                })
            
            original_size = ori_pressre.shape[-2]
            final_result, _ = merge_split_results(original_size, processed_results)
            
            # loss
            l2_error = final_result - ori_pressre
            l2_loss = torch.mean(torch.norm(l2_error, dim=-2) / (torch.norm(ori_pressre, dim=-2)))
            test_press_l2_all = test_press_l2_all + l2_loss.item()
            
            # for each coef
            
            pred_cd = pred_cd / len(splits_data)
            pred_cl_f = pred_cl_f / len(splits_data)
            pred_cl_r = pred_cl_r / len(splits_data)
            pred_cl = pred_cl / len(splits_data)

            test_cd_mse_1 = test_cd_mse_1 + ((pred_cd - coef[...,0])**2).mean().item()
            test_cd_mae_1 = test_cd_mae_1 + torch.mean(torch.abs(pred_cd - coef[...,0])).item()
            
            test_cl_mse_1 = test_cl_mse_1 + ((pred_cl - coef[...,1])**2).mean().item()
            test_cl_mae_1 = test_cl_mae_1 + torch.mean(torch.abs(pred_cl - coef[...,1])).item()
            
            test_cl_f_mse_1 = test_cl_f_mse_1 + ((pred_cl_f - coef[...,2])**2).mean().item()
            test_cl_f_mae_1 = test_cl_f_mae_1 + torch.mean(torch.abs(pred_cl_f - coef[...,2])).item()
            
            test_cl_r_mse_1 = test_cl_r_mse_1 + ((pred_cl_r - coef[...,3])**2).mean().item()
            test_cl_r_mae_1 = test_cl_r_mae_1 + torch.mean(torch.abs(pred_cl_r - coef[...,3])).item()
            
            num = num + 1
            
            # break
        
        test_press_l2_all = test_press_l2_all / num
        
        test_coef_mse_1 = {
            "cd_mse": test_cd_mse_1 / num,
            "cl_mse": test_cl_mse_1 / num,
            "cl_f_mse": test_cl_f_mse_1 / num,
            "cl_r_mse": test_cl_r_mse_1 / num
        }
        
        test_coef_mae_1 = {
            "cd_mae": test_cd_mae_1 / num,
            "cl_mae": test_cl_mae_1 / num,
            "cl_f_mae": test_cl_f_mae_1 / num,
            "cl_r_mae": test_cl_r_mae_1 / num
        }

    return test_press_l2_sample, test_press_l2_all, test_coef_mse_0, test_coef_mae_0, test_coef_mse_1, test_coef_mae_1

def get_dataset(args, local_rank):
    
    train_dataset = Car_Dataset(
        data_dir = args.dataset["data_path"], 
        mode="train",
        normalize = True,
        if_sample = True,
        sample_rate = args.dataset["train"]["sample_rate"],
        normalize_way = "z_score"
        )

    test_dataset_sample = Car_Dataset(
        data_dir = args.dataset["data_path"], 
        mode="test",
        normalize = True,
        if_sample = True,
        sample_rate = args.dataset["test"]["sample_rate"],
        normalize_way = "z_score"
        )
    
    test_dataset = Car_Dataset(
        data_dir = args.dataset["data_path"], 
        mode="test",
        normalize = True,
        if_sample = False,
        sample_rate = args.dataset["test"]["sample_rate"],
        normalize_way = "z_score"
        )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, seed=args.seed, rank=local_rank)
    test_sampler_sample = DistributedSampler(test_dataset_sample, num_replicas=world_size, shuffle= False, seed=args.seed, rank=local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle= False, seed=args.seed, rank=local_rank)
    
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=args.dataset["train"]["batchsize"], 
                        sampler=train_sampler,
                        num_workers=args.dataset["train"]["num_workers"]
                        )
        
    test_dataloader_sample = DataLoader(test_dataset_sample, 
                        batch_size=args.dataset["test"]["batchsize"], 
                        sampler=test_sampler_sample,
                        num_workers=args.dataset["test"]["num_workers"]
                        )
    
    test_dataloader = DataLoader(test_dataset, 
                        batch_size=args.dataset["test"]["batchsize"], 
                        sampler=test_sampler,
                        num_workers=args.dataset["test"]["num_workers"]
                        )
    
    return train_dataloader, test_dataloader_sample, test_dataloader

def init_model(args, local_rank):
    
    EPOCH = args.train["epoch"]
    
    model = Model(
        space_size = args.model["space_size"], 
        pos_enc_dim = args.model["pos_enc_dim"], 
        N_block = args.model["N_block"], 
        in_dim = args.model["in_dim"],
        out_dim = args.model["out_dim"],
        enc_dim = args.model["enc_dim"], 
        n_head = args.model["n_head"],
        n_token = args.model["n_token"],
        if_pred_cd = True
        ).to(local_rank)
    
    if args.model["if_init"]:
        model.apply(init_weights)     
    else:
        checkpoint_path = f"{path_nn}/{args.name}_{EPOCH-1}.pth" 
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint) 
        
    return model


def main(args, path_logs, path_nn, path_record):

    # setting
    local_rank, world_size = setup()
    EPOCH = args.train["epoch"]
    real_lr = float(args.train["lr"])
    ######################################
    ######################################
    # load data
    train_dataloader, test_dataloader_sample, test_dataloader = get_dataset(args, local_rank)
    
    # load model
    model =  init_model(args, local_rank)
    model = DDP(model, device_ids=[local_rank])
    # model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))

    if local_rank == 0:
        
        # print("---train_dataloader---")
        print("------------")
        print(f"No. of train batches: {len(train_dataloader)}")
        print(f"No. of test batches: {len(test_dataloader)}")
        print("#params:", params)
        print(f"EPOCH: {EPOCH}")        
        print("---------")
        
        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(f"No. of train batches: {len(train_dataloader)}\n")
            file.write(f"No. of test batches: {len(test_dataloader)}\n")
        
            file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
            file.write(f"{args.name}, #params: {params}\n")
            file.write(f"EPOCH: {EPOCH}\n")
            
        log_dir = f"{path_logs}/{args.name}/rank_{local_rank}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    # train
    ######################################
    optim = torch.optim.AdamW(model.parameters(), lr=real_lr, weight_decay=1e-5)
    if EPOCH == 1:
        scheduler = CosineAnnealingLR(optim, T_max= EPOCH, eta_min = real_lr)  
    else:
        scheduler = CosineAnnealingLR(optim, T_max= EPOCH, eta_min = real_lr/50)
    
    for epoch in range(EPOCH):
        start_time = time.time()
        train_loss, train_press_l2, train_coef_mse, train_coef_mae = train(args, model, train_dataloader, optim, local_rank)
        end_time = time.time()
        
        # update lr
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        current_lr = torch.tensor(current_lr, device=local_rank)
        
        # cd 
        train_cd_mse = train_coef_mse["cd_mse"]
        train_cd_mae = train_coef_mae["cd_mae"]
        train_cd_mse = gather_tensor(torch.tensor(train_cd_mse, device=local_rank), world_size)
        train_cd_mae = gather_tensor(torch.tensor(train_cd_mae, device=local_rank), world_size)
        
        # cl
        train_cl_mse = train_coef_mse["cl_mse"]
        train_cl_mae = train_coef_mae["cl_mae"]
        train_cl_mse = gather_tensor(torch.tensor(train_cl_mse, device=local_rank), world_size)
        train_cl_mae = gather_tensor(torch.tensor(train_cl_mae, device=local_rank), world_size)
        
        # cl_f
        train_cl_f_mse = train_coef_mse["cl_f_mse"]
        train_cl_f_mae = train_coef_mae["cl_f_mae"]
        train_cl_f_mse = gather_tensor(torch.tensor(train_cl_f_mse, device=local_rank), world_size)
        train_cl_f_mae = gather_tensor(torch.tensor(train_cl_f_mae, device=local_rank), world_size)
        
        # cl_r
        train_cl_r_mse = train_coef_mse["cl_r_mse"]
        train_cl_r_mae = train_coef_mae["cl_r_mae"]
        train_cl_r_mse = gather_tensor(torch.tensor(train_cl_r_mse, device=local_rank), world_size)
        train_cl_r_mae = gather_tensor(torch.tensor(train_cl_r_mae, device=local_rank), world_size)
        
        # else
        training_time = (end_time - start_time)
        training_time = torch.tensor(training_time, device=local_rank)
        training_time = gather_tensor(training_time, world_size)
        
        train_loss = gather_tensor(torch.tensor(train_loss, device=local_rank), world_size)
        train_press_l2 = gather_tensor(torch.tensor(train_press_l2, device=local_rank), world_size)

        if local_rank == 0:
            
            writer.add_scalar('Loss/train_loss', train_loss, epoch)
            writer.add_scalar('Loss/train_press_l2', train_press_l2, epoch)
            writer.add_scalar('Loss/train_cd_mse', train_cd_mse, epoch)
            writer.add_scalar('Loss/train_cd_mae', train_cd_mae, epoch)
            writer.add_scalar('Loss/train_cl_mse', train_cl_mse, epoch)
            writer.add_scalar('Loss/train_cl_mae', train_cl_mae, epoch)
            writer.add_scalar('Loss/train_cl_f_mse', train_cl_f_mse, epoch)
            writer.add_scalar('Loss/train_cl_f_mae', train_cl_f_mae, epoch)
            writer.add_scalar('Loss/train_cl_r_mse', train_cl_r_mse, epoch)
            writer.add_scalar('Loss/train_cl_r_mae', train_cl_r_mae, epoch)
            
            print(f"Epoch: {epoch+1}/{EPOCH}, Train Loss: {train_loss:.5e}, Train Pressure L2: {train_press_l2:.5e}")
            print(f"train_cd_mse: {train_cd_mse:.5e}, train_cd_mae: {train_cd_mae:.5e}")
            print(f"train_cl_mse: {train_cl_mse:.5e}, train_cl_mae: {train_cl_mae:.5e}")
            print(f"train_cl_f_mse: {train_cl_f_mse:.5e}, train_cl_f_mae: {train_cl_f_mae:.5e}")
            print(f"train_cl_r_mse: {train_cl_r_mse:.5e}, train_cl_r_mae: {train_cl_r_mae:.5e}")
            print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}")
            
            with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                file.write(f"Epoch: {epoch+1}/{EPOCH}, Train Loss: {train_loss:.5e}, Train Pressure L2: {train_press_l2:.5e}\n")
                file.write(f"train_cd_mse: {train_cd_mse:.5e}, train_cd_mae: {train_cd_mae:.5e}\n")
                file.write(f"train_cl_mse: {train_cl_mse:.5e}, train_cl_mae: {train_cl_mae:.5e}\n")
                file.write(f"train_cl_f_mse: {train_cl_f_mse:.5e}, train_cl_f_mae: {train_cl_f_mae:.5e}\n")
                file.write(f"train_cl_r_mse: {train_cl_r_mse:.5e}, train_cl_r_mae: {train_cl_r_mae:.5e}\n")
                file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n")
        
        ###############################################################
        if (epoch+1) % 5 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            
            start_time = time.time() 
            model.eval()
            test_press_l2_sample, test_press_l2, test_coef_mse_0, test_coef_mae_0, test_coef_mse_1, test_coef_mae_1 = infer(args, model, test_dataloader_sample, test_dataloader, local_rank)
            end_time = time.time()
            
            test_time = (end_time - start_time)
            test_time = torch.tensor(test_time, device=local_rank)
            test_time = gather_tensor(test_time, world_size)
            
            test_press_l2_sample = gather_tensor(torch.tensor(test_press_l2_sample, device=local_rank), world_size)
            test_press_l2 = gather_tensor(torch.tensor(test_press_l2, device=local_rank), world_size)
            
            test_cd_mse_0 = test_coef_mse_0["cd_mse"]
            test_cd_mae_0 = test_coef_mae_0["cd_mae"]
            test_cd_mse_1 = test_coef_mse_1["cd_mse"]
            test_cd_mae_1 = test_coef_mae_1["cd_mae"]
            
            test_cl_mse_0 = test_coef_mse_0["cl_mse"]
            test_cl_mae_0 = test_coef_mae_0["cl_mae"]
            test_cl_mse_1 = test_coef_mse_1["cl_mse"]
            test_cl_mae_1 = test_coef_mae_1["cl_mae"]
            
            test_cl_f_mse_0 = test_coef_mse_0["cl_f_mse"]
            test_cl_f_mae_0 = test_coef_mae_0["cl_f_mae"]
            test_cl_f_mse_1 = test_coef_mse_1["cl_f_mse"]
            test_cl_f_mae_1 = test_coef_mae_1["cl_f_mae"]
            
            test_cl_r_mse_0 = test_coef_mse_0["cl_r_mse"]
            test_cl_r_mae_0 = test_coef_mae_0["cl_r_mae"]
            test_cl_r_mse_1 = test_coef_mse_1["cl_r_mse"]
            test_cl_r_mae_1 = test_coef_mae_1["cl_r_mae"]

            test_cd_mse_0 = gather_tensor(torch.tensor(test_cd_mse_0, device=local_rank), world_size)
            test_cd_mae_0 = gather_tensor(torch.tensor(test_cd_mae_0, device=local_rank), world_size)
            test_cd_mse_1 = gather_tensor(torch.tensor(test_cd_mse_1, device=local_rank), world_size)
            test_cd_mae_1 = gather_tensor(torch.tensor(test_cd_mae_1, device=local_rank), world_size)
            
            test_cl_mse_0 = gather_tensor(torch.tensor(test_cl_mse_0, device=local_rank), world_size)
            test_cl_mae_0 = gather_tensor(torch.tensor(test_cl_mae_0, device=local_rank), world_size)
            test_cl_mse_1 = gather_tensor(torch.tensor(test_cl_mse_1, device=local_rank), world_size)
            test_cl_mae_1 = gather_tensor(torch.tensor(test_cl_mae_1, device=local_rank), world_size)
            
            test_cl_f_mse_0 = gather_tensor(torch.tensor(test_cl_f_mse_0, device=local_rank), world_size)
            test_cl_f_mae_0 = gather_tensor(torch.tensor(test_cl_f_mae_0, device=local_rank), world_size)
            test_cl_f_mse_1 = gather_tensor(torch.tensor(test_cl_f_mse_1, device=local_rank), world_size)
            test_cl_f_mae_1 = gather_tensor(torch.tensor(test_cl_f_mae_1, device=local_rank), world_size)
            
            test_cl_r_mse_0 = gather_tensor(torch.tensor(test_cl_r_mse_0, device=local_rank), world_size)
            test_cl_r_mae_0 = gather_tensor(torch.tensor(test_cl_r_mae_0, device=local_rank), world_size)
            test_cl_r_mse_1 = gather_tensor(torch.tensor(test_cl_r_mse_1, device=local_rank), world_size)
            test_cl_r_mae_1 = gather_tensor(torch.tensor(test_cl_r_mae_1, device=local_rank), world_size)
            
            if local_rank == 0:
                
                print("---Inference---")
                print(f"Epoch: {epoch+1}/{EPOCH}, Test L2 sample: {test_press_l2_sample:.5e}, out: {test_press_l2:.5e}")
                print(f"sample")
                print(f"test_cd_mse_0: {test_cd_mse_0:.5e}, test_cd_mae_0: {test_cd_mae_0:.5e}")
                print(f"test_cl_mse_0: {test_cl_mse_0:.5e}, test_cl_mae_0: {test_cl_mae_0:.5e}")
                print(f"test_cl_f_mse_0: {test_cl_f_mse_0:.5e}, test_cl_f_mae_0: {test_cl_f_mae_0:.5e}")
                print(f"test_cl_r_mse_0: {test_cl_r_mse_0:.5e}, test_cl_r_mae_0: {test_cl_r_mae_0:.5e}")
                print(f"all by mean")
                print(f"test_cd_mse_1: {test_cd_mse_1:.5e}, test_cd_mae_1: {test_cd_mae_1:.5e}")
                print(f"test_cl_mse_1: {test_cl_mse_1:.5e}, test_cl_mae_1: {test_cl_mae_1:.5e}")
                print(f"test_cl_f_mse_1: {test_cl_f_mse_1:.5e}, test_cl_f_mae_1: {test_cl_f_mae_1:.5e}")
                print(f"test_cl_r_mse_1: {test_cl_r_mse_1:.5e}, test_cl_r_mae_1: {test_cl_r_mae_1:.5e}")                
                print(f"time pre test epoch/s:{test_time:.2f}")
                print("--------------")
                
                with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                    file.write(f"Epoch: {epoch+1}/{EPOCH}, Test L2 sample: {test_press_l2_sample:.5e}, Test L2 all: {test_press_l2:.5e}\n")
                    file.write(f"sample\n")
                    file.write(f"test_cd_mse_0: {test_cd_mse_0:.5e}, test_cd_mae_0: {test_cd_mae_0:.5e}\n")
                    file.write(f"test_cl_mse_0: {test_cl_mse_0:.5e}, test_cl_mae_0: {test_cl_mae_0:.5e}\n")
                    file.write(f"test_cl_f_mse_0: {test_cl_f_mse_0:.5e}, test_cl_f_mae_0: {test_cl_f_mae_0:.5e}\n")
                    file.write(f"test_cl_r_mse_0: {test_cl_r_mse_0:.5e}, test_cl_r_mae_0: {test_cl_r_mae_0:.5e}\n")
                    file.write(f"all by mean\n")
                    file.write(f"test_cd_mse_1: {test_cd_mse_1:.5e}, test_cd_mae_1: {test_cd_mae_1:.5e}\n")
                    file.write(f"test_cl_mse_1: {test_cl_mse_1:.5e}, test_cl_mae_1: {test_cl_mae_1:.5e}\n")
                    file.write(f"test_cl_f_mse_1: {test_cl_f_mse_1:.5e}, test_cl_f_mae_1: {test_cl_f_mae_1:.5e}\n")
                    file.write(f"test_cl_r_mse_1: {test_cl_r_mse_1:.5e}, test_cl_r_mae_1: {test_cl_r_mae_1:.5e}\n")
                    file.write(f"time pre test epoch/s:{test_time:.2f}\n")
                
                writer.add_scalar('Loss/test_press_l2_sample', test_press_l2_sample, epoch)
                writer.add_scalar('Loss/test_press_l2', test_press_l2, epoch)
                
                writer.add_scalar('Loss/test_cd_mse_sample', test_cd_mse_0, epoch)
                writer.add_scalar('Loss/test_cd_mae_sample', test_cd_mae_0, epoch)
                writer.add_scalar('Loss/test_cl_mse_sample', test_cl_mse_0, epoch)
                writer.add_scalar('Loss/test_cl_mae_sample', test_cl_mae_0, epoch)
                writer.add_scalar('Loss/test_cl_f_mse_sample', test_cl_f_mse_0, epoch)
                writer.add_scalar('Loss/test_cl_f_mae_sample', test_cl_f_mae_0, epoch)
                writer.add_scalar('Loss/test_cl_r_mse_sample', test_cl_r_mse_0, epoch)
                writer.add_scalar('Loss/test_cl_r_mae_sample', test_cl_r_mae_0, epoch)
                
                writer.add_scalar('Loss/test_cd_mse_mean', test_cd_mse_1, epoch)
                writer.add_scalar('Loss/test_cd_mae_mean', test_cd_mae_1, epoch)
                writer.add_scalar('Loss/test_cl_mse_mean', test_cl_mse_1, epoch)
                writer.add_scalar('Loss/test_cl_mae_mean', test_cl_mae_1, epoch)
                writer.add_scalar('Loss/test_cl_f_mse_mean', test_cl_f_mse_1, epoch)
                writer.add_scalar('Loss/test_cl_f_mae_mean', test_cl_f_mae_1, epoch)
                writer.add_scalar('Loss/test_cl_r_mse_mean', test_cl_r_mse_1, epoch)
                writer.add_scalar('Loss/test_cl_r_mae_mean', test_cl_r_mae_1, epoch)

        if (epoch+1) % 100 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            
            nn_save_path = os.path.join(args.save_path, "nn")
            os.makedirs(nn_save_path, exist_ok=True)            
            torch.save(model.module.state_dict(), f"{nn_save_path}/{args.name}_epoch_{epoch}.pth")
        
    if local_rank == 0:
        writer.close()
    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    # save path
    path_logs = args.save_path + "/logs"
    path_nn = args.save_path + "/nn"
    path_record = args.save_path + "/record"
    
    # if not os.path.exists(path_logs):
    #     os.makedirs(path_logs)
    os.makedirs(path_logs, exist_ok=True)
    
    # if not os.path.exists(path_nn):
    #     os.makedirs(path_nn)
    os.makedirs(path_nn, exist_ok=True)
    
    # if not os.path.exists(path_record):
    #     os.makedirs(path_record)
    os.makedirs(path_record, exist_ok=True)
        
    
    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(str(args) + "\n")
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
        
    if args.seed is not None:
        set_seed(args.seed)

    world_size = torch.cuda.device_count()  # 获取GPU数量
    print(f"Let's use {world_size} GPUs!")

    main(args, path_logs, path_nn, path_record)
    
    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"time is {time.asctime( time.localtime(time.time()) )}\n")
        
        
'''
--------------
Epoch: 1/1, Train Loss: 1.12832e+00, Train Pressure L2: 1.13927e+00
time pre train epoch/s:783.28, current_lr:2.5000e-04
---Inference---
Epoch: 1/1, Test L2 sample: 9.45658e-01, out: 9.26093e-01
time pre test epoch/s:146.56
--------------
'''