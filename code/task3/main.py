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

def train(args, model, train_dataloader, optim, local_rank):
    
    model.train()
    train_loss = 0
    train_press_l2 = 0        
    num = 0
    
    for i, [splits_data, info] in enumerate(train_dataloader):
        
        optim.zero_grad()

        sampled_pressure = splits_data['pressure'].to(local_rank)
        sampled_pressure_norm = (sampled_pressure - args.dataset["press_mean"]) / args.dataset["press_std"]
        
        sampled_centroids = splits_data['centroids'].to(local_rank)
        sampled_areas = splits_data['areas'].to(local_rank)
        sampled_edges = splits_data['edges']
        
        # forward
        pred_pressure_norm = model(sampled_areas, sampled_centroids, sampled_edges, info)
        
        loss_l2_norm = get_l2_loss(pred_pressure_norm, sampled_pressure_norm)
        
        pred_pressure = pred_pressure_norm * args.dataset["press_std"] + args.dataset["press_mean"]

        loss_l2 = get_l2_loss(pred_pressure, sampled_pressure).item()
            
        # backward
        loss_l2_norm.backward()
        optim.step()

        train_loss += loss_l2_norm.item()
        train_press_l2 += loss_l2
        num = num + 1
        
        # break
    
    return train_loss / num, train_press_l2 / num

def infer(args, model, test_dataloader_sample, test_dataloader, local_rank):
    
    model.eval()
    test_press_l2_sample = 0
    test_press_l2_all = 0
    
    with torch.no_grad():
        
        # for sample
        num = 0 
        for i, [splits_data, info] in enumerate(test_dataloader_sample):

            sampled_pressure = splits_data['pressure'].to(local_rank)
            sampled_centroids = splits_data['centroids'].to(local_rank)
            sampled_areas = splits_data['areas'].to(local_rank)
            sampled_edges = splits_data['edges']
            
            # forward
            pred_pressure_norm = model(sampled_areas, sampled_centroids, sampled_edges, info)            
            pred_pressure = pred_pressure_norm * args.dataset["press_std"] + args.dataset["press_mean"]

            loss_l2 = get_l2_loss(pred_pressure, sampled_pressure).item()
    
            test_press_l2_sample += loss_l2
            num = num + 1
            
            # break
        
        test_press_l2_sample = test_press_l2_sample / num
        
        # for all
        num = 0 
        for i, batch_data in enumerate(test_dataloader):
            
            splits_data = batch_data['splits']
            info = batch_data['info']
            ori_pressre = batch_data['ori_pressre'].to(local_rank) 

            processed_results = []
            
            for split_id, split_data in splits_data.items():
                
                sampled_centroids = split_data['centroids'].to(local_rank) 
                sampled_areas = split_data['areas'].to(local_rank) 
                sampled_edges = split_data['edges']
                new_to_old = split_data['new_to_old']
                
                pred_pressure_norm = model(sampled_areas, sampled_centroids, sampled_edges, info)
                pred_pressure = pred_pressure_norm * args.dataset["press_std"] + args.dataset["press_mean"]

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
            
            num = num + 1
            
            # break
        
        test_press_l2_all = test_press_l2_all / num

    return test_press_l2_sample, test_press_l2_all

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
        n_token = args.model["n_token"]
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
        train_loss, train_press_l2 = train(args, model, train_dataloader, optim, local_rank)
        end_time = time.time()
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        current_lr = torch.tensor(current_lr, device=local_rank)

        training_time = (end_time - start_time)
        training_time = torch.tensor(training_time, device=local_rank)
        training_time = gather_tensor(training_time, world_size)
        
        train_loss = gather_tensor(torch.tensor(train_loss, device=local_rank), world_size)
        train_press_l2 = gather_tensor(torch.tensor(train_press_l2, device=local_rank), world_size)
        
        if local_rank == 0:
            
            writer.add_scalar('Loss/train_loss', train_loss, epoch)
            writer.add_scalar('Loss/train_press_l2', train_press_l2, epoch)
            
            print(f"Epoch: {epoch+1}/{EPOCH}, Train Loss: {train_loss:.5e}, Train Pressure L2: {train_press_l2:.5e}")
            print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}")
            
            with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                file.write(f"Epoch: {epoch+1}/{EPOCH}, Train Loss: {train_loss:.5e}, Train Pressure L2: {train_press_l2:.5e}\n")
                file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n")
        
        ###############################################################
        if (epoch+1) % 5 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            
            start_time = time.time() 
            model.eval()
            test_press_l2_sample, test_press_l2 = infer(args, model, test_dataloader_sample, test_dataloader, local_rank)
            end_time = time.time()
            
            test_time = (end_time - start_time)
            test_time = torch.tensor(test_time, device=local_rank)
            test_time = gather_tensor(test_time, world_size)
            
            test_press_l2_sample = gather_tensor(torch.tensor(test_press_l2_sample, device=local_rank), world_size)
            test_press_l2 = gather_tensor(torch.tensor(test_press_l2, device=local_rank), world_size)
            
            if local_rank == 0:
                
                print("---Inference---")
                print(f"Epoch: {epoch+1}/{EPOCH}, Test L2 sample: {test_press_l2_sample:.5e}, out: {test_press_l2:.5e}")
                print(f"time pre test epoch/s:{test_time:.2f}")
                print("--------------")
                
                with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                    file.write(f"Epoch: {epoch+1}/{EPOCH}, Test L2 sample: {test_press_l2_sample:.5e}, Test L2 all: {test_press_l2:.5e}\n")
                    file.write(f"time pre test epoch/s:{test_time:.2f}\n")
                
                writer.add_scalar('Loss/test_press_l2_sample', test_press_l2_sample, epoch)
                writer.add_scalar('Loss/test_press_l2', test_press_l2, epoch)

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

    world_size = torch.cuda.device_count()
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