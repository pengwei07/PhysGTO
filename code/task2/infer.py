import torch
import numpy as np
import os
import time

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR

from tensorboardX import SummaryWriter
import argparse
from types import SimpleNamespace
import json

# 1. dataset
from src.dataset.heat_flow import Heat_Flow_Dataset

# 2. model
from src.model.PhysGTO import Model

from utils import set_seed, init_weights, collate

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')  # Change the default config file name if needed
    parser.add_argument('--model_path', default='result/GNOT_cylinder2d/GNOT_cylinder2d_epo_1000_1000.nn', type=str, help='Path to model file')  # Change the default config file name if needed
    parser.add_argument('--horizon_test', default=5, type=int, help='Path to model file')  # Change the default config file name if needed

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)  # Load JSON instead of YAML
    
    args = SimpleNamespace(**config, **vars(args))
    
    return args

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

def main(args):
    # setting
    local_rank, world_size = setup()
    args.dataset["horizon_test"] = args.horizon_test
    # data
    ######################################
    # load data
    if args.dataset["name"] == "heat_flow":
        test_dataset = Heat_Flow_Dataset(
            data_path = args.dataset["data_path"], 
            mode="test",
            all_length = args.dataset["length"],
            delta_t = args.dataset["delta_t"],
            input_step = args.dataset["input_step"],
            window_length = args.horizon_test,
            model_name = args.model["name"]
        )
        
    # sampler
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,  shuffle=False, rank=local_rank)
    
    test_dataloader = DataLoader(test_dataset, 
                        batch_size=1, 
                        sampler= test_sampler,
                        num_workers=args.dataset["test"]["num_workers"],
                        collate_fn=collate)
     
    # model
    ######################################
    if args.model["name"] == "PhysGTO":
        model = Model(
                space_size = args.model["space_size"], 
                pos_enc_dim = args.model["pos_enc_dim"], 
                cond_dim = args.model["cond_dim"], 
                N_block = args.model["N_block"], 
                in_dim = args.model["in_dim"],  
                out_dim = args.model["out_dim"],
                enc_dim = args.model["enc_dim"], 
                n_head = args.model["n_head"],
                n_token = args.model["n_token"]
            ).to(local_rank)
    else:
        raise ValueError
    
    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
        
    model = DDP(model, device_ids=[local_rank])
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
     
    if local_rank == 0:
        
        print("---test_dataloader---")
        for i, data in enumerate(test_dataloader):
            for key in data.keys():
                print(key, data[key].shape)
            break

        print("---------")      
        print(f"No. of test samples: {len(test_dataset)}")
        print(f"No. of test batches: {len(test_dataloader)}")
        print("---------")
        print(f"#params: {params}")     
        
    
        if not os.path.exists(f"{args.save_path}/record/"):
            os.makedirs(f"{args.save_path}/record/")
            
        with open(f"{args.save_path}/record/{args.name}_testing_log.txt", "a") as file:
            file.write(f"No. of test samples: {len(test_dataset)}\n")
            file.write(f"No. of test batches: {len(test_dataloader)}\n")
            file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
            file.write(f"{args.name}, #params: {params}\n")

        log_dir = f"{args.save_path}/logs/{args.name}/rank_{local_rank}"
        os.makedirs(log_dir, exist_ok=True)
        
    # train
    ######################################

    if args.dataset["name"] == "cylinder_flow":
        from src.train.train_cylinder import train, validate
    elif args.dataset["name"] == "uav_flow":
        from src.train.train_uav import train, validate
    elif args.dataset["name"] == "plasma":
        from src.train.train_plasma import train, validate
    elif args.dataset["name"] == "heat_flow":
        from src.train.train_heat_flow import train, validate
        
    
    # test
    start_time = time.time() 
    test_error = validate(args, model, test_dataloader, local_rank, if_save=False)
    end_time = time.time()
    
    training_time1 = (end_time - start_time)
    training_time1 = torch.tensor(training_time1, device=local_rank)
    training_time1 = gather_tensor(training_time1, world_size)
    #######################
    if args.dataset["name"] == "cylinder_flow":
        test_L2_u = gather_tensor(torch.tensor(test_error['L2_u'], device=local_rank), world_size)
        test_L2_v = gather_tensor(torch.tensor(test_error['L2_v'], device=local_rank), world_size)
        test_L2_p = gather_tensor(torch.tensor(test_error['L2_p'], device=local_rank), world_size)
        test_mean_l2 = gather_tensor(torch.tensor(test_error['mean_l2'], device=local_rank), world_size)   
        
        test_RMSE_u = gather_tensor(torch.tensor(test_error['RMSE_u'], device=local_rank), world_size)
        test_RMSE_p = gather_tensor(torch.tensor(test_error['RMSE_p'], device=local_rank), world_size)
        
        test_each_t_l2 = gather_tensor(test_error['each_l2'].clone().detach().to(local_rank), world_size)
        
        if local_rank == 0:
            print("---Inference---")
            print(f"test_mean_l2: {test_mean_l2:.4e}")
            print(f"L2 loss: u {test_L2_u:.4e}, v: {test_L2_v:.4e}, p: {test_L2_p:.4e}")
            print(f"each time step loss: {test_each_t_l2}")
            print(f"RMSE loss: U {test_RMSE_u:.4e}, P: {test_RMSE_p:.4e}")
            print(f"time pre test epoch/s:{training_time1:.2f}")
            print("--------------")
            
            with open(f"{args.save_path}/record/{args.name}_testing_log.txt", "a") as file:
                file.write(f"Inference, test_mean_l2: {test_mean_l2:.4e}\n")
                file.write(f"Test: L2_u: {test_L2_u:.4e}, L2_v: {test_L2_v:.4e}, L2_p: {test_L2_p:.4e}\n")
                file.write(f"each time step loss: {test_each_t_l2}\n")
                file.write(f"Test: RMSE_u: {test_RMSE_u:.4e}, RMSE_p: {test_RMSE_p:.4e}\n")
                file.write(f"time pre test epoch/s:{training_time1:.2f}\n") 
    elif args.dataset["name"] == "uav_flow":
        test_L2_u = gather_tensor(torch.tensor(test_error['L2_u'], device=local_rank), world_size)
        test_L2_v = gather_tensor(torch.tensor(test_error['L2_v'], device=local_rank), world_size)
        test_L2_ps = gather_tensor(torch.tensor(test_error['L2_ps'], device=local_rank), world_size)
        test_L2_pg = gather_tensor(torch.tensor(test_error['L2_pg'], device=local_rank), world_size)
        test_mean_l2 = gather_tensor(torch.tensor(test_error['mean_l2'], device=local_rank), world_size)   
        
        test_RMSE_u = gather_tensor(torch.tensor(test_error['RMSE_u'], device=local_rank), world_size)
        test_RMSE_p = gather_tensor(torch.tensor(test_error['RMSE_p'], device=local_rank), world_size)
        
        test_each_t_l2 = gather_tensor(test_error['each_l2'].clone().detach().to(local_rank), world_size)
        
        if local_rank == 0:
            print("---Inference---")
            print(f"test_mean_l2: {test_mean_l2:.4e}")
            print(f"L2 loss: u {test_L2_u:.4e}, v: {test_L2_v:.4e}, ps: {test_L2_ps:.4e}, pg: {test_L2_pg:.4e}")
            print(f"each time step loss: {test_each_t_l2}")
            print(f"RMSE loss: U {test_RMSE_u:.4e}, P: {test_RMSE_p:.4e}")
            print(f"time pre test epoch/s:{training_time1:.2f}")
            print("--------------")
            
            with open(f"{args.save_path}/record/{args.name}_testing_log.txt", "a") as file:
                file.write(f"Inference, test_mean_l2: {test_mean_l2:.4e}\n")
                file.write(f"Test: L2_u: {test_L2_u:.4e}, L2_v: {test_L2_v:.4e}, L2_ps: {test_L2_ps:.4e}, L2_pg: {test_L2_pg:.4e}\n")
                file.write(f"each time step loss: {test_each_t_l2}\n")
                file.write(f"Test: RMSE_u: {test_RMSE_u:.4e}, RMSE_p: {test_RMSE_p:.4e}\n")
                file.write(f"time pre test epoch/s:{training_time1:.2f}\n") 
    elif args.dataset["name"] == "plasma":
        test_L2_ne = gather_tensor(torch.tensor(test_error['L2_ne'], device=local_rank), world_size)
        test_L2_te = gather_tensor(torch.tensor(test_error['L2_te'], device=local_rank), world_size)
        test_L2_v = gather_tensor(torch.tensor(test_error['L2_v'], device=local_rank), world_size)
        test_L2_T = gather_tensor(torch.tensor(test_error['L2_T'], device=local_rank), world_size)
        test_mean_l2 = gather_tensor(torch.tensor(test_error['mean_l2'], device=local_rank), world_size)   
        
        test_RMSE = gather_tensor(torch.tensor(test_error['RMSE'], device=local_rank), world_size)
        
        test_each_t_l2 = gather_tensor(test_error['each_l2'].clone().detach().to(local_rank), world_size)
        
        if local_rank == 0:
            print("---Inference---")
            print(f"test_mean_l2: {test_mean_l2:.4e}")
            print(f"L2 loss: ne {test_L2_ne:.4e}, te: {test_L2_te:.4e}, v: {test_L2_v:.4e}, T: {test_L2_T:.4e}")
            print(f"each time step loss: {test_each_t_l2}")
            print(f"RMSE loss: {test_RMSE:.4e}")
            print(f"time pre test epoch/s:{training_time1:.2f}")
            print("--------------")
            
            with open(f"{args.save_path}/record/{args.name}_testing_log.txt", "a") as file:
                file.write(f"Inference, test_mean_l2: {test_mean_l2:.4e}\n")
                file.write(f"Test: L2_ne: {test_L2_ne:.4e}, L2_te: {test_L2_te:.4e}, L2_v: {test_L2_v:.4e}, L2_T: {test_L2_T:.4e}\n")
                file.write(f"each time step loss: {test_each_t_l2}\n")
                file.write(f"Test: RMSE: {test_RMSE:.4e}\n")
                file.write(f"time pre test epoch/s:{training_time1:.2f}\n") 
    elif args.dataset["name"] == "heat_flow":
        test_L2_u = gather_tensor(torch.tensor(test_error['L2_u'], device=local_rank), world_size)
        test_L2_v = gather_tensor(torch.tensor(test_error['L2_v'], device=local_rank), world_size)
        test_L2_p = gather_tensor(torch.tensor(test_error['L2_p'], device=local_rank), world_size)
        test_L2_T = gather_tensor(torch.tensor(test_error['L2_T'], device=local_rank), world_size)
        test_mean_l2 = gather_tensor(torch.tensor(test_error['mean_l2'], device=local_rank), world_size)   
        
        test_RMSE_u = gather_tensor(torch.tensor(test_error['RMSE_u'], device=local_rank), world_size)
        test_RMSE_p = gather_tensor(torch.tensor(test_error['RMSE_p'], device=local_rank), world_size)
        test_RMSE_T = gather_tensor(torch.tensor(test_error['RMSE_T'], device=local_rank), world_size)
        
        test_each_t_l2 = gather_tensor(test_error['each_l2'].clone().detach().to(local_rank), world_size)
        
        if local_rank == 0:
            print("---Inference---")
            print(f"test_mean_l2: {test_mean_l2:.4e}")
            print(f"L2 loss: u {test_L2_u:.4e}, v: {test_L2_v:.4e}, p: {test_L2_p:.4e}, T: {test_L2_T:.4e}")
            print(f"each time step loss: {test_each_t_l2}")
            print(f"RMSE loss: U {test_RMSE_u:.4e}, P: {test_RMSE_p:.4e}, T: {test_RMSE_T:.4e}")
            print(f"time pre test epoch/s:{training_time1:.2f}")
            print("--------------")
            
            with open(f"{args.save_path}/record/{args.name}_testing_log.txt", "a") as file:
                file.write(f"Inference, test_mean_l2: {test_mean_l2:.4e}\n")
                file.write(f"Test: L2_u: {test_L2_u:.4e}, L2_v: {test_L2_v:.4e}, L2_p: {test_L2_p:.4e}, L2_T: {test_L2_T:.4e}\n")
                file.write(f"each time step loss: {test_each_t_l2}\n")
                file.write(f"Test: RMSE_u: {test_RMSE_u:.4e}, RMSE_p: {test_RMSE_p:.4e}, RMSE_T: {test_RMSE_T:.4e}\n")
                file.write(f"time pre test epoch/s:{training_time1:.2f}\n") 
        

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    if args.seed is not None:
        set_seed(args.seed)

    # main(args)
    if args.train["if_multi_gpu"]:
        world_size = torch.cuda.device_count()  # 获取GPU数量
        print(f"Let's use {world_size} GPUs!")
    
    main(args)