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

# 1. dataset
from src.dataset.heat_flow import Heat_Flow_Dataset

# 2. model
from src.model.PhysGTO import Model

from utils import set_seed, init_weights, parse_args, collate

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def main(args):
    # setting
    local_rank, world_size = setup()
    EPOCH = args.train["epoch"]
    real_lr = float(args.train["lr"])

    # data
    ######################################
    # load data
    if args.dataset["name"] == "heat_flow":
        train_dataset = Heat_Flow_Dataset(
            data_path = args.dataset["data_path"], 
            mode="train",
            all_length = args.dataset["length"],
            delta_t = args.dataset["delta_t"],
            input_step = args.dataset["input_step"],
            window_length = args.dataset["horizon_train"],
            model_name = args.model["name"]
        )
        test_dataset = Heat_Flow_Dataset(
            data_path = args.dataset["data_path"], 
            mode="test",
            all_length = args.dataset["length"],
            delta_t = args.dataset["delta_t"],
            input_step = args.dataset["input_step"],
            window_length = args.dataset["horizon_test"],
            model_name = args.model["name"]
        )

        
    # sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, seed=args.seed, rank=local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,  shuffle=False, rank=local_rank)
    
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=args.dataset["train"]["batchsize"], 
                        sampler=train_sampler,
                        num_workers=args.dataset["train"]["num_workers"],
                        collate_fn=collate)

    test_dataloader = DataLoader(test_dataset, 
                        batch_size=args.dataset["test"]["batchsize"], 
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
    
    if args.model["if_init"]:
        model.apply(init_weights)
        
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
     
    if local_rank == 0:
        
        print("---train_dataloader---")
        for i, data in enumerate(train_dataloader):
            for key in data.keys():
                print(key, data[key].shape)
            break
        print("---test_dataloader---")
        for i, data in enumerate(test_dataloader):
            for key in data.keys():
                print(key, data[key].shape)
            break

        print("---------")      
        print(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}")
        print(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}")
        print("---------")
        print(f"EPOCH: {EPOCH}, #params: {params}")       
    
        if not os.path.exists(f"{args.save_path}/record/"):
            os.makedirs(f"{args.save_path}/record/")
            
        with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
            file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
            file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
            file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
            file.write(f"{args.name}, #params: {params}\n")
            file.write(f"EPOCH: {EPOCH}\n")        

        log_dir = f"{args.save_path}/logs/{args.name}/rank_{local_rank}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
    # train
    ######################################
    real_lr = float(args.train["lr"])
    
    if args.model["name"] == "PhysGTO":
        optimizer = torch.optim.AdamW(model.parameters(), lr=real_lr, weight_decay=real_lr/100.0)
        if EPOCH == 1:
            scheduler = CosineAnnealingLR(optimizer, T_max= EPOCH, eta_min = real_lr)  
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max= EPOCH, eta_min = real_lr/50)
    else:
        raise ValueError
        
        
    if args.dataset["name"] == "heat_flow":
        from src.train.train_heat_flow import train, validate
        
    for epoch in range(EPOCH):
        start_time = time.time()
        train_error = train(args, model, train_dataloader, optimizer, local_rank)
        end_time = time.time()

        # 获取当前的学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  
        # get_last_lr返回的是列表，我们需要第一个元素，即当前的学习率
        
        training_time = (end_time - start_time)
        training_time = torch.tensor(training_time, device=local_rank)
        
        current_lr = torch.tensor(current_lr, device=local_rank)
        
        current_lr = gather_tensor(current_lr, world_size)
        training_time = gather_tensor(training_time, world_size)
        
        if args.dataset["name"] == "heat_flow":
            train_loss = gather_tensor(torch.tensor(train_error['loss'], device=local_rank), world_size)
        
            L2_u = gather_tensor(torch.tensor(train_error['L2_u'], device=local_rank), world_size)
            L2_v = gather_tensor(torch.tensor(train_error['L2_v'], device=local_rank), world_size)
            L2_p = gather_tensor(torch.tensor(train_error['L2_p'], device=local_rank), world_size)
            L2_T = gather_tensor(torch.tensor(train_error['L2_T'], device=local_rank), world_size)
            train_mean_l2 = gather_tensor(torch.tensor(train_error['mean_l2'], device=local_rank), world_size)

            RMSE_u = gather_tensor(torch.tensor(train_error['RMSE_u'], device=local_rank), world_size)
            RMSE_p = gather_tensor(torch.tensor(train_error['RMSE_p'], device=local_rank), world_size)
            RMSE_T = gather_tensor(torch.tensor(train_error['RMSE_T'], device=local_rank), world_size)
            
            each_t_l2 = gather_tensor(train_error['each_l2'].clone().detach().to(local_rank), world_size)
            
            if local_rank == 0:

                print(f"Training, Epoch: {epoch + 1}/{EPOCH}")
                print(f"Train Loss: {train_loss:.4e}")
                print(f"L2 loss: u {L2_u:.4e}, v: {L2_v:.4e}, p: {L2_p:.4e}, T: {L2_T:.4e}, train_mean_l2: {train_mean_l2:.4e}")
                print(f"each time step loss: {each_t_l2}")
                print(f"RMSE loss: U {RMSE_u:.4e}, P: {RMSE_p:.4e}, T: {RMSE_T:.4e}")
                print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}")
                
                writer.add_scalar('lr/lr', current_lr, epoch)
                writer.add_scalar('Loss/train', train_loss, epoch)
                
                writer.add_scalar('L2/train_mean_l2', train_mean_l2, epoch)
                writer.add_scalar('L2/train_L2_u', L2_u, epoch)
                writer.add_scalar('L2/train_L2_v', L2_v, epoch)
                writer.add_scalar('L2/train_L2_p', L2_p, epoch)
                writer.add_scalar('L2/train_L2_T', L2_T, epoch)
                
                writer.add_scalar('RMSE/train_RMSE_u', RMSE_u, epoch)
                writer.add_scalar('RMSE/train_RMSE_p', RMSE_p, epoch)
                writer.add_scalar('RMSE/train_RMSE_T', RMSE_T, epoch)
                
                with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                    file.write(f"Training, epoch: {epoch + 1}/{EPOCH}\n")
                    file.write(f"Train Loss: {train_loss:.4e}\n")
                    file.write(f"L2 loss: u {L2_u:.4e}, v: {L2_v:.4e}, p: {L2_p:.4e}, T: {L2_T:.4e}, train_mean_l2: {train_mean_l2:.4e}\n")
                    file.write(f"each time step loss: {each_t_l2}\n")
                    file.write(f"RMSE loss: U {RMSE_u:.4e}, P: {RMSE_p:.4e}, T: {RMSE_T:.4e}\n")
                    file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n")
        if (epoch+1) % 5 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            # test
            start_time = time.time() 
            test_error = validate(args, model, test_dataloader, local_rank)
            end_time = time.time()
            
            training_time1 = (end_time - start_time)
            training_time1 = torch.tensor(training_time1, device=local_rank)
            training_time1 = gather_tensor(training_time1, world_size)
            #######################
            if args.dataset["name"] == "heat_flow":
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
                    print(f"Epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}")
                    print(f"L2 loss: u {test_L2_u:.4e}, v: {test_L2_v:.4e}, p: {test_L2_p:.4e}, T: {test_L2_T:.4e}")
                    print(f"each time step loss: {test_each_t_l2}")
                    print(f"RMSE loss: U {test_RMSE_u:.4e}, P: {test_RMSE_p:.4e}, T: {test_RMSE_T:.4e}")
                    print(f"time pre test epoch/s:{training_time1:.2f}")
                    print("--------------")
                    
                    writer.add_scalar('L2/test_mean_l2', test_mean_l2, epoch)
                    writer.add_scalar('L2/test_L2_u', test_L2_u, epoch)
                    writer.add_scalar('L2/test_L2_v', test_L2_v, epoch)
                    writer.add_scalar('L2/test_L2_p', test_L2_p, epoch)
                    writer.add_scalar('L2/test_L2_T', test_L2_T, epoch)
                    
                    writer.add_scalar('RMSE/test_RMSE_u', test_RMSE_u, epoch)
                    writer.add_scalar('RMSE/test_RMSE_p', test_RMSE_p, epoch)
                    writer.add_scalar('RMSE/test_RMSE_T', test_RMSE_T, epoch)
                    
                    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                        file.write(f"Inference, epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}\n")
                        file.write(f"Test: L2_u: {test_L2_u:.4e}, L2_v: {test_L2_v:.4e}, L2_p: {test_L2_p:.4e}, L2_T: {test_L2_T:.4e}\n")
                        file.write(f"each time step loss: {test_each_t_l2}\n")
                        file.write(f"Test: RMSE_u: {test_RMSE_u:.4e}, RMSE_p: {test_RMSE_p:.4e}, RMSE_T: {test_RMSE_T:.4e}\n")
                        file.write(f"time pre test epoch/s:{training_time1:.2f}\n") 
        
        if (epoch+1) % 50 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            if args.if_save:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.train["if_multi_gpu"] else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': scheduler.get_last_lr()[0],  # 获取当前学习率
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                torch.save(checkpoint, f"{nn_save_path}/{args.name}_{epoch+1}.nn")

    if local_rank == 0:
        writer.close()
        
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