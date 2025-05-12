import os
import time
import numpy as np
import random
import h5py

import torch
from torch.utils.data import DataLoader

# load
from src.dataset import Car_Dataset
from src.sample import merge_split_results

from src.model.physGTO import Model
from src.utils import set_seed, collate, init_weights, parse_args,RelLpLoss,get_l2_loss

from thop import profile

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#print(device)

def infer(args, model, test_dataloader, local_rank):
    
    h5file = h5py.File('ahmed_body_pred.h5', 'w')
    
    model.eval()
    test_press_l2_all = 0
    
    with torch.no_grad():
        
        # for all
        num = 0 
        for i, [batch_data, idx] in enumerate(test_dataloader):
            
            splits_data = batch_data['splits']
            info = batch_data['info'].to(local_rank) 
            ori_pressre = batch_data['ori_pressre'].to(local_rank) 

            processed_results = []
            
            for split_id, split_data in splits_data.items():
                
                sampled_centroids = split_data['centroids'].to(local_rank) 
                sampled_areas = split_data['areas'].to(local_rank) 
                sampled_edges = split_data['edges'].to(local_rank) 
                new_to_old = split_data['new_to_old']
                
                pred_pressure_norm = model(sampled_areas, sampled_centroids, sampled_edges, info)
                
                macs, params = profile(model, inputs=(sampled_areas, sampled_centroids, sampled_edges, info))
                flops = macs * 2
                print(f"params: {params/1e6:.2f}M, macs: {macs/1e9:.2f}G, FLOPs: {flops/1e9:.2f}G")
                
                
                
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
            
            break
        
        test_press_l2_all = test_press_l2_all / num
    h5file.close()

    return test_press_l2_all

def get_dataset(args):
    
    test_dataset = Car_Dataset(
        data_dir = args.dataset["data_path"], 
        mode="test",
        normalize = True,
        if_sample = False,
        sample_rate = args.dataset["test"]["sample_rate"],
        normalize_way = "z_score"
        )
    
    test_dataloader = DataLoader(test_dataset, 
                        batch_size=args.dataset["test"]["batchsize"], 
                        num_workers=args.dataset["test"]["num_workers"]
                        )
    
    return test_dataloader

def init_model(args, local_rank, flag):
    
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
    
    if flag:
        model.apply(init_weights)     
    else:
        # checkpoint_path = f"result/nn/{args.name}_epoch_{EPOCH-1}.pth" 
        checkpoint_path = "./checkpoints/ahmed_body_base_epo_500_gpu_4_epoch_499.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint) 
        
    return model


def main(args):

    # setting
    
    ######################################
    ######################################
    # load data
    test_dataloader = get_dataset(args)
    
    # load model
    model =  init_model(args, device, flag=False)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
    print(f"#params: {params}")
    
    start_time = time.time() 
    model.eval()
    test_press_l2 = infer(args, model, test_dataloader, device)
    end_time = time.time()
        
    test_time = (end_time - start_time)

    print("---Inference---")
    print(f"Test L2: {test_press_l2:.5e}")
    print(f"time pre test epoch/s:{test_time:.2f}")
    print("--------------")
            

    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    if args.seed is not None:
        set_seed(args.seed)

    world_size = torch.cuda.device_count() 
    print(f"Let's use {world_size} GPUs!")

    main(args)
    
        
        
'''
--------------
Epoch: 1/1, Train Loss: 1.12832e+00, Train Pressure L2: 1.13927e+00
time pre train epoch/s:783.28, current_lr:2.5000e-04
---Inference---
Epoch: 1/1, Test L2 sample: 9.45658e-01, out: 9.26093e-01
time pre test epoch/s:146.56
--------------
'''