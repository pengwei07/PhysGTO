
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import time
import os

from utils import count_params, LpLoss, UnitGaussianNormalizer
# from model import PhysGTO  
# from model_only_atten import PhysGTO
# from transolver import Model  
from src.physGTO import PhysGTO  

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def get_single_edges(unique_edges):
    # Ensure swap_mask is on the same device as unique_edges
    device = unique_edges.device
    swap_mask = (torch.rand(unique_edges.shape[0], device=device) > 0.5)

    # Apply swap based on the mask
    senders_swapped = torch.where(swap_mask, unique_edges[:, 1], unique_edges[:, 0])
    receivers_swapped = torch.where(swap_mask, unique_edges[:, 0], unique_edges[:, 1])
    
    # Stack the swapped edges back together
    randomized_unique_edges = torch.stack([senders_swapped, receivers_swapped], dim=-1)
    
    return randomized_unique_edges

def make_edges(faces):

    edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0)
    
    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)
    
    packed_edges = torch.stack([senders, receivers], dim=-1).int() # dim = [edges, 2]
    unique_edges = torch.unique(packed_edges, dim=0)
    
    # unique_edges = get_single_edges(unique_edges)
    # unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0) # dim = [edges, 2]

    return unique_edges

def main(args):  

    PATH = args.data_dir
 
    ntrain = args.num_train
    ntest = args.num_test
    
    batch_size = args.batch_size
    learning_rate = args.lr
    
    epochs = args.epochs
    ################################################################
    # reading data
    ################################################################
     
    data = sio.loadmat(PATH)
    
    Points = data['MeshNodes'].T
    Elements = np.array(data['MeshElements'].T - 1, dtype=np.int32)
    
    node_pos = torch.tensor(Points, dtype=torch.float32)
    Elements = torch.tensor(Elements, dtype=torch.int32)
    edges = make_edges(Elements)
    
    xx  = (node_pos[:,0] - node_pos[:,0].min()) / (node_pos[:,0].max() - node_pos[:,0].min())
    yy  = (node_pos[:,1] - node_pos[:,1].min()) / (node_pos[:,1].max() - node_pos[:,1].min())
    node_pos = torch.stack((xx, yy), dim=-1)
    
    # xx = (node_pos[:,0] - node_pos[:,0].mean()) / (node_pos[:,0].std())
    # yy = (node_pos[:,1] - node_pos[:,1].mean()) / (node_pos[:,1].std())
    # node_pos = torch.stack((xx, yy), dim=-1)

    # Points: (2290, 2) Elements: (4338, 3) edges: torch.Size([6627, 2])
    
    y_dataIn = torch.Tensor(data['u_field']) 
    x_dataIn = torch.Tensor(data['c_field'])
    
    x_data = x_dataIn 
    y_data = y_dataIn 
    
    ################################################################
    # normalization
    ################################################################   
    
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    
    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    print('x_test:', x_test.shape, 'y_test:', y_test.shape)
    
    norm_x  = UnitGaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train)
    x_test  = norm_x.encode(x_test)

    norm_y  = UnitGaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)
    
    x_train = x_train.reshape(ntrain,-1,1)
    x_test = x_test.reshape(ntest,-1,1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    model = PhysGTO().to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
    print('Number of parameters:', params)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/100)
    
    myloss = LpLoss(size_average=False)
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    train_att_loss = np.zeros((epochs))

    for ep in range(epochs):
        model.train()
        train_mse = 0
        train_l2 = 0
        train_att = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
    
            optimizer.zero_grad()
            out = model(x, node_pos.to(device), edges.to(device))
    
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            
            l2.backward() 
            
            out_real = norm_y.decode(out.view(batch_size, -1).cpu())
            y_real = norm_y.decode(y.view(batch_size, -1).cpu())
            train_l2 += myloss(out_real, y_real).item()   

            optimizer.step()
            train_mse += mse.item()
    
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, node_pos.to(device), edges.to(device))
                
                out_real = norm_y.decode(out.view(batch_size, -1).cpu())
                y_real   = norm_y.decode(y.view(batch_size, -1).cpu())
                test_l2 += myloss(out_real, y_real).item()                

        train_mse /= len(train_loader)
        
        train_l2 /= ntrain
        test_l2  /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        train_att_loss[ep] = train_att
        
        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        
        if ep % 1 == 0:
            print('Step: %d, Train L2: %.5f, Test L2: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, T))
            print(f"current_lr:{scheduler.get_last_lr()[0]:.4e}")
            
        time_step = time.perf_counter()

    print("Training done...")
    
    if not os.path.exists("result"):
        os.makedirs("result")
    np.save(f'result/{args.case_name}_train_error.npy', train_error)
    np.save(f'result/{args.case_name}_test_error.npy', test_error)
    
    torch.save(model.state_dict(), f'result/{args.case_name}_model.pth')

def set_seed(seed: int = 0):    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            

    for args in [
                    { 
                    'seed': 42,
                    'edges_sample_ratio': 1,
                    'batch_size'    : 50, 
                    'epochs'    : 500,
                    'data_dir'  : '../../data/task1/Darcy.mat',
                    'num_train' : 1000, 
                    'num_test'  : 200,
                    'lr'        : 1e-3,
                    'case_name': "physGTO"
                    },
                ]:
            
        args = objectview(args)
                
    set_seed(args.seed)
    main(args)

    