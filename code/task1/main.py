
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import time
import pandas as pd
import os
from thop import profile

from functools import reduce
import operator
 
from src.model import physGTO
from src.get_mesh_info import get_mesh_edges
from src.utils import count_params,LpLoss,UnitGaussianNormalizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def main(args, save_path):  
    
    with open(f"{save_path}/{args.CaseName}_training_log.txt", "a") as file:
        file.write(str(args) + "\n")
        
    PATH = args.data_dir
 
    ntrain = args.num_train
    ntest = args.num_test
    
    batch_size = args.batch_size
    learning_rate = args.lr
    
    epochs = args.epochs
    ################################################################
    # reading data and calculating LBO basis
    ################################################################
    
    data = sio.loadmat(PATH)
    
    Points = data['MeshNodes'].T
    Elements = np.array(data['MeshElements'].T - 1, dtype=np.int32)
    
    node_pos = torch.tensor(Points, dtype=torch.float32)
    Elements = torch.tensor(Elements, dtype=torch.int32)
    edges = get_mesh_edges(Elements)

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
    
    print(f"node_pos: {node_pos.shape}, edges: {edges.shape}")
    
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
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    
    model = physGTO().to(device)
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

    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
    
            optimizer.zero_grad()
            out = model(x, node_pos.to(device), edges.to(device))
    
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() 
            
            out_real = norm_y.decode(out.view(batch_size, -1).cpu())
            y_real = norm_y.decode(y.view(batch_size, -1).cpu())
            train_l2 += myloss(out_real, y_real).item()   

            optimizer.step()
    
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

        train_l2 /= ntrain
        test_l2  /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        
        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        
        print('Step: %d, Train L2: %.5f, Test L2 error: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, T))
        with open(f"{save_path}/{args.CaseName}_training_log.txt", "a") as file:
            file.write(f"Step: {ep}, Train L2: {train_l2:.5f}, Test L2: {test_l2:.5f}, Time: {T:.3f}s\n")
            file.write(f"current_lr:{scheduler.get_last_lr()[0]:.4e}\n")
            
        time_step = time.perf_counter()

    print("Training done...")
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=1, shuffle=False)
    pre_train = torch.zeros(y_train.shape)
    y_train   = torch.zeros(y_train.shape)
    x_train   = torch.zeros(x_train.shape[0:2])

    with torch.no_grad():
        index = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, node_pos.to(device), edges.to(device))

            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())
            
            pre_train[index,:] = out_real
            y_train[index,:]   = y_real
            x_train[index]   = norm_x.decode(x.view(1, -1).cpu())
            
            index = index + 1
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)
    y_test   = torch.zeros(y_test.shape)
    x_test   = torch.zeros(x_test.shape[0:2])
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x, node_pos.to(device), edges.to(device))
            
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())
            x_real   = norm_x.decode(x.view(1, -1).cpu())
            
            pre_test[index,:] = out_real
            y_test[index,:] = y_real
            x_test[index] = x_real
            
            index = index + 1
            
    # ================ Save Data ====================
    current_directory = save_path
    sava_path = current_directory + "/logs/" + args.CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)
    
    dataframe = pd.DataFrame({'Test_loss' : [test_l2],
                              'Train_loss': [train_l2],
                              'num_paras' : [count_params(model)],
                              'train_time':[time_step_end - time_start]})
    
    dataframe.to_csv(sava_path + 'log.csv', index = False, sep = ',')
    
    loss_dict = {'train_error' :train_error,
                 'test_error'  :test_error}
    
    pred_dict = {   'pre_test' : pre_test.cpu().detach().numpy(),
                    'pre_train': pre_train.cpu().detach().numpy(),
                    'x_test'   : x_test.cpu().detach().numpy(),
                    'x_train'  : x_train.cpu().detach().numpy(),
                    'y_test'   : y_test.cpu().detach().numpy(),
                    'y_train'  : y_train.cpu().detach().numpy(),
                    }
    
    sio.savemat(sava_path +'NORM_loss.mat', mdict = loss_dict)                                                     
    sio.savemat(sava_path +'NORM_pre.mat', mdict = pred_dict)
    
    test_l2 = (myloss(y_test, pre_test).item())/ntest
    print('\nTesting error: %.3e'%(test_l2))
    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))
    
    torch.save(model.state_dict(), f'{save_path}/{args.CaseName}.pth')
    
def set_seed(seed: int = 0):    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    
    save_path = './result'
    os.makedirs(save_path, exist_ok=True)

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            
    for i in range(3):
        
        print('====================================')
        print('NO.'+str(i)+' repetition......')
        print('====================================')
        
        for args in [
                        { 
                          'batch_size'    : 50, 
                          'epochs'    : 500,
                          'data_dir'  : '../../data/task1/Darcy',
                          'num_train' : 1000, 
                          'num_test'  : 200,
                          'CaseName'  : 'physGTO_' + str(i), 
                          'lr'        : 1e-3},
                    ]:
            
            args = objectview(args)
        
        set_seed(42)
        main(args, save_path)

    