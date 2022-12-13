"""
Adversarial training for availability attack
"""
from tqdm import tqdm
from algorithms.milp import MILP_Avai
import os, sys
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import torch

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def train_epoch(model, train_loader, optimizer, criterion, config, args, flexible_feature, fixed_feature):
    
    # during the adversarial training, the model is updated for each batch
    # the attacker is also updated for each batch
    device_nn = config['device_nn']
    device_milp = config['device_milp']
    model.train()
    
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        data, target = data.to(device_nn), target.to(device_nn)
        # normal loss
        loss = criterion(model(data), target)
        
        # Consider both minimization and maximization attack
        data = data.to(device_milp) # send to cpu
        for mode in ['max', 'min']:
            args.mode = mode
            milp_solver = MILP_Avai(args, flexible_feature, fixed_feature, config, model)
            X_split, initial_bound_all, bounds_all = milp_solver.split_batch(data)
            
            with suppress_stdout():
                with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                    # construct the parallel pool
                    pool = executor.map(milp_solver.milp_attack, X_split, bounds_all)
                
                    for idx, res in enumerate(pool):
                        if idx == 0:
                            X_att = res[0]
                        else:
                            X_att = torch.cat((X_att, res[0]), dim=0)
            
            # to cuda
            X_att = X_att.to(device_nn)
            # print(X_att.shape, target.shape)
            loss = loss + criterion(model(X_att), target) * args.beta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_epoch(model, loader, criterion, config, loss_name):
    
    device = config['device_nn']
    model.eval()
    
    loss_summary = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            loss_summary += loss.item() * data.size(0)
    
    loss_summary = loss_summary / len(loader.dataset)
    
    print(loss_name, "MSE loss: ", round(loss_summary, 5))
    
    return loss_summary

if __name__ == "__main__":
    
    from utils.dataset import return_dataloader
    import yaml
    import os
    import numpy as np
    import random
    from utils.model import DNN
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type = int, default = 2048)
    parser.add_argument('-mbs', '--max_batch_size', type = int, default = 32)
    parser.add_argument("--max_workers", type = int, default = 80)
    parser.add_argument("--max_missing_no", type = int, default = 6)
    parser.add_argument("--impute_value", type = float, default = 0.5)
    parser.add_argument("--beta", type = float, default = 1.)
    parser.add_argument("--verbose", default = False, action = 'store_true')
    
    args = parser.parse_args()
    
    with open("utils/config.yml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    if not os.path.exists(config['nn_dir']):
        os.makedirs(config['nn_dir'])
    
    random_seed = config['random_seed']
    first_hidden_size = config['first_hidden_size']
    no_layer = config['no_layer']
    device = config['device_nn']
    
    flexible_feature = np.arange(0,6)
    fixed_feature = np.arange(6,12)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    
    train_dataloader, test_dataloader = return_dataloader(config, args)
    print('train size: ', len(train_dataloader.dataset), 'test size: ', len(test_dataloader.dataset))
    feature_size = train_dataloader.dataset[0][0].size(0)
    
    model = DNN(feature_size, no_layer, first_hidden_size)
    model.to(device)
    print(model.layer_list)
        
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])
    criterion = torch.nn.MSELoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=config['lr'] / 10, last_epoch=-1)
    
    save_dir = config['nn_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_loss = 1e5
    epochs = config['epochs']
    
    for epoch in range(1, epochs + 1):
        print('Epoch: ', epoch)
        train_epoch(model, train_dataloader, optimizer, criterion, config, args, flexible_feature, fixed_feature)
        train_loss = test_epoch(model, train_dataloader, criterion, config, 'Train')
        test_loss = test_epoch(model, test_dataloader, criterion, config, 'Test')
        # save the epoch with the best test loss
        if test_loss < best_loss:
            print('save', best_loss, '->', test_loss)
            best_loss = test_loss
            torch.save(model.state_dict(), save_dir + f'dnn_model_adver_{args.impute_value}.pt')
                
        scheduler.step()