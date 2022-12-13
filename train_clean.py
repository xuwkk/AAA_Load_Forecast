"""
train the load forecasting model
"""

import torch

def train_epoch(model, train_loader, optimizer, criterion, config):

    device = config['device_nn']
    model.train()
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) 
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
    parser.add_argument('-bs', '--batch_size', type = int, default = 64)
    args = parser.parse_args()
    
    with open("utils/config.yml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    if not os.path.exists(config['nn_dir']):
        os.makedirs(config['nn_dir'])
    
    random_seed = config['random_seed']
    first_hidden_size = config['first_hidden_size']
    no_layer = config['no_layer']
    device = config['device_nn']
    
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
    
    best_loss = 1e5
    epochs = config['epochs']
    
    for epoch in range(1, epochs + 1):
        print('Epoch: ', epoch)
        train_epoch(model, train_dataloader, optimizer, criterion, config)
        train_loss = test_epoch(model, train_dataloader, criterion, config, 'Train')
        test_loss = test_epoch(model, test_dataloader, criterion, config, 'Test')
        
        if config['watch_test']:
            if test_loss < best_loss:
                print('Saving model...')
                best_loss = test_loss
                torch.save(model.state_dict(), config['nn_dir'] + 'dnn_model.pt')
        else:
            if train_loss < best_loss:
                print('Saving model...')
                best_loss = train_loss
                torch.save(model.state_dict(), config['nn_dir'] + 'dnn_model.pt')
                
        scheduler.step()