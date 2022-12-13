"""
Evaluation script for PGD adversarial attack.
    1. Compare the result with the MILP attack
"""

if __name__ == "__main__":
    
    import argparse
    from algorithms.pgd import PGD
    import torch
    import numpy as np
    import yaml
    from utils.dataset import return_dataloader
    from utils.model import DNN
    from time import time
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--num_iter", type=int, default=30)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--random_start", type=int, default=1)
    parser.add_argument("--mode", type=str, default="max")
    parser.add_argument("--model", type = str, default = "plain")
    parser.add_argument("--save", default = False, action='store_true')
    
    args = parser.parse_args()
    
    flexible_feature = np.arange(0,6)
    fixed_feature = np.arange(6,12)
    
    with open("utils/config.yml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    device = config['device_nn']
    train_loader, test_loader = return_dataloader(config, args)
    
    print('train size: ', len(train_loader.sampler), 'test size: ', len(test_loader.sampler))
    feature_size = train_loader.dataset[0][0].size(0)
    model = DNN(feature_size, config['no_layer'], config['first_hidden_size'])
    model.to(config['device_nn'])
    model.eval()
    
    if args.model == 'plain':
        model.load_state_dict(torch.load(f'trained_models/dnn_model.pt'))
    else:
        print("I haven't implemented this yet")
        raise NotImplementedError
    
    Y_ADVER = []
    Y_PRED = []
    
    pgd = PGD(args, flexible_feature, fixed_feature, config, model)
    
    start_time = time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        initial_bound = pgd.initial_bound_clamp(data)
        X_att = pgd.pgd(data, initial_bound)
        
        y_pred = model(data).detach().cpu().numpy().flatten()
        y_att = model(X_att).detach().cpu().numpy().flatten()
        
        Y_PRED += (y_pred.tolist())
        Y_ADVER += (y_att.tolist())
    
    end_time = time()
    
    Y_PRED = np.array(Y_PRED)
    Y_ADVER = np.array(Y_ADVER)
    
    # evaluation
    print('average elapsed time: ', (end_time - start_time) / len(test_loader.dataset) * 1000, 'ms')
    print('average deviation (%): ', np.mean((Y_ADVER - Y_PRED)/Y_PRED) * 100, "%")
    print('max deviation (%): ', np.max((Y_ADVER - Y_PRED)/Y_PRED)* 100, "%")
    print('min deviation (%): ', np.min((Y_ADVER - Y_PRED)/Y_PRED)* 100, "%")
    
    zero_no = np.where(Y_ADVER == Y_PRED)[0].shape[0]
    print('zero no: ', zero_no)
    
    result_dir = 'pgd_result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Summary
    summary = {
        'time': (end_time - start_time) / len(test_loader.dataset) * 1000,
        'average deviation': np.mean(((Y_ADVER - Y_PRED)/Y_PRED)) * 100,
        'max deviation': np.max(((Y_ADVER - Y_PRED)/Y_PRED))* 100,
        'min deviation': np.min(((Y_ADVER - Y_PRED)/Y_PRED))* 100,
        'zero no': zero_no
    }
    
    if args.save:
        save_dir = os.path.join(result_dir, f'{args.epsilon}_{args.mode}_{args.random_start}')
        np.save(save_dir, summary)
    
