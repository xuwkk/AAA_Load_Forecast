"""
Evaluate the performance of the model
"""
import os, sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

if __name__ == "__main__":
    
    import argparse
    import numpy as np
    import yaml
    from utils.dataset import return_dataloader
    from utils.model import DNN
    from algorithms.milp import MILP_Inte, MILP_Avai
    import torch
    from time import time
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--model", type = str, default = "plain")
    parser.add_argument("--mode", type = str, default = 'max')
    parser.add_argument("--verbose", default = False, action='store_true')
    
    parser.add_argument("--milp", type = str)
    parser.add_argument("-p", "--is_parallel", default = False, action='store_true')
    parser.add_argument("--max_workers", type = int, default = 80)
    parser.add_argument("-eps","--epsilon", type = float, default = 0.2)
    parser.add_argument("-m", "--max_missing_no", type = int, default = 1)
    parser.add_argument("-i", "--impute_value", type = float, default = 0.0)
    parser.add_argument("--save", default = False, action='store_true')
    
    args = parser.parse_args()
    
    flexible_feature = np.arange(0,6)
    fixed_feature = np.arange(6,12)
    
    with open("utils/config.yml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    train_loader, test_loader = return_dataloader(config, args)
    
    print('train size: ', len(train_loader.sampler), 'test size: ', len(test_loader.sampler))
    feature_size = train_loader.dataset[0][0].size(0)
    model = DNN(feature_size, config['no_layer'], config['first_hidden_size'])
    model.to(config['device_milp'])
    model.eval()
    
    if args.model == 'plain':
        model.load_state_dict(torch.load(f'trained_models/dnn_model.pt'))
    elif args.model == '0.0':
        model.load_state_dict(torch.load(f'trained_models/dnn_model_adver_0.0.pt'))
    elif args.model == '0.5':
        model.load_state_dict(torch.load(f'trained_models/dnn_model_adver_0.5.pt'))
    else:
        raise ValueError('model name incorrect')
    
    model.layer_list
    
    if args.milp == "inte":
        milp_solver = MILP_Inte(args, flexible_feature, fixed_feature, config, model)
    else:
        print("Availability MILP", 'mode', args.mode, 'missing no', args.max_missing_no, 'impute value', args.impute_value)
        milp_solver = MILP_Avai(args, flexible_feature, fixed_feature, config, model)
        
    
    X_all = [] # original data
    X_att_all = [] # attacked data
    status_all = []
    actual_missing_no_all = [] # only for availability MILP
    
    start_time = time()
    if args.is_parallel:
        # parallel for availability MILP
        print("Parallel MILP")
        
        initial_bound_all = []
        bounds_all = []
        # Store all required input to the milp_attack in list
        # append the bounds of each batch
        # the parallel algorithm automatically allocate the batch to each worker
        # the return of the parallel milp_attack has the same batch allocation
        for X, y in test_loader:
            X_all.append(X) # append each batch
            initial_bound = milp_solver.initial_bound_clamp(X) # bound of this batch
            initial_bound_all.append(initial_bound) # append each batch
            bounds = milp_solver.bound_propagation(initial_bound)
            bounds_all.append(bounds)
        
        # solve the milp_attack in parallel
        if args.milp == "avai":
            with suppress_stdout():
                with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                    # construct the parallel pool
                    pool = executor.map(milp_solver.milp_attack, X_all, bounds_all)
                
                    for res in pool:
                        X_att_all.append(res[0])
                        actual_missing_no_all += res[1]
                        status_all += res[2]
        else: 
            with suppress_stdout():
                with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                    pool = executor.map(milp_solver.milp_attack, X_all, initial_bound_all, bounds_all)
                
                    for res in pool:
                        X_att_all.append(res[0])
                        status_all += res[1]
    else:
        print('Sequential MILP')
        for X, y in tqdm(test_loader):
            X, y = X.to(config['device_milp']), y.to(config['device_milp'])
            X_all.append(X)
            
            initial_bound = milp_solver.initial_bound_clamp(X) # for a batch of data
            bounds = milp_solver.bound_propagation(initial_bound) # calculate the bound on each layer
            with suppress_stdout():
                if args.milp == "inte":
                    X_att, status = milp_solver.milp_attack(X, initial_bound=initial_bound, propagate_bound=bounds)
                else:
                    X_att, missing_no, status = milp_solver.milp_attack(X, propagate_bound=bounds)
                    actual_missing_no_all += missing_no
            status_all += status
            X_att_all.append(X_att)
        
    end_time = time()
    
    # Evaluate the performance
    Y_PRED = []
    Y_MILP = []
    
    for X, X_att in zip(X_all, X_att_all):
        y_pred = model(X).flatten()
        y_milp = model(X_att).flatten()
        Y_PRED += y_pred.tolist()
        Y_MILP += y_milp.tolist()
        
    Y_PRED = np.array(Y_PRED)
    Y_MILP = np.array(Y_MILP)
    
    # evaluation
    print('average elapsed time: ', (end_time - start_time) / len(test_loader.dataset) * 1000, 'ms')
    print('average deviation (%): ', np.mean(((Y_MILP - Y_PRED)/Y_PRED)) * 100, "%")
    print('max deviation (%): ', np.max(((Y_MILP - Y_PRED)/Y_PRED))* 100, "%")
    print('min deviation (%): ', np.min(((Y_MILP - Y_PRED)/Y_PRED))* 100, "%")
    print('std deviation (%): ', np.std(((Y_MILP - Y_PRED)/Y_PRED))* 100, "%")
    
    if args.mode == 'max':
        violation_no = np.where(Y_MILP < Y_PRED)[0].shape[0]
    else:
        violation_no = np.where(Y_MILP > Y_PRED)[0].shape[0]
    print('violation no: ', violation_no)
    
    zero_no = np.where(Y_MILP == Y_PRED)[0].shape[0]
    print('zero no: ', zero_no)
    
    success_no = len(np.where(np.array(status_all) == 'optimal')[0])
    print('success rate: ', success_no / len(test_loader.dataset))
    
    result_dir = 'milp_result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Summary
    summary = {
        'time': (end_time - start_time) / len(test_loader.dataset) * 1000,
        'dev': (Y_MILP - Y_PRED)/Y_PRED * 100,
        'average deviation': np.mean(((Y_MILP - Y_PRED)/Y_PRED)) * 100,
        'max deviation': np.max(((Y_MILP - Y_PRED)/Y_PRED))* 100,
        'min deviation': np.min(((Y_MILP - Y_PRED)/Y_PRED))* 100,
        'std deviation': np.std(((Y_MILP - Y_PRED)/Y_PRED))* 100,
        'violation no': violation_no,
        'zero no': zero_no,
        'success rate': success_no / len(test_loader.dataset),
        'actual missing no': actual_missing_no_all if args.milp == "avai" else None,
    }   
    
    if args.save:
        if args.milp == "inte":
            save_dir = os.path.join(result_dir, f'{args.model}_{args.milp}_{args.mode}_{args.epsilon}_{args.is_parallel}')
        else:
            save_dir = os.path.join(result_dir, f'{args.model}_{args.milp}_{args.mode}_{args.max_missing_no}_{args.impute_value}_{args.is_parallel}')
        np.save(save_dir, summary)