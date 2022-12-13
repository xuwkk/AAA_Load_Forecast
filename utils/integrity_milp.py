"""
Using MIQP to solve the integrity adversarial attack problem.
"""
from copy import deepcopy
import torch
import cvxpy as cp
import torch.nn as nn
from contextlib import contextmanager
import os, sys

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class MILP_SOLVER:
    def __init__(self, args, flexible_feature, fixed_feature, config, network):
        self.epsilon = args.epsilon
        self.flexible_feature = flexible_feature
        self.fixed_feature = fixed_feature
        self.device = config['device_milp']
        self.network = deepcopy(network).to(self.device)
        self.mode = args.mode
        self.verbose = args.verbose
    
    def initial_bound_clamp(self, X):
        """
        given a batch of data (X), return the initial bound of the MILP. 
        fixed feature: bounded by it self
        flexible feature: bounded by epsilon and further clamped into [0,1]
        """
        
        X_min = deepcopy(X)
        X_max = deepcopy(X)
        
        X_min[:, self.flexible_feature] = (X[:, self.flexible_feature] - self.epsilon).clamp(min = 0)
        X_max[:, self.flexible_feature] = (X[:, self.flexible_feature] + self.epsilon).clamp(max = 1)
        
        # check
        assert torch.all(X_min[:, self.flexible_feature] >= 0)
        assert torch.all(X_max[:, self.flexible_feature] <= 1)
        assert torch.all(X_min[:, self.fixed_feature] == X[:, self.fixed_feature])
        assert torch.all(X_max[:, self.fixed_feature] == X[:, self.fixed_feature])
        assert torch.all(torch.all(X - X_min) < self.epsilon)
        assert torch.all(torch.all(X_max - X) < self.epsilon)
        
        return (X_min, X_max)
    
    def bound_propagation(self, initial_bound):
        """
        find the lower and upper bound of the output of each layer in MLP given the initial bound
        """
        
        l, u = initial_bound
        bounds = []
        
        for layer in self.network.layer_list:
            if isinstance(layer, torch.nn.Linear):
                    l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() 
                        + layer.bias[:,None]).t()
                    u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() 
                        + layer.bias[:,None]).t()
                    
            elif isinstance(layer, torch.nn.ReLU):
                l_ = l.clamp(min=0)
                u_ = u.clamp(min=0)
            
            else:
                continue
                
            bounds.append((l_.detach(), u_.detach()))
            l,u = l_, u_
    
        return bounds
    
    def form_milp(self, initial_bound, propagate_bound, X_idx):
        """
        formulate the MILP problem
        attack one data point at a time
        only attack on the flexible features
        X_idx: the index of the data in current batch
        """
    
        # select the linear layer and pack the layer corresponding bounds
        linear_layers = []
        linear_layers = linear_layers + [(layer, bound) for layer, bound in zip(self.network.layer_list, propagate_bound) if isinstance(layer, torch.nn.Linear)]
        # print('no of layer', len(linear_layers))
        d = len(linear_layers)-1 # exclude the output layer
        
        # create cvxpy variables
        # z: the input of each layer + the output of the last layer
        # d+1
        z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] + [cp.Variable(linear_layers[-1][0].out_features)])
        
        # v: the output of each layer (last layer excluded) because the integer variable is associated to the ReLU activation
        v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers[:-1]]
        
        # extract relevant matrices
        # linear layer weight and bias
        # d-1
        W = [layer.weight.detach().cpu().numpy() for layer,_ in linear_layers]
        b = [layer.bias.detach().cpu().numpy() for layer,_ in linear_layers]
    
        # lower bound of the output of each linear layer
        l = [l[X_idx].detach().cpu().numpy() for _, (l,_) in linear_layers]
        u = [u[X_idx].detach().cpu().numpy() for _, (_,u) in linear_layers]
        
        # the bounds on the input
        l0 = initial_bound[0][X_idx].view(-1).detach().cpu().numpy()
        u0 = initial_bound[1][X_idx].view(-1).detach().cpu().numpy()
        
        # add ReLU constraints
        constraints = []
        for i in range(len(linear_layers)-1):
            # for each non-last layer, add a constraint
            constraints += [z[i+1] >= W[i] @ z[i] + b[i], 
                            z[i+1] >= 0,
                            cp.multiply(v[i], u[i]) >= z[i+1],
                            W[i] @ z[i] + b[i] >= z[i+1] + cp.multiply((1-v[i]), l[i])]
    
        # final linear constraint
        constraints += [z[d+1] == W[d] @ z[d] + b[d]]
        
        # initial bound constraints
        constraints += [z[0] >= l0, z[0] <= u0]
    
        if self.mode == 'max':
            return cp.Problem(cp.Maximize(z[d+1]), constraints), (z, v)
        elif self.mode == 'min':
            return cp.Problem(cp.Minimize(z[d+1]), constraints), (z, v)
        
    # MIP attack
    def milp_attack(self, X, initial_bound, propagate_bound):
        
        """
        solve the MILP problem for a batch of data
        mode: 'max' or 'min' to maximize or minimize the prediction
        """
        
        X_att_milp = torch.zeros_like(X)
        for i in range(len(X)):
            prob, (z, v) = self.form_milp(initial_bound, propagate_bound, X_idx=i)
            prob.solve(solver=cp.GUROBI, verbose=self.verbose, MIPGap=1e-4)
            X_att_ = torch.tensor(z[0].value)
            
            X_att_milp[i,:] = X_att_
        
        # verify the bound
        assert torch.all(X_att_milp[:, self.flexible_feature] >= 0-1e-5)
        assert torch.all(X_att_milp[:, self.flexible_feature] <= 1+1e-5)
        assert torch.all(torch.abs(X_att_milp[:, self.fixed_feature] - X[:, self.fixed_feature]) <= 1e-5)
        assert torch.all(torch.abs(X_att_milp - X) < self.epsilon * 1.00001)
        
        return X_att_milp    
    
if __name__ == "__main__":
    
    import numpy as np
    import argparse
    import yaml
    from dataset import return_dataloader
    from model import DNN
    from time import time
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    
    # train
    parser.add_argument("--batch_size", type = int, default = 10)
    parser.add_argument("--epsilon", type = float, default = 0.1)
    parser.add_argument("--mode", type = str, default = 'max')
    parser.add_argument("--max_workers", type = int, default = 40)
    parser.add_argument("--is_parallel", type = bool, default = False)
    parser.add_argument("--verbose", default = False, action='store_true')
    
    args = parser.parse_args()
    print(args)
    
    with open("utils/config.yml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    device = config['device_milp']
    
    train_loader, test_loader = return_dataloader(config, args)
    feature_size = train_loader.dataset[0][0].size(0)
    model = DNN(feature_size, config['no_layer'], config['first_hidden_size'])
    model.load_state_dict(torch.load('trained_models/dnn_model.pt'))
    model.to(device)
    
    print(model.layer_list)
    
    flexible_feature = np.arange(0, 6)
    fixed_feature = np.arange(6, 12)
    milp_solver = MILP_SOLVER(args, flexible_feature, fixed_feature, config, model)
    
    if args.is_parallel:
        pass
    else:
        # sequential solve the MILP
        X_all = []
        X_att_all = []
        
        start_time = time()
        for X,y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            X_all.append(X)
            
            initial_bound = milp_solver.initial_bound_clamp(X) # for a batch of data
            bounds = milp_solver.bound_propagation(initial_bound) # calculate the bound on each layer
            with suppress_stdout():
                X_att_milp = milp_solver.milp_attack(X, initial_bound, bounds)
            X_att_all.append(X_att_milp)
            
        end_time = time()
    
    # evaluation
    Y_PRED = []
    Y_MILP = []
    for X, X_att in zip(X_all, X_att_all):
        y_pred = milp_solver.network(X).detach().cpu().numpy().flatten()
        y_milp = milp_solver.network(X_att).detach().cpu().numpy().flatten()
        Y_PRED += y_pred.tolist()
        Y_MILP += y_milp.tolist()

    Y_PRED = np.array(Y_PRED)
    Y_MILP = np.array(Y_MILP)

    # evaluation
    print('average elapsed time: ', (end_time - start_time) / len(test_loader.dataset) * 1000, 'ms')
    print('average deviation (%): ', np.mean((Y_MILP - Y_PRED)/Y_PRED) * 100, "%")
    print('max deviation (%): ', np.max((Y_MILP - Y_PRED)/Y_PRED)* 100, "%")
    print('min deviation (%): ', np.min((Y_MILP - Y_PRED)/Y_PRED)* 100, "%")