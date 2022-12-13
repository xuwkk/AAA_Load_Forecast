"""
Functions for mixed integer linear programming (MILP) problems.
    1. Integrity adversarial attack
    2. Availability adversarial attack
"""

from copy import deepcopy
import torch
import cvxpy as cp
import numpy as np

"""
Integrity adversarial attack
"""
class MILP_Inte:
    def __init__(self, args, flexible_feature, fixed_feature, config, network):
        
        self.flexible_feature = flexible_feature
        self.fixed_feature = fixed_feature
        self.device = config['device_milp']
        self.network = deepcopy(network).to(self.device)
        self.mode = args.mode
        self.verbose = args.verbose
        try:
            self.max_batch_size = args.max_batch_size
        except:
            self.max_batch_size = None
        try:
            self.epsilon = args.epsilon
        except:
            self.epsilon = None
        self.network.eval()
    
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
        status = []
        X_att_milp = torch.zeros_like(X)
        for i in range(len(X)):
            prob, (z, v) = self.form_milp(initial_bound, propagate_bound, X_idx=i)
            prob.solve(solver=cp.GUROBI, verbose=self.verbose)
            status.append(prob.status)
            X_att_ = torch.tensor(z[0].value)
            
            X_att_milp[i,:] = X_att_
        
        # verify the bound
        assert torch.all(X_att_milp[:, self.flexible_feature] >= 0-1e-5)
        assert torch.all(X_att_milp[:, self.flexible_feature] <= 1+1e-5)
        assert torch.all(torch.abs(X_att_milp[:, self.fixed_feature] - X[:, self.fixed_feature]) <= 1e-5)
        assert torch.all(torch.abs(X_att_milp - X) < self.epsilon * 1.00001)
        
        return X_att_milp, status

"""
Available MILP solvers:
"""
class MILP_Avai(MILP_Inte):
    """
    Availability attack
    """
    
    def __init__(self, args, flexible_feature, fixed_feature, config, network):
        super().__init__(args, flexible_feature, fixed_feature, config, network)
        
        self.impute_value = args.impute_value  # use mean for the missing data
        self.max_missing_no = args.max_missing_no
        
    def initial_bound_clamp(self, X):
        """
        rewrite the initial bound function
        given a batch of data, return the initial bound of the MILP for missing data attack. 
        fixed feature: bounded by it self
        flexible feature: 
            mean: is assumed as 0.5 as we use minmax scaler
            feature > mean: [mean, feature]
            feature < mean: [feature, mean]
        """
        
        # bounds
        X_min = deepcopy(X)
        X_max = deepcopy(X)
        X_min[:, self.flexible_feature] = X[:, self.flexible_feature].clamp(max = self.impute_value) 
        X_max[:, self.flexible_feature] = X[:, self.flexible_feature].clamp(min = self.impute_value)
        X_min = X_min - 0.2
        X_max = X_max + 0.2 # add a small value to avoid numerical issue
        
        # if self.impute_value == 0:
        #     X_min = deepcopy(X)
        #     X_max = deepcopy(X)
            
        #     X_min[:, self.flexible_feature] = -0.01  # assign a small value for bound propagation purpose. no influence on the final result
            
        # else:
        
        #     X_min = deepcopy(X)
        #     X_max = deepcopy(X)
            
        #     X_min[:, self.flexible_feature] = X[:, self.flexible_feature].clamp(max = self.impute_value)
        #     X_max[:, self.flexible_feature] = X[:, self.flexible_feature].clamp(min = self.impute_value)
            
        #     assert torch.all(X_min[:, self.flexible_feature] <= self.impute_value)
        #     assert torch.all(X_max[:, self.flexible_feature] >= self.impute_value)
        
        # # the bound is [0,X]
        # X_min = deepcopy(X)
        # X_max = deepcopy(X)
        # X_max[:, self.flexible_feature] = X_max[:, self.flexible_feature] * 1.2
        # X_min[:, self.flexible_feature] = -0.01  # assign a small value for bound propagation purpose. no influence on the final result
        
            # check
        try:
            assert torch.all(X_min <= X_max)
        except:
            print(X_max - X_min)
        # assert torch.all(X_min[:, self.fixed_feature] == X[:, self.fixed_feature])
        # assert torch.all(X_max[:, self.fixed_feature] == X[:, self.fixed_feature])
        
        return (X_min, X_max)
    
    def split_batch(self, X_batch):
        """
        split a batch of data into several smaller batcher
        X_batch: (batch_size, feature_dim)
        max_batch_size: maximum small batch size
        """
        
        # split X_batch into several small batches
        full_index = int(len(X_batch)/self.max_batch_size)
        X_all = [X_batch[i*self.max_batch_size:(i+1)*self.max_batch_size] for i in range(full_index)]
        if full_index*self.max_batch_size != len(X_batch):
            # adds on the remaining batch
            X_all.append(X_batch[full_index * self.max_batch_size:])
        
        initial_bound_all = []
        bounds_all = []
        
        for X in X_all:
            
            initial_bound = self.initial_bound_clamp(X)
            initial_bound_all.append(initial_bound)
            
            bounds = self.bound_propagation(initial_bound)
            bounds_all.append(bounds)
        
        return X_all, initial_bound_all, bounds_all
        
    
    def form_milp(self, x, propagate_bound, X_idx):
    
        """
        formulate the MILP problem
        only attack on the flexible features
        because MILP is solved for each sample:
            X_idx: the index of the data in current batch
        """
    
        # select the linear layer and pack the layer corresponding bounds
        linear_layers = []
        linear_layers = linear_layers + [(layer, bound) for layer, bound in zip(self.network.layer_list, propagate_bound) if isinstance(layer, torch.nn.Linear)]
        d = len(linear_layers)-1
        
        # create cvxpy variables
        # z: the input of each layer + the output of the last layer
        z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] + [cp.Variable(linear_layers[-1][0].out_features)])
        
        # v: the output of each layer (last layer excluded) because the integer variable is associated to the ReLU activation
        v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers[:-1]]
        
        # m: the integer variable for the missing data
        m = cp.Variable(len(self.flexible_feature), boolean=True)
        
        # n: the constant to select the fixed data
        n = np.ones(len(self.fixed_feature))
        
        m_n = cp.hstack((m, n)) # stack the integer variable and the constant
        
        # extract relevant matrices
        # linear layer weight and bias
        W = [layer.weight.detach().cpu().numpy() for layer,_ in linear_layers]
        b = [layer.bias.detach().cpu().numpy() for layer,_ in linear_layers]
    
        # lower bound of the output of each linear layer
        l = [l[X_idx].detach().cpu().numpy() for _, (l,_) in linear_layers]
        u = [u[X_idx].detach().cpu().numpy() for _, (_,u) in linear_layers]
        
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
        if self.impute_value == 0:
            constraints += [z[0] == cp.diag(m_n) @ x]
        else:
            constraints += [z[0] == cp.diag(m_n) @ x + (1-m_n) * self.impute_value]
        
        # constraint on the missing data number
        # m = 0: missing
        constraints += [cp.sum(m) >= len(self.flexible_feature) - self.max_missing_no]
        
        if self.mode == 'max':
            return cp.Problem(cp.Maximize(z[d+1]), constraints), (z, v, m)
        elif self.mode == 'min':
            return cp.Problem(cp.Minimize(z[d+1]), constraints), (z, v, m)
    
    # MIP attack
    def milp_attack(self, X, propagate_bound):
        
        """
        solve the MILP problem for a batch of data,
        this setting is beneficial for the adversarial training
        """
        
        X_att_milp = torch.zeros_like(X)
        actual_missing_no = []
        status = []
        m_summary = []
        
        for i in range(len(X)):
            y_pred = self.network(X[i].unsqueeze(0)).detach().cpu().numpy()
            prob, (z, v, m) = self.form_milp(X[i], propagate_bound, X_idx=i)
            prob.solve(solver=cp.GUROBI, verbose=self.verbose, Presolve = -1)
            
            status.append(prob.status)
            
            # calculate new x_att
            # if self.mode == 'max' and prob.value < y_pred:
            #     m = np.ones(len(self.flexible_feature)).astype(int)
            
            # elif self.mode == 'min' and prob.value > y_pred:
            #     m = np.ones(len(self.flexible_feature)).astype(int)
            # else:
            #     m = np.rint(m.value)
            m = np.rint(m.value)
            #m = m.value
            n = np.ones(len(self.fixed_feature))
            m_n = torch.tensor(np.hstack((m, n)), dtype=torch.float)

            # explicit imputation to avoid numerical error
            x_att = torch.diag(m_n) @ X[i] + (1-m_n) * torch.tensor(self.impute_value, dtype=torch.float)
            
            # save solution
            X_att_milp[i,:] = x_att
            
            actual_missing_no.append(len(self.flexible_feature) - round(np.sum(m)))
            m_summary.append(m)
        
        # verify the bound
        assert torch.all(torch.abs(X_att_milp[:, self.fixed_feature] - X[:, self.fixed_feature]) <= 1e-7)
        
        return X_att_milp, actual_missing_no, status