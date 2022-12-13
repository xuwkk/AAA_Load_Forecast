"""
Projected gradient descent (PGD) attack for integrity attack.
"""
from copy import deepcopy
import torch

class PGD:
    def __init__(self, args, flexible_feature, fixed_feature, config, network):
        self.epsilon = args.epsilon
        self.num_iter = args.num_iter
        self.step_size = args.step_size
        self.flexible_feature = flexible_feature
        self.fixed_feature = fixed_feature
        self.device = config['device_nn']
        self.random_start = args.random_start
        self.mode = args.mode
        self.network = deepcopy(network).to(self.device).eval()
        
    def initial_bound_clamp(self, X):
        """
        given a batch of data, return the initial bound on the batch of the data. 
        fixed feature: bounded by it self
        flexible feature: further clamped into [0,1]
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
    
    def pgd(self, X, initial_bound):
        
        # X: original batch of data
        l_flex, u_flex = initial_bound[0][:, self.flexible_feature], initial_bound[1][:, self.flexible_feature]
        
        for s in range(self.random_start):
            # initialization
            X_flex = torch.rand_like(l_flex, requires_grad=True) # only update flexible features, range is [0,1]
            X_fix = X[:, self.fixed_feature]
            X_fix.requires_grad = False
            X_flex.data = l_flex + (u_flex - l_flex) * X_flex.data # range is [l_flex, u_flex]
            
            for t in range(self.num_iter):
                with torch.enable_grad():
                    try:
                        loss_unpack = self.network(torch.concat((X_flex, X_fix), 1)) # concat flexible and fixed features
                    except:
                        print(X_flex.shape, X_fix.shape, l_flex.shape, u_flex.shape)
                    loss = loss_unpack.mean()
                
                loss.backward()
                if self.mode == 'max':
                    # gradient ascent (normalized by linf norm)
                    X_flex.data = X_flex.data + self.step_size * torch.sign(X_flex.grad)
                elif self.mode == 'min':
                    # gradient descent (normalized by linf norm)
                    X_flex.data = X_flex.data - self.step_size * torch.sign(X_flex.grad)
                
                X_flex.data.clamp_(min = l_flex, max = u_flex)
                
                X_flex.grad.zero_()
            
            # select the best multi-run results
            if s == 0:
                # first run
                optimal_value = loss_unpack.detach() # for each data point in the batch
                X_flex_optimal = X_flex.detach()
            else:
                # multi-run
                if self.mode == 'max':
                    for u in range(len(optimal_value)):
                        if optimal_value[u] < loss_unpack.detach()[u]:
                            optimal_value[u] = loss_unpack.detach()[u]
                            X_flex_optimal[u] = X_flex.detach()[u]
                elif self.mode == 'min':
                    for u in range(len(optimal_value)):
                        if optimal_value[u] > loss_unpack.detach()[u]:
                            optimal_value[u] = loss_unpack.detach()[u]
                            X_flex_optimal[u] = X_flex.detach()[u]
            
        X_att = torch.cat((X_flex_optimal, X_fix), 1).detach()    
        
        assert torch.all(X_att[:, self.flexible_feature] >= l_flex)
        assert torch.all(X_att[:, self.flexible_feature] <= u_flex)
        assert torch.all(X_att[:, self.fixed_feature] == X[:, self.fixed_feature])

        return X_att