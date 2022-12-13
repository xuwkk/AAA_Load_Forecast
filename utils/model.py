"""
NN models
"""
import torch
import torch.nn as nn

class DNN(torch.nn.Module):
    
    def __init__(self, no_feature, no_layer, first_hidden_size):
        super().__init__()
        
        model_list = []
        input_feature = no_feature
        for i in range(no_layer-1):
            model_list.append(torch.nn.Linear(input_feature, int(first_hidden_size/(2**i))))
            model_list.append(torch.nn.ReLU())
            input_feature = int(first_hidden_size/(2**i))
        
        model_list.append(torch.nn.Linear(input_feature, 1)) # output layer
        self.layer_list = torch.nn.ModuleList(model_list)
        
    def forward(self, x):
        for i, layer in enumerate(self.layer_list):
            x = layer(x)
        
        return x.flatten()