# Imports

import torch
import torch.nn as nn

from torch import Tensor


# Vanilla MLP

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], output_dim: int, activation: str='relu') -> None:
        super(MLP, self).__init__()

        # Initialize activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation function')

        # Initialize input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])

        # Initialize hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim)-1):
            self.hidden_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))

        # Initialize output layer
        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)

    
    def forward(self, x: Tensor) -> Tensor:
        # Apply input layer
        x = self.input_layer(x)
        x = self.activation(x)

        # Apply hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)

        # Apply output layer
        x = self.output_layer(x)

        return x
    
    
# DeepONet
    
class DeepONet(nn.Module):
    def __init__(self, branch_dim: list[int], trunk_dim: list[int], activation: str='tanh') -> None:
        super(DeepONet, self).__init__()

        self.branch = MLP(branch_dim[0], branch_dim[1:-1], branch_dim[-1], activation)
        self.trunk = MLP(trunk_dim[0], trunk_dim[1:-1], trunk_dim[-1], activation)
    
    def forward(self, u: Tensor, x: Tensor) -> Tensor:
        # Input u is of shape (batch_size, gridpoints_num)
        # Input x is of shape (gridpoints_num, 1)
        
        branch_val = self.branch(u)
        trunk_val = self.trunk(x)
        out = torch.vmap(torch.vmap(lambda bs, ts: torch.sum(bs * ts), in_dims=(None, 0)), in_dims=(0, None))(branch_val, trunk_val)
        return out
    

# MIONet: Multi-input version of DeepONet
    
class MIONet(nn.Module):
    def __init__(self, input_num: int, branch_dim_list: list[list[int]], trunk_dim: list[int], activation: str='relu'):
        super(MIONet, self).__init__()
        assert(len(branch_dim_list) == input_num)

        self.input_num = input_num
        self.branch_list = nn.ModuleList()
        for branch_dim in branch_dim_list:
            self.branch_list.append(MLP(branch_dim[0], branch_dim[1:-1], branch_dim[-1], activation))
        self.trunk = MLP(trunk_dim[0], trunk_dim[1:-1], trunk_dim[-1], activation)
    
    def forward(self, func_list: list[Tensor], x: Tensor) -> Tensor:
        # Input func_list is a list of input functions with length self.input_num,
        # and each element is of shape (batch_size, gridpoints_num)
        # Input x is of shape (gridpoints_num, 1)
        
        assert(len(self.branch_list) > 0)
        assert(len(func_list) == self.input_num)
        branch_val_prod = self.branch_list[0](func_list[0])
        for i in range(1, self.input_num):
            branch_val_prod *= self.branch_list[i](func_list[i])
        trunk_val = self.trunk(x)
        out = torch.vmap(torch.vmap(lambda bs, ts: torch.sum(bs * ts), in_dims=(None, 0)), in_dims=(0, None))(branch_val_prod, trunk_val)
        return out
