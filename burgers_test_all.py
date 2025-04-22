"""
Script for testing the trained DeepONet and MIONet surrogate models
    of the stationary Burgers equation.
"""

# Imports

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from typing import Literal

from models.deeponet import DeepONet, MIONet
from utils.utils_deeponet import loss_l2, loss_l2_rel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Configurations

branch_layer_dim_cts = [101, 101, 101, 101]
trunk_layer_dim_cts = [1, 101, 101, 101]

input_num_gradadj = 2
branch_layer_z_dim_gradadj = [101, 101, 101, 101]
branch_layer_yh_dim_gradadj = [101, 101, 101, 101]
branch_layers_dim_list_gradadj = [branch_layer_z_dim_gradadj, 
                                  branch_layer_yh_dim_gradadj]
trunk_layer_dim_gradadj = [1, 101, 101, 101]

activation = 'relu'

save_pred: Literal['off', 'npy', 'mat'] = 'off'

# Load test data

TEST_PATH = './data/burgers/burgers_cts_gradadj_test.npz'
n_test = 10000

test_reader = np.load(TEST_PATH)
test_u = torch.tensor(test_reader['u'][-n_test:, :], dtype=torch.float).to(device)
test_s_cts = torch.tensor(test_reader['y'][-n_test:, :], dtype=torch.float).to(device)
test_p = torch.tensor(test_reader['z'][-n_test:, :], dtype=torch.float).to(device)
test_yh = torch.tensor(test_reader['y'][-n_test:, :], dtype=torch.float).to(device)
test_s_gradadj = torch.tensor(test_reader['p'][-n_test:, :], dtype=torch.float).to(device)
gridpoints = torch.tensor(test_reader['x'], dtype=torch.float).to(device)


''' Testing control-to-state model '''

print(f"\nTesting control-to-state model:\n")

# Load trained model

model_cts = DeepONet(branch_layer_dim_cts, trunk_layer_dim_cts, activation).to(device)
model_cts.load_state_dict(torch.load(
    './trained_models/burgers_model_cts_deeponet_hc_param.pt'))

test_loader_cts = DataLoader(
    TensorDataset(test_u, test_s_cts), 
    batch_size=1, 
    shuffle=False
)

hc_cts = gridpoints * (1.0 - gridpoints)

# Testing error and record prediction

pred_cts = torch.zeros(test_s_cts.shape)
index = 0
test_cts_l2_abs = []
test_cts_l2_rel = []
with torch.no_grad():
    for u, s in test_loader_cts:
        out = hc_cts * model_cts(u, gridpoints.T)
        pred_cts[index] = out
        test_cts_l2_abs.append(loss_l2(out, s).detach().cpu().item())
        test_cts_l2_rel.append(loss_l2_rel(out, s).detach().cpu().item())
test_cts_l2_abs = np.array(test_cts_l2_abs)
test_cts_l2_rel = np.array(test_cts_l2_rel)
print(f"Mean of of absolute L2 error of s: {np.mean(test_cts_l2_abs):.4e}")
print(f"SD of of absolute L2 error of s: {np.std(test_cts_l2_abs):.4e}")
print(f"Mean of relative L2 error of s: {np.mean(test_cts_l2_rel):.4e}")
print(f"SD of relative L2 error of s: {np.std(test_cts_l2_rel):.4e}")

if save_pred == 'npy':
    np.save('./burgers_pred_cts_deeponet.npy', pred_cts.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('./burgers_pred_cts_deeponet.mat', mdict={'pred': pred.cpu().numpy()})


''' Testing gradient-adjoint model '''

print(f"\nTesting gradient-adjoint model:\n")

# Load trained model

model_gradadj = MIONet(input_num_gradadj, branch_layers_dim_list_gradadj, 
                       trunk_layer_dim_gradadj, activation).to(device)
model_gradadj.load_state_dict(torch.load(
    './trained_models/burgers_model_gradadj_mionet_hc_param.pt'))
model_gradadj.eval()

test_loader_gradadj = DataLoader(
    TensorDataset(test_p, test_yh, test_s_gradadj), 
    batch_size=1, 
    shuffle=False
)

hc_gradadj = gridpoints * (1.0 - gridpoints)

# Testing error and record prediction

pred_gradadj = torch.zeros(test_s_cts.shape)
index = 0
test_gradadj_l2_abs = []
test_gradadj_l2_rel = []
with torch.no_grad():
    for p, yh, s in test_loader_gradadj:
        out = hc_gradadj * model_gradadj([p, yh], gridpoints.T)
        pred_gradadj[index] = out
        test_gradadj_l2_abs.append(loss_l2(out, s).detach().cpu().item())
        test_gradadj_l2_rel.append(loss_l2_rel(out, s).detach().cpu().item())
test_gradadj_l2_abs = np.array(test_gradadj_l2_abs)
test_gradadj_l2_rel = np.array(test_gradadj_l2_rel)
print(f"Mean of of absolute L2 error of s: {np.mean(test_gradadj_l2_abs):.4e}")
print(f"SD of of absolute L2 error of s: {np.std(test_gradadj_l2_abs):.4e}")
print(f"Mean of relative L2 error of s: {np.mean(test_gradadj_l2_rel):.4e}")
print(f"SD of relative L2 error of s: {np.std(test_gradadj_l2_rel):.4e}")

if save_pred == 'npy':
    np.save('./burgers_pred_gradadj_deeponet.npy', pred_cts.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('./burgers_pred_gradadj_deeponet.mat', mdict={'pred': pred.cpu().numpy()})
