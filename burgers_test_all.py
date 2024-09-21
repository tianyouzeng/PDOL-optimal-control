"""
Script for testing the trained DeepONet and MIONet surrogate models
    of the stationary Burgers equation.
"""

# Imports

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.deeponet import DeepONet, MIONet
from utils.utils_deeponet import loss_l2, loss_l2_rel

import matplotlib.pyplot as plt
import scipy.io as sio

# Load test data

TEST_PATH = './data/burgers/burgers_data_test.mat'
N_TEST = 10000

mat_contents = sio.loadmat(TEST_PATH)
u_test = torch.tensor(mat_contents['u'][-N_TEST:,:], dtype=torch.float)
s_cts_test = torch.tensor(mat_contents['y'][-N_TEST:,:], dtype=torch.float)
p_test = torch.tensor(mat_contents['z'][-N_TEST:,:], dtype=torch.float)
yh_test = torch.tensor(mat_contents['y'][-N_TEST:,:], dtype=torch.float)
s_gradadj_test = torch.tensor(mat_contents['p'][-N_TEST:,:], dtype=torch.float)
gridpoints = torch.tensor(mat_contents['x'], dtype=torch.float)


########## Testing control-to-state model ##########

# Load trained model

branch_layer_dim = [101, 101, 101, 101]
trunk_layer_dim = [1, 101, 101, 101]
activation = 'relu'

model_cts = DeepONet(branch_layer_dim, trunk_layer_dim, activation)
model_cts.load_state_dict(torch.load('./trained_models/burgers_model_cts_deeponet_hc_param.pt'))
model_cts.eval()

test_loader = DataLoader(TensorDataset(u_test, s_cts_test), batch_size=1, shuffle=False)

hc = gridpoints * (1.0 - gridpoints)

# Testing error and record prediction

pred = torch.zeros(s_cts_test.shape)
index = 0
test_cts_l2 = []
test_cts_l2_rel = []
with torch.no_grad():
    for u, s in test_loader:
        out = hc * model_cts(u, gridpoints.T)
        pred[index] = out
        test_cts_l2.append(loss_l2(out, s).detach().cpu().item())
        test_cts_l2_rel.append(loss_l2_rel(out, s).detach().cpu().item())
test_cts_l2 = np.array(test_cts_l2)
test_cts_l2_rel = np.array(test_cts_l2_rel)
print("Mean of of absolute L2 error of s: {:.4e}".format(np.mean(test_cts_l2)))
print("SD of of absolute L2 error of s: {:.4e}".format(np.std(test_cts_l2)))
print("Mean of relative L2 error of s: {:.4e}".format(np.mean(test_cts_l2_rel)))
print("SD of relative L2 error of s: {:.4e}".format(np.std(test_cts_l2_rel)))

# sio.savemat('./trained_models/burgers_pred_cts_deeponet.mat', mdict={'pred': pred.cpu().numpy()})

# Plot the results

index = np.array(np.random.randint(N_TEST))
with torch.no_grad():
    s_pred = hc * model_cts(u_test, gridpoints.T)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(gridpoints.flatten(), s_cts_test[index, :], label='Exact s', lw=2)
plt.plot(gridpoints.flatten(), s_pred[index, :], '--', label='Predicted s', lw=2)
plt.xlabel('y')
plt.ylabel('s(y)')
plt.tight_layout()
plt.legend()

plt.subplot(1,2,2)
plt.plot(gridpoints.flatten(), s_pred[index, :] - s_cts_test[index, :], '--', lw=2, label='error')
plt.tight_layout()
plt.legend()
plt.show()


########## Testing gradient-adjoint model ##########

# Load trained model

input_num = 2
branch_layer_z_dim = [101, 101, 101, 101]
branch_layer_yh_dim = [101, 101, 101, 101]
branch_layers_dim_list = [branch_layer_z_dim, branch_layer_yh_dim]
trunk_layer_dim = [1, 101, 101, 101]
activation = 'relu'

model_gradadj = MIONet(input_num, branch_layers_dim_list, trunk_layer_dim, activation)
model_gradadj.load_state_dict(torch.load('./trained_models/burgers_model_gradadj_mionet_hc_param.pt'))
model_gradadj.eval()

test_loader = DataLoader(TensorDataset(p_test, yh_test, s_gradadj_test), batch_size=1, shuffle=False)

hc = gridpoints * (1.0 - gridpoints)

# Testing error and record prediction

pred = torch.zeros(s_cts_test.shape)
index = 0
test_gradadj_l2 = []
test_gradadj_l2_rel = []
with torch.no_grad():
    for p, yh, s in test_loader:
        out = hc * model_gradadj([p, yh], gridpoints.T)
        pred[index] = out
        test_gradadj_l2.append(loss_l2(out, s).detach().cpu().item())
        test_gradadj_l2_rel.append(loss_l2_rel(out, s).detach().cpu().item())
test_gradadj_l2 = np.array(test_gradadj_l2)
test_gradadj_l2_rel = np.array(test_gradadj_l2_rel)
print("Mean of of absolute L2 error of s: {:.4e}".format(np.mean(test_gradadj_l2)))
print("SD of of absolute L2 error of s: {:.4e}".format(np.std(test_gradadj_l2)))
print("Mean of relative L2 error of s: {:.4e}".format(np.mean(test_gradadj_l2_rel)))
print("SD of relative L2 error of s: {:.4e}".format(np.std(test_gradadj_l2_rel)))

# sio.savemat('./trained_models/burgers_pred_cts_deeponet.mat', mdict={'pred': pred.cpu().numpy()})

# Plot the results

index = np.array(np.random.randint(N_TEST))
with torch.no_grad():
    s_pred = hc * model_gradadj([p_test, yh_test], gridpoints.T)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(gridpoints.flatten(), s_gradadj_test[index, :], label='Exact s', lw=2)
plt.plot(gridpoints.flatten(), s_pred[index, :], '--', label='Predicted s', lw=2)
plt.xlabel('y')
plt.ylabel('s(y)')
plt.tight_layout()
plt.legend()

plt.subplot(1,2,2)
plt.plot(gridpoints.flatten(), s_pred[index, :] - s_gradadj_test[index, :], '--', lw=2, label='error')
plt.tight_layout()
plt.legend()
plt.show()
