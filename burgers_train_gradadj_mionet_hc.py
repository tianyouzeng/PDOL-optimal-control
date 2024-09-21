"""
Script for training the DeepONet surrogate model for the control-to-state operator 
    of the optimal control problem of stationary Burgers equation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

from models.deeponet import MIONet
from utils.utils_deeponet import loss_mse, loss_l2, loss_l2_rel

torch.manual_seed(43)
np.random.seed(43)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Configs

TRAIN_PATH = './data/burgers/burgers_data_train.mat'
TEST_PATH = './data/burgers/burgers_data_test.mat'

N_TRAIN = 5000
N_TEST = 10000

input_num = 2
branch_layer_z_dim = [101, 101, 101, 101]
branch_layer_yh_dim = [101, 101, 101, 101]
branch_layers_dim_list = [branch_layer_z_dim, branch_layer_yh_dim]
trunk_layer_dim = [1, 101, 101, 101]
activation = 'relu'

batch_size = 100   # require this to devide N_TRAIN
epochs = 2000
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.8

# Load data

mat_contents = sio.loadmat(TRAIN_PATH)
p_train = torch.tensor(mat_contents['z'][:N_TRAIN,:], dtype=torch.float).to(device)
yh_train = torch.tensor(mat_contents['y'][:N_TRAIN,:], dtype=torch.float).to(device)
s_train = torch.tensor(mat_contents['p'][:N_TRAIN,:], dtype=torch.float).to(device)
gridpoints = torch.tensor(mat_contents['x'], dtype=torch.float).to(device)

mat_contents = sio.loadmat(TEST_PATH)
p_test = torch.tensor(mat_contents['z'][-N_TEST:,:], dtype=torch.float).to(device)
yh_test = torch.tensor(mat_contents['y'][-N_TEST:,:], dtype=torch.float).to(device)
s_test = torch.tensor(mat_contents['p'][-N_TEST:,:], dtype=torch.float).to(device)

S = gridpoints.shape[1]

train_loader = DataLoader(TensorDataset(p_train, yh_train, s_train), batch_size=batch_size, shuffle=True)

# Define model and optimizer

model = MIONet(input_num, branch_layers_dim_list, trunk_layer_dim, activation).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# Hard constraint function

hc = gridpoints * (1.0 - gridpoints)    # dimension (1, gridpoints_num), which will be broadcasted correctly.

# Train model

for ep in range(epochs):

    model.train()
    train_mse = 0
    train_l2 = 0
    for p, yh, s in train_loader:
        optimizer.zero_grad()
        out = hc * model([p, yh], gridpoints.T)
        mse = loss_mse(out, s)
        mse.backward()
        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss_l2(out, s).item()
    scheduler.step()
    train_mse /= len(train_loader)
    train_l2 /= len(train_loader)

    if ep % 50 == 0:
        print(ep, train_mse, train_l2)

torch.save(model.state_dict(), "./trained_models/burgers_model_gradadj_mionet_hc_param.pt")

# Test model

test_loader = DataLoader(TensorDataset(p_test, yh_test, s_test), batch_size=1, shuffle=False)
pred = torch.zeros(s_test.shape)
index = 0
model.eval()
test_l2 = []
test_l2_rel = []
with torch.no_grad():
    for p, yh, s in test_loader:
        out = hc * model([p, yh], gridpoints.T)
        pred[index] = out
        test_l2.append(loss_l2(out, s).detach().cpu().item())
        test_l2_rel.append(loss_l2_rel(out, s).detach().cpu().item())
    test_l2 = np.array(test_l2)
    test_l2_rel = np.array(test_l2_rel)
    print("Mean of of absolute L2 error of s: {:.4e}".format(np.mean(test_l2)))
    print("SD of of absolute L2 error of s: {:.4e}".format(np.std(test_l2)))
    print("Mean of relative L2 error of s: {:.4e}".format(np.mean(test_l2_rel)))
    print("SD of relative L2 error of s: {:.4e}".format(np.std(test_l2_rel)))

# sio.savemat('./trained_models/burgers_pred_gradadj_mionet_hc.mat', mdict={'pred': pred.cpu().numpy()})

