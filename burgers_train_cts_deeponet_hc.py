"""
Script for training the DeepONet surrogate model for the control-to-state operator 
    of the optimal control problem of stationary Burgers equation.
"""

import timeit
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from typing import Literal

from models.deeponet import DeepONet
from utils.utils_deeponet import loss_mse, loss_l2, loss_l2_rel

torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Configs

TRAIN_PATH = './data/burgers/burgers_cts_gradadj_train.npz'
TEST_PATH = './data/burgers/burgers_cts_gradadj_test.npz'

n_train = 5000
n_test = 10000

branch_layer_dim = [101, 101, 101, 101]
trunk_layer_dim = [1, 101, 101, 101]
activation = 'relu'

batch_size = 100   # require this to devide n_train
epochs = 2000
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.8

print(f'Configurations:\n')
print(f'{n_train=}\n{n_test=}\n{branch_layer_dim=}\n{trunk_layer_dim=}')
print(f'{batch_size=}\n{epochs=}\n{learning_rate=}\n{scheduler_step=}\n{scheduler_gamma=}\n')

S = 101

record_time = True
save_pred: Literal['off', 'npy', 'mat'] = 'off'

# Load data

train_reader = np.load(TRAIN_PATH)
train_u = torch.tensor(train_reader['u'][:n_train, :], dtype=torch.float).to(device)
train_s = torch.tensor(train_reader['y'][:n_train, :], dtype=torch.float).to(device)
gridpoints = torch.tensor(train_reader['x'], dtype=torch.float).to(device)
del train_reader

test_reader = np.load(TEST_PATH)
test_u = torch.tensor(test_reader['u'][-n_test:, :], dtype=torch.float).to(device)
test_s = torch.tensor(test_reader['y'][-n_test:, :], dtype=torch.float).to(device)
del test_reader

train_loader = DataLoader(
    TensorDataset(train_u, train_s),
    batch_size=batch_size,
    shuffle=True
)

# Define model and optimizer

model = DeepONet(branch_layer_dim, trunk_layer_dim, activation).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=scheduler_step,
    gamma=scheduler_gamma
)

# Hard constraint function

hc = gridpoints * (1.0 - gridpoints)    # dimension (1, gridpoints_num), which will be broadcasted correctly.

# Train model

t_start = 0
if record_time:
    t_start = timeit.default_timer()

for ep in range(epochs):

    model.train()
    train_mse = 0
    train_l2 = 0
    for u, s in train_loader:
        optimizer.zero_grad()
        out = hc * model(u, gridpoints.T)
        mse = loss_mse(out, s)
        mse.backward()
        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss_l2(out, s).item()
    scheduler.step()
    train_mse /= len(train_loader)
    train_l2 /= len(train_loader)

    if ep % 50 == 0:
        print(f'epoch: {ep}, training mse: {train_mse}, training l2: {train_l2}')

if record_time:
    t_end = timeit.default_timer()
    print(f'Total training time: {t_end - t_start}')

torch.save(model.state_dict(), "./burgers_model_cts_deeponet_hc_param.pt")

# Test model

test_loader = DataLoader(
    TensorDataset(test_u, test_s),
    batch_size=1,
    shuffle=False
)
pred = torch.zeros(test_s.shape)
index = 0
model.eval()
test_l2 = []
test_l2_rel = []
with torch.no_grad():
    for u, s in test_loader:
        out = hc * model(u, gridpoints.T)
        pred[index] = out
        test_l2.append(loss_l2(out, s).detach().cpu().item())
        test_l2_rel.append(loss_l2_rel(out, s).detach().cpu().item())
    test_l2 = np.array(test_l2)
    test_l2_rel = np.array(test_l2_rel)
    print(f"Mean of of absolute L2 error of s: {np.mean(test_l2):.4e}")
    print(f"SD of of absolute L2 error of s: {np.std(test_l2):.4e}")
    print(f"Mean of relative L2 error of s: {np.mean(test_l2_rel):.4e}")
    print(f"SD of relative L2 error of s: {np.std(test_l2_rel):.4e}")

if save_pred == 'npy':
    np.save('./burgers_pred_cts_deeponet_hc.npy', pred.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('./burgers_pred_cts_deeponet_hc.mat', mdict={'pred': pred.cpu().numpy()})
