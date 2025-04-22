"""
Script for training the FNO surrogate model for the control-to-state operator 
    of the bilinear parabolic optimal control problem.
This file is adapted from the official implementation of FNO in:
    - https://github.com/khassibi/fourier-neural-operator/fourier_3d.py
"""


# Imports and initializations

from timeit import default_timer
from typing import Literal

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

from models.fno import FNO3d
from utils.utils_fno import Adam, LpLoss, count_params

torch.manual_seed(114514)
np.random.seed(114514)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Configurations

TRAIN_PATH = './data/bilinparab/bp_cts_gradadj_train.npz'
TEST_PATH = './data/bilinparab/bp_cts_gradadj_test.npz'

n_train = 2048
n_test = 256

modes = [8, 8, 8]
width = 16
input_num = 2
paddings = [8, 8, 8]

batch_size = 8
epochs = 300
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.5

print(f'Configurations:\n')
print(f'{n_train=}\n{n_test=}\n{modes=}\n{width=}\n{input_num=}\n{paddings=}')
print(f'{batch_size=}\n{epochs=}\n{learning_rate=}\n{scheduler_step=}\n{scheduler_gamma=}\n')

sub = 1
S = 64 // sub
T = 64

record_time = True
save_pred: Literal['off', 'npy', 'mat'] = 'off'

t1 = 0
if record_time:
    runtime = np.zeros(2, )
    t1 = default_timer()


# Load data

train_reader = np.load(TRAIN_PATH)
train_u = torch.tensor(train_reader["u"])[:n_train, ::sub, ::sub, :T].to(device)
train_f = torch.tensor(train_reader["f"])[:n_train, ::sub, ::sub, :T].to(device)
train_s = torch.tensor(train_reader["s"])[:n_train, ::sub, ::sub, :T].to(device)
del train_reader

test_reader = np.load(TEST_PATH)
test_u = torch.tensor(test_reader["u"])[-n_test:, ::sub, ::sub, :T].to(device)
test_f = torch.tensor(test_reader["f"])[-n_test:, ::sub, ::sub, :T].to(device)
test_s = torch.tensor(test_reader["s"])[-n_test:, ::sub, ::sub, :T].to(device)
del test_reader

print(f'training data shape: {train_u.shape}')
print(f'testing data shape: {test_u.shape}')
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_u = train_u.reshape(n_train, S, S, T, 1)
train_f = train_f.reshape(n_train, S, S, T, 1)
test_u = test_u.reshape(n_test, S, S, T, 1)
test_f = test_f.reshape(n_test, S, S, T, 1)
train_uf = torch.cat((train_u, train_f), dim=-1)
test_uf = torch.cat((test_u, test_f), dim=-1)
del train_u, train_f, test_u, test_f

train_loader = DataLoader(
    TensorDataset(train_uf, train_s),
    batch_size=batch_size,
    shuffle=True
)

if record_time:
    t2 = default_timer()
    print(f'preprocessing finished, time used: {t2 - t1}')
print(f'\n')


# Training

model = FNO3d(input_num, modes, width, paddings).to(device)
print(f'FNO parameter count: {count_params(model)}\n')
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=scheduler_step,
    gamma=scheduler_gamma
)
myloss = LpLoss(size_average=False)

t_start = 0
if record_time:
    t_start = default_timer()

for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0

    for uf, s in train_loader:
        optimizer.zero_grad()
        out = model(uf).squeeze(dim=-1)
        mse = F.mse_loss(out, s, reduction='mean')
        l2 = myloss(out.view(batch_size, -1), s.view(batch_size, -1))
        l2.backward()
        optimizer.step()

        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    train_mse /= len(train_loader)
    train_l2 /= n_train

    t2 = default_timer()
    print(f'epoch: {ep}, time: {t2-t1}, training mse: {train_mse}, training l2: {train_l2}')

if record_time:
    t_end = default_timer()
    print(f'Total training time: {t_end - t_start}')

torch.save(model.state_dict(), "./bp_model_cts_gradadj_fno3d_param_playground.pt")


# Test the trained model

test_loader = DataLoader(
    TensorDataset(test_uf, test_s),
    batch_size=1,
    shuffle=False
)
pred = torch.zeros(test_s.shape)
test_l2 = np.zeros(n_test)
index = 0
with torch.no_grad():
    for uf, s in test_loader:
        out = model(uf).view(S, S, T)
        pred[index] = out
        test_l2[index] = myloss(out.view(1, -1), s.view(1, -1)).item()
        index = index + 1
    print('testing l2:', np.mean(test_l2))

if save_pred == 'npy':
    np.save('bp_pred_all_fno3d.npy', pred.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('bp_pred_all_fno3d.mat', mdict={'pred': pred.cpu().numpy()})
