"""
Script for training the FNO surrogate model for the derivative of the adjoint operator 
    of the semilinear parabolic optimal control problem.
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

torch.manual_seed(1919810)
np.random.seed(1919810)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Configurations

TRAIN_PATH = './data/semilinparab/sp_gradadj_train.npz'
TEST_PATH = './data/semilinparab/sp_gradadj_test.npz'

n_train = 2048
n_test = 256 

modes = [8, 8, 8]
width = 16
input_num = 2
paddings = [8, 8, 8]

batch_size = 8
epochs = 3
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.6

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
train_yh = torch.tensor(train_reader['yh'])[:n_train, ::sub, ::sub, :T].to(device)
train_f = torch.tensor(train_reader['f'])[:n_train, ::sub, ::sub, :T].to(device)
train_s = torch.tensor(train_reader['s'])[:n_train, ::sub, ::sub, :T].to(device)
del train_reader

test_reader = np.load(TEST_PATH)
test_yh = torch.tensor(test_reader['yh'])[-n_test:, ::sub, ::sub, :T].to(device)
test_f = torch.tensor(test_reader['f'])[-n_test:, ::sub, ::sub, :T].to(device)
test_s = torch.tensor(test_reader['s'])[-n_test:, ::sub, ::sub, :T].to(device)
del test_reader

train_yh = train_yh.reshape(n_train, S, S, T, 1)
train_f = train_f.reshape(n_train, S, S, T, 1)
test_yh = test_yh.reshape(n_test, S, S, T, 1)
test_f = test_f.reshape(n_test, S, S, T, 1)
train_yh_f = torch.cat((train_yh, train_f), dim=-1)
test_yh_f = torch.cat((test_yh, test_f), dim=-1)
del train_yh, train_f, test_yh, test_f

train_loader = DataLoader(
    TensorDataset(train_yh_f, train_s),
    batch_size=batch_size,
    shuffle=True
)


# Hard constraints

size_x = size_y = size_z = 64
gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, 1, size_x, 1).repeat([1, size_y, 1, size_z])
gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, size_y, 1, 1).repeat([1, 1, size_x, size_z])
gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z).repeat([1, size_x, size_y, 1])
hc_single = gridx * (1.0 - gridx) * gridy * (1.0 - gridy) * (1.0 - gridz**0.4)
x = gridx.repeat((batch_size, 1, 1, 1))
y = gridy.repeat((batch_size, 1, 1, 1))
t = gridz.repeat((batch_size, 1, 1, 1))
hc = x * (1.0 - x) * y * (1.0 - y) * (1.0 - t**0.4)
hc_single = hc_single.to(device)
hc = hc.to(device)
del gridx, gridy, gridz, x, y, t


if record_time:
    t2 = default_timer()
    print(f'preprocessing finished, time used: {t2-t1}')
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

if record_time:
    t_start = default_timer()

for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for yh_f, s in train_loader:
        optimizer.zero_grad()
        out = hc * model(yh_f).view(batch_size, S, S, T)

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

torch.save(model.state_dict(), "./sp_model_gradadj_fno3d_hc_param.pt")


# Testing the trained model

test_loader = DataLoader(
    TensorDataset(test_yh_f, test_s),
    batch_size=1,
    shuffle=False
)
pred = torch.zeros(test_s.shape)
test_l2 = np.zeros(n_test)
index = 0
with torch.no_grad():
    for yh_f, s in test_loader:
        out = (hc_single.to(device) * model(yh_f).view(1, S, S, T)).view(S, S, T)
        pred[index] = out
        test_l2[index] = myloss(out.view(1, -1), s.view(1, -1)).item()
        index = index + 1
    print('testing l2:', np.mean(test_l2))

if save_pred == 'npy':
    np.save('./sp_pred_gradadj_fno3d_hc.npy', pred.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('./sp_pred_gradadj_fno3d_hc.mat', mdict={'pred': pred.cpu().numpy()})
