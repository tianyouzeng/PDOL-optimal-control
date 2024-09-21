"""
Script for training the FNO surrogate model for the derivative of the adjoint operator 
    of the semilinear parabolic optimal control problem.
This file is adapted from the official implementation of FNO in:
    - https://github.com/khassibi/fourier-neural-operator/fourier_3d.py
"""


# Imports and initializations

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

from models.fno import FNO3d_M
from utils.utils_fno import MatReader, Adam, LpLoss, count_params

from timeit import default_timer

torch.manual_seed(1919810)
np.random.seed(1919810)


# Configurations

TRAIN_PATH = ('./data/semilinparab/sp_gradadj_train_yh.mat', 
              './data/semilinparab/sp_gradadj_train_f.mat', 
              './data/semilinparab/sp_gradadj_train_y.mat')
TEST_PATH = ('./data/semilinparab/sp_gradadj_test_yh.mat', 
             './data/semilinparab/sp_gradadj_test_f.mat', 
             './data/semilinparab/sp_gradadj_test_y.mat')

ntrain = 2048
ntest = 256 
ntestopt = 1

modes = 8
width = 16
input_num = 2
paddings = 8

batch_size = 8
epochs = 500
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.6

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64 // sub
T = 64


# Load data

reader = MatReader(TRAIN_PATH[0])
train_u = reader.read_field('yh')[:ntrain,::sub,::sub,:T]
reader = MatReader(TRAIN_PATH[1])
train_f = reader.read_field('f')[:ntrain,::sub,::sub,:T]
reader = MatReader(TRAIN_PATH[2])
train_s = reader.read_field('s')[:ntrain,::sub,::sub,:T]

reader = MatReader(TEST_PATH[0])
test_u = reader.read_field('yh')[-ntest:,::sub,::sub,:T]
reader = MatReader(TEST_PATH[1])
test_f = reader.read_field('f')[-ntest:,::sub,::sub,:T]
reader = MatReader(TEST_PATH[2])
test_s = reader.read_field('s')[-ntest:,::sub,::sub,:T]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_u = train_u.reshape(ntrain,S,S,T,1)
train_f = train_f.reshape(ntrain,S,S,T,1)
test_u = test_u.reshape(ntest,S,S,T,1)
test_f = test_f.reshape(ntest,S,S,T,1)
train_uf = torch.cat((train_u, train_f), axis=-1)
test_uf = torch.cat((test_u, test_f), axis=-1)

train_loader = DataLoader(TensorDataset(train_uf, train_s), batch_size=batch_size, shuffle=True)

size_x = size_y = size_z = 64
gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, 1, size_x, 1).repeat([1, size_y, 1, size_z])
gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, size_y, 1, 1).repeat([1, 1, size_x, size_z])
gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z).repeat([1, size_x, size_y, 1])
hc_single = gridx * (1.0 - gridx) * gridy * (1.0 - gridy) * (1.0 - gridz**0.4)
x = gridx.repeat((batch_size,1,1,1))
y = gridy.repeat((batch_size,1,1,1))
t = gridz.repeat((batch_size,1,1,1))
hc = x * (1.0 - x) * y * (1.0 - y) * (1.0 - t**0.4)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')


# Training

model = FNO3d_M(input_num, modes, modes, modes, width, paddings).cuda()
print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for uf, s in train_loader:
        uf, s = uf.cuda(), s.cuda()

        optimizer.zero_grad()
        out = hc.cuda() * model(uf).view(batch_size, S, S, T)

        mse = F.mse_loss(out, s, reduction='mean')
        l2 = myloss(out.view(batch_size, -1), s.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    train_mse /= len(train_loader)
    train_l2 /= ntrain

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2)

torch.save(model.state_dict(), "./sp_model_gradadj_fno3d_hc_param.pt")


# Testing the trained model

pred = torch.zeros(test_s.shape)
index = 0
test_loader = DataLoader(TensorDataset(test_uf, test_s), batch_size=1, shuffle=False)
with torch.no_grad():
    for uf, s in test_loader:
        test_l2 = 0
        uf, s = uf.cuda(), s.cuda()

        out = (hc_single.cuda() * model(uf).view(1, S, S, T)).view(S, S, T)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), s.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1

# sio.savemat('./sp_pred_gradadj_fno3d_hc_42.mat', mdict={'pred': pred.cpu().numpy()})
