"""
Script for testing the trained FNO surrogate models 
    of the semilinear parabolic optimal control problem.
"""


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

from models.fno import FNO3d, FNO3d_M
from utils.utils_fno import MatReader, LpLoss, count_params


########## Testing control-to-state model ##########


# Configs

modes = 8
width = 20
input_num = 1
paddings = 8

sub = 1
S = 64 // sub
T = 64


# Load data

TEST_PATH = ('./data/semilinparab/sp_cts_test_u.mat', 
             './data/semilinparab/sp_cts_test_y.mat')
ntest = 256

reader = MatReader(TEST_PATH[0])
test_u = reader.read_field('u')[-ntest:,::sub,::sub,:T]
reader = MatReader(TEST_PATH[1])
test_s = reader.read_field('s')[-ntest:,::sub,::sub,:T]

test_u = test_u.reshape(ntest,S,S,T,1)

size_x = size_y = size_z = 64
gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, 1, size_x, 1).repeat([1, size_y, 1, size_z])
gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, size_y, 1, 1).repeat([1, 1, size_x, size_z])
gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z).repeat([1, size_x, size_y, 1])
hc_single = gridx * (1.0 - gridx) * gridy * (1.0 - gridy) * gridz**0.4


# Evaluation

model_cts = FNO3d(modes, modes, modes, width, paddings).cuda()
model_cts.load_state_dict(torch.load('./trained_models/sp_model_cts_fno3d_hc_param.pt'))
print(count_params(model_cts))
model_cts.eval()

myloss = LpLoss(size_average=False)

pred_cts = torch.zeros(test_s.shape)
index = 0
test_loader = DataLoader(TensorDataset(test_u, test_s), batch_size=1, shuffle=False)
test_cts_l2_rel = torch.zeros(ntest)
test_cts_l2_abs = torch.zeros(ntest)
with torch.no_grad():
    for u, s in test_loader:
        u, s = u.cuda(), s.cuda()

        out = hc_single.cuda() * model_cts(u).view(S, S, T)
        pred_cts[index] = out[None,:,:,:]

        test_cts_l2_rel[index] = myloss(out, s).item()
        test_cts_l2_abs[index] = myloss.abs(out, s).item()
        print(index, test_cts_l2_abs[index].item(), test_cts_l2_rel[index].item())
        index = index + 1

print("Absolute L2 error mean: ", torch.mean(test_cts_l2_abs).item())
print("Relative L2 error mean: ", torch.mean(test_cts_l2_rel).item())

print("Absolute L2 error SD: ", torch.std(test_cts_l2_abs).item())
print("Relative L2 error SD: ", torch.std(test_cts_l2_rel).item())

# sio.savemat('sp_pred_cts_fno3d.mat', mdict={'pred': pred_cts.cpu().numpy()})


########## Testing gradient-adjoint model ##########


# Configs

modes = 8
width = 16
input_num = 2
paddings = 8

sub = 1
S = 64 // sub
T = 64


# Load data

TEST_PATH = ('./data/semilinparab/sp_gradadj_test_yh.mat', 
             './data/semilinparab/sp_gradadj_test_f.mat',
             './data/semilinparab/sp_gradadj_test_y.mat')
ntest = 256

reader = MatReader(TEST_PATH[0])
test_u = reader.read_field('yh')[-ntest:,::sub,::sub,:T]
reader = MatReader(TEST_PATH[1])
test_f = reader.read_field('f')[-ntest:,::sub,::sub,:T]
reader = MatReader(TEST_PATH[2])
test_s = reader.read_field('s')[-ntest:,::sub,::sub,:T]

test_u = test_u.reshape(ntest,S,S,T,1)
test_f = test_f.reshape(ntest,S,S,T,1)
test_uf = torch.cat((test_u, test_f), axis=-1)

size_x = size_y = size_z = 64
gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, 1, size_x, 1).repeat([1, size_y, 1, size_z])
gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, size_y, 1, 1).repeat([1, 1, size_x, size_z])
gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z).repeat([1, size_x, size_y, 1])
hc_single = gridx * (1.0 - gridx) * gridy * (1.0 - gridy) * (1.0 - gridz**0.4)


# Evaluation

model_gradadj = FNO3d_M(input_num, modes, modes, modes, width, paddings).cuda()
model_gradadj.load_state_dict(torch.load('./trained_models/sp_model_gradadj_fno3d_hc_param.pt'))
print(count_params(model_gradadj))
model_gradadj.eval()

myloss = LpLoss(size_average=False)

pred_gradadj = torch.zeros(test_s.shape)
index = 0
test_loader = DataLoader(TensorDataset(test_uf, test_s), batch_size=1, shuffle=False)
test_gradadj_l2_rel = torch.zeros(ntest)
test_gradadj_l2_abs = torch.zeros(ntest)
with torch.no_grad():
    for uf, s in test_loader:
        uf, s = uf.cuda(), s.cuda()

        out = hc_single.cuda() * model_gradadj(uf).view(S, S, T)
        pred_gradadj[index] = out[None,:,:,:]

        test_gradadj_l2_rel[index] = myloss(out, s).item()
        test_gradadj_l2_abs[index] = myloss.abs(out, s).item()
        print(index, test_gradadj_l2_abs[index].item(), test_gradadj_l2_rel[index].item())
        index = index + 1

print("Absolute L2 error mean: ", torch.mean(test_gradadj_l2_abs).item())
print("Relative L2 error mean: ", torch.mean(test_gradadj_l2_rel).item())

print("Absolute L2 error SD: ", torch.std(test_gradadj_l2_abs).item())
print("Relative L2 error SD: ", torch.std(test_gradadj_l2_rel).item())

# sio.savemat('sp_pred_gradadj_fno3d.mat', mdict={'pred': pred_gradadj.cpu().numpy()})
