"""
Script for testing the trained FNO surrogate model 
    of the bilinear parabolic optimal control problem.
"""


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

from models.fno import FNO3d_M
from utils.utils_fno import LpLoss, count_params, MatReader


# Configs

modes = 8
width = 16
input_num = 2
paddings = 8

sub = 1
S = 64 // sub
T = 64


# Load data

TEST_PATH = ('./data/bilinparab/bp_cts_gradadj_test_u.mat', 
             './data/bilinparab/bp_cts_gradadj_test_y.mat')
ntest = 256

reader = MatReader(TEST_PATH[0])
test_u = reader.read_field('u')[-ntest:,::sub,::sub,:T]
reader = MatReader(TEST_PATH[1])
test_f = reader.read_field('f')[-ntest:,::sub,::sub,:T]
reader = MatReader(TEST_PATH[2])
test_s = reader.read_field('s')[-ntest:,::sub,::sub,:T]

test_u = test_u.reshape(ntest,S,S,T,1)
test_f = test_f.reshape(ntest,S,S,T,1)
test_uf = torch.cat((test_u, test_f), axis=-1)

device = torch.device('cuda')

# Evaluation

model = FNO3d_M(input_num, modes, modes, modes, width, paddings).cuda()
model.load_state_dict(torch.load('./trained_models/bp_model_cts_gradadj_fno3d_param.pt'))
print(count_params(model))
model.eval()

myloss = LpLoss(d=3, size_average=False)

pred = torch.zeros(test_s.shape)
index = 0
test_loader = DataLoader(TensorDataset(test_uf, test_s), batch_size=1, shuffle=False)
test_l2_rel = torch.zeros(ntest)
test_l2_abs = torch.zeros(ntest)
with torch.no_grad():
    for uf, s in test_loader:
        uf, s = uf.cuda(), s.cuda()

        out = model(uf).view(1, S, S, T)
        pred[index] = out[None,:,:,:]

        test_l2_rel[index] = myloss(out, s).item()
        test_l2_abs[index] = myloss.abs(out, s).item()
        print(index, test_l2_abs[index].item(), test_l2_rel[index].item())
        index = index + 1

print("Absolute L2 error mean: ", torch.mean(test_l2_abs).item())
print("Relative L2 error mean: ", torch.mean(test_l2_rel).item())

print("Absolute L2 error SD: ", torch.std(test_l2_abs).item())
print("Relative L2 error SD: ", torch.std(test_l2_rel).item())

# sio.savemat('bp_pred_fno3d.mat', mdict={'pred': pred.cpu().numpy()})




