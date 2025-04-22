"""
Script for testing the trained FNO surrogate models 
    of the semilinear parabolic optimal control problem.
"""


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from typing import Literal

from models.fno import FNO3d
from utils.utils_fno import LpLoss, count_params

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Configurations

TEST_PATH_CTS = './data/semilinparab/sp_cts_test.npz'
TEST_PATH_GRADADJ = './data/semilinparab/sp_gradadj_test.npz'
n_test_cts = 256
n_test_gradadj = 256

modes_cts = [16, 16, 16]
width_cts = 32
input_num_cts = 1
paddings_cts = [8, 8, 8]

modes_gradadj = [8, 8, 8]
width_gradadj = 16
input_num_gradadj = 2
paddings_gradadj = [8, 8, 8]

sub = 1
S = 64 // sub
T = 64

save_pred: Literal['off', 'npy', 'mat'] = 'off'


# Grid points

size_x = size_y = size_z = 64
gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, 1, size_x, 1).repeat([1, size_y, 1, size_z])
gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, size_y, 1, 1).repeat([1, 1, size_x, size_z])
gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z).repeat([1, size_x, size_y, 1])


''' Test control-to-state model '''

# Load data

test_reader_cts = np.load(TEST_PATH_CTS)
test_u_cts = torch.tensor(test_reader_cts['u'])[-n_test_cts:, ::sub, ::sub, :T].to(device)
test_s_cts = torch.tensor(test_reader_cts['s'])[-n_test_cts:, ::sub, ::sub, :T].to(device)
del test_reader_cts

test_u_cts = test_u_cts.reshape(n_test_cts,S,S,T,1)

test_loader_cts = DataLoader(
    TensorDataset(test_u_cts, test_s_cts),
    batch_size=1,
    shuffle=False
)

hc_single_cts = gridx * (1.0 - gridx) * gridy * (1.0 - gridy) * gridz**0.4
hc_single_cts = hc_single_cts.to(device)


# Evaluation

print(f"\nTesting control-to-state model:\n")

model_cts = FNO3d(input_num_cts, modes_cts, width_cts, paddings_cts).to(device)
model_cts.load_state_dict(torch.load('./trained_models/sp_model_cts_fno3d_hc_param.pt'))
print(f'FNO parameter count: {count_params(model_cts)}\n')

myloss = LpLoss(size_average=False)
pred_cts = torch.zeros(test_s_cts.shape)
index = 0

test_cts_l2_rel = torch.zeros(n_test_cts)
test_cts_l2_abs = torch.zeros(n_test_cts)
with torch.no_grad():
    for u, s in test_loader_cts:
        out = hc_single_cts * model_cts(u).view(S, S, T)
        pred_cts[index] = out[None,:,:,:]
        test_cts_l2_rel[index] = myloss(out, s).item()
        test_cts_l2_abs[index] = myloss.abs(out, s).item()
        index = index + 1

print(f"Absolute L2 error mean: {torch.mean(test_cts_l2_abs).item():.4e}")
print(f"Relative L2 error mean: {torch.mean(test_cts_l2_rel).item():.4e}")

print(f"Absolute L2 error SD: {torch.std(test_cts_l2_abs).item():.4e}")
print(f"Relative L2 error SD: {torch.std(test_cts_l2_rel).item():.4e}")

if save_pred == 'npy':
    np.save('./sp_pred_cts_fno3d.npy', pred_cts.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('./sp_pred_cts_fno3d.mat', mdict={'pred': pred_cts.cpu().numpy()})


''' Testing gradient-adjoint model '''


# Load data

test_reader = np.load(TEST_PATH_GRADADJ)
test_u_gradadj = torch.tensor(test_reader['yh'])[-n_test_gradadj:, ::sub, ::sub, :T].to(device)
test_f_gradadj = torch.tensor(test_reader['f'])[-n_test_gradadj:, ::sub, ::sub, :T].to(device)
test_s_gradadj = torch.tensor(test_reader['s'])[-n_test_gradadj:, ::sub, ::sub, :T].to(device)
del test_reader

test_u_gradadj = test_u_gradadj.reshape(n_test_gradadj,S,S,T,1)
test_f_gradadj = test_f_gradadj.reshape(n_test_gradadj,S,S,T,1)
test_uf_gradadj = torch.cat((test_u_gradadj, test_f_gradadj), dim=-1)
del test_u_gradadj, test_f_gradadj

test_loader_gradadj = DataLoader(
    TensorDataset(test_uf_gradadj, test_s_gradadj),
    batch_size=1,
    shuffle=False
)

hc_single_gradadj = gridx * (1.0 - gridx) * gridy * (1.0 - gridy) * (1.0 - gridz**0.4)
hc_single_gradadj = hc_single_gradadj.to(device)


# Evaluation

print(f"\nTesting gradient-adjoint model:\n")

model_gradadj = FNO3d(input_num_gradadj, modes_gradadj, 
                      width_gradadj, paddings_gradadj).to(device)
model_gradadj.load_state_dict(
    torch.load('./trained_models/sp_model_gradadj_fno3d_hc_param.pt'))
print(f'FNO parameter count: {count_params(model_gradadj)}\n')

myloss = LpLoss(size_average=False)
pred_gradadj = torch.zeros(test_s_gradadj.shape)
index = 0
test_gradadj_l2_rel = torch.zeros(n_test_gradadj)
test_gradadj_l2_abs = torch.zeros(n_test_gradadj)

with torch.no_grad():
    for uf, s in test_loader_gradadj:
        out = hc_single_gradadj * model_gradadj(uf).view(S, S, T)
        pred_gradadj[index] = out[None,:,:,:]
        test_gradadj_l2_rel[index] = myloss(out, s).item()
        test_gradadj_l2_abs[index] = myloss.abs(out, s).item()
        index = index + 1

print(f"Absolute L2 error mean: {torch.mean(test_gradadj_l2_abs).item():.4e}")
print(f"Relative L2 error mean: {torch.mean(test_gradadj_l2_rel).item():.4e}")

print(f"Absolute L2 error SD: {torch.std(test_gradadj_l2_abs).item():.4e}")
print(f"Relative L2 error SD: {torch.std(test_gradadj_l2_rel).item():.4e}")

if save_pred == 'npy':
    np.save('./sp_pred_gradadj_fno3d.npy', pred_gradadj.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('./sp_pred_gradadj_fno3d.mat', mdict={'pred': pred_gradadj.cpu().numpy()})
