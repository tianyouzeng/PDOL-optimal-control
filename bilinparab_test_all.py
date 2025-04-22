"""
Script for testing the trained FNO surrogate model 
    of the bilinear parabolic optimal control problem.
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

TEST_PATH = './data/bilinparab/bp_cts_gradadj_test.npz'
n_test = 256

modes = [8, 8, 8]
width = 16
input_num = 2
paddings = [8, 8, 8]

sub = 1
S = 64 // sub
T = 64

save_pred: Literal['off', 'npy', 'mat'] = 'off'


# Load data

test_reader = np.load(TEST_PATH)
test_u = torch.tensor(test_reader["u"])[-n_test:, ::sub, ::sub, :T].to(device)
test_f = torch.tensor(test_reader["f"])[-n_test:, ::sub, ::sub, :T].to(device)
test_s = torch.tensor(test_reader["s"])[-n_test:, ::sub, ::sub, :T].to(device)
del test_reader

test_u = test_u.reshape(n_test,S,S,T,1)
test_f = test_f.reshape(n_test,S,S,T,1)
test_uf = torch.cat((test_u, test_f), dim=-1)
del test_u, test_f

test_loader = DataLoader(TensorDataset(test_uf, test_s), batch_size=1, shuffle=False)


# Evaluation

model = FNO3d(input_num, modes, width, paddings).to(device)
model.load_state_dict(torch.load('./trained_models/bp_model_cts_gradadj_fno3d_param.pt'))
print(f'FNO parameter count: {count_params(model)}\n')

myloss = LpLoss(d=3, size_average=False)
pred = torch.zeros(test_s.shape)
index = 0
test_l2_rel = torch.zeros(n_test)
test_l2_abs = torch.zeros(n_test)

with torch.no_grad():
    for uf, s in test_loader:
        out = model(uf).view(1, S, S, T)
        pred[index] = out[None,:,:,:]

        test_l2_rel[index] = myloss(out, s).item()
        test_l2_abs[index] = myloss.abs(out, s).item()
        index = index + 1

print(f"Absolute L2 error mean: {torch.mean(test_l2_abs).item():.4e}")
print(f"Relative L2 error mean: {torch.mean(test_l2_rel).item():.4e}")

print(f"Absolute L2 error SD: {torch.std(test_l2_abs).item():.4e}")
print(f"Relative L2 error SD: {torch.std(test_l2_rel).item():.4e}")

if save_pred == 'npy':
    np.save('./bp_pred_cts_gradadj_fno3d.npy', pred.cpu().numpy())
elif save_pred == 'mat':
    sio.savemat('bp_pred_cts_gradadj_fno3d.mat', mdict={'pred': pred.cpu().numpy()})
