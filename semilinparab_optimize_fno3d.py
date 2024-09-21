''' 
This script solves the following semilinear parabolic optimal control problem 
    with the pretrained FNO surrogate models on the spatial-temporal domain [0, 1]^2 x [0, 1]:
min     J(u, y) = 0.5 * ||y - yd||_2^2 + 0.5 * alpha * ||u||_2^2 + beta * ||u||_1^2
s.t.    \partial_t y - \Delta y + y * (y - 0.25) * (y + 1) = u,
        y = 0 on the boundary,
        y(0) = 0 when T=0,
        u_a <= u <= u_b.
'''


# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation

from models.fno import FNO3d, FNO3d_M

from torch import Tensor    # for type hints

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Load trained models

MODEL_PATH_CTS = './trained_models/sp_model_cts_fno3d_hc_param.pt'
MODEL_PATH_GRADADJ = './trained_models/sp_model_gradadj_fno3d_hc_param.pt'

modes = 8
width_cts = 20
width_gradadj = 16
input_num_gradadj = 2
paddings = 8

model_cts = FNO3d(modes, modes, modes, width_cts, paddings)
model_cts.load_state_dict(torch.load(MODEL_PATH_CTS))
model_cts.to(device).eval()

model_gradadj = FNO3d_M(input_num_gradadj, modes, modes, modes, width_gradadj, paddings)
model_gradadj.load_state_dict(torch.load(MODEL_PATH_GRADADJ))
model_gradadj.to(device).eval()


# Descritization scheme

S = 64  # spacial discretization resolution
T = 64  # temporal discretization resolution
xlist, ylist, tlist = np.linspace(0, 1, S), np.linspace(0, 1, S), np.linspace(0, 1, T)
xmesh, ymesh = np.meshgrid(xlist, ylist)
x = np.repeat(xmesh[:,:,None], tlist.shape[0], axis=2)
y = np.repeat(ymesh[:,:,None], tlist.shape[0], axis=2)
t = np.repeat(np.repeat(tlist[None,None,:], xmesh.shape[0], axis=0), xmesh.shape[1], axis=1)
hc = torch.tensor(x * (1.0 - x) * y * (1.0 - y) * (1.0 - t**0.4)).float()
hc_cts = torch.tensor(x * (1.0 - x) * y * (1.0 - y) * t**0.4).float()


# Problem constants and coefficients

T_MAX = 1.0
u_a = -10.0
u_b = 20.0
alpha = 0.0001
beta = 0.004

def yd_func(x, y, t):
    return np.exp(-20.0 * ((x - 0.2)**2 + (y - 0.2)**2 + (t - 0.2)**2)) \
        + np.exp(-20.0 * ((x - 0.7)**2 + (y - 0.7)**2 + (t - 0.9)**2))

yd = torch.tensor(yd_func(x, y, t)).float()


# Primal-dual algorithm parameters

tau = 500.0
sigma = 0.4
omega = 1.0
rho = 1.8   # control relaxation stepsize, set to 1.0 for no-relaxation
MAX_ITER = 1000


# Proximal operators

def prox_G(u: Tensor, alpha: float, beta: float, tau: float, u_a: float, u_b: float) -> Tensor:
    prod = tau * beta
    val1 = (u - prod) / (alpha * tau + 1)
    val2 = (u + prod) / (alpha * tau + 1)
    proxval = torch.zeros(u.shape)
    proxval[u > prod] = val1[u > prod]
    proxval[u < -prod] = val2[u < -prod]
    proxval = torch.maximum(u_a * torch.ones_like(proxval), torch.minimum(u_b * torch.ones_like(proxval), proxval))
    return proxval

def prox_F_conj(y: Tensor, yd: Tensor, sigma: float) -> Tensor:
    proxval = (y - sigma * yd) / (1.0 + sigma)
    return proxval


# Initialization

u_prev = torch.zeros((S, S, T))
p_prev = torch.zeros((S, S, T))
u = torch.zeros((S, S, T))
p = torch.zeros((S, S, T))
s = torch.zeros((S, S, T))


# Optimization

with torch.no_grad():
    for iter in range(MAX_ITER):
        print(iter + 1)
        u_prev = u.detach().clone()
        p_prev = p.detach().clone()

        y_hat = hc_cts * model_cts(u_prev[None,:,:,:,None].to(device)).view(S, S, T).detach().cpu()
        S_adj_p_prod = hc * model_gradadj(torch.cat((y_hat[None,:,:,:,None].to(device), 
                          p_prev[None,:,:,:,None].to(device)), axis=-1)).view(S, S, T).detach().cpu()
        u_step = tau * S_adj_p_prod
        u = prox_G(u_prev - u_step, alpha, beta, tau, u_a, u_b)

        u_extra = u + omega * (u - u_prev)

        cts_u_prod = hc_cts * model_cts(u_extra[None,:,:,:,None].to(device)).view(S, S, T).detach().cpu()    # match the input requirement of FNO
        p_step = sigma * cts_u_prod
        p = prox_F_conj(p_prev + p_step, yd, sigma)

        # Relaxation steps, optional
        u = u_prev + 1.8 * (u - u_prev)
        p = p_prev + 1.8 * (p - p_prev)

        diff_u = torch.norm(u - u_prev).item() / torch.max(torch.tensor([(S * S * T)**(0.5), torch.norm(u_prev).item()])).item()
        diff_p = torch.norm(p - p_prev).item() / torch.max(torch.tensor([(S * S * T)**(0.5), torch.norm(p_prev).item()])).item()
        print(diff_u)
        print(diff_p)
        if iter >= 3 and diff_u <= 1e-5 and diff_p <= 1e-5:
            break


# Solve the state variable

s = hc_cts * model_cts(u[None,:,:,:,None].to(device)).view(S, S, T).detach().cpu()


# Test the result

import scipy.io as sio

sol_mat_contents = sio.loadmat("./trad_alg/sp_sol_solvepde_beta_0.004.mat")
u_exact = sol_mat_contents['u'].transpose(1,0,2)
s_exact = sol_mat_contents['s'].transpose(1,0,2)

u = u.detach().cpu().numpy()
p = p.detach().cpu().numpy()
s = s.detach().cpu().numpy()

# np.savez("./results/sp_computed_u_fno3d_beta_0.004.npz", xmesh, ymesh, u)

fig, ax = plt.subplots()
ctf = ax.pcolormesh(xmesh, ymesh, u[:,:,16], vmin=-5.1, vmax=20.1, cmap=cm.coolwarm, shading='gouraud', rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-5, 0, 5, 10, 15, 20])
# plt.savefig("./results/sp_computed_u_fno_0.25_beta_0.004.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(xmesh, ymesh, u[:,:,32], vmin=-5.1, vmax=20.1, cmap=cm.coolwarm, shading='gouraud', rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-5, 0, 5, 10, 15, 20])
# plt.savefig("./results/sp_computed_u_fno_0.5_beta_0.004.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(xmesh, ymesh, u[:,:,48], vmin=-5.1, vmax=20.1, cmap=cm.coolwarm, shading='gouraud', rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-5, 0, 5, 10, 15, 20])
# plt.savefig("./results/sp_computed_u_fno_0.75_beta_0.004.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(xmesh, ymesh, s[:,:,16], vmin=-0.1, vmax=0.3, cmap=cm.coolwarm, shading='gouraud', rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-0.1, 0, 0.1, 0.2, 0.3])
# plt.savefig("./results/sp_computed_y_fno_0.25_beta_0.004.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(xmesh, ymesh, s[:,:,32], vmin=-0.1, vmax=0.3, cmap=cm.coolwarm, shading='gouraud', rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-0.1, 0, 0.1, 0.2, 0.3])
# plt.savefig("./results/sp_computed_y_fno_0.5_beta_0.004.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(xmesh, ymesh, s[:,:,48], vmin=-0.1, vmax=0.3, cmap=cm.coolwarm, shading='gouraud', rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-0.1, 0, 0.1, 0.2, 0.3])
# plt.savefig("./results/sp_computed_y_fno_0.75_beta_0.004.pdf")
plt.show()

err_u_abs = np.linalg.norm(u - u_exact) / np.sqrt(S * S * T)
err_u_rel = np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)
print("absolute error of control: ", err_u_abs)
print("relative error of control: ", err_u_rel)

err_s_abs = np.linalg.norm(s - s_exact) / np.sqrt(S * S * T)
err_s_rel = np.linalg.norm(s - s_exact) / np.linalg.norm(s_exact)
print("absolute error of state: ", err_s_abs)
print("relative error of state: ", err_s_rel)


