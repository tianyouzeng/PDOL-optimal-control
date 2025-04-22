''' 
This script solves the following semilinear parabolic optimal control problem 
    with the pretrained FNO surrogate models on the spatial-temporal domain [0, 1]^2 x [0, 1]:
min     J(u, y) = 0.5 * ||y - yd||_2^2 + 0.5 * alpha * ||u||_2^2 + beta * ||u||_1^2
s.t.    partial_t y - Delta y + y * (y - 0.25) * (y + 1) = u,
        y = 0 on the boundary,
        y(0) = 0 when T=0,
        u_a <= u <= u_b.
'''


# Imports

import timeit

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.fno import FNO3d
from utils.semilinparab_prob_param import yd_func

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Primal-dual algorithm parameters

tau = 500.0
sigma = 0.4
omega = 1.0
rho_u = rho_p = 1.8     # control relaxation stepsize, set to 1.0 for no-relaxation
MAX_ITER = 1000

verbose = True          # print the iteration information if True
record_time = True      # print the computation time if True
save_results = False    # save the computed optimal control and state if True
save_plots = False      # save plotted results if True


# Descritization scheme

S = 64  # spacial discretization resolution
T = 64  # temporal discretization resolution
x_list, y_list, t_list = \
    np.linspace(0, 1, S), np.linspace(0, 1, S), np.linspace(0, 1, T)
x_mesh, y_mesh = np.meshgrid(x_list, y_list)
x = np.repeat(x_mesh[:,:,None], t_list.shape[0], axis=2)
y = np.repeat(y_mesh[:,:,None], t_list.shape[0], axis=2)
t = np.repeat(np.repeat(t_list[None,None,:], x_mesh.shape[0], axis=0), 
              x_mesh.shape[1], axis=1)
hc = torch.tensor(x * (1.0 - x) * y * (1.0 - y) * (1.0 - t**0.4)).float()
hc_cts = torch.tensor(x * (1.0 - x) * y * (1.0 - y) * t**0.4).float()


# Load trained models

MODEL_PATH_CTS = './trained_models/sp_model_cts_fno3d_hc_param.pt'
MODEL_PATH_GRADADJ = './trained_models/sp_model_gradadj_fno3d_hc_param.pt'

modes_cts = [16, 16, 16]
modes_gradadj = [8, 8, 8]
width_cts = 32
width_gradadj = 16
input_num_cts = 1
input_num_gradadj = 2
train_S = 64
train_T = 64
paddings = [8 * S // train_S, 8 * S // train_S, 8 * T // train_T]

model_cts = FNO3d(input_num_cts, modes_cts, width_cts, paddings)
model_cts.load_state_dict(torch.load(MODEL_PATH_CTS))
model_cts.to(device).eval()

model_gradadj = FNO3d(input_num_gradadj, modes_gradadj, width_gradadj, paddings)
model_gradadj.load_state_dict(torch.load(MODEL_PATH_GRADADJ))
model_gradadj.to(device).eval()


# Problem constants and coefficients

T_MAX = 1.0
u_a = -10.0
u_b = 20.0
alpha = 0.0001
beta = 0.004    # use 0 instead of 0.0 to generate correct file names

yd = torch.tensor(yd_func(x, y, t)).float()


# Proximal operators

def prox_G(u: torch.Tensor, alpha: float, beta: float, 
           tau: float, u_a: float, u_b: float) -> torch.Tensor:
    prod = tau * beta
    val1 = (u - prod) / (alpha * tau + 1)
    val2 = (u + prod) / (alpha * tau + 1)
    proxval = torch.zeros(u.shape)
    proxval[u > prod] = val1[u > prod]
    proxval[u < -prod] = val2[u < -prod]
    proxval = torch.maximum(
        u_a * torch.ones_like(proxval),
        torch.minimum(u_b * torch.ones_like(proxval), proxval)
    )
    return proxval

def prox_F_conj(y: torch.Tensor, yd: torch.Tensor, sigma: float) -> torch.Tensor:
    proxval = (y - sigma * yd) / (1.0 + sigma)
    return proxval


# Initialization

time_start = 0
if record_time:
    time_start = timeit.default_timer()

u_prev = torch.zeros((S, S, T))
p_prev = torch.zeros((S, S, T))
u = torch.zeros((S, S, T))
p = torch.zeros((S, S, T))
s = torch.zeros((S, S, T))


# Optimization

with torch.no_grad():
    for iter in range(MAX_ITER):
        if verbose:
            print(f"iter: {iter + 1}")
        u_prev = u.detach().clone()
        p_prev = p.detach().clone()

        y_hat = hc_cts * model_cts(
            u_prev[None,:,:,:,None].to(device)).view(S, S, T).detach().cpu()
        S_adj_p_prod = hc * model_gradadj(
            torch.cat((
                y_hat[None,:,:,:,None].to(device),
                p_prev[None,:,:,:,None].to(device)
            ), dim=-1)
        ).view(S, S, T).detach().cpu()
        u_step = tau * S_adj_p_prod
        u = prox_G(u_prev - u_step, alpha, beta, tau, u_a, u_b)

        u_extra = u + omega * (u - u_prev)

        cts_u_prod = hc_cts * model_cts(
            u_extra[None,:,:,:,None].to(device)).view(S, S, T).detach().cpu()    # match the input requirement of FNO
        p_step = sigma * cts_u_prod
        p = prox_F_conj(p_prev + p_step, yd, sigma)

        # Relaxation steps, optional
        u = u_prev + rho_u * (u - u_prev)
        p = p_prev + rho_p * (p - p_prev)

        diff_u = torch.norm(u - u_prev).item() / torch.max(torch.tensor([
            (S * S * T)**(0.5),
            torch.norm(u_prev).item()
        ])).item()
        diff_p = torch.norm(p - p_prev).item() / torch.max(torch.tensor([
            (S * S * T)**(0.5),
            torch.norm(p_prev).item()
        ])).item()
        if verbose:
            print(f"{diff_u=}")
            print(f"{diff_p=}")
        if iter >= 3 and diff_u <= 1e-5 and diff_p <= 1e-5:
            break


# Solve the state variable

s = hc_cts * model_cts(u[None,:,:,:,None].to(device)).view(S, S, T).detach().cpu()

if record_time:
    time_end = timeit.default_timer()
    print("computation time: ", time_end - time_start)


# Plot the result

u = u.detach().cpu().numpy()
p = p.detach().cpu().numpy()
s = s.detach().cpu().numpy()

if save_results:
    np.savez(f"./results/sp_computed_fno_beta_{beta}.npz", x_mesh, y_mesh, u, s)

fig, ax = plt.subplots()
ctf = ax.pcolormesh(x_mesh, y_mesh, u[:,:,16], vmin=-5.1, vmax=20.1,
                    cmap=mpl.colormaps['coolwarm'], shading='gouraud',
                    rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-5, 0, 5, 10, 15, 20])
if save_plots:
    plt.savefig(f"./results/sp_computed_u_fno_0.25_beta_{beta}.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(x_mesh, y_mesh, u[:,:,32], vmin=-5.1, vmax=20.1,
                    cmap=mpl.colormaps['coolwarm'], shading='gouraud',
                    rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-5, 0, 5, 10, 15, 20])
if save_plots:
    plt.savefig(f"./results/sp_computed_u_fno_0.5_beta_{beta}.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(x_mesh, y_mesh, u[:,:,48], vmin=-5.1, vmax=20.1,
                    cmap=mpl.colormaps['coolwarm'], shading='gouraud',
                    rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-5, 0, 5, 10, 15, 20])
if save_plots:
    plt.savefig(f"./results/sp_computed_u_fno_0.75_beta_{beta}.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(x_mesh, y_mesh, s[:,:,16], vmin=-0.1, vmax=0.3,
                    cmap=mpl.colormaps['coolwarm'], shading='gouraud',
                    rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-0.1, 0, 0.1, 0.2, 0.3])
if save_plots:
    plt.savefig(f"./results/sp_computed_y_fno_0.25_beta_{beta}.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(x_mesh, y_mesh, s[:,:,32], vmin=-0.1, vmax=0.3,
                    cmap=mpl.colormaps['coolwarm'], shading='gouraud',
                    rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-0.1, 0, 0.1, 0.2, 0.3])
if save_plots:
    plt.savefig(f"./results/sp_computed_y_fno_0.5_beta_{beta}.pdf")
plt.show()

fig, ax = plt.subplots()
ctf = ax.pcolormesh(x_mesh, y_mesh, s[:,:,48], vmin=-0.1, vmax=0.3,
                    cmap=mpl.colormaps['coolwarm'], shading='gouraud',
                    rasterized=True, linewidth=0)
fig.colorbar(ctf, shrink=0.5, aspect=5, ticks=[-0.1, 0, 0.1, 0.2, 0.3])
if save_plots:
    plt.savefig(f"./results/sp_computed_y_fno_0.75_beta_{beta}.pdf")
plt.show()
