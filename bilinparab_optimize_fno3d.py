''' 
This script solves the following semilinear parabolic optimal control problem 
    with the pretrained FNO surrogate models on the spatial-temporal domain [0, 1]^2 x [0, 1]:
min     J(u, y) = 0.5 * ||y - yd||_2^2 + 0.5 * alpha * ||u||_2^2 + beta * ||u||_1^2
s.t.    \partial_t y - \Delta y + u * y = f,
        y = 0 on the boundary,
        y(0) = 0 when T=0,
        u_a <= u <= u_b.
'''

# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from models.fno import FNO3d_M
from utils.bilinparab_prob_param import yd_func, f_func

from torch import Tensor    # for type hints

torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Load trained models

MODEL_PATH = './trained_models/bp_model_cts_gradadj_fno3d_param.pt'

input_num = 2
modes = 8
width = 16
paddings = 8

model = FNO3d_M(input_num, modes, modes, modes, width, paddings).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Descritization scheme

S = 64  # spacial discretization resolution
T = 64  # temporal discretization resolution
xlist, ylist, tlist = np.linspace(0, 1, S), np.linspace(0, 1, S), np.linspace(0, 1, T)
xmesh, ymesh = np.meshgrid(xlist, ylist)
x = np.repeat(xmesh[:,:,None], tlist.shape[0], axis=2)
y = np.repeat(ymesh[:,:,None], tlist.shape[0], axis=2)
t = np.repeat(np.repeat(tlist[None,None,:], xmesh.shape[0], axis=0), xmesh.shape[1], axis=1)


# Problem constants and coefficients

T_MAX = 1.0
u_a = -1.0
u_b = 2.0
alpha = 0.01
beta = 0.01

yd = torch.tensor(yd_func(alpha, beta, u_a, u_b, x, y, t)).float()
f = torch.tensor(f_func(alpha, beta, u_a, u_b, x, y, t)).float()


# Primal-dual algorithm parameters

tau = 200.0
sigma = 0.4
omega = 1.0
r = 1.618
s = 1.618   # control relaxation stepsize, set to 1.0 for no-relaxation
MAX_ITER = 100


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
u: Tensor = torch.zeros((S, S, T))
p: Tensor = torch.zeros((S, S, T))


# Optimization
with torch.no_grad():
    for iter in range(MAX_ITER):
        print(iter + 1)
        u_prev = u.detach().clone()
        p_prev = p.detach().clone()

        uf_prev = torch.cat((u_prev[None,:,:,:,None], 0.25 * f[None,:,:,:,None]), axis=-1)
        y_hat = 4.0 * model(uf_prev.to(device)).view(S, S, T).detach().cpu()
        adjeq_sol = model(torch.cat((u_prev.flip((-1))[None,:,:,:,None].to(device), 
                          p_prev.flip((-1))[None,:,:,:,None].to(device)), axis=-1)).view(S, S, T).flip((-1)).detach().cpu()
        S_adj_p_prod = -y_hat * adjeq_sol
        u_step = tau * S_adj_p_prod
        u = prox_G(u_prev - u_step, alpha, beta, tau, u_a, u_b)

        u_extra = u + omega * (u - u_prev)

        cts_u_prod = 4.0 * model(torch.cat((u_extra[None,:,:,:,None].to(device), 0.25 * f[None,:,:,:,None].to(device)), axis=-1)).view(S, S, T).detach().cpu()    # match the input requirement of FNO
        p_step = sigma * cts_u_prod
        p = prox_F_conj(p_prev + p_step, yd, sigma)

        u = u_prev + r * (u - u_prev)
        p = p_prev + s * (p - p_prev)

        diff_u = torch.norm(u - u_prev).item() / torch.max(torch.tensor([(S * S * T)**(0.5), torch.norm(u_prev).item()])).item()
        diff_p = torch.norm(p - p_prev).item() / torch.max(torch.tensor([(S * S * T)**(0.5), torch.norm(p_prev).item()])).item()
        print(diff_u)
        print(diff_p)
        if iter >= 3 and diff_u <= 1e-5 and diff_p <= 1e-5:
            break


# Solve the state variable

s = 4.0 * model(torch.cat((u[None,:,:,:,None].to(device), 0.25 * f[None,:,:,:,None].to(device)), axis=-1)).view(S, S, T).detach().cpu()


# Test the result

from utils.bilinparab_prob_param import u_func, p_func, s_func

u_exact = u_func(alpha, beta, u_a, u_b, x, y, t)
p_exact = p_func(beta, x, y, t)
s_exact = s_func(beta, x, y, t)

u = u.detach().cpu().numpy()
p = p.detach().cpu().numpy()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 6.4))
surf = ax.plot_surface(xmesh, ymesh, u[:,:,16], cmap=cm.coolwarm, linewidth=0.3, edgecolors='black', rasterized=True)
# plt.savefig("./results/bp_computed_u_fno_0.25.pdf")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 6.4))
surf = ax.plot_surface(xmesh, ymesh, u_exact[:,:,16], cmap=cm.coolwarm, linewidth=0.3, edgecolors='black', rasterized=True)
# plt.savefig("./results/bp_exact_u_fno_0.25.pdf")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 6.4))
surf = ax.plot_surface(xmesh, ymesh, u[:,:,32], cmap=cm.coolwarm, linewidth=0.3, edgecolors='black', rasterized=True)
# plt.savefig("./results/bp_computed_u_fno_0.5.pdf")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 6.4))
surf = ax.plot_surface(xmesh, ymesh, u_exact[:,:,32], cmap=cm.coolwarm, linewidth=0.3, edgecolors='black', rasterized=True)
# plt.savefig("./results/bp_exact_u_fno_0.5.pdf")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 6.4))
surf = ax.plot_surface(xmesh, ymesh, u[:,:,48], cmap=cm.coolwarm, linewidth=0.3, edgecolors='black', rasterized=True)
# plt.savefig("./results/bp_computed_u_fno_0.75.pdf")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 6.4))
surf = ax.plot_surface(xmesh, ymesh, u_exact[:,:,48], cmap=cm.coolwarm, linewidth=0.3, edgecolors='black', rasterized=True)
# plt.savefig("./results/bp_exact_u_fno_0.75.pdf")
plt.show()

err_u_abs = np.linalg.norm(u - u_exact) / np.sqrt(S * S * T)
err_u_rel = np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)
print("absolute error of control: ", err_u_abs)
print("relative error of control: ", err_u_rel)

err_s_abs = np.linalg.norm(s - s_exact) / np.sqrt(S * S * T)
err_s_rel = np.linalg.norm(s - s_exact) / np.linalg.norm(s_exact)
print("absolute error of state: ", err_s_abs)
print("relative error of state: ", err_s_rel)

