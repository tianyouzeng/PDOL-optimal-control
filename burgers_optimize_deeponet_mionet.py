''' 
This script solves the following optimal control problem of stationary Burgers equation
    with the pretrained DeepONet and MIONet surrogate models on the spatial domain [0, 1]:
min     J(u, y) = 0.5 * ||y - yd||_2^2 + 0.5 * alpha * ||u||_2^2
s.t.    -nv * y'' + y * y' = u
        y(0) = 0, y(1) = 0
        u_a <= u <= u_b
'''

# Imports

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.deeponet import DeepONet, MIONet

from torch import Tensor    # for type hints

torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Load trained models

branch_layer_dim = [101, 101, 101, 101]
trunk_layer_dim = [1, 101, 101, 101]
branch_layers_dim_list = [branch_layer_dim, branch_layer_dim]
trunk_layer_dim = [1, 101, 101, 101]
activation = 'relu'

model_cts = DeepONet(branch_layer_dim, trunk_layer_dim, activation)
model_cts.load_state_dict(torch.load('./trained_models/burgers_model_cts_deeponet_hc_param.pt'))
model_cts.to(device)
model_cts.eval()

model_gradadj = MIONet(2, branch_layers_dim_list, trunk_layer_dim, activation).to(device)
model_gradadj.load_state_dict(torch.load('./trained_models/burgers_model_gradadj_mionet_hc_param.pt'))
model_gradadj.to(device)
model_gradadj.eval()


# Descritization scheme

S = 101
gridpoints = torch.linspace(0, 1, S).to(device)
hc = gridpoints * (1.0 - gridpoints)


# Problem constants and coefficients

u_a = -torch.inf * torch.ones(S).to(device)
u_b = 0.3 * torch.ones(S).to(device)
alpha = 0.1

def yd_func(x: Tensor) -> Tensor:
    return 0.2 * torch.ones_like(x)

yd = yd_func(gridpoints).to(device)


# Primal-dual algorithm parameters

tau = 1.0
sigma = 0.2
omega = 1.0
rho = 1.0
MAX_ITER = 1000
eps = 1e-5

# Proximal operators

def prox_G(u: Tensor, alpha: float, tau: float, u_a: Tensor, u_b: Tensor) -> Tensor:
    inner = u / (alpha * tau + 1.0)
    return torch.maximum(u_a, torch.minimum(u_b, inner))

def prox_F_conj(y: Tensor, yd: Tensor, sigma: float) -> Tensor:
    return (y - sigma * yd) / (1.0 + sigma)


# Initialization

u_prev = torch.zeros(S).to(device)
p_prev = torch.zeros(S).to(device)
u = torch.zeros(S).to(device)
p = torch.zeros(S).to(device)
s = torch.zeros(S).to(device)


# Optimization

with torch.no_grad():
    for iter in range(MAX_ITER):
        print(iter + 1)
        u_prev = u.clone().detach()
        p_prev = p.clone().detach()

        y_hat = hc * model_cts(u_prev[None,:], gridpoints[:,None]).flatten()
        S_gradadj_p_prod = hc * model_gradadj([p_prev[None,:], y_hat[None,:]], gridpoints[:,None]).flatten()
        u_step = tau * S_gradadj_p_prod
        u = prox_G(u_prev - u_step, alpha, tau, u_a, u_b)

        u_extra = u + omega * (u - u_prev)

        cts_u_prod = hc * model_cts(u_extra[None,:], gridpoints[:,None]).flatten()    # match the input requirement of FNO
        p_step = sigma * cts_u_prod
        p = prox_F_conj(p_prev + p_step, yd, sigma)

        u = u_prev + rho * (u - u_prev)
        p = p_prev + rho * (p - p_prev)

        diff_u = torch.norm(u - u_prev).item() / torch.max(torch.tensor([S**(0.5), torch.norm(u_prev).item()])).item()
        diff_p = torch.norm(p - p_prev).item() / torch.max(torch.tensor([S**(0.5), torch.norm(p_prev).item()])).item()
        print(diff_u)
        print(diff_p)
        if iter >= 3 and diff_u <= eps and diff_p <= eps:
            break


# Solve the state variable

s = hc * model_cts(u[None, :], gridpoints[:, None]).flatten()


# Plot results

u = u.detach().cpu()
s = s.detach().cpu()
gridpoints = gridpoints.detach().cpu()

plt.plot(gridpoints, u)
# plt.savefig("./results/burgers_computed_u_alpha_0.1_yd_0.3.pdf")
plt.show()

plt.plot(gridpoints, s)
# plt.savefig("./results/burgers_computed_y_alpha_0.1_yd_0.3.pdf")
plt.show()


# Test results with traditional SSN

ssn_results = np.load("./trad_alg/computed_u_y_ssn_alpha_0.1_yd_0.2.npz")
u_ssn = torch.cat((torch.zeros(1), torch.tensor(ssn_results['u']).flatten(), torch.zeros(1)))
s_ssn = torch.cat((torch.zeros(1), torch.tensor(ssn_results['y']).flatten(), torch.zeros(1)))

plt.plot(gridpoints, u - u_ssn)
plt.plot(gridpoints, s - s_ssn)
plt.show()

err_u_abs = torch.norm(u - u_ssn) / torch.sqrt(torch.tensor(S)).item()
err_u_rel = torch.norm(u - u_ssn) / torch.norm(u_ssn)
print("absolute error of control: ", err_u_abs.item())
print("relative error of control: ", err_u_rel.item())

err_s_abs = torch.norm(s - s_ssn) / torch.sqrt(torch.tensor(S)).item()
err_s_rel = torch.norm(s - s_ssn) / torch.norm(s_ssn)
print("absolute error of state: ", err_s_abs.item())
print("relative error of state: ", err_s_rel.item())

