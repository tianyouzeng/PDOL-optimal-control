''' 
This script solves the following optimal control problem of stationary Burgers equation
    with the pretrained DeepONet and MIONet surrogate models on the spatial domain [0, 1]:
min     J(u, y) = 0.5 * ||y - yd||_2^2 + 0.5 * alpha * ||u||_2^2
s.t.    -nv * y'' + y * y' = u
        y(0) = 0, y(1) = 0
        u_a <= u <= u_b
'''

# Imports

import timeit

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.deeponet import DeepONet, MIONet

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Primal-dual algorithm parameters

tau = 1.0
sigma = 0.2
omega = 1.0
rho = 1.0
MAX_ITER = 1000
eps = 1e-5

verbose = True          # print the iteration information if True
record_time = True      # print the computation time if True
save_results = False    # save the computed optimal control and state if True
save_plots = False      # save plotted results if True


# Descritization scheme

S_IN = 100      # input discretization resolution, which follows from the training settings
S_OUT = 2000    # target output discretization resolution
S_IN, S_OUT = S_IN + 1, S_OUT + 1
assert((S_OUT - 1) % (S_IN - 1) == 0)
INTERP_POINTS = (S_OUT - 1) // (S_IN - 1)
gridpoints = torch.linspace(0, 1, S_OUT).to(device)
hc = gridpoints * (1.0 - gridpoints)


# Load trained models

input_num_gradadj = 2
branch_layer_dim_cts = [101, 101, 101, 101]
trunk_layer_dim_cts = [1, 101, 101, 101]
branch_layers_dim_list_gradadj = [branch_layer_dim_cts, branch_layer_dim_cts]
trunk_layer_dim_gradadj = [1, 101, 101, 101]
activation = 'relu'

model_cts = DeepONet(branch_layer_dim_cts, trunk_layer_dim_cts, activation)
model_cts.load_state_dict(
    torch.load('./trained_models/burgers_model_cts_deeponet_hc_param.pt'))
model_cts.to(device)
model_cts.eval()

model_gradadj = MIONet(input_num_gradadj, branch_layers_dim_list_gradadj, 
                       trunk_layer_dim_gradadj, activation).to(device)
model_gradadj.load_state_dict(
    torch.load('./trained_models/burgers_model_gradadj_mionet_hc_param.pt'))
model_gradadj.to(device)
model_gradadj.eval()


# Problem constants and coefficients

u_a = -torch.inf * torch.ones(S_OUT).to(device)
u_b = 0.3 * torch.ones(S_OUT).to(device)
alpha = 0.1

def yd_func(x: torch.Tensor) -> torch.Tensor:
    return 0.3 * torch.ones_like(x)

yd = yd_func(gridpoints).to(device)


# Proximal operators

def prox_G(u: torch.Tensor, alpha: float, tau: float, 
           u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
    inner = u / (alpha * tau + 1.0)
    return torch.maximum(u_a, torch.minimum(u_b, inner))

def prox_F_conj(y: torch.Tensor, yd: torch.Tensor, sigma: float) -> torch.Tensor:
    return (y - sigma * yd) / (1.0 + sigma)


# Initialization

time_start = 0
if record_time:
    time_start = timeit.default_timer()

u_prev = torch.zeros(S_OUT).to(device)
p_prev = torch.zeros(S_OUT).to(device)
u = torch.zeros(S_OUT).to(device)
p = torch.zeros(S_OUT).to(device)
s = torch.zeros(S_OUT).to(device)


# Optimization

with torch.no_grad():
    for iter in range(MAX_ITER):
        if verbose:
            print(f"iter: {iter + 1}")
        u_prev = u.clone().detach()
        p_prev = p.clone().detach()

        y_hat = hc * model_cts(
            u_prev[None, ::INTERP_POINTS],
            gridpoints[:, None]
        ).flatten()
        S_gradadj_p_prod = hc * model_gradadj([
            p_prev[None, ::INTERP_POINTS], 
            y_hat[None, ::INTERP_POINTS]], 
            gridpoints[:, None]
        ).flatten()
        u_step = tau * S_gradadj_p_prod
        u = prox_G(u_prev - u_step, alpha, tau, u_a, u_b)

        u_extra = u + omega * (u - u_prev)

        cts_u_prod = hc * model_cts(
            u_extra[None, ::INTERP_POINTS],
            gridpoints[:, None]
        ).flatten()
        p_step = sigma * cts_u_prod
        p = prox_F_conj(p_prev + p_step, yd, sigma)

        u = u_prev + rho * (u - u_prev)
        p = p_prev + rho * (p - p_prev)

        diff_u = torch.norm(u - u_prev).item() / torch.max(torch.tensor([
            S_IN**(0.5),
            torch.norm(u_prev).item()
        ])).item()
        diff_p = torch.norm(p - p_prev).item() / torch.max(torch.tensor([
            S_IN**(0.5),
            torch.norm(p_prev).item()
        ])).item()
        if verbose:
            print(f"{diff_u=}")
            print(f"{diff_p=}")
        if iter >= 3 and diff_u <= eps and diff_p <= eps:
            break


# Solve the state variable

s = hc * model_cts(u[None, ::INTERP_POINTS], gridpoints[:, None]).flatten()

if record_time:
    time_end = timeit.default_timer()
    print("computation time: ", time_end - time_start)


# Plot results

u = u.detach().cpu()
p = p.detach().cpu()
s = s.detach().cpu()
gridpoints = gridpoints.detach().cpu()

if save_results:
    np.savez(f"./results/burgers_computed_deeponet_mionet_alpha_{alpha:.1f}_yd_{yd[0]:.1f}_res_{S_OUT-1}.npz", u, s)

plt.figure()
plt.plot(gridpoints, u)
if save_plots:
    plt.savefig(f"./results/burgers_computed_u_alpha_{alpha:.1f}_yd_{yd[0]:.1f}_res_{S_OUT-1}.pdf")
plt.show()

plt.figure()
plt.plot(gridpoints, s)
if save_plots:
    plt.savefig(f"./results/burgers_computed_y_alpha_{alpha:.1f}_yd_{yd[0]:.1f}_res_{S_OUT-1}.pdf")
plt.show()


# Test results with traditional SSN

# You may need to run tral_alg/ssn_fem/burgers_optimize_ssn.py 
#   to get the results for comparison

ssn_results = np.load(f"./trad_alg/results/computed_u_y_ssn_alpha_{alpha:.1f}_yd_{yd[0]:.1f}_res_{S_OUT-1}.npz")
u_ssn = torch.cat((
    torch.zeros(1),
    torch.tensor(ssn_results['u']).flatten(),
    torch.zeros(1)
))
s_ssn = torch.cat((
    torch.zeros(1),
    torch.tensor(ssn_results['y']).flatten(),
    torch.zeros(1)
))

plt.plot(gridpoints, u - u_ssn)
plt.plot(gridpoints, s - s_ssn)
plt.show()

err_u_abs = torch.norm(u - u_ssn) / torch.sqrt(torch.tensor(S_OUT)).item()
err_u_rel = torch.norm(u - u_ssn) / torch.norm(u_ssn)
print(f"absolute error of control: {err_u_abs.item()}")
print(f"relative error of control: {err_u_rel.item()}")

err_s_abs = torch.norm(s - s_ssn) / torch.sqrt(torch.tensor(S_OUT)).item()
err_s_rel = torch.norm(s - s_ssn) / torch.norm(s_ssn)
print(f"absolute error of state: {err_s_abs.item()}")
print(f"relative error of state: {err_s_rel.item()}")
