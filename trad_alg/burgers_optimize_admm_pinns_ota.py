"""
Created on Mon Jul 25 14:16:32 2022
Modified on 20 Dec 24 17:18:11 2024

@author: yuehr
@modified by: tianyouzeng
"""

import timeit
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

"geom"
class Space1d():
    def __init__(self,a,b):
        self.a=a
        self.b=b

    # def on_boundary(self,xt):
    #     return np.any(np.isclose(xt[:, :-1], [self.a, self.b]), axis=-1)
    # def on_initial(self,xt):
    #     return np.isclose(xt[:, -1:],self.t0).flatten()
    def uniform_points(self,n,boundary=True):
        if boundary:
            x=np.linspace(self.a, self.b, num=n)[:, None]
        else:
            x=np.linspace(self.a, self.b, num=n+1,endpoint=False)[1:, None]
        return x
    def random_points(self,nx):
        xr=np.random.rand(nx)[:, None]
        x=(self.b-self.a)*xr+self.a
        return x
    def uniform_boundary_points(self):
        return np.vstack((self.a, self.b))
N = 100
h = 1/N
domain=Space1d(0,1)
N=100
n=N+1
x=domain.uniform_points(n)
x_torch=torch.from_numpy(x).to(torch.float32).clone().to(device)
    
"model"
zd=0.3
nu=1/12
alpha=0.1
b=0.3
def yd_torch(x_torch):
    return 0*x_torch+zd
def BC_torch(x_torch):
    return 0*x_torch
"Network"
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.yy = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.linear_Tanh_stack_p = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.yy(x)*x*(1-x),self.linear_Tanh_stack_p(x)*x*(1-x)
        # logits = self.linear_Tanh_stack_q(x),self.uu(x),self.vv(x)
        return logits
model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()
"trainning function"
Quiet=True
def train(model, loss_fn, optimizer):
    optimizer.zero_grad()
    loss = data.losses(model, loss_fn)
    loss.backward()
    optimizer.step()
    # loss= loss.item()
    if not Quiet:
        print(f"loss: {loss:>7f} ") 
def train_BFGS( model, loss_fn, optimizer):
    def closure():
        optimizer.zero_grad()
        loss = data.losses(model, loss_fn)
        loss.backward()
        return loss
    optimizer.step(closure)
    loss= closure().item()
    if not Quiet:
        print(f"loss: {loss:>7f} ") 
"""
ADMM
"""
# def pde(x_torch,y_torch,u_torch,u_prox):
#     "primal"
#     x_np=x_torch.cpu().detach().numpy()
#     x_np=np.round(x*(n-1))
#     i_a=np.int32(x_np)
#     u_prox_p=u_prox[i_a,0]
#     u_prox_torch=torch.from_numpy(u_prox_p).to(torch.float32).clone()
    
#     dy=torch.autograd.grad(y_torch, x_torch, grad_outputs=torch.ones_like(y_torch), create_graph=True)[0]
#     ddy=torch.autograd.grad(dy, x_torch, grad_outputs=torch.ones_like(dy), create_graph=True)[0]
#     error_pde=-nu*ddy+dy*y_torch-u_torch
#     error_prox=np.sqrt(beta)*(u_torch-u_prox_torch)+np.sqrt(alpha)*u_torch
#     error_yd=(y_torch-yd_torch(x_torch))
#     return torch.vstack((error_pde,error_yd,error_prox))        
class Burgers():
    def __init__(
        self,
        geometry,
        pde,
        bcs,
        u_prox,
        num_domain,
        para_bd,
    ):
        self.geom = geometry
        self.pde = pde
        self.para_bd=para_bd
        self.BC_torch = bcs
        self.u_prox=u_prox
        self.num_domain=num_domain
        
    def losses(self, model, loss_fn,random=False):
        if random:
            data_point=torch.from_numpy(self.geom.random_points(self.num_domain)).to(device)
        else:
            data_point=torch.from_numpy(self.geom.uniform_points(self.num_domain)).to(device)    
        x=data_point.to(torch.float32).clone().detach().requires_grad_(True)
        y= model(x)[0]
        p= model(x)[1]
        error_domain=self.pde(x,y,p,self.u_prox)
#         x_DBC=torch.from_numpy(self.geom.uniform_boundary_points()).to(torch.float32).clone()
#         error_DBC=model(x_DBC)[0]-self.BC_torch(x_DBC)
#         dual_DBC=model(x_DBC)[0]
#         pred=torch.vstack((error_domain,self.para_bd*error_DBC,self.para_bd*dual_DBC))
        pred=error_domain
        loss = loss_fn(pred, torch.zeros_like(pred))
        return loss
    
def kkt(x_torch,y_torch,p_torch,u_prox):
    x_np=x_torch.cpu().detach().numpy()
    x_np=np.round(x*(n-1))
    i_a=np.int32(x_np)
    u_prox_p=u_prox[i_a,0]
    u_prox_torch=torch.from_numpy(u_prox_p).to(torch.float32).clone().to(device)
    "opt"
    u_torch=(-p_torch+beta*u_prox_torch)/(alpha+beta)
    "primal"
    dy=torch.autograd.grad(y_torch, x_torch, grad_outputs=torch.ones_like(y_torch), create_graph=True)[0]
    ddy=torch.autograd.grad(dy, x_torch, grad_outputs=torch.ones_like(dy), create_graph=True)[0]
    primal=-nu*ddy+dy*y_torch-u_torch
    "dual"
    dp=torch.autograd.grad(p_torch, x_torch, grad_outputs=torch.ones_like(p_torch), create_graph=True)[0]
    ddp=torch.autograd.grad(dp, x_torch, grad_outputs=torch.ones_like(dp), create_graph=True)[0]
    dual=-nu*ddp-dp*y_torch-(y_torch-yd_torch(x_torch))
    return torch.vstack((primal,dual))


beta=0.1

t_start = timeit.default_timer()

u_iter=0*x
z_iter=u_iter
dual_iter=0*x
u_prox=0*x
for i_outer in range(20):
    "k subproblem"
    u_prox=z_iter+dual_iter/beta
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = Burgers(domain,kkt,BC_torch,u_prox,num_domain=n,para_bd=1)
    epochs =5000
    for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
        train(model, loss_fn, optimizer)
    print(f"ADMM iteration {i_outer + 1} finished")
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=1000, max_eval=None, tolerance_grad=1e-09, tolerance_change=1e-09, history_size=100, line_search_fn= 'strong_wolfe' )
    epochs =1
    for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
        train_BFGS(model, loss_fn, optimizer)
#     print("Done!")
    "z subproblem"
    y=model(x_torch)[0]
    y_iter=y.cpu().detach().numpy()
    p=model(x_torch)[1]
    p_iter=p.cpu().detach().numpy()
    u_iter=(-p_iter+beta*u_prox)/(alpha+beta)
    u_temp=u_iter-dual_iter/beta
    z_iter=np.where(u_temp<=b,u_temp,b)
    plt.figure()
    plt.plot(x, u_iter, "-", label="u_iter")
    plt.plot(x, z_iter, "-", label="z_iter")
    plt.legend()
    plt.show()
    "dual update"
    dual_iter=dual_iter-beta*(u_iter-z_iter)

y=model(x_torch)[0]
p=model(x_torch)[1]
y_iter=y.cpu().detach().numpy()
p_iter=p.cpu().detach().numpy()
u_iter=(-p_iter+beta*u_prox)/(alpha+beta)

t_end = timeit.default_timer()
print(f"Elapsed time: {t_end - t_start}")

# plt.figure()
# plt.plot(x, u_iter, "-", label="u")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(x, y_iter, "--", label="y")
# plt.legend()
# plt.show()


ssn_results = np.load(f"./trad_alg/results/computed_u_y_ssn_alpha_{alpha:.1f}_yd_{zd:.1f}_res_{N}.npz")
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

u = torch.tensor(u_iter).flatten()
y = torch.tensor(y_iter).flatten()

err_u_abs = torch.norm(u - u_ssn) / torch.sqrt(torch.tensor(N)).item()
err_u_rel = torch.norm(u - u_ssn) / torch.norm(u_ssn)
print(f"absolute error of control: {err_u_abs.item()}")
print(f"relative error of control: {err_u_rel.item()}")

err_s_abs = torch.norm(y - s_ssn) / torch.sqrt(torch.tensor(N)).item()
err_s_rel = torch.norm(y - s_ssn) / torch.norm(s_ssn)
print(f"absolute error of state: {err_s_abs.item()}")
print(f"relative error of state: {err_s_rel.item()}")
