"""
Created on Sat Jul 23 13:42:40 2022

@author: yhr
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
from scipy.sparse import spdiags

"geom"
x_range = (0, 1)
N = 100
h = 1/N
xp = np.arange(x_range[0], x_range[1]+h, h)
x = np.zeros([N+1, 1])
x[:, 0] = xp
"model"
zd=0.3
nu=1/12
alpha=0.1
b=0.3
"Matrix"
e = np.ones(N-1)
ee=0*x[1:-1]+1
diag_data_A = np.array([-2*e, e, e])
A = spdiags(diag_data_A, [0, 1, -1], N-1, N-1)
# A = A.todense()
diag_data_D = np.array([e, -e])
D = spdiags(diag_data_D, [0,  -1], N-1, N-1)
# D = D.todense()
I=spdiags(ee[:,0], 0, N-1, N-1)
"ini"
u=0*x[1:-1]
y=0*x[1:-1]
p=0*x[1:-1]
lambda_dual=0*x[1:-1]

'SSN'
for i in range(20):
    Dy=D*y
    DTp=D.transpose()*p
    Y=spdiags(y[:,0], 0, N-1, N-1)
    DY=spdiags(Dy[:,0], 0, N-1, N-1)
    DTP=spdiags(DTp[:,0], 0, N-1, N-1)
    inact=np.float64(u+lambda_dual<=b*ee)
    InAct=spdiags(inact[:,0], 0, N-1, N-1)
    act=np.float64(u+lambda_dual>b*ee)
    Act=spdiags(act[:,0], 0, N-1, N-1)
    K11=-nu/h**2*A+Y*D/h+DY/h
    K12=-InAct/alpha
    K21=I+DTP/h
    K22=-nu/h**2*A+Y*D.transpose()/h
    K1=scp.sparse.hstack((K11,K12))
    K2=scp.sparse.hstack((K21,K22))
    K=scp.sparse.vstack((K1,K2))
    K_dense=K.todense()
    righthand1= nu/h**2*A*y-(Y*D*y)/h+Act*ee*b +InAct*p/alpha
    righthand2=zd*ee-y+ nu/h**2*A*p+(Y*D*p)/h
    righthand=np.vstack((righthand1,righthand2))
    big_y=np.linalg.solve(K_dense,righthand)
    delta_y=big_y[:N-1,:]
    delta_p=big_y[N-1:,:]
    y=y+delta_y
    p=p+delta_p
    u=Act*ee*b +InAct*p/alpha
    lambda_dual=p-alpha*u



plt.figure()
plt.plot(x[1:-1], u, "-", label="u")
plt.legend()
plt.show()

plt.figure()
plt.plot(x[1:-1], y, "--", label="y")
plt.legend()
plt.show()

np.savez("trad_alg/computed_u_y_ssn_alpha_0.1_yd_0.3.npz", u=u, y=y)

# x_plot=x[1:-1]
# x_plot.tofile("x_plot_FEM.bin")
# u.tofile("u_FEM.bin")
# y.tofile("y_FEM.bin")


