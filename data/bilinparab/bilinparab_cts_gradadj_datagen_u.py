import numpy as np
from scipy.fft import idstn
import scipy.io as sio

s = 253  # grid size on Fourier space, this is (m-1)*4+1
m = 64  # grid size for down-sampling
n = 2048  # dataset size
alpha = 3
tau = 3

k1, k2, k3 = np.meshgrid(range(s), range(s), range(s))
coef = np.sqrt(2.0) * (4.0 * np.pi**2 * (k1**2 + k2**2 + k3**2) + tau**2)**(-0.5 * alpha)
u = np.zeros((n, m, m, m))

xf, yf, tf = np.linspace(0, 1, s), np.linspace(0, 1, s), np.linspace(0, 1, s)
xfg, yfg, tfg = np.meshgrid(xf, yf, tf)
x, y, t = np.linspace(0, 1, m), np.linspace(0, 1, m), np.linspace(0, 1, m)
xg, yg, tg = np.meshgrid(x, y, t)
X, Y = np.meshgrid(x, y)

for i in range(n):

    if i % 64 == 63:
        print(i)

    xi = np.random.standard_normal((s, s, s))
    L = 2024 * s**3 * coef * xi
    L[0,0,0] = 0
    uf = idstn(L, norm="backward")

    thre = 3.0
    fac = 3.0
    u_a = -1.0
    u_b = 2.0
    uf[(-thre<uf)&(uf<thre)] = np.zeros(uf[(-thre<uf)&(uf<thre)].shape)
    uf[uf>thre] = (uf[uf>thre] - thre) / (fac + 1)
    uf[uf<-thre] = (uf[uf<-thre] + thre) / (fac + 1)
    uf[uf > u_b] = 0.0 * uf[uf > u_b] + u_b
    uf[uf < u_a] = 0.0 * uf[uf < u_a] + u_a

    u[i, :, :, :] = uf[::4, ::4, ::4]  # down-sampling

mat_fname = "bp_cts_gradadj_train_u_play.mat"
sio.savemat(mat_fname, {"u": u})
