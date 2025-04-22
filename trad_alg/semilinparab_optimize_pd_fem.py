''' 
This script solves the following semilinear parabolic optimal control problem 
    with the FEM-based primal-dual method on the spatial-temporal 
    domain [0, 1]^2 x [0, 1]:
min     J(u, y) = 0.5 * ||y - yd||_2^2 + 0.5 * alpha * ||u||_2^2 + beta * ||u||_1^2
s.t.    partial_t y - Delta y + y * (y - 0.25) * (y + 1) = u,
        y = 0 on the boundary,
        y(0) = 0 when T=0,
        u_a <= u <= u_b.
'''


''' Import '''


import numpy as np
import scipy.sparse, scipy.sparse.linalg
import sparse
import numpy.typing as npt

import itertools
import timeit
import multiprocessing
import functools
import contextlib

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir) 
from utils.fem_mesh import rectangleMesh, quadpts, TriMesh2D


''' Multiprocessing configs '''

use_multiprocessing = True
process_count = 64
pool = multiprocessing.Pool(processes=process_count)


''' Problem parameters '''

alpha = 0.0001
beta = 0.004
u_a = -10.0
u_b = 20.0

def yd_func(pos: npt.NDArray, t: float) -> npt.NDArray:
    ''' Desired state '''
    x1 = pos[:, 0]
    x2 = pos[:, 1]
    yd = (np.exp(-20.0 * ((x1 - 0.2)**2 + (x2 - 0.2)**2 + (t - 0.2)**2))
        + np.exp(-20.0 * ((x1 - 0.7)**2 + (x2 - 0.7)**2 + (t - 0.9)**2)))
    return yd

def boundary_cond_func(pos: npt.NDArray) -> npt.NDArray:
    ''' Dirichlet boundary condition '''
    x1 = pos[:, 0]
    return 0 * x1

def init_cond_func(pos: npt.NDArray) -> npt.NDArray:
    ''' Initial condition '''
    x1 = pos[:, 0]
    return 0 * x1

def R_func(y: npt.NDArray) -> npt.NDArray:
    return y * (y - 0.25) * (y + 1.0)

def R_grad_func(y: npt.NDArray) -> npt.NDArray:
    ''' Frechet derivative of R(y) '''
    # Note that R'(u) acts on any w as a pointwise product
    # For convenience, we return the R'(u) as the function multiplied to w
    return (3.0 * y**2 + 1.5 * y - 0.25)


''' Space geometry '''

x_range = (0.0, 1.0)
y_range = (0.0, 1.0)
resol_space = 64
h = 1 / resol_space
node, elem = rectangleMesh(x_range, y_range, h)
tri_mesh = TriMesh2D(node,elem)
tri_mesh.update_auxstructure()
tri_mesh.update_gradbasis()
is_bd_node = tri_mesh.isBdNode
der_phi = tri_mesh.Dlambda
area: npt.NDArray = tri_mesh.area
phi, weight = quadpts()
num_quad = len(phi)
num_node = num_dof = len(node)
num_tri = len(elem)
free_node = ~is_bd_node
free_node_idx_list = free_node * np.cumsum(free_node) - 1
num_free_node = np.sum(free_node)


''' Time geometry '''

t_range = (0, 1)
resol_time = 64
delta_t = 1.0 / resol_time
tp = np.arange(t_range[0], t_range[1] + delta_t, delta_t)
t = np.zeros([resol_time + 1, 1])
t[:, 0]=tp


''' Desired state '''

yd = np.zeros([num_node, resol_time + 1])
for n in range(resol_time + 1):
    t_now = n * delta_t
    yd[:, n] = yd_func(node, t_now)


''' FEM matrices '''

# For most of the cases, only store the free_node submatrix is necessary for computation

# Stiff matrix
matrix_a = scipy.sparse.csr_matrix((num_free_node, num_free_node))
for i in range(3):
    for j_val in range(3):
        matrix_a_ij = area*(der_phi[..., i] * der_phi[..., j_val]).sum(axis=-1)
        coords_i_temp = free_node_idx_list[elem[:, i]]
        coords_j_temp = free_node_idx_list[elem[:, j_val]]
        ij_is_free_node = (coords_i_temp != -1) & (coords_j_temp != -1)
        coords_i = coords_i_temp[ij_is_free_node]
        coords_j = coords_j_temp[ij_is_free_node]
        matrix_a += scipy.sparse.csr_matrix(
            (matrix_a_ij[ij_is_free_node], (coords_i, coords_j)),
            shape=(num_free_node, num_free_node)
        )  

# Mass matrix for the right hand side
matrix_m = scipy.sparse.csr_matrix((num_free_node, num_free_node))
for i in range(3):
    for j_val in range(3):
        matrix_m_ij = (1.0 + (i == j_val)) / 12.0 * area
        coords_i_temp = free_node_idx_list[elem[:, i]]
        coords_j_temp = free_node_idx_list[elem[:, j_val]]
        ij_is_free_node = (coords_i_temp != -1) & (coords_j_temp != -1)
        coords_i = coords_i_temp[ij_is_free_node]
        coords_j = coords_j_temp[ij_is_free_node]
        matrix_m += scipy.sparse.csr_matrix(
            (matrix_m_ij[ij_is_free_node], (coords_i, coords_j)),
            shape=(num_free_node, num_free_node)
        )

# Mass matrix (full), for computing inner product
matrix_m_full = scipy.sparse.csr_matrix((num_node, num_node))
for i in range(3):
    for j_val in range(3):
        matrix_m_ij = (1.0 + (i == j_val)) / 12.0 * area
        matrix_m_full += scipy.sparse.csr_matrix(
            (matrix_m_ij, (elem[:,i], elem[:,j_val])),
            shape=(num_dof, num_dof)
        )

# "Mass tensor" for term (u * y)
tensor_t = sparse.COO(coords=[], shape=(num_free_node, num_free_node, num_node))
for i, j_val, k in itertools.product(range(3), repeat=3):
    if i == j_val == k:
        tensor_t_ijk = 6.0 * area / 60.0
    elif i == j_val or j_val == k or k == i:
        tensor_t_ijk = 2.0 * area / 60.0
    else:
        tensor_t_ijk = 1.0 * area / 60.0
    coords_i_temp = free_node_idx_list[elem[:, i]]
    coords_j_temp = free_node_idx_list[elem[:, j_val]]
    ij_is_free_node = (coords_i_temp != -1) & (coords_j_temp != -1)
    coords_i = coords_i_temp[ij_is_free_node]
    coords_j = coords_j_temp[ij_is_free_node]
    coords_k = elem[:, k][ij_is_free_node]
    tensor_t += sparse.COO(
        coords=(coords_i, coords_j, coords_k), 
        data=tensor_t_ijk[ij_is_free_node], 
        shape=(num_free_node, num_free_node, num_dof)
    )

# Invariant part of the FEM coefficient matrix in each PD iteration
matrix_k_invariant = matrix_a + matrix_m / delta_t 

def update_matrix_k_single(
        time: int,
        u: npt.NDArray, 
        tensor_t: sparse.COO,
        matrix_k_invariant: npt.NDArray) -> scipy.sparse.csr_matrix:
    ''' 
    Update the variant part of the FEM coefficient matrix 
        in each PD iteration, for a single time step
    '''
    return sparse.COO.tocsr(matrix_k_invariant + tensor_t.dot(sparse.COO(u[:, time])))    # .dot: # \sum_k T_ijk U_kt

def update_matrix_k_list(
        u: npt.NDArray, 
        ) -> list[scipy.sparse.csr_matrix]:
    ''' 
    Update the variant part of the FEM coefficient matrix in each PD iteration
    '''

    '''
    For updating this FEM matrix, two strategies can be used:
        1. Compute a ndof x ndof x resol_time sparse array. In each time step
            for solving the PDEs, use the corresponding slice of the sparse array.
        2. Compute a list of ndof x ndof sparse matrices. In each time step
            for solving the PDEs, use the corresponding matrix in the list.
    The first strategy requires to slice the sparse array in each time step;
        the second approach cannot utilize the efficient tensor-matrix multilication
        (instead, we have to apply a for loop)
    Here, we use the second approach and alleviate the computational costs of the
        for loop by multiprocessing.
    '''

    if use_multiprocessing:
        func = functools.partial(update_matrix_k_single, u=u, tensor_t=tensor_t, 
                                 matrix_k_invariant=matrix_k_invariant)
        assert pool is not None
        matrix_k_list = pool.map(func, range(resol_time + 1))
    else:
        matrix_k_list = [scipy.sparse.csr_matrix((num_free_node, num_free_node)) 
                         for _ in range(resol_time + 1)]
        for time in range(resol_time + 1):
            matrix_k_list[time] += update_matrix_k_single(time, u, tensor_t, matrix_k_invariant)
    return matrix_k_list

def inner_prod(u: npt.NDArray, v: npt.NDArray) -> float:
    ''' Compute the inner product of two functions u and v '''

    mu = matrix_m_full.dot(u)
    vmu: npt.NDArray = np.multiply(mu,v)
    return vmu.sum() * delta_t


''' Operator evaluations '''

'''
The following operators are defined:
    S(u): the solution operator of the state equation
    G(u):= (S'(u))^* (S(u) - yd)
'''


def S_func(u: npt.NDArray, max_iter: int=20) -> npt.NDArray:
    y = np.zeros([num_node, resol_time+1])
    y[:, 0] = init_cond_func(node)
    iter = 0

    while (iter < max_iter 
           and (iter == 0 or np.sqrt(inner_prod(y - y_prev, y - y_prev)) > 1e-5)):
        y_prev = y.copy()
        R_y_prev = R_func(y_prev)

        for n in range(resol_time):
            y_old = y[:, n:n+1]
            y_now = np.zeros([num_dof, 1])
            y_now[is_bd_node, 0] = boundary_cond_func(node[is_bd_node])

            # right hand side
            u_now = u[:, n+1:n+2]
            R_y_prev_now = R_y_prev[:, n+1:n+2]
            rhs_n = np.zeros([num_node, 1])
            rhs_n[free_node, 0] = ((matrix_m / delta_t) @ (y_old[free_node, 0])
                + matrix_m @ u_now[free_node, 0] - matrix_m @ R_y_prev_now[free_node, 0])
            
            # solve FEM linear system
            y_now[free_node, 0] = scipy.sparse.linalg.cg(
                matrix_k_invariant,
                rhs_n[free_node]
            )[0]
            y[:, n + 1] = y_now[:, 0]
        iter += 1

    return y


def S_grad_adj_func(
        u: npt.NDArray,
        w: npt.NDArray,
        S_u: npt.NDArray|None = None,
        ) -> npt.NDArray:
    
    if S_u is None:
        S_u = S_func(u)
    
    R_grad_Su = R_grad_func(S_u)
    matrix_k_list = update_matrix_k_list(R_grad_Su)

    p = np.zeros([num_node, resol_time + 1])
    p[:, resol_time] = init_cond_func(node)

    for i in range(resol_time):
        n = resol_time - i - 1
        w_now = np.zeros([num_dof, 1])
        w_now[:, 0] = w[:, n]
        p_later = np.zeros([num_dof, 1])
        p_later[:, 0] = p[:, n + 1]
        rhs_n = np.zeros([num_node, 1])
        rhs_n[free_node, 0] = ((matrix_m / delta_t) @ p_later[free_node,0] 
                               + matrix_m @ w_now[free_node,0])
        p_now = np.zeros([num_dof, 1])
        p_now[is_bd_node, 0] = boundary_cond_func(node[is_bd_node])

        matrix_k = matrix_k_list[n]

        p_now[free_node, 0] = scipy.sparse.linalg.cg(
            matrix_k,
            rhs_n[free_node]
        )[0]
        p[:, n] = p_now[:, 0]
    
    return p


''' Proximal operators '''


def prox_G(u: npt.NDArray, alpha: float, beta: float, 
           tau: float, u_a: float, u_b: float) -> npt.NDArray:
    prod = tau * beta
    val1 = (u - prod) / (alpha * tau + 1)
    val2 = (u + prod) / (alpha * tau + 1)
    proxval = np.zeros_like(u)
    proxval[u > prod] = val1[u > prod]
    proxval[u < -prod] = val2[u < -prod]
    proxval = np.maximum(
        u_a * np.ones_like(proxval),
        np.minimum(u_b * np.ones_like(proxval), proxval)
    )
    return proxval


def prox_F_conj(y: npt.NDArray, yd: npt.NDArray, sigma: float) -> npt.NDArray:
    proxval = (y - sigma * yd) / (1.0 + sigma)
    return proxval
    

''' PD Iterations '''

if __name__ == "__main__":
    max_iter_pd = 20
    tau = 500.0
    sigma = 0.4
    omega = 1.0
    rho_u = rho_p = 1.0     # control relaxation stepsize, set to 1.0 for no-relaxation

    u = np.zeros((num_node, resol_time + 1))
    p = np.zeros((num_node, resol_time + 1))

    with multiprocessing.Pool(processes=process_count) if use_multiprocessing \
            else contextlib.nullcontext() as pool:
        start_time = timeit.default_timer()

        for iter_pd in range(max_iter_pd):
            print(f"PD iteration {iter_pd + 1}")
            u_prev = u.copy()
            p_prev = p.copy()

            S_u = S_func(u_prev)
            S_grad_adj_u_p = S_grad_adj_func(u_prev, p_prev, S_u)
            u_step = tau * S_grad_adj_u_p
            u = prox_G(u_prev - u_step, alpha, beta, tau, u_a, u_b)
            u_extra = u + omega * (u - u_prev)
            S_uextra = S_func(u_extra)
            p_step = sigma * S_uextra
            p = prox_F_conj(p_prev + p_step, yd, sigma)

            # Relaxation steps, optional
            u = u_prev + rho_u * (u - u_prev)
            p = p_prev + rho_p * (p - p_prev)
            
            diff_u = (np.sqrt(inner_prod(u - u_prev, u - u_prev)) 
                      / np.max([1.0, np.sqrt(inner_prod(u_prev, u_prev))])).item()
            diff_p = (np.sqrt(inner_prod(p - p_prev, p - p_prev)) 
                      / np.max([1.0, np.sqrt(inner_prod(p_prev, p_prev))])).item()
            print(f"{diff_u=}")
            print(f"{diff_p=}")
            if iter_pd >= 3 and diff_u <= 1e-5 and diff_p <= 1e-5:
                break

        y = S_func(u)

        end_time = timeit.default_timer()
        print(f"Elapsed time: {end_time - start_time}")
