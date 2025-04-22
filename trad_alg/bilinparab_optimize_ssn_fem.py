''' 
This script solves the following bilinear optimal control problem 
    with the FEM-based semismooth Newton methods on the spatial-temporal 
    domain [0, 1]^2 x [0, 1]:
min     J(u, y) = 0.5 * ||y - yd||_2^2 + 0.5 * alpha * ||u||_2^2 + beta * ||u||_1^2
s.t.    partial_t y - Delta y + u * y = f,
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


''' Problem parameters '''

alpha = 0.01
beta = 0.01
u_a = -1.0
u_b = 2.0

def y_func(pos: npt.NDArray, t: float, beta: float) -> npt.NDArray:
    ''' Exact state '''
    x1 = pos[:, 0]
    x2 = pos[:, 1]
    return 5.0 * np.sqrt(beta) * t * np.sin(3.0 * np.pi * x1) * np.sin(np.pi * x2)

def bd_cond_func(pos: npt.NDArray) -> npt.NDArray:
    ''' Dirichlet boundary condition '''
    x1 = pos[:, 0]
    return 0 * x1

def init_cond_func(pos: npt.NDArray) -> npt.NDArray:
    ''' Initial condition '''
    x1 = pos[:, 0]
    return 0 * x1

def p_func(pos: npt.NDArray, t: float, beta: float) -> npt.NDArray:
    ''' Exact adjoint '''
    x1 = pos[:, 0]
    x2 = pos[:, 1]
    return 5.0 * np.sqrt(beta) * (t - 1.0) * np.sin(np.pi * x1) * np.sin(np.pi * x2)

def u_func(pos: npt.NDArray, t: float, 
           alpha: float, beta: float, u_a: float, u_b: float) -> npt.NDArray:
    ''' Exact control '''
    p_exact = p_func(pos, t, beta)
    y_exact = y_func(pos, t, beta)
    u1 = (np.where((-p_exact * y_exact + beta) / alpha > u_a,
                   (-p_exact * y_exact + beta) / alpha, u_a) 
                   * (p_exact * y_exact > beta))
    u2 = (np.where((-p_exact * y_exact - beta) / alpha < u_b,
                   (-p_exact * y_exact - beta) / alpha, u_b) 
                   * (p_exact * y_exact < -beta))
    return u1 + u2

def f_func(pos: npt.NDArray, t: float,
           alpha: float, beta: float, u_a: float, u_b: float) -> npt.NDArray:
    ''' Source function '''
    x1 = pos[:, 0]
    x2 = pos[:, 1]
    y_partial_t = 5.0 * np.sqrt(beta) \
        * np.sin(3.0 * np.pi * x1) * np.sin(np.pi * x2)
    minus_delta_y = -5.0 * np.sqrt(beta) * t * (-10.0 * np.pi**2) \
        * np.sin(3.0 * np.pi * x1) * np.sin(np.pi * x2)
    u_exact = u_func(pos, t, alpha, beta, u_a, u_b)
    y_exact = y_func(pos, t, beta)
    return y_partial_t + minus_delta_y + u_exact * y_exact

def yd_func(pos: npt.NDArray, t: float,
            alpha: float, beta: float, u_a: float, u_b: float) -> npt.NDArray:
    ''' Desired state '''
    x1 = pos[:, 0]
    x2 = pos[:, 1]
    p_partial_t = 5.0 * np.sqrt(beta) * np.sin(np.pi * x1) * np.sin(np.pi * x2)
    minus_delta_p = -5.0 * np.sqrt(beta) * (t - 1.0) * (-2.0 * np.pi**2) \
        * np.sin(np.pi * x1) * np.sin(np.pi * x2)
    y_exact = y_func(pos, t, beta)
    u_exact = u_func(pos, t, alpha, beta, u_a, u_b)
    p_exact = p_func(pos, t, beta)
    return -p_partial_t + minus_delta_p + u_exact * p_exact + y_exact


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
tau = 1.0 / resol_time
tp = np.arange(t_range[0], t_range[1] + tau, tau)
t = np.zeros([resol_time + 1, 1])
t[:, 0] = tp


''' Ground truth optimal control '''

u_exact = np.zeros([num_node, resol_time + 1])
for n in range(resol_time + 1):
    t_now = n * tau
    u_exact[:, n] = u_func(node, t_now, alpha, beta, u_a, u_b)


''' Desired state '''

yd = np.zeros([num_node, resol_time + 1])
for n in range(resol_time + 1):
    t_now = n * tau
    yd[:, n] = yd_func(node, t_now, alpha, beta, u_a, u_b)


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

# Invariant part of the FEM coefficient matrix in each ADMM iteration
matrix_k_invariant = matrix_a + matrix_m / tau 

def update_matrix_k_single(
        time: int,
        u: npt.NDArray, 
        tensor_t: sparse.COO,
        matrix_k_invariant: npt.NDArray):
    ''' 
    Update the variant part of the FEM coefficient matrix 
        in each SSN iteration, for a single time step
    '''
    return sparse.COO.tocsr(matrix_k_invariant + tensor_t.dot(sparse.COO(u[:, time])))    # .dot: # \sum_k T_ijk U_kt

def update_matrix_k_list(
        u: npt.NDArray, 
        tensor_t: sparse.COO,
        matrix_k_invariant: npt.NDArray,
        use_multiprocessing: bool = False,
        ) -> list[scipy.sparse.spmatrix]:
    ''' 
    Update the variant part of the FEM coefficient matrix in each SSN iteration
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
        matrix_k_list = [sparse.COO(coords=[], shape=(num_free_node, num_free_node)) 
                         for _ in range(resol_time + 1)]
        for time in range(resol_time + 1):
            matrix_k_list[time] += update_matrix_k_single(time, u, tensor_t, 
                                                          matrix_k_invariant)
    return matrix_k_list

def inner_prod(u: npt.NDArray, v: npt.NDArray) -> float:
    ''' Compute the inner product of two functions u and v '''

    mu = matrix_m_full.dot(u)
    vmu: npt.NDArray = np.multiply(mu,v)
    return vmu.sum() * tau


''' Operator evaluations '''

'''
The following operators are defined:
    S(u): the solution operator of the state equation
    G(u):= (S'(u))^* (S(u) - yd)
'''

def S_func(
        u: npt.NDArray, 
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None,
        ) -> npt.NDArray:
    ''' Solve the state equation y = S(u) '''

    if matrix_k_list is None:
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )

    y = np.zeros([num_node, resol_time+1])
    y[:, 0] = init_cond_func(node)
     
    for n in range(resol_time):
        t_now = (n + 1) * tau
        y_old = y[:, n:n+1]
        y_now = np.zeros([num_dof, 1])
        y_now[is_bd_node, 0] = bd_cond_func(node[is_bd_node])
        
        # right hand side
        b = np.zeros(num_dof)
        b_tri = np.zeros((num_tri, 3))
        for p in range(num_quad):
            # quadrature points in the x-y coordinate
            pos = phi[p, 0] * node[elem[:, 0]] + phi[p, 1] * node[elem[:, 1]] \
                + phi[p, 2] * node[elem[:, 2]]
            f_pos = f_func(pos, t_now, alpha, beta, u_a, u_b)
            for i in range(3):
                b_tri[:, i] += weight[p] * phi[p, i] * f_pos
        b_tri *= area.reshape(-1,1)
        b = np.bincount(elem.ravel(), weights=b_tri.ravel())
        f_n = b
        rhs_n = np.zeros([num_node, 1])
        rhs_n[free_node, 0] = (matrix_m / tau) * y_old[free_node, 0] + f_n[free_node]

        matrix_k = matrix_k_list[n + 1]

        y_now[free_node, 0] = scipy.sparse.linalg.cg(
            matrix_k,
            rhs_n[free_node]
        )[0]
        y[:, n + 1] = y_now[:, 0]

    return y

def S_grad_func(
        u: npt.NDArray,
        w: npt.NDArray,
        S_u: npt.NDArray|None = None,
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None
        ) -> npt.NDArray:
    ''' Solve S'(u)w '''

    if matrix_k_list is None:
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )
    if S_u is None:
        S_u = S_func(u, matrix_k_list)

    y = np.zeros([num_node, resol_time+1])
    y[:, 0] = init_cond_func(node)
     
    for n in range(resol_time):
        y_old = np.zeros([num_dof, 1])
        y_old[:, 0] = y[:, n]
        y_now = np.zeros([num_dof, 1])
        y_now[is_bd_node, 0] = bd_cond_func(node[is_bd_node])
        
        # right hand side
        w_now = w[:, n+1:n+2]
        rhs_n = np.zeros([num_node, 1])
        rhs_n[free_node, 0] = ((matrix_m / tau) * y_old[free_node,0] 
                               + matrix_m * (-w_now[free_node,0] * S_u[free_node, n + 1]))

        matrix_k = matrix_k_list[n + 1]

        y_now[free_node, 0] = scipy.sparse.linalg.cg(
            matrix_k,
            rhs_n[free_node]
        )[0]
        y[:, n + 1] = y_now[:, 0]

    return y


def S_grad_adj_func(
        u: npt.NDArray,
        w: npt.NDArray,
        S_u: npt.NDArray|None = None,
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None
        ) -> npt.NDArray:
    ''' Solve (S'(u))* w '''

    if matrix_k_list is None:
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )
    if S_u is None:
        S_u = S_func(u, matrix_k_list)

    p = np.zeros([num_node, resol_time + 1])
    p[:, resol_time] = init_cond_func(node)

    for i in range(resol_time):
        n = resol_time - i - 1
        w_now = np.zeros([num_dof, 1])
        w_now[:, 0] = w[:, n]
        p_later = np.zeros([num_dof, 1])
        p_later[:, 0] = p[:, n + 1]
        rhs_n = np.zeros([num_node, 1])
        rhs_n[free_node, 0] = ((matrix_m / tau) * p_later[free_node,0] 
                               + matrix_m * w_now[free_node,0])
        p_now = np.zeros([num_dof, 1])
        p_now[is_bd_node, 0] = bd_cond_func(node[is_bd_node])

        matrix_k = matrix_k_list[n]

        p_now[free_node, 0] = scipy.sparse.linalg.cg(
            matrix_k,
            rhs_n[free_node]
        )[0]
        p[:, n] = p_now[:, 0]
    
    S_grad_adj_u_w = -S_u * p
    return S_grad_adj_u_w


def S_grad_grad_adj_func(
        u: npt.NDArray,
        w1: npt.NDArray,
        w2: npt.NDArray,
        S_grad_u_w1: npt.NDArray|None = None,
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None
        ) -> npt.NDArray:
    ''' Solve (S''(u) w1)* w2 '''

    if matrix_k_list is None:
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )
    if S_grad_u_w1 is None:
        S_grad_u_w1 = S_grad_func(u, w1, matrix_k_list)

    p = np.zeros([num_node, resol_time + 1])
    p[:, resol_time] = init_cond_func(node)

    for i in range(resol_time):
        n = resol_time - i - 1
        w2_now = np.zeros([num_dof, 1])
        w2_now[:, 0] = w2[:, n]
        p_later = np.zeros([num_dof, 1])
        p_later[:, 0] = p[:, n + 1]
        rhs_n = np.zeros([num_node, 1])
        rhs_n[free_node, 0] = ((matrix_m / tau) * p_later[free_node,0] 
                               + matrix_m * w2_now[free_node,0])
        p_now = np.zeros([num_dof, 1])
        p_now[is_bd_node, 0] = bd_cond_func(node[is_bd_node])

        matrix_k = matrix_k_list[n]

        p_now[free_node, 0] = scipy.sparse.linalg.cg(
            matrix_k,
            rhs_n[free_node]
        )[0]
        p[:, n] = p_now[:, 0]
    
    w1_p_prod = w1 * p
    S_grad_adj_u_w1p = S_grad_adj_func(u, w1_p_prod, matrix_k_list=matrix_k_list)
    res = -(S_grad_u_w1 * p + S_grad_adj_u_w1p)
    return res


def G_func(
        u: npt.NDArray,
        yd: npt.NDArray,
        S_u: npt.NDArray|None = None,
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None,
        ) -> npt.NDArray:
    ''' Return y = G(u) := (S'(u))* (S(u) - yd) '''

    if matrix_k_list is None:
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )
    if S_u is None:
        S_u = S_func(u, matrix_k_list)
    
    y = S_grad_adj_func(u, S_u - yd, S_u, matrix_k_list)
    return y


def G_grad_func(
        u: npt.NDArray,
        w: npt.NDArray,
        yd: npt.NDArray,
        S_u : npt.NDArray|None = None,
        S_grad_u_w: npt.NDArray|None = None,
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None
        ) -> npt.NDArray:
    '''
    Return G'(u) w
    '''

    if matrix_k_list is None:
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )
    if S_u is None:
        S_u = S_func(u, matrix_k_list)
    if S_grad_u_w is None:
        S_grad_u_w = S_grad_func(u, w, S_u, matrix_k_list)
    
    term1 = S_grad_adj_func(u, S_grad_u_w, S_u, matrix_k_list)
    term2 = S_grad_grad_adj_func(u, w, S_u - yd, S_grad_u_w, matrix_k_list)
    res = term1 + term2
    return res


def psi_func(
        t: npt.NDArray,
        alpha: float,
        beta: float,
        u_a: float,
        u_b: float
        ) -> npt.NDArray:
    ''' Scalar valued function in the superposition operator '''

    proj_minus_t = (-t).clip(-beta, beta)
    shrk = -(1.0 / alpha) * (t + proj_minus_t)
    psi = shrk.clip(u_a, u_b)
    return psi


def psi_grad_func(
        t: npt.NDArray,
        alpha: float,
        beta: float,
        u_a: float,
        u_b: float
        ) -> npt.NDArray:
    ''' Clarke subdifferential of psi '''

    psi_grad = -(1.0 / alpha) * (t > -beta - alpha * u_b) * (t < -beta) \
        -(1.0 / alpha) * (t > beta) * (t < beta - alpha * u_a)
    return psi_grad


def phi_func(
        u: npt.NDArray,
        yd: npt.NDArray,
        alpha: float,
        beta: float,
        u_a: float,
        u_b: float,
        G_u: npt.NDArray|None = None,
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None,
        ) -> npt.NDArray:
    ''' The nonlinear equation (phi(u) = 0) to be solved by SSN '''

    if G_u is None:
        G_u = G_func(u, yd, matrix_k_list=matrix_k_list)
    psi_u = psi_func(G_u, alpha, beta, u_a, u_b)
    phi = u - psi_u
    return phi


def get_ssn_step(
        u: npt.NDArray,
        yd: npt.NDArray,
        alpha: float,
        beta: float,
        u_a: float,
        u_b: float,
        S_u: npt.NDArray|None = None,
        G_u: npt.NDArray|None = None,
        matrix_k_list: list[scipy.sparse.spmatrix]|None = None,
        tol: float = 1e-5,
        max_cg_iter: int = 6
        ) -> npt.NDArray:
    '''
    Return the step of SSN iteration p by applying CG on solving:
        (\partial\Phi(u)) v = (-\Phi(u))
    Note that \partial\Phi(u) is not self-adjoint.
    However, for our example, it suffices to use the self-adjoint version of CG,
        which produces accurate result and is more efficient.
    '''
    
    if matrix_k_list is None:
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )
    if S_u is None:
        S_u = S_func(u, matrix_k_list)
    if G_u is None:
        G_u = G_func(u, yd, S_u, matrix_k_list)
    
    p = np.zeros([num_node, resol_time + 1])

    phi_u = phi_func(u, yd, alpha, beta, u_a, u_b, G_u, matrix_k_list)
    is_inactive_nodes = ((G_u > -beta - alpha * u_b) * (G_u < -beta) 
                       + (G_u > beta) * (G_u < beta - alpha * u_a))
    is_active_nodes = ~is_inactive_nodes

    p[is_active_nodes] = -phi_u[is_active_nodes]
    p_active = is_active_nodes * p
    S_grad_u_pa = S_grad_func(u, p_active, S_u, matrix_k_list)

    p_inactive = is_inactive_nodes * p
    S_grad_u_pi = S_grad_func(u, p_inactive, S_u, matrix_k_list)
    r = (is_inactive_nodes * (p + (1.0 / alpha) * G_grad_func(
            u, p_inactive, yd, S_u, S_grad_u_pi, matrix_k_list
        )) + is_inactive_nodes * (phi_u + (1.0 / alpha) * G_grad_func(
            u, p_active, yd, S_u, S_grad_u_pa, matrix_k_list
        ))
    )
    r_norm = np.sqrt(inner_prod(r, r)).item()
    q = -r
    iter = 0
    
    while r_norm > tol and iter < max_cg_iter:
        print(f"CG iteration {iter + 1}: {r_norm=}")
        r_prev = r.copy()
        r_prev_norm = r_norm
        S_grad_u_q = S_grad_func(u, is_inactive_nodes * q, S_u, matrix_k_list)
        A_q = is_inactive_nodes * (q + (1.0 / alpha) * G_grad_func(
            u, is_inactive_nodes * q, yd, S_u, S_grad_u_q, matrix_k_list
        ))
        alpha_cg = r_prev_norm**2 / inner_prod(q, A_q)
        p = p + alpha_cg * q
        r = r_prev + alpha_cg * A_q
        r_norm = np.sqrt(inner_prod(r, r)).item()
        beta_cg = r_norm**2 / r_prev_norm**2
        q = -r + beta_cg * q
        iter += 1

    return p
    

''' SSN Iterations '''

if __name__ == "__main__":
    max_iter_ssn = 6
    u_exact_norm = np.sqrt(inner_prod(u_exact, u_exact))    # for printing relative error

    with (multiprocessing.Pool(processes=process_count) if use_multiprocessing 
            else contextlib.nullcontext() as pool):
        start_time = timeit.default_timer()

        u = 0 * u_exact
        matrix_k_list = update_matrix_k_list(
            u, tensor_t, matrix_k_invariant, use_multiprocessing
        )
        S_u = S_func(u, matrix_k_list)
        G_u = G_func(u, yd, S_u, matrix_k_list, )

        for iter_ssn in range(max_iter_ssn):
            print(f"SSN iteration {iter_ssn + 1}")
            ssn_step = get_ssn_step(u, yd, alpha, beta, u_a, u_b, 
                                    S_u=S_u, G_u=G_u, matrix_k_list=matrix_k_list)
            u = u + ssn_step

            S_u = S_func(u, matrix_k_list)
            G_u = G_func(u, yd, S_u, matrix_k_list, )
            matrix_k_list = update_matrix_k_list(
                u, tensor_t, matrix_k_invariant, use_multiprocessing
            )

            error_u = u - u_exact
            error_u_rel = np.sqrt(inner_prod(error_u, error_u)) \
                / u_exact_norm
            phi_val = phi_func(u, yd, alpha, beta, u_a, u_b, G_u, matrix_k_list)
            phi_val_norm = np.sqrt(inner_prod(phi_val, phi_val)).item()
            print(f"relative error: {error_u_rel}")
            print(f"phi norm: {phi_val_norm}")

        end_time = timeit.default_timer()
        print(f"Elapsed time: {end_time - start_time}")
