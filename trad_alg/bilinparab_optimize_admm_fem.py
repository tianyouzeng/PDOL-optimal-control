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
import multiprocessing, multiprocessing.pool
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

def g_D(pos: npt.NDArray) -> npt.NDArray:
    ''' Dirichlet boundary condition '''
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
t[:, 0]=tp


''' Ground truth optimal control '''
u_exact = np.zeros([num_node, resol_time+1])
for n in range(resol_time+1):
    t_now = n * tau
    u_exact[:, n] = u_func(node, t_now, alpha, beta, u_a, u_b)

''' Desired state '''
yd = np.zeros([num_node, resol_time+1])
for n in range(resol_time+1):
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
        in each ADMM iteration, for a single time step
    '''
    return matrix_k_invariant + tensor_t.dot(sparse.COO(u[:, time]))       

def update_matrix_k_list(
        u: npt.NDArray, 
        tensor_t: sparse.COO,
        matrix_k_invariant: npt.NDArray,
        use_multiprocessing: bool = False,
        pool: multiprocessing.pool.Pool|None = None
        ) -> list[scipy.sparse.spmatrix]:
    ''' Update the Variant part of the FEM coefficient matrix in each ADMM iteration '''

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


''' Solve the state and adjoint equations '''

def solve_state(
        u: npt.NDArray, 
        matrix_k: sparse.COO,
        y_cache: npt.NDArray|None = None
        ) -> npt.NDArray:
    ''' Solve the state variable y from the state equation '''

    y = np.zeros([num_node, resol_time+1])
    y[:, 0] = y_func(node, 0, beta)   # TODO: change this to init condition
     
    for n in range(resol_time):
        t_now = (n + 1) * tau
        y_old = np.zeros([num_dof, 1])
        y_old[:, 0] = y[:, n]
        y_now = np.zeros([num_dof, 1])
        y_now[is_bd_node, 0] = g_D(node[is_bd_node])
        
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

        matrix_k_now = sparse.COO.tocsc(matrix_k[n + 1])

        # Solve the linear system
        if y_cache is not None:
            y_now[free_node, 0] = scipy.sparse.linalg.cg(
                matrix_k_now,
                rhs_n[free_node],
                y_cache[free_node, n + 1]
            )[0]    # warm start
        else:
            y_now[free_node, 0] = scipy.sparse.linalg.cg(
                matrix_k_now,
                rhs_n[free_node]
            )[0]
        y[:, n + 1] = y_now[:, 0]

    return y

def solve_adjoint(
        u: npt.NDArray,
        w: npt.NDArray,
        matrix_k: sparse.COO,
        p_cache: npt.NDArray|None = None
        ) -> npt.NDArray:
    ''' 
    Solve the adjoint variable z from the (linear) adjoint equation
    p: adjoint variable
    w: right hand side
    '''

    p = np.zeros([num_node, resol_time + 1])
    p[:, resol_time] = 0 * y_func(node, 0, beta)

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
        p_now[is_bd_node, 0] = 0 * g_D(node[is_bd_node])

        matrix_k_now = sparse.COO.tocsc(matrix_k[n])

        if p_cache is not None:
            p_now[free_node, 0] = scipy.sparse.linalg.cg(
                matrix_k_now,
                rhs_n[free_node],
                p_cache[free_node, n]
            )[0]
        else:
            p_now[free_node, 0] = scipy.sparse.linalg.cg(
                matrix_k_now,
                rhs_n[free_node]
            )[0]
        p[:, n] = p_now[:, 0]

    return p


def j_func(
        u: npt.NDArray, 
        u_prox: npt.NDArray, 
        yd: npt.NDArray,
        tensor_t: sparse.COO,
        y_cache: npt.NDArray|None = None,
        p_cache: npt.NDArray|None = None
        ) -> tuple[float, npt.NDArray, npt.NDArray, npt.NDArray]:
    ''' Compute the ADMM-FEM-subproblem objective function and its gradient '''

    matrix_k = update_matrix_k_list(u, tensor_t, matrix_k_invariant)
    y = solve_state(u, matrix_k, y_cache)
    w = y - yd
    p = solve_adjoint(u, w, matrix_k, p_cache)
    j = 0.5 * inner_prod(w, w) + 0.5 * alpha * inner_prod(u, u) \
        + 0.5 * gamma * inner_prod(u - u_prox, u - u_prox)
    der_j = u * alpha - y * p + gamma * (u - u_prox)
    return j, der_j, y, p


''' ADMM iterations '''

if __name__ == "__main__":
    max_iter_admm = 10
    max_iter_bb = 10
    init_gd_stepsize = 50.0
    gamma = 0.01    # 'beta' for ADMM

    u = 0 * u_exact
    z = u
    dual = 0 * u

    with (multiprocessing.Pool(processes=process_count) if use_multiprocessing 
            else contextlib.nullcontext() as pool):
        start_time = timeit.default_timer()

        for iter_admm in range(max_iter_admm):
            u_prox = z + dual / gamma
            print(f"ADMM iteration {iter_admm + 1}")
            u_old = None
            der_j_old = None
            y = p = None
            for iter_grad in range(max_iter_bb):      # gradient steps
                j_val, der_j, y, p= j_func(u, u_prox, yd, tensor_t, y, p)
                grad_norm = inner_prod(der_j, der_j)
                print(f"Gradient step {iter_grad}, J={j_val}, grad_norm={grad_norm}")
                if grad_norm < 1e-10:
                    break
                p_gd = -der_j
                if u_old is None:
                    stepsize = init_gd_stepsize
                    denom = None
                else:
                    u_diff = u - u_old
                    der_j_diff = der_j - der_j_old
                    denom = inner_prod(u_diff, der_j_diff)
                    if denom < 1e-15:
                        break
                    stepsize = inner_prod(u_diff, u_diff) / denom
                u_old = u.copy()
                der_j_old = der_j.copy()
                u = u + stepsize * p_gd
            u_temp = u - dual / gamma
            z_old = z.copy()
            z = (np.where((gamma * u_temp + beta) / (gamma) > u_a,
                        (gamma * u_temp + beta) / (gamma), u_a)
                        * (-gamma * u_temp > beta)
                        + np.where((gamma * u_temp - beta) / (gamma) < u_b,
                                    (gamma * u_temp - beta) / (gamma), u_b)
                                    * (-gamma * u_temp < -beta))
            dual = dual - gamma * (u - z)
            error_primal = z - z_old
            error_dual = u - z
            error_u = u - u_exact
            error_u_rel = np.sqrt(inner_prod(error_u, error_u)) \
                / np.sqrt(inner_prod(u_exact, u_exact))
            print(f"relative error: {error_u_rel.item()}") 

        end_time = timeit.default_timer()
        print(f"Elapsed time: {end_time - start_time}")
    
