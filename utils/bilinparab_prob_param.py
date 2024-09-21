# Auxiliary functions for generating problem parameters for bilinparab_optimize.py

import numpy as np

def s_func(beta, x, y, t) -> np.ndarray:
    s_val = 5.0 * np.sqrt(beta) * t * np.sin(3.0*np.pi*x) * np.sin(np.pi*y)
    return s_val

def s_t_func(beta, x, y, t) -> np.ndarray:
    s_t_val = 5.0 * np.sqrt(beta) * np.sin(3.0*np.pi*x) * np.sin(np.pi*y)
    return s_t_val

def s_laplace_func(beta, x, y, t) -> np.ndarray:
    s_laplace_val = -50.0 * np.pi**2 * np.sqrt(beta) * t * np.sin(3.0*np.pi*x) * np.sin(np.pi*y)
    return s_laplace_val

def p_func(beta, x, y, t)  -> np.ndarray:
    p_val = 5.0 * np.sqrt(beta) * (t-1.0) * np.sin(np.pi*x) * np.sin(np.pi*y)
    return p_val

def p_t_func(beta, x, y, t) -> np.ndarray:
    p_t_val = 5.0 * np.sqrt(beta) * np.sin(np.pi*x) * np.sin(np.pi*y)
    return p_t_val

def p_laplace_func(beta, x, y, t) -> np.ndarray:
    p_laplace_val = -10.0 * np.pi**2 * np.sqrt(beta) * (t-1.0) * np.sin(np.pi*x) * np.sin(np.pi*y)
    return p_laplace_val

def u_func(alpha, beta, u_a, u_b, x, y, t) -> np.ndarray:
    mult_val = p_func(beta, x, y, t) * s_func(beta, x, y, t)
    func_val_1 = (-mult_val + beta) / alpha
    func_val_2 = (-mult_val - beta) / alpha
    u_val = np.zeros(mult_val.shape)
    u_val[mult_val > beta] = np.maximum(func_val_1[mult_val > beta], u_a)
    u_val[mult_val < -beta] = np.minimum(func_val_2[mult_val < -beta], u_b)
    return u_val

def yd_func(alpha: float, beta: float, u_a: float, u_b: float, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    yd = -p_t_func(beta, x, y, t) - p_laplace_func(beta, x, y, t) + s_func(beta, x, y, t) + u_func(alpha, beta, u_a, u_b, x, y, t) * p_func(beta, x, y, t)
    return yd

def f_func(alpha: float, beta: float, u_a: float, u_b: float, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    f = s_t_func(beta, x, y, t) - s_laplace_func(beta, x, y, t) + u_func(alpha, beta, u_a, u_b, x, y, t) * s_func(beta, x, y, t)
    return f
