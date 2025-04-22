# Auxiliary functions for generating problem parameters for semilinparab_optimize.py

import numpy as np
import numpy.typing as npt

def yd_func(x: npt.NDArray, y: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
    return np.exp(-20.0 * ((x - 0.2)**2 + (y - 0.2)**2 + (t - 0.2)**2)) \
        + np.exp(-20.0 * ((x - 0.7)**2 + (y - 0.7)**2 + (t - 0.9)**2))