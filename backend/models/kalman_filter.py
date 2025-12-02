import numpy as np
from numba import njit

@njit(fastmath=True)
def kalman_filter_numba(z, q=1e-5, r=1e-2):
    n = len(z)
    smooth = np.empty(n)
    x, P = z[0], 1.0
    for t in range(n):
        P_prior = P + q
        K = P_prior / (P_prior + r)
        x = x + K * (z[t] - x)
        P = (1 - K) * P_prior
        smooth[t] = x
    return smooth
