import numpy as np
from numba import njit

@njit(fastmath=True)
def bucket_numba(val, low, high):
    if val < low: return 0
    elif val > high: return 2
    else: return 1

@njit(fastmath=True)
def q_learning_episode_numba(close, ma10, ma50, rsi, Q, alpha, gamma, eps):
    n = len(close)
    cash, pos = 10000.0, 0.0
    eq = np.empty(n)
    for t in range(n - 1):
        s0 = bucket_numba(close[t] / ma10[t], 0.98, 1.02)
        s1 = bucket_numba(ma10[t] / ma50[t], 0.98, 1.02)
        s2 = bucket_numba(rsi[t], 30.0, 70.0)
        a = np.random.randint(3) if np.random.rand() < eps else np.argmax(Q[s0, s1, s2])
        if a == 1 and pos == 0: pos, cash = cash / close[t], 0
        elif a == 2 and pos > 0: cash, pos = pos * close[t], 0
        nxt = close[t + 1]
        r = (cash + pos * nxt) - (cash + pos * close[t])
        s0n = bucket_numba(close[t+1]/ma10[t+1], 0.98, 1.02)
        s1n = bucket_numba(ma10[t+1]/ma50[t+1], 0.98, 1.02)
        s2n = bucket_numba(rsi[t+1], 30.0, 70.0)
        td = r + gamma * np.max(Q[s0n, s1n, s2n])
        Q[s0, s1, s2, a] += alpha * (td - Q[s0, s1, s2, a])
        eq[t] = cash + pos * nxt
    eq[-1] = cash + pos * close[-1]
    return Q, eq
