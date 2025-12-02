import numpy as np
from numpy.linalg import lstsq

def arima_forecast(series, steps=50):
    diff = np.diff(series)
    Xmat = np.array([diff[i-2:i] for i in range(2, len(diff))])
    yvec = diff[2:]
    params = lstsq(Xmat, yvec, rcond=None)[0]
    mean_val = np.mean(diff)
    preds = []
    for _ in range(steps):
        ar_term = np.dot(params, diff[-2:][::-1])
        pred = mean_val + ar_term
        preds.append(pred)
        diff = np.append(diff, pred)
    return np.r_[series[-1], preds].cumsum()[-steps:]
