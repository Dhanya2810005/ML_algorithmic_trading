import numpy as np

def compute_performance(eq):
    eq = np.asarray(eq)
    ret = np.diff(eq) / eq[:-1]
    cagr = (eq[-1]/eq[0])**(1/(len(eq)/252)) - 1
    sharpe = np.mean(ret)*252 / (np.std(ret)*np.sqrt(252)) if np.std(ret)!=0 else 0
    dd = np.min((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)) if len(eq)>0 else 0
    return {'CAGR': cagr, 'Sharpe': sharpe, 'MaxDD': dd}
