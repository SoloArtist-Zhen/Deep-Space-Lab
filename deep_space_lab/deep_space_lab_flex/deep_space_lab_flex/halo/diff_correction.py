
import numpy as np
from deep_space_lab_flex.integrators.propagate_with_stm import propagate_with_stm, monodromy

def single_shoot_periodic(state0, mu, extra, T, max_iter=6):
    s = state0.copy()
    for _ in range(max_iter):
        ts, Y = propagate_with_stm(s, T/2, mu, extra, with_stm=True)
        y_mid = Y[-1,:6]; Phi = Y[-1,6:].reshape((6,6))
        g = np.array([ y_mid[1], y_mid[3] ])  # y=0, vx=0
        dgdvy = Phi[[1,3],4]
        dv = - np.linalg.pinv(dgdvy.reshape(-1,1)) @ g.reshape(-1,1)
        s[4] += float(dv)
        if np.linalg.norm(g) < 1e-9: break
    return s, 2*ts[-1]
