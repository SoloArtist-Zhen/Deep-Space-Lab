
import numpy as np
from deep_space_lab_fast.integrators.propagate_with_stm import propagate_with_stm, monodromy

def single_shoot_periodic(state0, mu, extra, T, max_iter=6):
    s = state0.copy()
    for _ in range(max_iter):
        ts, Y = propagate_with_stm(s, T/2, mu, extra, with_stm=True)
        y_mid = Y[-1,:6]; Phi = Y[-1,6:].reshape((6,6))
        g = np.array([ y_mid[1], y_mid[3] ])  # y=0, vx=0 at half-period
        dgdvy = Phi[[1,3],4]
        dv = - np.linalg.pinv(dgdvy.reshape(-1,1)) @ g.reshape(-1,1)
        s[4] += float(dv)
        if np.linalg.norm(g) < 1e-9: break
    return s, 2*ts[-1]

def multi_shoot_periodic(state0, mu, extra, T, segments=5, max_iter=6):
    dt = T/segments
    y0 = state0.copy(); vys = np.full(segments, y0[4])
    for _ in range(max_iter):
        R_list = []; J = np.zeros((2*segments, segments))
        s = y0.copy()
        for i in range(segments):
            s[4] = vys[i]
            ts, Y = propagate_with_stm(s, dt, mu, extra, with_stm=True)
            end = Y[-1,:6]; Phi = Y[-1,6:].reshape((6,6))
            res = np.array([ end[1]-y0[1], end[3]-y0[3] ])
            R_list.append(res); J[2*i:2*i+2, i] = Phi[[1,3],4]
            s = end
        R = np.concatenate(R_list)
        dv = - np.linalg.lstsq(J, R, rcond=None)[0]
        vys += dv
        if np.linalg.norm(R) < 1e-8: break
    return vys
