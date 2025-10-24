
import numpy as np
from deep_space_lab.integrators.propagate_with_stm import propagate_with_stm, monodromy

def single_shoot_periodic(state0, mu, extra, T, max_iter=8):
    s = state0.copy()
    for _ in range(max_iter):
        ts, Y = propagate_with_stm(s, T/2, mu, extra, with_stm=True)
        y_mid = Y[-1,:6]; Phi = Y[-1,6:].reshape((6,6))
        g = np.array([ y_mid[1], y_mid[3] ])  # y=0, vx=0 at half-period
        dgdvy = Phi[[1,3],4]
        dv = - np.linalg.pinv(dgdvy.reshape(-1,1)) @ g.reshape(-1,1)
        s[4] += float(dv)
        if np.linalg.norm(g) < 1e-10: break
    return s, 2*ts[-1]

def multi_shoot_periodic(state0, mu, extra, T, segments=6, max_iter=8):
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
        if np.linalg.norm(R) < 1e-9: break
    return vys

def collocation_hs(state0, mu, extra, T, nodes=16, max_iter=10):
    x = np.tile(state0, (nodes,1))
    tgrid = np.linspace(0,T,nodes); h=tgrid[1]-tgrid[0]
    def f(s):
        from deep_space_lab.dynamics.crtbp import full_dynamics_with_jac
        f,_ = full_dynamics_with_jac(s, mu, extra); return f
    hist=[]
    for _ in range(max_iter):
        R=[]
        for k in range(0, nodes-2, 2):
            s0=x[k]; s2=x[k+2]; s1=0.5*(s0+s2)
            f0=f(s0); f2=f(s2); f1=f(s1)
            defect = s0 - s2 + (h/6.0)*(f0 + 4*f1 + f2)
            R.append(defect)
        R=np.concatenate(R); nr=np.linalg.norm(R); hist.append(nr)
        if nr<1e-8: break
        x[1:-1:2] -= R.reshape(-1,6)*0.1
        x[0:-2:2] -= R.reshape(-1,6)*0.02
        x[2::2]   += R.reshape(-1,6)*0.02
    return x, np.array(hist)
