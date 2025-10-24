
import numpy as np
from deep_space_lab_flex.dynamics.crtbp import full_dynamics_with_jac

def rkf45_step(f, t, y, h):
    c2=1/5; c3=3/10; c4=4/5; c5=8/9
    a21=1/5
    a31=3/40; a32=9/40
    a41=44/45; a42=-56/15; a43=32/9
    a51=19372/6561; a52=-25360/2187; a53=64448/6561; a54=-212/729
    a61=9017/3168; a62=-355/33; a63=46732/5247; a64=49/176; a65=-5103/18656
    b1=35/384; b2=0; b3=500/1113; b4=125/192; b5=-2187/6784; b6=11/84
    e1=b1-5179/57600; e2=b2-0; e3=b3-7571/16695; e4=b4-393/640; e5=b5-92097/339200; e6=b6-187/2100
    k1 = f(t, y)
    k2 = f(t + c2*h, y + h*(a21*k1))
    k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
    k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
    k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
    k6 = f(t + h,     y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
    y5 = y + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
    err = h*np.linalg.norm(e1*k1 + e2*k2 + e3*k3 + e4*k4 + e5*k5 + e6*k6, ord=np.inf)
    return y5, err

def propagate_with_stm(x0, T, mu=0.0121505856, extra=None, h0=2e-3, atol=1e-9, rtol=1e-8, with_stm=True):
    if extra is None: extra = {}
    def dyn(t, y):
        state = y[:6]
        f, A = full_dynamics_with_jac(state, mu, extra)
        if with_stm:
            Phi = y[6:].reshape((6,6))
            dPhi = A @ Phi
            return np.concatenate([f, dPhi.reshape(-1)])
        else:
            return f
    y = np.zeros(6 + (36 if with_stm else 0)); y[:6] = x0.copy()
    if with_stm: y[6:] = np.eye(6).reshape(-1)
    t = 0.0; h = h0; traj=[y.copy()]; ts=[0.0]
    while t < T - 1e-12:
        if t + h > T: h = T - t
        y_new, err = rkf45_step(lambda tt,yy: dyn(tt,yy), t, y, h)
        tol = atol + rtol*np.linalg.norm(y_new, ord=np.inf)
        if err < tol or h<1e-7:
            t += h; y = y_new; traj.append(y.copy()); ts.append(t)
        s = 2.0 if err==0 else 0.9*(tol/err)**0.25; s = min(2.0, max(0.2, s)); h *= s
    return np.array(ts), np.array(traj)

def monodromy(x0, T, mu, extra):
    ts, Y = propagate_with_stm(x0, T, mu, extra, h0=2e-3, with_stm=True)
    return Y[-1,6:].reshape((6,6))
