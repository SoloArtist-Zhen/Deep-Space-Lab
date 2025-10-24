
import numpy as np
from deep_space_lab.integrators.propagate_with_stm import propagate_with_stm

def multiple_shoot_periodic(state0, mu, extra, T, segments=8, max_iter=8):
    dt=T/segments; xk=[state0.copy() for _ in range(segments)]; hist=[]
    for _ in range(max_iter):
        F=[]; G=[]
        for i in range(segments):
            s=xk[i]; ts,Y=propagate_with_stm(s, dt, mu, extra, with_stm=True)
            end=Y[-1,:6]; Phi=Y[-1,6:].reshape((6,6)); nxt=xk[(i+1)%segments]
            res=end-nxt; F.append(res); G.append((i, Phi-np.eye(6)))
        R=np.concatenate(F); hist.append(np.linalg.norm(R))
        if hist[-1]<1e-9: break
        for i,(idx,Jc) in enumerate(G):
            dx = - np.linalg.lstsq(Jc, F[i], rcond=None)[0]
            xk[idx] += dx
    return xk, np.array(hist)
