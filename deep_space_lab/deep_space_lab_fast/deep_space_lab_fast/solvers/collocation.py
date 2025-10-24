
import numpy as np

def run_collocation(state0, mu, extra, T, nodes=10, max_iter=8):
    x = np.tile(state0, (nodes,1))
    tgrid = np.linspace(0,T,nodes); h=tgrid[1]-tgrid[0]
    def f(s):
        from deep_space_lab_fast.dynamics.crtbp import full_dynamics_with_jac
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
        x[1:-1:2] -= R.reshape(-1,6)*0.12
        x[0:-2:2] -= R.reshape(-1,6)*0.02
        x[2::2]   += R.reshape(-1,6)*0.02
    return x, np.array(hist)
