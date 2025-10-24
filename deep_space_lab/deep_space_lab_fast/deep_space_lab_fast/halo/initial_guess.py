
import numpy as np

def l1_position(mu):
    x = 0.8
    for _ in range(20):
        r1 = x + mu; r2 = x - (1-mu)
        f = x - (1-mu)*(x+mu)/abs(r1)**3 - mu*(x-(1-mu))/abs(r2)**3
        df = 1 - (1-mu)*(1/abs(r1)**3 - 3*(x+mu)**2/abs(r1)**5) - mu*(1/abs(r2)**3 - 3*(x-(1-mu))**2/abs(r2)**5)
        x -= f/df
    return x

def halo_richardson_like(mu, Az=0.03, around="L1"):
    gamma = abs(l1_position(mu)) if around=="L1" else abs(1-l1_position(mu))
    nu = np.sqrt( ((mu)/(gamma**3)) + ((1-mu)/((1-gamma)**3)) + 1 )
    x0 = (1 - mu) - gamma + 1e-3 if around=="L1" else (1 - mu) + gamma - 1e-3
    return np.array([x0,0.0,Az, 0.0, 0.1*Az, 0.0]), 2*np.pi/nu
