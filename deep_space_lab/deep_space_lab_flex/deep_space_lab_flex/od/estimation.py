
import numpy as np

def H_range(x): return np.linalg.norm(x[:3])
def H_doppler(x):
    r = np.linalg.norm(x[:3])+1e-9
    return (x[:3]@x[3:])/r
def H_angles(x):
    x1,y1,z1 = x[:3]
    az = np.arctan2(y1, x1)
    el = np.arctan2(z1, np.sqrt(x1*x1 + y1*y1)+1e-9)
    return np.array([az, el])

def simulate_measurements(traj, noise):
    zs=[]
    for s in traj:
        z = [H_range(s) + noise.get("range",0)*np.random.randn(),
             H_doppler(s) + noise.get("doppler",0)*np.random.randn()]
        ang = H_angles(s) + np.array([noise.get("ang",0)*np.random.randn(), noise.get("ang",0)*np.random.randn()])
        zs.append( np.array([z[0], z[1], ang[0], ang[1]]) )
    return np.array(zs)

def jacobian_meas(x):
    xr, yr, zr, vx, vy, vz = x
    r = np.linalg.norm(x[:3]) + 1e-12
    H = np.zeros((4,6))
    H[0,:3] = x[:3]/r
    rr = (x[:3]@x[3:])/r
    H[1,:3] = (x[3:]*r - (x[:3]*rr))/r**2; H[1,3:] = x[:3]/r
    H[2,0] = -yr/(xr*xr + yr*yr + 1e-12); H[2,1] =  xr/(xr*xr + yr*yr + 1e-12)
    rho = np.sqrt(xr*xr + yr*yr) + 1e-12
    H[3,0] = -xr*zr/(rho*r*rho); H[3,1] = -yr*zr/(rho*r*rho); H[3,2] =  rho/(r*rho)
    return H

def batch_ls(traj_f, z, R):
    N=len(traj_f); Hs=[]; rs=[]
    for k in range(N):
        Hk=jacobian_meas(traj_f[k])
        zk=np.array([H_range(traj_f[k]), H_doppler(traj_f[k]), *H_angles(traj_f[k])])
        Hs.append(Hk); rs.append(z[k]-zk)
    H=np.vstack(Hs); r=np.concatenate(rs); W=np.kron(np.eye(N), np.linalg.inv(R))
    A=H.T@W@H; b=H.T@W@r; dx=np.linalg.solve(A+1e-9*np.eye(6), b); P=np.linalg.inv(A+1e-12*np.eye(6))
    return dx, P

def ekf(traj_dyn, z, R, Q):
    x=traj_dyn[0].copy(); P=np.eye(6)*1e-2; est=[x.copy()]; Ps=[P.copy()]
    for k in range(1,len(traj_dyn)):
        x=traj_dyn[k]; P=P+Q
        H=jacobian_meas(x)
        y=z[k]-np.array([H_range(x), H_doppler(x), *H_angles(x)])
        S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S); x=x+K@y; P=(np.eye(6)-K@H)@P
        est.append(x.copy()); Ps.append(P.copy())
    return np.array(est), np.array(Ps)
