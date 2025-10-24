
import numpy as np

def crtbp_acc_jac(state, mu):
    x,y,z,vx,vy,vz = state
    mu1 = 1.0 - mu
    r1 = np.array([x + mu, y, z])
    r2 = np.array([x - mu1, y, z])
    d1 = np.linalg.norm(r1); d2 = np.linalg.norm(r2)
    d13 = d1**3; d15 = d1**5
    d23 = d2**3; d25 = d2**5
    Ux = x - mu1*(x+mu)/d13 - mu*(x-mu1)/d23
    Uy = y - mu1*y/d13     - mu*y/d23
    Uz =    - mu1*z/d13     - mu*z/d23
    ax = 2*vy + Ux; ay = -2*vx + Uy; az = Uz
    a = np.array([ax, ay, az])
    Uxx = 1 - mu1*(1/d13 - 3*(r1[0]**2)/d15) - mu*(1/d23 - 3*(r2[0]**2)/d25)
    Uyy = 1 - mu1*(1/d13 - 3*(y**2)/d15)     - mu*(1/d23 - 3*(y**2)/d25)
    Uzz =    - mu1*(1/d13 - 3*(z**2)/d15)    - mu*(1/d23 - 3*(z**2)/d25)
    Uxy =    - mu1*(-3*(r1[0]*y)/d15)        - mu*(-3*(r2[0]*y)/d25)
    Uxz =    - mu1*(-3*(r1[0]*z)/d15)        - mu*(-3*(r2[0]*z)/d25)
    Uyz =    - mu1*(-3*(y*z)/d15)            - mu*(-3*(y*z)/d25)
    A = np.zeros((6,6)); A[0,3]=1; A[1,4]=1; A[2,5]=1
    A[3,0]=Uxx; A[3,1]=Uxy; A[3,2]=Uxz; A[3,4]=2
    A[4,0]=Uxy; A[4,1]=Uyy; A[4,2]=Uyz; A[4,3]=-2
    A[5,0]=Uxz; A[5,1]=Uyz; A[5,2]=Uzz
    return a, A

def srp_acc_and_jac(state, params):
    x,y,z, *_ = state
    c = params.get("c_srp", 0.0); sunx = params.get("sun_x", 10.0)
    r = np.array([x - sunx, y, z]); d = np.linalg.norm(r)
    if d < 1e-9: return np.zeros(3), np.zeros((3,3))
    d3 = d**3; d5 = d**5
    a = c * r / d3
    I = np.eye(3); J = c*(I/d3 - 3*np.outer(r,r)/d5)
    return a, J

def j2_acc_and_jac(state, params, mu_primary, J2, R_ref, primary_pos):
    x,y,z, *_ = state
    rx, ry, rz = primary_pos
    r2 = rx*rx+ry*ry+rz*rz; r = np.sqrt(r2)
    if r < 1e-9 or J2==0.0: return np.zeros(3), np.zeros((3,3))
    mu = mu_primary; R2 = R_ref*R_ref
    k = mu*J2*R2 / (r**5)
    z2 = rz*rz; fac1 = 1 - 5*z2/r2
    a = -k * np.array([fac1*rx, fac1*ry, fac1*rz + 2*rz*r])
    eps=1e-6; base=np.array([rx,ry,rz])
    def acc_from_pos(p):
        px,py,pz=p; rr2=px*px+py*py+pz*pz; rr=np.sqrt(rr2); 
        if rr<1e-9: return np.zeros(3)
        kk=mu*J2*R2/(rr**5); fac=1-5*pz*pz/rr2
        return -kk*np.array([fac*px, fac*py, fac*pz+2*pz*rr])
    J=np.zeros((3,3))
    for i in range(3):
        dp=np.zeros(3); dp[i]=eps
        J[:,i]=(acc_from_pos(base+dp)-acc_from_pos(base-dp))/(2*eps)
    return a, J

def full_dynamics_with_jac(state, mu, extra):
    a_crt, A = crtbp_acc_jac(state, mu)
    asrp, Jsrp = srp_acc_and_jac(state, extra)
    mu1 = 1.0 - mu
    r1 = np.array([state[0] + mu, state[1], state[2]])
    r2 = np.array([state[0] - mu1, state[1], state[2]])
    aJ2_e, JJ2_e = j2_acc_and_jac(state, extra, mu_primary=mu1, J2=extra.get("J2_e",0.0),
                                  R_ref=extra.get("Re_bar", 0.01), primary_pos=r1)
    aJ2_m, JJ2_m = j2_acc_and_jac(state, extra, mu_primary=mu, J2=extra.get("J2_m",0.0),
                                  R_ref=extra.get("Rm_bar", 0.003), primary_pos=r2)
    a = a_crt + asrp + aJ2_e + aJ2_m
    A2 = A.copy(); A2[3:6,0:3] += (Jsrp + JJ2_e + JJ2_m)
    f = np.array([state[3],state[4],state[5], *(a.tolist())])
    return f, A2
