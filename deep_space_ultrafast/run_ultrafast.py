
import os, sys, time, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------ Dynamics (2D CRTBP + light SRP/J2) ------------------
def crtbp_f(s, mu):
    x,y,vx,vy = s
    mu1 = 1.0 - mu
    r1 = np.array([x + mu, y]); r2 = np.array([x - mu1, y])
    d1 = (r1[0]**2 + r1[1]**2)**0.5; d2 = (r2[0]**2 + r2[1]**2)**0.5
    Ux = x - mu1*(x+mu)/d1**3 - mu*(x-mu1)/d2**3
    Uy = y - mu1*y/d1**3     - mu*y/d2**3
    ax = 2*vy + Ux; ay = -2*vx + Uy
    return np.array([vx,vy,ax,ay])

def add_srp_j2(s, mu, c_srp=1.2e-6, sunx=10.0, J2e=8e-4, Re=0.01):
    # Very light surrogates for speed in 2D: SRP central repulsion from sun, J2-ish radial correction term near Earth
    x,y,vx,vy = s
    # SRP
    rs = np.array([x - sunx, y]); ds = (rs[0]**2 + rs[1]**2)**0.5 + 1e-9
    a_srp = c_srp * rs / ds**3
    # J2-like (2D surrogate): small radial term around Earth (-mu at x=-mu)
    rE = np.array([x + mu, y]); dE = (rE[0]**2 + rE[1]**2)**0.5 + 1e-9
    a_j2 = -J2e * Re**2 * rE / dE**5
    return np.array([0,0, a_srp[0]+a_j2[0], a_srp[1]+a_j2[1]])

def rk4_step(f, s, h, *args):
    k1 = f(s, *args); k2 = f(s + 0.5*h*k1, *args)
    k3 = f(s + 0.5*h*k2, *args); k4 = f(s + h*k3, *args)
    return s + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ------------------ Fig 1: orbit point cloud ------------------
def fig_orbit_scatter():
    mu = 0.0121505856
    s = np.array([0.82, 0.06, 0.0, -0.23])
    h = 0.01; N = 500
    P = np.zeros((N,4))
    for k in range(N):
        P[k] = s
        s = rk4_step(crtbp_f, s, h, mu)
    t = np.linspace(0,1,N)
    plt.figure(figsize=(5.0,4.2))
    plt.scatter(P[:,0], P[:,1], c=t, s=6, cmap="viridis")
    plt.scatter([-mu, 1-mu], [0,0], s=25, c=["k","k"], marker="x")
    plt.axis("equal"); plt.grid(True, alpha=0.3)
    plt.title("CRTBP orbit point cloud (colored by time)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"crtbp_orbit_scatter.png"), dpi=150); plt.close()

# ------------------ Fig 2: SRP+J2 overlay scatter ------------------
def fig_srp_j2_overlay():
    mu = 0.0121505856
    s0 = np.array([0.85, 0.04, 0.0, -0.21])
    h = 0.01; N = 400
    P0 = np.zeros((N,4)); Pp = np.zeros((N,4))
    s = s0.copy(); sp = s0.copy()
    for k in range(N):
        P0[k] = s; Pp[k] = sp
        s  = rk4_step(crtbp_f, s,  h, mu)
        sp = rk4_step(lambda ss,mu: crtbp_f(ss,mu) + add_srp_j2(ss,mu), sp, h, mu)
    plt.figure(figsize=(5.0,4.2))
    plt.scatter(P0[:,0], P0[:,1], s=5, alpha=0.7, label="CRTBP")
    plt.scatter(Pp[:,0], Pp[:,1], s=5, alpha=0.7, label="CRTBP + SRP + J2")
    plt.legend(loc="best", fontsize=8)
    plt.axis("equal"); plt.grid(True, alpha=0.3)
    plt.title("Overlay: SRP & J2 effect (scatter)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"srp_j2_overlay_scatter.png"), dpi=150); plt.close()

# ------------------ Fig 3: mini NSGA-II (3D scatter) ------------------
def mini_nsga2(objective, bounds, pop=8, gen=8, pc=0.9, pm=0.2):
    D = len(bounds)
    X = np.array([[np.random.uniform(lo,hi) for (lo,hi) in bounds] for _ in range(pop)])
    F = np.array([objective(xi) for xi in X])
    def dom(a,b): return (np.all(a<=b) and np.any(a<b))
    def fronts_of(F):
        S=[[] for _ in range(len(F))]; n=[0]*len(F); fronts=[[]]
        for p in range(len(F)):
            for q in range(len(F)):
                if dom(F[p],F[q]): S[p].append(q)
                elif dom(F[q],F[p]): n[p]+=1
            if n[p]==0: fronts[0].append(p)
        i=0; ranks=[None]*len(F)
        while fronts[i]:
            Q=[]
            for p in fronts[i]:
                ranks[p]=i
                for q in S[p]:
                    n[q]-=1
                    if n[q]==0: Q.append(q)
            i+=1; fronts.append(Q)
        return fronts[:-1], ranks
    for g in range(gen):
        fronts, ranks = fronts_of(F)
        def pick():
            i,j = np.random.randint(len(X)), np.random.randint(len(X))
            return i if ranks[i] < ranks[j] else j
        Y=[]
        for _ in range(len(X)//2):
            p1=X[pick()]; p2=X[pick()]
            c1,c2=p1.copy(),p2.copy()
            if np.random.rand()<pc:
                a=np.random.rand(D); c1=a*p1+(1-a)*p2; c2=a*p2+(1-a)*p1
            for d in range(D):
                if np.random.rand()<pm:
                    lo,hi=bounds[d]; c1[d]=np.clip(c1[d]+0.2*(hi-lo)*np.random.randn(), lo,hi)
                if np.random.rand()<pm:
                    lo,hi=bounds[d]; c2[d]=np.clip(c2[d]+0.2*(hi-lo)*np.random.randn(), lo,hi)
            Y += [c1,c2]
        Y=np.array(Y); FY=np.array([objective(y) for y in Y])
        Z=np.vstack([X,Y]); G=np.vstack([F,FY])
        fronts,_=fronts_of(G); X_new=[]; F_new=[]
        for fr in fronts:
            for i in fr:
                if len(X_new) < len(X):
                    X_new.append(Z[i]); F_new.append(G[i])
        X=np.array(X_new); F=np.array(F_new)
    fronts,_=fronts_of(F)
    return X[fronts[0]], F[fronts[0]]

def fig_pareto():
    mu = 0.0121505856
    def dyn(s):
        x,y,vx,vy = s
        mu1=1-mu
        r1=np.array([x+mu,y]); r2=np.array([x-mu1,y])
        d1=(r1[0]**2+r1[1]**2)**0.5; d2=(r2[0]**2+r2[1]**2)**0.5
        Ux = x - mu1*(x+mu)/d1**3 - mu*(x-mu1)/d2**3
        Uy = y - mu1*y/d1**3 - mu*y/d2**3
        return np.array([vx,vy, 2*vy+Ux, -2*vx+Uy])
    def prop(s,T,dt=0.02):
        N=max(1,int(T/dt))
        for _ in range(N):
            k1=dyn(s); k2=dyn(s+0.5*dt*k1); k3=dyn(s+0.5*dt*k2); k4=dyn(s+dt*k3)
            s = s + (dt/6)*(k1+2*k2+2*k3+k4)
        return s
    s0=np.array([0.88,0.0,0.0,-0.19]); target=np.array([0.80,0.10,0.0,0.0])
    def objective(y):
        t1=y[0]; dv1=y[1:3]; t2=y[3]; dv2=y[4:6]
        sA=prop(s0,t1); sA[2:]+=dv1
        sB=prop(sA,t2); sB[2:]+=dv2
        sf=prop(sB,0.2)
        dV=np.linalg.norm(dv1)+np.linalg.norm(dv2)
        tof=t1+t2+0.2
        miss=np.linalg.norm(sf[:2]-target[:2])
        return np.array([dV,tof,miss])
    bounds=[(0.1,0.6),(-0.05,0.05),(-0.05,0.05),(0.1,0.6),(-0.05,0.05),(-0.05,0.05)]
    X,F = mini_nsga2(objective, bounds, pop=8, gen=8)
    # 3D scatter
    fig=plt.figure(figsize=(5.2,4.4))
    ax=fig.add_subplot(111, projection="3d")
    ax.scatter(F[:,0], F[:,1], F[:,2], s=28)
    ax.set_xlabel("ΔV"); ax.set_ylabel("TOF"); ax.set_zlabel("Miss")
    ax.set_title("Mini Pareto Front (3 objectives)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"pareto_front_3d.png"), dpi=150); plt.close(fig)
    return {"pareto_min": [float(np.min(F[:,0])), float(np.min(F[:,1])), float(np.min(F[:,2]))]}

# ------------------ Fig 4: Algorithm comparison (shooting vs collocation) ------------------
def toy_bvp_compare():
    # Problem: drive from s(0)=sA to s(T)=sB under dyn, adjust control knobs.
    mu = 0.0121505856
    def dyn(s):
        x,y,vx,vy = s
        mu1=1-mu
        r1=np.array([x+mu,y]); r2=np.array([x-mu1,y])
        d1=(r1[0]**2+r1[1]**2)**0.5; d2=(r2[0]**2+r2[1]**2)**0.5
        Ux = x - mu1*(x+mu)/d1**3 - mu*(x-mu1)/d2**3
        Uy = y - mu1*y/d1**3 - mu*y/d2**3
        return np.array([vx,vy, 2*vy+Ux, -2*vx+Uy])
    def prop(s,T,dt=0.02):
        N=max(1,int(T/dt))
        for _ in range(N):
            k1=dyn(s); k2=dyn(s+0.5*dt*k1); k3=dyn(s+0.5*dt*k2); k4=dyn(s+dt*k3)
            s = s + (dt/6)*(k1+2*k2+2*k3+k4)
        return s

    sA=np.array([0.86,0.04,0.0,-0.20]); sB=np.array([0.80,0.08,0.0,0.0]); T=0.6

    # --- Single shooting: adjust initial velocity (2 iters) ---
    t0=time.perf_counter()
    v0 = sA[2:].copy()
    for _ in range(2):
        sA_try = sA.copy()
        sA_try[2:] = v0
        sf = prop(sA_try, T)
        res = sf[:2]-sB[:2]   # match terminal position only
        # finite-diff Jacobian wrt v0
        J = np.zeros((2,2)); eps=1e-3
        for i in range(2):
            dv = np.zeros(2); dv[i]=eps
            sA_p = sA_try.copy(); sA_p[2:] = v0 + dv
            sA_m = sA_try.copy(); sA_m[2:] = v0 - dv
            sfp = prop(sA_p, T)[:2]; sfm = prop(sA_m, T)[:2]
            J[:,i] = (sfp - sfm)/(2*eps)
        dv = - np.linalg.pinv(J) @ res
        v0 = v0 + dv
    shoot_time = time.perf_counter() - t0
    shoot_res = float(np.linalg.norm(res))

    # --- Collocation (Hermite–Simpson, 3 nodes): adjust middle state (1 iter) ---
    t0=time.perf_counter()
    X0 = sA.copy(); X2 = sB.copy()
    X1 = 0.5*(X0+X2)  # mid guess
    h = T/2
    # defects on state using HS
    f0 = dyn(X0); f1 = dyn(X1); f2 = dyn(X2)
    defect = X0 - X2 + (h/3.0)*(f0 + 4*f1 + f2)
    # one Gauss–Newton-like correction on X1 (position components only to keep tiny)
    J = np.zeros((4,2)); eps=1e-3
    for i in range(2):  # perturb x,y at midpoint
        d = np.zeros(2); d[i]=eps
        X1p = X1.copy(); X1p[0]+=d[0]; X1p[1]+=d[1]
        fp = dyn(X1p)
        defect_p = X0 - X2 + (h/3.0)*(f0 + 4*fp + f2)
        J[:,i] = (defect_p - defect)/eps
    # target: zero defect on x,y,vx,vy
    rhs = -defect
    dx = np.linalg.lstsq(J, rhs, rcond=None)[0]
    X1[0]+=dx[0]; X1[1]+=dx[1]
    # updated defect
    f1 = dyn(X1); defect2 = X0 - X2 + (h/3.0)*(f0 + 4*f1 + f2)
    coll_time = time.perf_counter() - t0
    coll_res = float(np.linalg.norm(defect2))

    # scatter: residual vs time
    plt.figure(figsize=(5.0,4.2))
    plt.scatter([shoot_res], [shoot_time], marker="o", s=120, label="Single Shooting")
    plt.scatter([coll_res],  [coll_time],  marker="^", s=120, label="Collocation (HS, 1-iter)")
    plt.xlabel("Residual norm"); plt.ylabel("Runtime (s)")
    plt.title("Algorithm comparison (toy BVP)")
    plt.legend(loc="best"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"alg_compare_scatter.png"), dpi=150); plt.close()
    return {"shoot_res":shoot_res, "shoot_time":shoot_time, "coll_res":coll_res, "coll_time":coll_time}

def main():
    t0 = time.time()
    fig_orbit_scatter()
    fig_srp_j2_overlay()
    pareto_summary = fig_pareto()
    algo_summary = toy_bvp_compare()
    # save summary
    summary = {"pareto": pareto_summary, "algo_compare": algo_summary, "elapsed_s": float(time.time()-t0)}
    with open(os.path.join(OUT,"summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[Done] Figures saved to ./outputs")

if __name__=="__main__":
    main()
