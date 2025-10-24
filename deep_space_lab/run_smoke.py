
import os, sys, time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

OUT = "smoke_outputs"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(msg); sys.stdout.flush()

# -------------------- Stage 1: Tiny CRTBP propagation (2D quick) --------------------
def crtbp_acc(state, mu):
    x,y,vx,vy = state
    mu1 = 1.0 - mu
    r1 = np.array([x + mu, y])
    r2 = np.array([x - mu1, y])
    d1 = np.hypot(r1[0], r1[1]); d2 = np.hypot(r2[0], r2[1])
    Ux = x - mu1*(x+mu)/d1**3 - mu*(x-mu1)/d2**3
    Uy = y - mu1*y/d1**3     - mu*y/d2**3
    ax = 2*vy + Ux
    ay = -2*vx + Uy
    return np.array([vx, vy, ax, ay])

def rk4_step(f, s, h, *args):
    k1 = f(s, *args)
    k2 = f(s + 0.5*h*k1, *args)
    k3 = f(s + 0.5*h*k2, *args)
    k4 = f(s + h*k3, *args)
    return s + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def stage1_crtbp():
    log("[Stage1] CRTBP tiny propagation…")
    mu = 0.0121505856
    s = np.array([0.8, 0.05, 0.0, -0.25])  # x,y,vx,vy seed
    h = 0.01; N = 600
    traj = np.zeros((N,4))
    for k in range(N):
        traj[k] = s
        s = rk4_step(crtbp_acc, s, h, mu)
        if (k+1) % 200 == 0: log(f"  progress: {k+1}/{N}")
    plt.figure(figsize=(5,4))
    plt.plot(traj[:,0], traj[:,1], lw=1.2)
    plt.plot([-mu, 1-mu], [0,0], 'ko', ms=4)  # primaries (rough)
    plt.axis('equal'); plt.grid(True); plt.title("CRTBP (tiny) trajectory")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "crtbp_tiny_xy.png"), dpi=140); plt.close()
    log("  saved: smoke_outputs/crtbp_tiny_xy.png")

# -------------------- Stage 2: Mini NSGA-II (6 pop × 6 gen) --------------------
def mini_nsga2(objective, bounds, pop=6, gen=6, pc=0.9, pm=0.2):
    D = len(bounds)
    X = np.array([[np.random.uniform(lo,hi) for (lo,hi) in bounds] for _ in range(pop)])
    F = np.array([objective(xi) for xi in X])
    def dominates(a,b): return np.all(a<=b) and np.any(a<b)
    def sort_front(F):
        S=[[] for _ in range(len(F))]; n=[0]*len(F); fronts=[[]]
        for p in range(len(F)):
            for q in range(len(F)):
                if dominates(F[p],F[q]): S[p].append(q)
                elif dominates(F[q],F[p]): n[p]+=1
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
    hv_hist=[]
    for g in range(gen):
        fronts, ranks = sort_front(F)
        def tour():
            i,j=np.random.randint(pop), np.random.randint(pop)
            return i if ranks[i] < ranks[j] else j
        Y=[]
        for _ in range(pop//2):
            p1=X[tour()]; p2=X[tour()]
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
        fronts,_=sort_front(G)
        X_new=[]; F_new=[]
        for fr in fronts:
            for i in fr:
                if len(X_new) < pop:
                    X_new.append(Z[i]); F_new.append(G[i])
        X=np.array(X_new); F=np.array(F_new)
        hv_hist.append(np.mean(np.sum(F,axis=1)))  # cheap proxy
    fronts,_=sort_front(F)
    return X[fronts[0]], F[fronts[0]], np.array(hv_hist)

def simple_objective():
    # 2-impulse min [ΔV, TOF, miss]
    mu = 0.0121505856
    def dyn2(s):
        x,y,vx,vy = s
        mu1=1-mu
        r1=np.array([x+mu,y]); r2=np.array([x-mu1,y])
        d1=np.hypot(*r1); d2=np.hypot(*r2)
        Ux = x - mu1*(x+mu)/d1**3 - mu*(x-mu1)/d2**3
        Uy = y - mu1*y/d1**3 - mu*y/d2**3
        return np.array([vx,vy, 2*vy+Ux, -2*vx+Uy])
    s0=np.array([0.9,0.0,0.0,-0.18]); target=np.array([0.8,0.1,0,0])
    def prop(s, T, dt=0.02):
        N=max(1,int(T/dt))
        for _ in range(N):
            k1=dyn2(s); k2=dyn2(s+0.5*dt*k1); k3=dyn2(s+0.5*dt*k2); k4=dyn2(s+dt*k3)
            s=s+(dt/6)*(k1+2*k2+2*k3+k4)
        return s
    def obj(y):
        t1=y[0]; dv1=y[1:3]; t2=y[3]; dv2=y[4:6]
        sA=prop(s0,t1); sA[2:]+=dv1
        sB=prop(sA,t2); sB[2:]+=dv2
        sf=prop(sB,0.2)
        dV=np.linalg.norm(dv1)+np.linalg.norm(dv2)
        tof=t1+t2+0.2
        miss=np.linalg.norm(sf[:2]-target[:2])
        return np.array([dV,tof,miss])
    bounds=[(0.1,0.6), (-0.05,0.05),(-0.05,0.05), (0.1,0.6), (-0.05,0.05),(-0.05,0.05)]
    return obj, bounds

def stage2_pareto():
    log("[Stage2] Mini NSGA-II…")
    obj,bounds = simple_objective()
    X,F,hv = mini_nsga2(obj, bounds, pop=6, gen=6)
    # plots
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(F[:,0],F[:,1],F[:,2], s=25)
    ax.set_xlabel("ΔV"); ax.set_ylabel("TOF"); ax.set_zlabel("Miss")
    ax.set_title("Mini Pareto (smoke)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"pareto_smoke.png"), dpi=140); plt.close(fig)
    plt.figure(figsize=(5,3)); plt.plot(hv); plt.xlabel("Gen"); plt.ylabel("Proxy HV"); plt.title("Proxy Hypervolume"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"hv_proxy.png"), dpi=140); plt.close()
    log("  saved: smoke_outputs/pareto_smoke.png, hv_proxy.png")

# -------------------- Stage 3: Mini OD residuals --------------------
def stage3_od():
    log("[Stage3] Orbit Determination (tiny)…")
    def H_range(x): return np.linalg.norm(x[:2])
    def H_ang(x): return math.atan2(x[1], x[0])
    mu=0.0121505856; s=np.array([0.85,0.05,0.0,-0.2]); h=0.015; N=200
    traj=[]; z=[]
    def fwrap(ss): 
        x,y,vx,vy=ss
        mu1=1-mu
        r1=np.array([x+mu,y]); r2=np.array([x-mu1,y])
        d1=np.hypot(*r1); d2=np.hypot(*r2)
        Ux = x - mu1*(x+mu)/d1**3 - mu*(x-mu1)/d2**3
        Uy = y - mu1*y/d1**3 - mu*y/d2**3
        ax = 2*vy + Ux; ay = -2*vx + Uy
        return np.array([vx,vy,ax,ay])
    for k in range(N):
        traj.append(s.copy())
        rng=H_range(s)+1e-3*np.random.randn(); ang=H_ang(s)+np.deg2rad(0.03)*np.random.randn()
        z.append([rng,ang])
        # RK4 step
        k1=fwrap(s); k2=fwrap(s+0.5*h*k1); k3=fwrap(s+0.5*h*k2); k4=fwrap(s+h*k3); s=s+(h/6)*(k1+2*k2+2*k3+k4)
    traj=np.array(traj); z=np.array(z)
    pred=np.column_stack([np.linalg.norm(traj[:,:2],axis=1), np.arctan2(traj[:,1], traj[:,0])])
    res=z-pred
    t=np.arange(N)*h
    plt.figure(figsize=(6,3)); plt.plot(t,res[:,0],label="Range"); plt.plot(t,np.rad2deg(res[:,1]),label="Angle [deg]")
    plt.xlabel("t"); plt.ylabel("Residual"); plt.title("OD residuals (smoke)"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"od_residuals_smoke.png"), dpi=140); plt.close()
    log("  saved: smoke_outputs/od_residuals_smoke.png")

if __name__ == "__main__":
    t0=time.time()
    stage1_crtbp()
    stage2_pareto()
    stage3_od()
    log("[Done] All smoke figures saved into ./smoke_outputs")
