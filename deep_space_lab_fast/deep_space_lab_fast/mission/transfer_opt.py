
import numpy as np

def hypervolume(front, ref=(3.0,3.0,3.0)):
    if len(front)==0: return 0.0
    F=np.minimum(np.array(front), ref); idx=np.argsort(F[:,0]); hv=0.0; prev0=ref[0]
    for i in idx[::-1]:
        a0=prev0-F[i,0]; a1=ref[1]-F[i,1]; a2=ref[2]-F[i,2]
        if a0>0 and a1>0 and a2>0: hv+=a0*a1*a2; prev0=F[i,0]
    return hv

def dominates(a,b): return np.all(a<=b) and np.any(a<b)

def fast_nondominated_sort(F):
    S=[[] for _ in range(len(F))]; n=[0]*len(F); rank=[0]*len(F); fronts=[[]]
    for p in range(len(F)):
        for q in range(len(F)):
            if dominates(F[p],F[q]): S[p].append(q)
            elif dominates(F[q],F[p]): n[p]+=1
        if n[p]==0: rank[p]=0; fronts[0].append(p)
    i=0
    while fronts[i]:
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0: rank[q]=i+1; Q.append(q)
        i+=1; fronts.append(Q)
    return fronts[:-1], rank

def crowding_distance(F, idxs):
    D=np.zeros(len(idxs))
    if len(idxs)==0: return D
    m=F.shape[1]
    for j in range(m):
        order=sorted(range(len(idxs)), key=lambda k:F[idxs[k],j])
        D[order[0]]=D[order[-1]]=1e9
        fj=F[idxs,j]; fmin,fmax=np.min(fj),np.max(fj)
        if fmax-fmin<1e-12: continue
        for k in range(1,len(idxs)-1):
            i1,i2=order[k-1], order[k+1]
            D[order[k]] += (F[idxs[i2],j]-F[idxs[i1],j])/(fmax-fmin)
    return D

def nsga2(objective, bounds, pop=14, gen=10, pc=0.9, pm=0.2):
    D=len(bounds)
    X=np.array([[np.random.uniform(lo,hi) for (lo,hi) in bounds] for _ in range(pop)])
    F=np.array([objective(xi) for xi in X]); hv_hist=[]
    for g in range(gen):
        fronts, rank=fast_nondominated_sort(F)
        def tournament():
            i,j=np.random.randint(len(X)),np.random.randint(len(X))
            if rank[i] < rank[j]: return i
            if rank[i] > rank[j]: return j
            fr_i = next(k for k,fr in enumerate(fronts) if i in fr)
            fr_j = next(k for k,fr in enumerate(fronts) if j in fr)
            Di=crowding_distance(F, fronts[fr_i])[fronts[fr_i].index(i)]
            Dj=crowding_distance(F, fronts[fr_j])[fronts[fr_j].index(j)]
            return i if Di>Dj else j
        Y=[]
        for _ in range(len(X)//2):
            p1=X[tournament()]; p2=X[tournament()]
            c1,c2=p1.copy(),p2.copy()
            if np.random.rand()<pc:
                alpha=np.random.rand(D); c1=alpha*p1+(1-alpha)*p2; c2=alpha*p2+(1-alpha)*p1
            for d in range(D):
                if np.random.rand()<pm:
                    lo,hi=bounds[d]; c1[d]=np.clip(c1[d]+0.2*(hi-lo)*np.random.randn(), lo, hi)
                if np.random.rand()<pm:
                    lo,hi=bounds[d]; c2[d]=np.clip(c2[d]+0.2*(hi-lo)*np.random.randn(), lo, hi)
            Y += [c1,c2]
        Y=np.array(Y); FY=np.array([objective(yi) for yi in Y])
        Z=np.vstack([X,Y]); G=np.vstack([F,FY]); fronts,_=fast_nondominated_sort(G)
        X_new=[]; F_new=[]
        for fr in fronts:
            if len(X_new)+len(fr)<=len(X):
                X_new += [Z[i] for i in fr]; F_new += [G[i] for i in fr]
            else:
                Dd=crowding_distance(G, fr); order=np.argsort(-Dd)
                for k in order:
                    if len(X_new) < len(X):
                        X_new.append(Z[fr[k]]); F_new.append(G[fr[k]])
                    else: break
                break
        X=np.array(X_new); F=np.array(F_new); hv_hist.append( hypervolume(G[fronts[0]]) )
    fronts,_=fast_nondominated_sort(F)
    return X[fronts[0]], F[fronts[0]], np.array(hv_hist)

def quick_objective(mu, extra, start_state, target_state, T_bounds):
    # 2-impulse, objectives: [ΔV, TOF, terminal miss]
    def dyn(s):
        from deep_space_lab_fast.dynamics.crtbp import full_dynamics_with_jac
        f,_ = full_dynamics_with_jac(s, mu, extra); return f
    def prop(s, T, dt=0.02):
        x=s.copy()
        for _ in range(max(1,int(T/dt))):
            f1=dyn(x); f2=dyn(x+0.5*dt*f1); f3=dyn(x+0.5*dt*f2); f4=dyn(x+dt*f3)
            x=x+(dt/6)*(f1+2*f2+2*f3+f4)
        return x
    def objective(y):
        t1=y[0]; dv1=y[1:4]; coast=y[4]; dv2=y[5:8]
        t1=np.clip(t1, T_bounds[0], T_bounds[1]); coast=max(0.05, min(coast, T_bounds[1]-t1))
        s1=prop(start_state, t1); s1[3:]+=dv1
        s2=prop(s1, coast); s2[3:]+=dv2
        sf=prop(s2, 0.3)
        dV=np.linalg.norm(dv1)+np.linalg.norm(dv2); tof=t1+coast+0.3
        miss=np.linalg.norm(sf[:3]-target_state[:3])
        return np.array([dV,tof,miss])
    return objective
