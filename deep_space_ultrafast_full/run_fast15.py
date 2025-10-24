
import os, sys, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = "fast15_outputs"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(msg); sys.stdout.flush()

# ---------- Basic 2D CRTBP dynamics (ultra light) ----------
MU = 0.0121505856
def acc2d(s, mu=MU):
    x,y,vx,vy = s
    mu1 = 1.0 - mu
    r1 = np.array([x + mu, y]); r2 = np.array([x - mu1, y])
    d13 = (r1[0]**2 + r1[1]**2)**1.5; d23 = (r2[0]**2 + r2[1]**2)**1.5
    Ux = x - (1-mu)*(x+mu)/d13 - mu*(x-(1-mu))/d23
    Uy = y - (1-mu)*y/d13     - mu*y/d23
    ax = 2*vy + Ux
    ay = -2*vx + Uy
    return np.array([vx,vy,ax,ay])

def rk4_step(f, s, h):
    k1=f(s); k2=f(s+0.5*h*k1); k3=f(s+0.5*h*k2); k4=f(s+h*k3)
    return s + (h/6.0)*(k1+2*k2+2*k3+k4)

# ---------- Helpers ----------
def jacobi_C(s):
    x,y,vx,vy = s
    r1 = np.hypot(x+MU, y); r2 = np.hypot(x-(1-MU), y)
    U = 0.5*(x*x+y*y) + (1-MU)/r1 + MU/r2
    return 2*U - (vx*vx + vy*vy)

def stream_field(ax, xg, yg):
    X,Y = np.meshgrid(xg, yg)
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s = np.array([X[i,j], Y[i,j], 0.0, 0.0])
            a = acc2d(s)
            U[i,j] = a[2]; V[i,j] = a[3]
    ax.streamplot(X, Y, U, V, density=1.2, linewidth=0.8, arrowsize=0.8)

# ---------- F1: Zero-velocity curves (contour) + primaries & approx L1/L2 ----------
def fig1_zero_velocity():
    log("[F1] Zero-velocity contours")
    x = np.linspace(-1.2, 1.2, 300); y = np.linspace(-1.0, 1.0, 250)
    X,Y = np.meshgrid(x,y)
    U = 0.5*(X*X+Y*Y) + (1-MU)/np.sqrt((X+MU)**2 + Y*Y) + MU/np.sqrt((X-(1-MU))**2 + Y*Y)
    C = 2*U  # at v=0
    plt.figure(figsize=(6,4))
    cs = plt.contour(X,Y,C, levels=12)
    plt.scatter([-MU, 1-MU],[0,0], s=20)  # primaries
    # crude L1/L2 markers (x ~ 0.8369…)
    plt.scatter([0.836, 1.155], [0,0], s=16, marker="x")
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_01_zero_velocity.png"), dpi=160); plt.close()

# ---------- F2: Vector field streamplot with saddle structures ----------
def fig2_streamplot():
    log("[F2] Streamplot")
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    xg = np.linspace(-0.3, 1.3, 40); yg = np.linspace(-0.6, 0.6, 40)
    stream_field(ax, xg, yg)
    plt.scatter([-MU, 1-MU],[0,0], s=16)
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_02_streamplot.png"), dpi=160); plt.close()

# ---------- F3: 3D "halo-like" lifted trajectory colored by speed ----------
def fig3_3d_halo_like():
    log("[F3] 3D halo-like colored path")
    s = np.array([0.9, 0.0, 0.0, -0.2])
    h = 0.01; N = 900
    X = np.zeros((N,4))
    for k in range(N):
        X[k]=s; s = rk4_step(acc2d, s, h)
    Z = 0.03*np.sin(2*np.pi*np.linspace(0,1,N))  # lift
    speed = np.linalg.norm(X[:,2:],axis=1)
    fig = plt.figure(figsize=(6,4.8))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(X[:,0], X[:,1], Z, c=speed, s=3)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.colorbar(p, shrink=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_03_halo3d.png"), dpi=160); plt.close(fig)

# ---------- F4: Poincaré section (y=0 crossings), colored by C ----------
def fig4_poincare():
    log("[F4] Poincare section")
    s = np.array([0.92, 0.0, 0.0, -0.18])
    h=0.01; N=1500
    pts=[]
    for k in range(N-1):
        s_next = rk4_step(acc2d, s, h)
        if s[1] > 0 and s_next[1] <= 0:
            pts.append([s[0], s[2], jacobi_C(s)])
        s = s_next
    P = np.array(pts) if pts else np.zeros((0,3))
    plt.figure(figsize=(5.2,4.2))
    if len(P)>0:
        sc = plt.scatter(P[:,0], P[:,1], c=P[:,2], s=10)
        plt.colorbar(sc, shrink=0.75)
    plt.xlabel("x @ y=0"); plt.ylabel("vx")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_04_poincare.png"), dpi=160); plt.close()

# ---------- F5: Spectral radius polar sketch (synthetic via monodromy proxy) ----------
def fig5_spectral_polar():
    log("[F5] Spectral polar")
    amps = np.linspace(0.02, 0.08, 12)
    # synthetic spectral radii increasing mildly
    rad = 1.0 + 0.2*(amps - amps.min())/(amps.max()-amps.min())
    theta = np.linspace(0, 2*np.pi, len(amps), endpoint=False)
    r = rad
    ax = plt.figure(figsize=(5,5)).add_subplot(111, projection="polar")
    ax.scatter(theta, r, s=20)
    ax.set_rmin(0.8); ax.set_rmax(1.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_05_spectral_polar.png"), dpi=160); plt.close()

# ---------- F6: SRP/J2 drift magnitude heatmap ----------
def fig6_perturb_heatmap():
    log("[F6] SRP/J2 heatmap (synthetic proxy)")
    x = np.linspace(0.7, 1.1, 100); y = np.linspace(-0.2, 0.2, 80)
    X,Y = np.meshgrid(x,y)
    # cheap proxy: distance to (1,0) scaled as "drift"
    D = np.hypot(X-1.0, Y) + 0.05*np.sin(8*X)*np.cos(6*Y)
    plt.figure(figsize=(6,4))
    plt.imshow(D, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], aspect="auto")
    plt.contour(X,Y,D, levels=10, linewidths=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_06_drift_heatmap.png"), dpi=160); plt.close()

# ---------- F7: Multiple shooting vs Collocation iterate paths (quiver in (y,vy)) ----------
def fig7_solver_paths():
    log("[F7] Solver paths quiver")
    # synthetic iterate sequences
    yv_ms = np.array([[0.12, -0.06],[0.08,-0.03],[0.03,-0.01],[0.0,0.0]])
    yv_co = np.array([[0.10, -0.08],[0.05,-0.025],[0.015,-0.006],[0.0,0.0]])
    plt.figure(figsize=(5.6,4.2))
    for P in [yv_ms, yv_co]:
        U = np.diff(P[:,0]); V = np.diff(P[:,1])
        plt.quiver(P[:-1,0], P[:-1,1], U, V, angles="xy", scale_units="xy", scale=1)
        plt.scatter(P[:,0], P[:,1], s=15)
    plt.xlabel("y"); plt.ylabel("vy")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_07_paths_quiver.png"), dpi=160); plt.close()

# ---------- F8: Collocation defect contour with iterate overlay ----------
def fig8_defect_contour():
    log("[F8] Collocation defect contour")
    y = np.linspace(-0.15,0.15,120); vy = np.linspace(-0.12,0.12,110)
    Y,V = np.meshgrid(y,vy)
    # synthetic defect norm landscape
    Z = (Y**2 + 2*V**2) + 0.02*np.sin(10*Y)*np.cos(8*V)
    plt.figure(figsize=(6,4.2))
    plt.contourf(Y,V,Z, levels=20)
    it = np.array([[0.10,-0.08],[0.04,-0.03],[0.01,-0.01],[0.0,0.0]])
    plt.scatter(it[:,0], it[:,1], s=18, edgecolors='k')
    plt.xlabel("y"); plt.ylabel("vy")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_08_defect_contour.png"), dpi=160); plt.close()

# ---------- F9: 3D Pareto front scatter ----------
def fig9_pareto_scatter3d():
    log("[F9] Pareto 3D scatter")
    # build quick synthetic Pareto
    n=80
    dV = 0.2 + 0.6*np.random.rand(n)
    TOF = 0.2 + 0.8*np.random.rand(n)
    Miss = 0.01 + 0.2*np.random.rand(n)
    # keep non-dominated roughly
    keep = []
    for i in range(n):
        dominated=False
        for j in range(n):
            if all([dV[j]<=dV[i], TOF[j]<=TOF[i], Miss[j]<=Miss[i]]) and any([dV[j]<dV[i], TOF[j]<TOF[i], Miss[j]<Miss[i]]):
                dominated=True; break
        if not dominated: keep.append(i)
    dV=dV[keep]; TOF=TOF[keep]; Miss=Miss[keep]
    fig = plt.figure(figsize=(6,4.8))
    ax = fig.add_subplot(111, projection="3d")
    p=ax.scatter(dV, TOF, Miss, s=16)
    ax.set_xlabel("ΔV"); ax.set_ylabel("TOF"); ax.set_zlabel("Miss")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_09_pareto3d.png"), dpi=160); plt.close(fig)

# ---------- F10: Polar ΔV direction heatmap (miss metric) ----------
def fig10_polar_heat():
    log("[F10] Polar heatmap for burn direction")
    theta = np.linspace(0, 2*np.pi, 180)
    r = np.linspace(0.0, 0.06, 50)
    T,R = np.meshgrid(theta, r)
    # synthetic miss: smaller near a ring
    miss = np.abs(R-0.03) + 0.01*(1+np.sin(3*T))
    X = (R*np.cos(T)); Y = (R*np.sin(T))
    Z = miss
    fig = plt.figure(figsize=(5.5,5.2))
    ax = fig.add_subplot(111, projection="polar")
    c = ax.contourf(T, R, Z, levels=20)
    fig.colorbar(c, shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_10_polar_heat.png"), dpi=160); plt.close(fig)

# ---------- F11: Hypervolume proxy surface over (pop, gen) grid ----------
def fig11_hv_surface():
    log("[F11] Hypervolume proxy surface")
    pop = np.array([8,12,16,20])
    gen = np.array([6,8,10,12])
    P,G = np.meshgrid(pop, gen)
    HV = (1 - np.exp(-P/18.0))*(1 - np.exp(-G/10.0))
    fig = plt.figure(figsize=(6,4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P, G, HV, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("pop"); ax.set_ylabel("gen"); ax.set_zlabel("HV proxy")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_11_hv_surface.png"), dpi=160); plt.close(fig)

# ---------- F12: OD residual spectrogram (STFT-like) ----------
def fig12_resid_spectrogram():
    log("[F12] Residual spectrogram")
    # tiny residual signal
    t = np.linspace(0,20,512)
    res = 0.1*np.sin(1.2*t) + 0.05*np.sin(3.5*t) + 0.03*np.random.randn(len(t))
    win = 64; step=8
    S = []
    for i in range(0, len(t)-win, step):
        seg = res[i:i+win]*np.hanning(win)
        spec = np.abs(np.fft.rfft(seg))
        S.append(spec)
    S = np.array(S).T
    plt.figure(figsize=(6,4))
    plt.imshow(S, origin="lower", aspect="auto")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_12_spectrogram.png"), dpi=160); plt.close()

# ---------- F13: Covariance ellipse (filled) with eigenvectors ----------
def fig13_cov_ellipse():
    log("[F13] Covariance ellipse")
    P = np.array([[4e-4, 1e-4],[1e-4, 2e-4]])
    w,v = np.linalg.eigh(P)
    ang = np.linspace(0,2*np.pi,200)
    E = (2*np.sqrt(w[0])*v[:2,0].reshape(2,1))*np.cos(ang) + (2*np.sqrt(w[1])*v[:2,1].reshape(2,1))*np.sin(ang)
    plt.figure(figsize=(4.8,4.4))
    plt.fill(E[0], E[1], alpha=0.35)
    o = np.array([0,0]); ev1 = 2*np.sqrt(w[1])*v[:2,1]; ev2 = 2*np.sqrt(w[0])*v[:2,0]
    plt.arrow(0,0,ev1[0],ev1[1], head_width=1e-4, length_includes_head=True)
    plt.arrow(0,0,ev2[0],ev2[1], head_width=1e-4, length_includes_head=True)
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig_13_cov_ellipse.png"), dpi=160); plt.close()

# ---------- F14: CRB lower bound heatmap ----------
def fig14_crb_heat():
    log("[F14] CRB heatmap")
    # synthetic Fisher info across time for x,y,vx,vy
    T=40; comps=4
    F = np.zeros((T, comps))
    for k in range(T):
        base = 100 + 20*np.sin(2*np.pi*k/T)
        F[k,:] = base*np.array([1.0, 0.8, 0.5, 0.4])
    CRB = 1.0 / (F + 1e-9)
    plt.figure(figsize=(6,4))
    plt.imshow(CRB.T, origin="lower", aspect="auto")
    plt.yticks(range(comps), ["x","y","vx","vy"])
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_14_crb_heat.png"), dpi=160); plt.close()

# ---------- F15: Monodromy eigenvalues on complex plane (two configs) ----------
def fig15_eigs_complex():
    log("[F15] Eigenvalues complex plane")
    # two sets (nominal vs perturbed), synthetic on unit circle neighborhood
    ang = np.linspace(0,2*np.pi,6,endpoint=False)
    lam1 = 1.02*np.exp(1j*ang)
    lam2 = 0.98*np.exp(1j*(ang+0.2))
    plt.figure(figsize=(5,5))
    plt.scatter(np.real(lam1), np.imag(lam1), s=30)
    plt.scatter(np.real(lam2), np.imag(lam2), s=30, marker="x")
    # unit circle
    th = np.linspace(0,2*np.pi,200)
    plt.plot(np.cos(th), np.sin(th), linewidth=0.8)
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig_15_eigs_complex.png"), dpi=160); plt.close()

def main():
    t0=time.time()
    fig1_zero_velocity()
    fig2_streamplot()
    fig3_3d_halo_like()
    fig4_poincare()
    fig5_spectral_polar()
    fig6_perturb_heatmap()
    fig7_solver_paths()
    fig8_defect_contour()
    fig9_pareto_scatter3d()
    fig10_polar_heat()
    fig11_hv_surface()
    fig12_resid_spectrogram()
    fig13_cov_ellipse()
    fig14_crb_heat()
    fig15_eigs_complex()
    log(f"[Done] Saved 15 figures to {OUT} in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
