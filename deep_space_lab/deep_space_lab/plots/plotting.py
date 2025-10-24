
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def fig_halo_family(ampl, period, energy):
    plt.figure(figsize=(6,4)); plt.plot(ampl, period, marker='o'); plt.xlabel("z-amplitude"); plt.ylabel("Period"); plt.title("Halo: Period vs Amplitude"); plt.grid(True); plt.tight_layout()
    plt.figure(figsize=(6,4)); plt.plot(energy, period, marker='x'); plt.xlabel("Jacobi-like energy"); plt.ylabel("Period"); plt.title("Energy–Period"); plt.grid(True); plt.tight_layout()

def fig_monodromy_spectral(ampl, rad, lam_minmax):
    plt.figure(figsize=(6,4)); plt.plot(ampl, rad, marker='o'); plt.xlabel("z-amplitude"); plt.ylabel("Spectral radius"); plt.title("Monodromy spectral radius"); plt.grid(True); plt.tight_layout()

def fig_iter_curves(ms_hist, col_hist, t_ms, t_col):
    plt.figure(figsize=(6,4)); plt.semilogy(ms_hist, label="Multiple shooting"); plt.semilogy(col_hist, label="Collocation"); plt.xlabel("Iteration"); plt.ylabel("Residual norm"); plt.title("Residual"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.figure(figsize=(4,3)); plt.bar([0,1], [t_ms, t_col]); plt.xticks([0,1], ["Shooting","Collocation"]); plt.ylabel("Runtime (s)"); plt.title("Compute time"); plt.tight_layout()

def fig_pareto_3d(F):
    fig=plt.figure(figsize=(6,4.5)); ax=fig.add_subplot(111, projection='3d'); ax.scatter(F[:,0],F[:,1],F[:,2],s=18); ax.set_xlabel("ΔV"); ax.set_ylabel("TOF"); ax.set_zlabel("Robustness"); ax.set_title("Pareto front"); plt.tight_layout()

def fig_hypervolume(hv_hist):
    plt.figure(figsize=(5,3.4)); plt.plot(hv_hist); plt.xlabel("Generation"); plt.ylabel("Hypervolume"); plt.title("Hypervolume convergence"); plt.grid(True); plt.tight_layout()

def fig_srp_j2_compare(t, base_xy, pert_xy):
    plt.figure(figsize=(6,4)); plt.plot(base_xy[:,0], base_xy[:,1], label="CRTBP"); plt.plot(pert_xy[:,0], pert_xy[:,1], label="CRTBP+SRP+J2"); plt.axis('equal'); plt.grid(True); plt.legend(); plt.title("SRP & J2 drift"); plt.tight_layout()

def fig_od_residuals(t, res):
    plt.figure(figsize=(6,4)); plt.plot(t, res[:,0], label="Range"); plt.plot(t, res[:,1], label="Doppler"); plt.plot(t, res[:,2], label="Az"); plt.plot(t, res[:,3], label="El"); plt.xlabel("Time"); plt.ylabel("Residual"); plt.title("OD residuals"); plt.legend(); plt.grid(True); plt.tight_layout()

def fig_cov_ellipse(Ps):
    P=Ps[-1][:2,:2]; w,v=np.linalg.eigh(P); ang=np.linspace(0,2*np.pi,200)
    E=(2*np.sqrt(w[0])*v[:2,0].reshape(2,1))*np.cos(ang) + (2*np.sqrt(w[1])*v[:2,1].reshape(2,1))*np.sin(ang)
    plt.figure(figsize=(4,4)); plt.plot(E[0], E[1]); plt.axis('equal'); plt.grid(True); plt.title("Final 2σ ellipse"); plt.tight_layout()
