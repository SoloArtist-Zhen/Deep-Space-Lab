
import os, sys, time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deep_space_lab_fast.halo.initial_guess import halo_richardson_like
from deep_space_lab_fast.halo.diff_correction import single_shoot_periodic, multi_shoot_periodic
from deep_space_lab_fast.solvers.collocation import run_collocation
from deep_space_lab_fast.integrators.propagate_with_stm import propagate_with_stm, monodromy
from deep_space_lab_fast.dynamics.crtbp import full_dynamics_with_jac
from deep_space_lab_fast.mission.transfer_opt import nsga2, quick_objective
from deep_space_lab_fast.od.estimation import simulate_measurements, batch_ls, ekf, H_angles
from deep_space_lab_fast.plots.plotting import *

OUT = "deep_space_lab_fast/outputs"
os.makedirs(OUT, exist_ok=True)

def log(s): print(s); sys.stdout.flush()

def do_halo_suite():
    log("[halo] start")
    mu = 0.0121505856
    extra0 = dict(c_srp=0.0, J2_e=0.0, J2_m=0.0, Re_bar=0.01, Rm_bar=0.003, sun_x=10.0)
    extraP = dict(c_srp=1.6e-6, J2_e=7e-4, J2_m=1.2e-4, Re_bar=0.01, Rm_bar=0.003, sun_x=10.0)

    amps = np.linspace(0.02, 0.06, 3)
    periods=[]; energies=[]; rads=[]
    for Az in amps:
        s0, T0 = halo_richardson_like(mu, Az=Az, around="L1")
        s_corr, T = single_shoot_periodic(s0, mu, extra0, T0, max_iter=6)
        periods.append(T)
        ts, Y = propagate_with_stm(s_corr, T, mu, extra0, h0=2e-3, with_stm=False)
        X=Y[:,:6]; U=[]
        for st in X:
            x,y,z,vx,vy,vz = st
            r1=np.linalg.norm([x+mu,y,z]); r2=np.linalg.norm([x-(1-mu),y,z])
            U.append(0.5*(x*x+y*y) + (1-mu)/r1 + mu/r2)
        C = 2*np.mean(U) - np.mean(np.sum(X[:,3:]**2,axis=1))
        energies.append(C)
    s0, T0 = halo_richardson_like(mu, Az=amps[-1], around="L1")
    s_corr, T = single_shoot_periodic(s0, mu, extra0, T0, max_iter=6)
    PhiT = monodromy(s_corr, T, mu, extra0); eigs = np.linalg.eigvals(PhiT); rads = [np.max(np.abs(eigs))]*len(amps)

    fig_halo_family(amps, periods, energies); fig_monodromy_spectral(amps, rads)

    # SRP/J2 drift (half period)
    ts, Y0 = propagate_with_stm(s_corr, T/2, mu, extra0, h0=2e-3, with_stm=False)
    ts, YP = propagate_with_stm(s_corr, T/2, mu, extraP, h0=2e-3, with_stm=False)
    fig_srp_j2_compare(Y0[:,:2], YP[:,:2])

    for i in range(len(plt.get_fignums())):
        plt.figure(i+1).savefig(f"{OUT}/halo_{i:02d}.png", dpi=170)
    plt.close('all')
    log("[halo] saved figures.")

def do_pareto_suite():
    log("[pareto] start")
    mu = 0.0121505856
    extra = dict(c_srp=1.2e-6, J2_e=6e-4, J2_m=1e-4, Re_bar=0.01, Rm_bar=0.003, sun_x=10.0)
    s0, T0 = halo_richardson_like(mu, Az=0.03, around="L1")
    ts, Y = propagate_with_stm(s0, 0.25, mu, extra, h0=2e-3, with_stm=False)
    start = Y[-1,:6]; target = start.copy(); target[:3] += np.array([0.04, -0.015, 0.01])

    obj = quick_objective(mu, extra, start, target, T_bounds=(0.2,0.9))
    bounds=[(0.2,0.8), (-0.04,0.04),(-0.04,0.04),(-0.04,0.04), (0.2,0.8), (-0.04,0.04),(-0.04,0.04),(-0.04,0.04)]
    X, F, hv = nsga2(obj, bounds, pop=14, gen=10, pc=0.9, pm=0.2)
    fig_pareto_3d(F); fig_hypervolume(hv)
    # ΔV composition
    plt.figure(figsize=(6,3.2))
    best_idx = np.argsort(F[:,0])[:5]
    dv1=[np.linalg.norm(X[i][1:4]) for i in best_idx]; dv2=[np.linalg.norm(X[i][5:8]) for i in best_idx]
    idx=np.arange(len(best_idx)); plt.bar(idx, dv1, width=0.35, label="impulse1"); plt.bar(idx, dv2, width=0.35, bottom=dv1, label="impulse2")
    plt.xlabel("Solution idx"); plt.ylabel("|ΔV|"); plt.title("ΔV composition (FAST)"); plt.legend(); plt.tight_layout()

    for i in range(len(plt.get_fignums())):
        plt.figure(i+1).savefig(f"{OUT}/pareto_{i:02d}.png", dpi=170)
    plt.close('all')
    log("[pareto] saved figures.")

def do_od_suite():
    log("[od] start")
    mu = 0.0121505856; extra=dict(c_srp=0.0,J2_e=0.0,J2_m=0.0, Re_bar=0.01,Rm_bar=0.003, sun_x=10.0)
    s0, T0 = halo_richardson_like(mu, Az=0.035, around="L1")
    ts, Y = propagate_with_stm(s0, 0.45, mu, extra, h0=2e-3, with_stm=False); traj=Y[:,:6]
    noise=dict(range=4e-4,doppler=1e-4, ang=np.deg2rad(0.02))
    z=simulate_measurements(traj, noise); R=np.diag([noise["range"]**2, noise["doppler"]**2, noise["ang"]**2, noise["ang"]**2])
    dx, P = batch_ls(traj, z, R); Q=np.diag([0,0,0,5e-6,5e-6,5e-6]); est, Ps = ekf(traj, z, R, Q)
    res=[]
    for k in range(len(traj)):
        pred=np.array([np.linalg.norm(est[k,:3]), (est[k,:3]@est[k,3:])/(np.linalg.norm(est[k,:3])+1e-9), *H_angles(est[k])])
        res.append(z[k]-pred)
    res=np.array(res)
    t=ts[:len(res)]
    fig_od_residuals(t, res); fig_cov_ellipse(Ps)
    for i in range(len(plt.get_fignums())):
        plt.figure(i+1).savefig(f"{OUT}/od_{i:02d}.png", dpi=170)
    plt.close('all')
    log("[od] saved figures.")

if __name__=="__main__":
    part = sys.argv[1] if len(sys.argv)>1 else "all"
    if part in ("all","halo"): do_halo_suite()
    if part in ("all","pareto"): do_pareto_suite()
    if part in ("all","od"): do_od_suite()
    print("Figures saved to", OUT)
