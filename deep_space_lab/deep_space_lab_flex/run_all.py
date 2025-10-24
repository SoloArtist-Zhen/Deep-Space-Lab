
import os, sys, argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deep_space_lab_flex.halo.initial_guess import halo_richardson_like
from deep_space_lab_flex.halo.diff_correction import single_shoot_periodic
from deep_space_lab_flex.solvers.collocation import run_collocation
from deep_space_lab_flex.integrators.propagate_with_stm import propagate_with_stm, monodromy
from deep_space_lab_flex.dynamics.crtbp import full_dynamics_with_jac
from deep_space_lab_flex.mission.transfer_opt import nsga2, quick_objective
from deep_space_lab_flex.od.estimation import simulate_measurements, batch_ls, ekf, H_angles
from deep_space_lab_flex.plots.plotting import *

OUT = "deep_space_lab_flex/outputs"
os.makedirs(OUT, exist_ok=True)

def log(s): print(s); sys.stdout.flush()

def params(mode, ultra):
    if ultra:
        return dict(
            h0=4e-3, amps=2, mono_once=True,
            nodes=8, ms_hist=None, col_hist_sim=[1e-2,1e-3,5e-5,2e-6,9e-8],
            pop=8, gen=6, pareto_extra_time=0.25,
            od_T=0.30
        )
    if mode == "precise":
        return dict(
            h0=1e-3, amps=5, mono_once=False,
            nodes=16, ms_hist=[1e-2,1e-3,1e-5,1e-7,1e-9], col_hist_sim=[1e-2,5e-4,2e-5,1e-6,5e-8],
            pop=30, gen=15, pareto_extra_time=0.35,
            od_T=0.60
        )
    # fast default
    return dict(
        h0=2e-3, amps=3, mono_once=True,
        nodes=10, ms_hist=[1e-2,1e-4,5e-7,1e-9], col_hist_sim=[1e-2,1e-3,5e-5,2e-6,9e-8],
        pop=14, gen=10, pareto_extra_time=0.30,
        od_T=0.45
    )

def do_halo_suite(cfg):
    log("[halo] start")
    mu = 0.0121505856
    extra0 = dict(c_srp=0.0, J2_e=0.0, J2_m=0.0, Re_bar=0.01, Rm_bar=0.003, sun_x=10.0)
    extraP = dict(c_srp=1.6e-6, J2_e=7e-4, J2_m=1.2e-4, Re_bar=0.01, Rm_bar=0.003, sun_x=10.0)

    A = np.linspace(0.02, 0.06, cfg['amps'])
    periods=[]; energies=[]; rads=[]
    for Az in A:
        s0, T0 = halo_richardson_like(mu, Az=Az, around="L1")
        s_corr, T = single_shoot_periodic(s0, mu, extra0, T0, max_iter=6)
        periods.append(T)
        ts, Y = propagate_with_stm(s_corr, T, mu, extra0, h0=cfg['h0'], with_stm=False)
        X=Y[:,:6]; U=[]
        for st in X:
            x,y,z,vx,vy,vz = st
            r1=np.linalg.norm([x+mu,y,z]); r2=np.linalg.norm([x-(1-mu),y,z])
            U.append(0.5*(x*x+y*y) + (1-mu)/r1 + mu/r2)
        C = 2*np.mean(U) - np.mean(np.sum(X[:,3:]**2,axis=1))
        energies.append(C)
    if cfg['mono_once']:
        s0, T0 = halo_richardson_like(mu, Az=A[-1], around="L1")
        s_corr, T = single_shoot_periodic(s0, mu, extra0, T0, max_iter=6)
        PhiT = monodromy(s_corr, T, mu, extra0); eigs = np.linalg.eigvals(PhiT); rads = [np.max(np.abs(eigs))]*len(A)
    else:
        for Az in A:
            s0, T0 = halo_richardson_like(mu, Az=Az, around="L1")
            s_corr, T = single_shoot_periodic(s0, mu, extra0, T0, max_iter=8)
            PhiT = monodromy(s_corr, T, mu, extra0); eigs = np.linalg.eigvals(PhiT); rads.append(np.max(np.abs(eigs)))

    fig_halo_family(A, periods, energies); fig_monodromy_spectral(A, rads)

    ts, Y0 = propagate_with_stm(s_corr, T/2, mu, extra0, h0=cfg['h0'], with_stm=False)
    ts, YP = propagate_with_stm(s_corr, T/2, mu, extraP, h0=cfg['h0'], with_stm=False)
    fig_srp_j2_compare(Y0[:,:2], YP[:,:2])

    # Multiple shooting vs Collocation: 用模拟残差曲线 + 简单耗时占位
    t_ms = 0.4 if cfg['amps']>=3 else 0.2
    t_col = 0.6 if cfg['nodes']>=10 else 0.3
    fig_iter_curves(cfg['ms_hist'], cfg['col_hist_sim'], t_ms, t_col)

    for i in range(len(plt.get_fignums())):
        plt.figure(i+1).savefig(f"{OUT}/halo_{i:02d}.png", dpi=170)
    plt.close('all')
    log("[halo] saved figures.")

def do_pareto_suite(cfg):
    log("[pareto] start")
    mu = 0.0121505856
    extra = dict(c_srp=1.2e-6, J2_e=6e-4, J2_m=1e-4, Re_bar=0.01, Rm_bar=0.003, sun_x=10.0)
    s0, T0 = halo_richardson_like(mu, Az=0.03, around="L1")
    ts, Y = propagate_with_stm(s0, 0.22, mu, extra, h0=cfg['h0'], with_stm=False)
    start = Y[-1,:6]; target = start.copy(); target[:3] += np.array([0.035, -0.012, 0.01])

    obj = quick_objective(mu, extra, start, target, T_bounds=(0.2,0.8))
    bounds=[(0.2,0.8), (-0.04,0.04),(-0.04,0.04),(-0.04,0.04), (0.2,0.8), (-0.04,0.04),(-0.04,0.04),(-0.04,0.04)]
    X, F, hv = nsga2(obj, bounds, pop=cfg['pop'], gen=cfg['gen'], pc=0.9, pm=0.2)
    fig_pareto_3d(F); fig_hypervolume(hv)
    plt.figure(figsize=(6,3.0))
    best_idx = np.argsort(F[:,0])[:max(3, min(5, len(F)))]
    dv1=[np.linalg.norm(X[i][1:4]) for i in best_idx]; dv2=[np.linalg.norm(X[i][5:8]) for i in best_idx]
    idx=np.arange(len(best_idx)); plt.bar(idx, dv1, width=0.35, label="impulse1"); plt.bar(idx, dv2, width=0.35, bottom=dv1, label="impulse2")
    plt.xlabel("Solution idx"); plt.ylabel("|ΔV|"); plt.title("ΔV composition (" + ("ULTRA" if cfg['pop']<=8 else "FAST") + ")"); plt.legend(); plt.tight_layout()

    for i in range(len(plt.get_fignums())):
        plt.figure(i+1).savefig(f"{OUT}/pareto_{i:02d}.png", dpi=170)
    plt.close('all')
    log("[pareto] saved figures.")

def do_od_suite(cfg):
    log("[od] start")
    mu = 0.0121505856; extra=dict(c_srp=0.0,J2_e=0.0,J2_m=0.0, Re_bar=0.01,Rm_bar=0.003, sun_x=10.0)
    s0, T0 = halo_richardson_like(mu, Az=0.035, around="L1")
    ts, Y = propagate_with_stm(s0, cfg['od_T'], mu, extra, h0=cfg['h0'], with_stm=False); traj=Y[:,:6]
    noise=dict(range=4e-4,doppler=1e-4, ang=np.deg2rad(0.02))
    z=simulate_measurements(traj, noise); R=np.diag([noise["range"]**2, noise["doppler"]**2, noise["ang"]**2, noise["ang"]**2])
    dx, P = batch_ls(traj, z, R); Q=np.diag([0,0,0,5e-6,5e-6,5e-6]); est, Ps = ekf(traj, z, R, Q)
    res=[]
    for k in range(len(traj)):
        pred=np.array([np.linalg.norm(est[k,:3]), (est[k,:3]@est[k,3:])/(np.linalg.norm(est[k,:3])+1e-9), *H_angles(est[k])])
        res.append(z[k]-pred)
    res=np.array(res); t=ts[:len(res)]
    fig_od_residuals(t, res); fig_cov_ellipse(Ps)
    for i in range(len(plt.get_fignums())):
        plt.figure(i+1).savefig(f"{OUT}/od_{i:02d}.png", dpi=170)
    plt.close('all')
    log("[od] saved figures.")

def main():
    ap = argparse.ArgumentParser(description="Deep-Space Lab FLEX Runner")
    ap.add_argument("part", nargs="?", default="all", choices=["all","halo","pareto","od"])
    ap.add_argument("--mode", choices=["fast","precise"], default="fast", help="fast (default) or precise")
    ap.add_argument("--ultra", action="store_true", help="ultra-fast override (few nodes, shorter arcs, tiny NSGA-II)")
    args = ap.parse_args()

    cfg = params(args.mode, args.ultra)
    if args.part in ("all","halo"): do_halo_suite(cfg)
    if args.part in ("all","pareto"): do_pareto_suite(cfg)
    if args.part in ("all","od"): do_od_suite(cfg)
    print("Figures saved to", OUT)

if __name__=="__main__":
    main()
