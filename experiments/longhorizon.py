import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_states, central_velocity, MATERIALS
from experiments.rollout import load_run, test_split, traj_frames_np, rollout_frames_np
from experiments.simkit import make_ball, simulate
from src.dataset import FrameCache

CTX = 5


def ball_stats(pos, vel, vok, radius, mass, T_end):
    T = min(pos.shape[0], T_end)
    N = pos.shape[1]
    E = np.full((T, N), np.nan)
    H = np.full((T, N), np.nan)
    rest = np.full((T, N), np.nan)
    for i in range(N):
        if not np.isfinite(radius[i]):
            continue
        y = pos[:T, i, 1] - radius[i]
        v2 = (vel[:T, i] ** 2).sum(1)
        ok = vok[:T, i] & np.isfinite(y)
        E[ok, i] = 0.5 * mass[i] * v2[ok] + mass[i] * 0.5 * np.maximum(y[ok], 0.0)
        H[ok, i] = np.maximum(y[ok], 0.0)
        rest[ok, i] = ((np.sqrt(v2[ok]) < 0.35) & (y[ok] < 0.8)).astype(float)
    return E, H, rest


def pool_curves(per_traj, T_end):
    E_num = np.zeros(T_end)
    E_den = np.zeros(T_end)
    H_sum = np.zeros(T_end)
    H_n = np.zeros(T_end)
    R_sum = np.zeros(T_end)
    R_n = np.zeros(T_end)
    for E, H, rest, E0 in per_traj:
        T, N = E.shape
        for i in range(N):
            e0 = E0[i]
            if not np.isfinite(e0) or e0 <= 0:
                continue
            ok = np.isfinite(E[:, i])
            E_num[:T][ok] += E[ok, i]
            E_den[:T][ok] += e0
            H_sum[:T][ok] += H[ok, i]
            H_n[:T][ok] += 1
            R_sum[:T][ok] += rest[ok, i]
            R_n[:T][ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return (E_num / E_den, H_sum / np.maximum(H_n, 1e-9), R_sum / np.maximum(R_n, 1e-9), H_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_11__08_36_56__bouncing_20k_b25_flow")
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import wasserstein_distance
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    all_trajs, test_trajs = test_split(cfg["data_dir"], args.seed)
    test_trajs = test_trajs[:args.n_traj]
    T_end = CTX + args.steps
    print(f"Long-horizon fidelity | {len(test_trajs)} trajs x {args.steps} steps "
          f"(training horizon was 100 frames)")

    fc = FrameCache(all_trajs, cache_device="cpu",
                    disk_cache_path=os.path.join(cfg["data_dir"], "frames_cache.pt"))
    real_frames = [traj_frames_np(fc, td) for td in test_trajs]

    t0 = time.time()
    sim_per_traj, model_inputs = [], []
    speeds_sim = {}
    for td, frames in zip(test_trajs, real_frames):
        st = extract_states(frames)
        gt_pos = np.load(os.path.join(td, "positions.npy"))
        gt_vel = np.load(os.path.join(td, "velocities.npy"))
        if gt_pos.shape[1] != st["pos"].shape[1]:
            continue
        cost = np.full((st["pos"].shape[1], gt_pos.shape[1]), 1e6)
        for i in range(st["pos"].shape[1]):
            d = np.linalg.norm(st["pos"][:, i, None] - gt_pos[:100], axis=2)
            md = np.nanmean(np.where(st["valid"][:, i, None], d, np.nan), axis=0)
            cost[i] = np.where(np.isfinite(md), md, 1e6)
        rows, cols = linear_sum_assignment(cost)
        balls = []
        bad = False
        track_of_gt = {int(c): int(r) for r, c in zip(rows, cols)}
        for j in range(gt_pos.shape[1]):
            tr = track_of_gt.get(j)
            if tr is None or not np.isfinite(st["radius"][tr]):
                bad = True
                break
            balls.append(make_ball(st["mats"][tr], st["radius"][tr],
                                   gt_pos[CTX - 1, j, 0], gt_pos[CTX - 1, j, 1],
                                   gt_vel[CTX - 1, j, 0], gt_vel[CTX - 1, j, 1]))
        if bad:
            continue
        pos, vel, _, _, _ = simulate(balls, args.steps + 1, render=False)
        radius = np.array([b["radius"] for b in balls], float)
        mass = np.array([b["mass"] for b in balls])
        full_pos = np.full((T_end, len(balls), 2), np.nan)
        full_vel = np.full((T_end, len(balls), 2), np.nan)
        full_pos[CTX - 1:] = pos[:T_end - CTX + 1]
        full_vel[CTX - 1:] = vel[:T_end - CTX + 1]
        vok = np.isfinite(full_vel[:, :, 0])
        E, H, rest = ball_stats(full_pos, full_vel, vok, radius, mass, T_end)
        E0 = E[CTX]
        sim_per_traj.append((E, H, rest, E0))
        for tq in (30, 150, args.steps - 2):
            t = CTX + tq
            if t < T_end:
                speeds_sim.setdefault(tq, []).append(np.linalg.norm(full_vel[t], axis=1))
        model_inputs.append((td, frames, st, E0))
    print(f"[1/3] Sim continuations for {len(sim_per_traj)} trajs ({time.time() - t0:.0f}s)")

    t0 = time.time()
    mf_all = rollout_frames_np(ae, dit, [m[1] for m in model_inputs], CTX, args.steps,
                               device=device, batch=64)
    print(f"[2/3] Rolled out {len(mf_all)} x {args.steps} model frames ({time.time() - t0:.0f}s)")

    t0 = time.time()
    model_per_traj = []
    integ_sum = np.zeros(T_end)
    integ_n = np.zeros(T_end)
    integ_real_sum = np.zeros(100)
    integ_real_n = np.zeros(100)
    speeds_model = {}
    for (td, frames, st_r, E0_sim), mf in zip(model_inputs, mf_all):
        seq = np.concatenate([frames[:CTX], mf], axis=0)
        st = extract_states(seq, inventory=st_r["inventory"])
        present = st["valid"] | st["merged"]
        Tm, N = present.shape
        integ_sum[:Tm] += present.sum(1)
        integ_n[:Tm] += N
        pr = st_r["valid"] | st_r["merged"]
        integ_real_sum[:pr.shape[0]] += pr.sum(1)[:100]
        integ_real_n[:pr.shape[0]] += pr.shape[1]
        vel, vok = central_velocity(st["pos"], st["valid"])
        E, H, rest = ball_stats(st["pos"], vel, vok, st["radius"], st["mass"], T_end)
        E0 = E[CTX + 1]
        model_per_traj.append((E, H, rest, E0))
        for tq in (30, 150, args.steps - 2):
            t = CTX + tq
            if t < Tm:
                sp = np.linalg.norm(vel[t], axis=1)
                speeds_model.setdefault(tq, []).append(sp[vok[t]])
    print(f"[3/3] Extracted model rollouts ({time.time() - t0:.0f}s)")

    E_sim, H_sim, R_sim, n_sim = pool_curves(sim_per_traj, T_end)
    E_mod, H_mod, R_mod, n_mod = pool_curves(model_per_traj, T_end)
    integrity = integ_sum / np.maximum(integ_n, 1e-9)
    integ_real = integ_real_sum / np.maximum(integ_real_n, 1e-9)

    wd = {}
    for tq in speeds_sim:
        s = np.concatenate(speeds_sim[tq])
        m = np.concatenate(speeds_model.get(tq, [np.array([])]))
        m = m[np.isfinite(m)]
        if m.size and s.size:
            wd[str(tq)] = {"wasserstein_px_per_frame": float(wasserstein_distance(s, m)),
                           "sim_mean": float(s.mean()), "model_mean": float(m.mean()),
                           "n_sim": int(s.size), "n_model": int(m.size)}

    Tc = T_end - 2
    ts = np.arange(Tc)
    results = {
        "run": args.run, "n_traj": len(sim_per_traj), "steps": args.steps,
        "energy_ratio": {"t": ts.tolist(), "sim": np.nan_to_num(E_sim[:Tc]).tolist(),
                         "model": np.nan_to_num(E_mod[:Tc]).tolist()},
        "mean_height": {"sim": np.nan_to_num(H_sim[:Tc]).tolist(),
                        "model": np.nan_to_num(H_mod[:Tc]).tolist()},
        "resting_frac": {"sim": np.nan_to_num(R_sim[:Tc]).tolist(),
                         "model": np.nan_to_num(R_mod[:Tc]).tolist()},
        "integrity": {"model": integrity[:Tc].tolist(), "real_baseline": integ_real.tolist()},
        "speed_wasserstein": wd,
        "summary": {
            "integrity_at_100": float(integrity[min(100, T_end - 1)]),
            "integrity_at_300": float(integrity[T_end - 1]),
            "energy_ratio_model_at_150": float(np.nan_to_num(E_mod)[min(CTX + 150, T_end - 1)]),
            "energy_ratio_sim_at_150": float(np.nan_to_num(E_sim)[min(CTX + 150, T_end - 1)]),
        },
    }
    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "longhorizon.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    make_plots(results, speeds_sim, speeds_model, plot_dir, args.steps)

    print(f"\nIntegrity (all balls tracked): t=100 {results['summary']['integrity_at_100']:.1%} | "
          f"t={T_end - 1} {results['summary']['integrity_at_300']:.1%}")
    for tq, v in sorted(wd.items(), key=lambda kv: int(kv[0])):
        print(f"Speed dist at t=+{tq}: sim mean {v['sim_mean']:.2f} vs model {v['model_mean']:.2f} "
              f"px/f | Wasserstein {v['wasserstein_px_per_frame']:.3f}")
    print(f"Saved to {out_dir}/longhorizon.json + plots/")


def make_plots(results, speeds_sim, speeds_model, plot_dir, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = CTX + 1
    ts = np.array(results["energy_ratio"]["t"])[t0:]
    for key in ("energy_ratio", "mean_height", "resting_frac"):
        results[key]["sim"] = results[key]["sim"][t0:]
        results[key]["model"] = results[key]["model"][t0:]
    results["integrity"]["model"] = results["integrity"]["model"][t0:]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(ts, results["energy_ratio"]["sim"], color="#0b8fa8", label="simulator")
    ax.plot(ts, results["energy_ratio"]["model"], color="#c2410c", label="model rollout")
    ax.set_title("Mechanical energy (pooled, normalized to t=5)", fontsize=10)
    ax.set_ylabel("E(t) / E(5)")
    ax.set_ylim(0, 1.15)

    ax = axes[0, 1]
    ax.plot(ts, results["mean_height"]["sim"], color="#0b8fa8", label="simulator")
    ax.plot(ts, results["mean_height"]["model"], color="#c2410c", label="model rollout")
    ax.set_title("Mean height above floor", fontsize=10)
    ax.set_ylabel("<y - r>  [px]")

    ax = axes[1, 0]
    ax.plot(ts, results["resting_frac"]["sim"], color="#0b8fa8", label="simulator")
    ax.plot(ts, results["resting_frac"]["model"], color="#c2410c", label="model rollout")
    ax.set_title("Fraction of balls at rest on the floor", fontsize=10)
    ax.set_ylabel("resting fraction")
    ax.set_ylim(0, 1.02)

    ax = axes[1, 1]
    ax.plot(np.arange(t0, len(results["integrity"]["real_baseline"])),
            results["integrity"]["real_baseline"][t0:], color="#0b8fa8",
            label="extractor on real frames (baseline)")
    ax.plot(ts, results["integrity"]["model"], color="#c2410c", label="model rollout")
    ax.set_title("Ball integrity: fraction of balls still cleanly tracked", fontsize=10)
    ax.set_ylabel("tracked fraction")
    ax.set_ylim(0, 1.02)

    for ax in axes.ravel():
        ax.axvline(100, color="gray", lw=1, ls=":")
        ax.set_xlabel("frame (training data ends at 100)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Long-horizon statistical fidelity: {results['n_traj']} rollouts x {steps} steps",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "longhorizon_curves.png"), dpi=140)
    plt.close(fig)

    if not speeds_sim:
        return
    tqs = sorted(speeds_sim.keys())
    fig, axes = plt.subplots(1, len(tqs), figsize=(4 * len(tqs), 3.4), sharey=True)
    for ax, tq in zip(np.atleast_1d(axes), tqs):
        s = np.concatenate(speeds_sim[tq])
        m = np.concatenate(speeds_model.get(tq, [np.array([])]))
        m = m[np.isfinite(m)]
        bins = np.linspace(0, max(6.0, np.percentile(s, 99)), 40)
        ax.hist(s, bins=bins, density=True, alpha=0.55, color="#0b8fa8", label="simulator")
        if m.size:
            ax.hist(m, bins=bins, density=True, alpha=0.55, color="#c2410c", label="model")
        ax.set_title(f"ball speeds at t = 5+{tq}", fontsize=10)
        ax.set_xlabel("|v|  [px/frame]")
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("density")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "longhorizon_speeds.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
