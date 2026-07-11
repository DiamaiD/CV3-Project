import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_states, WORLD, GRAVITY, AIR_DRAG_COEFF
from experiments.rollout import load_run, test_split, traj_frames_np, rollout_frames_np
from src.dataset import FrameCache


def free_flight_samples(pos, valid, radius, mass, t_min=0, min_run=6,
                        margin_wall=1.5, margin_ball=2.0):
    T, N = pos.shape[:2]
    rows = []
    ok = valid.copy()

    vel = np.full_like(pos, np.nan)
    if T >= 3:
        vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0
    if T >= 2:
        vel[0] = pos[1] - pos[0]
        vel[-1] = pos[-1] - pos[-2]
    speed = np.nan_to_num(np.linalg.norm(vel, axis=2), nan=99.0)

    for i in range(N):
        r = radius[i]
        if not np.isfinite(r):
            ok[:, i] = False
            continue
        x, y = pos[:, i, 0], pos[:, i, 1]
        m = r + margin_wall + speed[:, i]
        with np.errstate(invalid="ignore"):
            ok[:, i] &= np.where(np.isfinite(x) & np.isfinite(y),
                                 (x >= m) & (x <= WORLD - m) &
                                 (y >= m) & (y <= WORLD - m), False)
        for j in range(N):
            if j == i:
                continue
            d = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
            relspeed = np.nan_to_num(np.linalg.norm(vel[:, i] - vel[:, j], axis=1), nan=99.0)
            near = np.where(np.isfinite(d),
                            d < radius[i] + radius[j] + margin_ball + relspeed, False)
            ok[near, i] = False

    n_seg = 0
    for i in range(N):
        t = t_min
        while t < T:
            if not ok[t, i]:
                t += 1
                continue
            t1 = t
            while t1 + 1 < T and ok[t1 + 1, i]:
                t1 += 1
            if t1 - t + 1 >= min_run:
                n_seg += 1
                p = pos[t:t1 + 1, i]
                d2 = p[2:] - 2 * p[1:-1] + p[:-2]
                vc = (p[2:] - p[:-2]) / 2.0
                for s in range(d2.shape[0]):
                    rows.append((d2[s, 0], d2[s, 1], vc[s, 0], vc[s, 1], radius[i], mass[i]))
            t = t1 + 1
    return np.array(rows), n_seg


def fit_gravity(S, iters=3, k_mad=6.0):
    n = len(S)
    wy = S[:, 3] * np.abs(S[:, 3]) * S[:, 4] / S[:, 5]
    wx = S[:, 2] * np.abs(S[:, 2]) * S[:, 4] / S[:, 5]
    A = np.concatenate([np.stack([np.ones(n), -wy], 1), np.stack([np.zeros(n), -wx], 1)])
    b = np.concatenate([S[:, 1], S[:, 0]])
    keep = np.ones(len(b), bool)
    sol = np.zeros(2)
    for _ in range(iters):
        sol, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        resid = A @ sol - b
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med)) + 1e-9
        keep = np.abs(resid - med) < k_mad * mad
    resid = A @ sol - b
    return float(sol[0]), float(sol[1]), float(np.sqrt((resid[keep] ** 2).mean())), float(keep.mean())


def summarize(name, S, n_seg, n_traj):
    if len(S) == 0:
        print(f"{name:>18}: no free-flight samples")
        return None
    g_med = float(np.median(S[:, 1]))
    g_mad = float(np.median(np.abs(S[:, 1] - g_med)))
    d2x_med = float(np.median(S[:, 0]))
    g_fit, c_fit, rms, inlier = fit_gravity(S)
    r = {"n_traj": n_traj, "n_segments": n_seg, "n_samples": int(len(S)),
         "g_median_d2y": g_med, "g_mad": g_mad, "d2x_median": d2x_med,
         "g_fit": g_fit, "drag_fit": c_fit, "fit_resid_rms": rms, "fit_inlier_frac": inlier}
    print(f"{name:>18}: g_med {g_med:+.4f} (MAD {g_mad:.4f}) | joint fit g {g_fit:+.4f}, "
          f"drag {c_fit:.4f} | d2x_med {d2x_med:+.5f} | resid {rms:.4f} "
          f"(inliers {inlier:.1%}) | {len(S)} samples / {n_seg} segs / {n_traj} trajs")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_11__08_36_56__bouncing_20k_b25_flow")
    ap.add_argument("--n-traj", type=int, default=300)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    ctx_len = cfg["context_len"]
    all_trajs, test_trajs = test_split(cfg["data_dir"], args.seed)
    test_trajs = test_trajs[:args.n_traj]
    print(f"Gravity recovery | run {os.path.basename(args.run)} | {len(test_trajs)} test trajs | "
          f"{args.steps}-step rollouts | true g {GRAVITY}, drag {AIR_DRAG_COEFF}")

    fc = FrameCache(all_trajs, cache_device="cpu",
                    disk_cache_path=os.path.join(cfg["data_dir"], "frames_cache.pt"))

    t0 = time.time()
    real_frames = [traj_frames_np(fc, td) for td in test_trajs]
    print(f"[1/4] Loaded {len(real_frames)} real trajs ({time.time() - t0:.0f}s)")

    S_gt, seg_gt, S_real, seg_real, S_model, seg_model = [], 0, [], 0, [], 0
    real_states = []
    t0 = time.time()
    for td, frames in zip(test_trajs, real_frames):
        st = extract_states(frames)
        real_states.append(st)
        S, ns = free_flight_samples(st["pos"], st["valid"], st["radius"], st["mass"])
        if len(S):
            S_real.append(S)
            seg_real += ns

        gt_pos = np.load(os.path.join(td, "positions.npy"))
        if gt_pos.shape[1] == st["pos"].shape[1]:
            from scipy.optimize import linear_sum_assignment
            cost = np.full((st["pos"].shape[1], gt_pos.shape[1]), 1e6)
            for i in range(st["pos"].shape[1]):
                d = np.linalg.norm(st["pos"][:, i, None] - gt_pos, axis=2)
                md = np.nanmean(np.where(st["valid"][:, i, None], d, np.nan), axis=0)
                cost[i] = np.where(np.isfinite(md), md, 1e6)
            rows, cols = linear_sum_assignment(cost)
            order = np.argsort(rows)
            gt_matched = gt_pos[:, cols[order]]
            S, ns = free_flight_samples(gt_matched, np.ones(gt_matched.shape[:2], bool),
                                        st["radius"], st["mass"])
            if len(S):
                S_gt.append(S)
                seg_gt += ns
    print(f"[2/4] Extracted real + GT controls ({time.time() - t0:.0f}s)")

    t0 = time.time()
    model_frames = rollout_frames_np(ae, dit, real_frames, ctx_len, args.steps,
                                     num_steps=args.num_steps, device=device)
    print(f"[3/4] Rolled out {len(model_frames)} x {args.steps} model frames ({time.time() - t0:.0f}s)")

    t0 = time.time()
    n_lost = 0
    for frames, mf, st_r in zip(real_frames, model_frames, real_states):
        seq = np.concatenate([frames[:ctx_len], mf], axis=0)
        st = extract_states(seq, inventory=st_r["inventory"])
        if st["valid"][ctx_len:].mean() < 0.2:
            n_lost += 1
            continue
        S, ns = free_flight_samples(st["pos"], st["valid"], st["radius"], st["mass"],
                                    t_min=ctx_len + 1)
        if len(S):
            S_model.append(S)
            seg_model += ns
    print(f"[4/4] Extracted model rollouts ({time.time() - t0:.0f}s)"
          + (f" | {n_lost} trajs dropped (track loss)" if n_lost else ""))

    print(f"\nSecond-difference gravity fit (true: g {GRAVITY}, drag {AIR_DRAG_COEFF}):")
    S_all = {"gt_npy": np.concatenate(S_gt) if S_gt else np.empty((0, 6)),
             "extractor_real": np.concatenate(S_real) if S_real else np.empty((0, 6)),
             "model_rollout": np.concatenate(S_model) if S_model else np.empty((0, 6))}
    results = {"run": args.run, "n_traj": len(test_trajs), "rollout_steps": args.steps,
               "true_g": GRAVITY, "true_drag": AIR_DRAG_COEFF,
               "gt_npy": summarize("GT positions.npy", S_all["gt_npy"], seg_gt, len(test_trajs)),
               "extractor_real": summarize("extractor on real", S_all["extractor_real"], seg_real, len(test_trajs)),
               "model_rollout": summarize("MODEL rollout", S_all["model_rollout"], seg_model, len(test_trajs) - n_lost)}

    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "gravity.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    make_plots(S_all, results, plot_dir)
    print(f"Saved to {out} + plots/")


def make_plots(S_all, results, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-1.0, 0.0, 80)
    styles = [("gt_npy", "ground-truth states", "#374151"),
              ("extractor_real", "extractor on real frames", "#0b8fa8"),
              ("model_rollout", "MODEL rollouts", "#c2410c")]
    for key, label, color in styles:
        S = S_all[key]
        if not len(S):
            continue
        r = results[key]
        ax.hist(np.clip(S[:, 1], -1, 0), bins=bins, density=True, histtype="step",
                lw=2, color=color, label=f"{label}  (median {r['g_median_d2y']:+.4f})")
    ax.axvline(GRAVITY, color="black", lw=1.2, ls="--", label="true g = -0.5")
    ax.set_xlabel("per-frame vertical acceleration  d²y  [px/frame²]")
    ax.set_ylabel("density")
    ax.set_title("Gravity read out of free flight: second differences of ball height", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "gravity_hist.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
