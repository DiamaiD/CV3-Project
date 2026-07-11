import os
import sys
import glob
import json
import time
import random
import argparse
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_states, central_velocity, MAT_NAMES


def load_frames(traj_dir):
    paths = sorted(glob.glob(os.path.join(traj_dir, "frame_*.png")))
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bouncing_20k_b25")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/results/extractor_validation.json")
    args = ap.parse_args()

    random.seed(args.seed)
    all_trajs = glob.glob(os.path.join(args.data, "traj-*"))
    random.shuffle(all_trajs)
    n_train, n_val = int(len(all_trajs) * 0.8), int(len(all_trajs) * 0.1)
    test_trajs = all_trajs[n_train + n_val:][:args.n]
    print(f"Validating extractor on {len(test_trajs)} test trajectories from {args.data}")

    pos_err_clean, pos_err_merged, vel_err_cd, vel_err_inst, gt_disc = [], [], [], [], []
    mat_err = {m: [] for m in MAT_NAMES}
    radius_dev, inv_ok, inv_bad = [], 0, []
    n_ball_frames = n_valid = n_merged = n_coast = 0
    t0 = time.time()

    for ti, td in enumerate(test_trajs):
        frames = load_frames(td)
        gt_pos = np.load(os.path.join(td, "positions.npy"))
        gt_vel = np.load(os.path.join(td, "velocities.npy"))
        T, Ng = gt_pos.shape[:2]

        st = extract_states(frames)
        P, V, M = st["pos"], st["valid"], st["merged"]
        N = P.shape[1]
        if N == Ng:
            inv_ok += 1
        else:
            inv_bad.append((os.path.basename(td), N, Ng))

        cost = np.full((N, Ng), 1e6)
        for i in range(N):
            d = np.linalg.norm(P[:, i, None] - gt_pos, axis=2)
            mean_d = np.nanmean(np.where(V[:, i, None], d, np.nan), axis=0)
            cost[i] = np.where(np.isfinite(mean_d), mean_d, 1e6)
        rows, cols = linear_sum_assignment(cost)

        gt_cd = np.full_like(gt_pos, np.nan)
        gt_cd[1:-1] = (gt_pos[2:] - gt_pos[:-2]) / 2.0
        vel, vok = central_velocity(P, V)
        gt_disc.append(np.linalg.norm(gt_cd[1:-1] - gt_vel[1:-1], axis=2).ravel())

        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e6:
                continue
            d = np.linalg.norm(P[:, i] - gt_pos[:, j], axis=1)
            pos_err_clean.append(d[V[:, i]])
            mat_err[st["mats"][i]].append(d[V[:, i]])
            mm = M[:, i] & np.isfinite(d)
            if mm.any():
                pos_err_merged.append(d[mm])
            vm = vok[:, i] & np.isfinite(gt_cd[:, j, 0])
            if vm.any():
                vel_err_cd.append(np.linalg.norm(vel[vm][:, i] - gt_cd[vm][:, j], axis=1))
                vel_err_inst.append(np.linalg.norm(vel[vm][:, i] - gt_vel[vm][:, j], axis=1))
            r = st["radius"][i]
            if np.isfinite(r):
                radius_dev.append(abs(r - round(r)))

        n_ball_frames += T * N
        n_valid += int(V.sum())
        n_merged += int(M.sum())
        n_coast += int((~V & ~M).sum())
        if (ti + 1) % 25 == 0 or ti + 1 == len(test_trajs):
            print(f"  {ti + 1}/{len(test_trajs)} trajs | {time.time() - t0:.0f}s", flush=True)

    def stats(chunks):
        if not chunks:
            return None
        e = np.concatenate(chunks)
        return {"n": int(e.size), "rmse": float(np.sqrt((e ** 2).mean())),
                "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
                "max": float(e.max())}

    results = {
        "data": args.data, "n_trajs": len(test_trajs),
        "inventory_correct": inv_ok, "inventory_mismatches": inv_bad,
        "ball_frames": n_ball_frames,
        "rate_clean": n_valid / n_ball_frames,
        "rate_merged": n_merged / n_ball_frames,
        "rate_coasted": n_coast / n_ball_frames,
        "pos_err_px_clean": stats(pos_err_clean),
        "pos_err_px_merged_split": stats(pos_err_merged),
        "pos_err_px_by_material": {m: stats(v) for m, v in mat_err.items() if v},
        "vel_err_vs_gt_central_diff": stats(vel_err_cd),
        "vel_err_vs_gt_instantaneous": stats(vel_err_inst),
        "gt_central_diff_vs_instantaneous": stats(gt_disc),
        "radius_dev_from_int": stats([np.array(radius_dev)]) if radius_dev else None,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nInventory correct: {inv_ok}/{len(test_trajs)} trajs"
          + (f"  (mismatches: {inv_bad})" if inv_bad else ""))
    print(f"Ball-frames: {n_ball_frames} | clean {results['rate_clean']:.1%} | "
          f"merged-split {results['rate_merged']:.1%} | coasted {results['rate_coasted']:.1%}")
    for label, key in [("Position err (clean) [px]", "pos_err_px_clean"),
                       ("Position err (merged-split) [px]", "pos_err_px_merged_split"),
                       ("Velocity err vs GT c-diff [px/f]", "vel_err_vs_gt_central_diff"),
                       ("Velocity err vs GT instant [px/f]", "vel_err_vs_gt_instantaneous"),
                       ("GT c-diff vs instant (no extractor)", "gt_central_diff_vs_instantaneous")]:
        s = results[key]
        if s:
            print(f"{label:>36}: rmse {s['rmse']:.4f} | median {s['median']:.4f} | "
                  f"p95 {s['p95']:.4f} | max {s['max']:.4f}")
    print(f"{'Per-material pos rmse [px]':>36}: " + " | ".join(
        f"{m} {results['pos_err_px_by_material'][m]['rmse']:.4f}"
        for m in MAT_NAMES if m in results["pos_err_px_by_material"]))
    if results["radius_dev_from_int"]:
        print(f"{'Track radius dev from integer':>36}: median {results['radius_dev_from_int']['median']:.4f} | "
              f"max {results['radius_dev_from_int']['max']:.4f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
