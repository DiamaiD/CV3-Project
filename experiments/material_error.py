"""1-step prediction error on each ball's own pixels, by event AND by material.

For each target frame in held-out benchmark trajectories: predict frame t from
the GT context [t-CTX, t), decode to pixels, and score each ball's region
(box r+2, as in latent_analysis part A) against the ground-truth frame. Every
scored ball is classified into the same four exclusive event classes as the
canonical frame labels in src/eval (velocity-kick test, identical thresholds):
ball-ball contact > post ball-ball (1-3f) > wall bounce > free flight.

One measurement, two views: the by-event margin and the material x event matrix
share windows, regions and statistics, so the dashboard's "error by event" and
"error by material" tables are row/column sums of the same experiment and can
never disagree. Ball-region errors are naturally larger than whole-frame
numbers -- no white background dilutes them.

  python -m experiments.material_error --run runs/<run> [--data data/bench_v2_b25]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from experiments.latent_analysis import load_traj_meta
from experiments.rollout import load_run, rollout, test_split
from src.frameio import load_frames

MAT_NAMES = ["Superball", "Rubber", "Steel", "Sponge"]
EVENTS = ["free flight", "wall bounce", "post ball-ball (1-3f)", "ball-ball contact"]


def _ball_event_sets(pos, vel, gravity=-0.5, kick_thresh=1.5, pair_dist=18.0):
    """Per-transition kick sets, same thresholds as src.eval._traj_event_labels:
    hit[t] = balls whose velocity change at t->t+1 exceeds kick_thresh beyond
    gravity; paired[t] = the subset explained by a nearby second kicked ball."""
    T = len(pos)
    dv = vel[1:T] - vel[:T - 1]
    dv[:, :, 1] -= gravity
    mag = np.linalg.norm(dv, axis=2)
    hit_sets, pair_sets = [], []
    for t in range(T - 1):
        hit = np.where(mag[t] > kick_thresh)[0]
        paired = set()
        for a in range(len(hit)):
            for b in range(a + 1, len(hit)):
                i, j = hit[a], hit[b]
                if np.linalg.norm(pos[t, i] - pos[t, j]) < pair_dist:
                    paired.update((i, j))
        hit_sets.append(set(hit.tolist()))
        pair_sets.append(paired)
    return hit_sets, pair_sets


def _ball_event(i, f, hit_sets, pair_sets, post_frames=3):
    t = f - 1
    if t >= len(pair_sets):
        return "free flight"
    if i in pair_sets[t]:
        return "ball-ball contact"
    if any(i in pair_sets[u] for u in range(max(0, t - post_frames), t)):
        return "post ball-ball (1-3f)"
    if i in hit_sets[t]:
        return "wall bounce"
    return "free flight"


def _agg(errs):
    """errs = per-ball region MSE on the 0-1 scale."""
    if not errs:
        return {"n": 0, "px_mean": float("nan"), "px_std": float("nan"),
                "mse": float("nan"), "psnr": float("nan")}
    e = np.asarray(errs, np.float64)
    px = np.sqrt(e) * 255.0
    return {"n": int(e.size), "px_mean": float(px.mean()), "px_std": float(px.std()),
            "mse": float(e.mean()), "psnr": float(10.0 * np.log10(1.0 / e.mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/bench_v2_b25")
    ap.add_argument("--n-traj", type=int, default=150)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    ctx_len = cfg["context_len"]
    _, test = test_split(args.data)
    test = sorted(test)[: args.n_traj]
    print(f"Ball-region error matrix | run {os.path.basename(args.run.rstrip('/'))} | "
          f"data {args.data} | {len(test)} held-out trajs | 1-step, ctx {ctx_len}")

    trajs = []
    for td in test:
        pos = np.load(os.path.join(td, "positions.npy"))
        vel = np.load(os.path.join(td, "velocities.npy"))
        meta = load_traj_meta(td, pos[0])
        if meta is None:
            continue
        radii, mats = meta
        frames = load_frames(td)[:pos.shape[0]]
        trajs.append((frames, pos, radii, mats, *_ball_event_sets(pos, vel)))
    windows = [(ti, t) for ti, (f, *_ ) in enumerate(trajs)
               for t in range(ctx_len, f.shape[0], args.stride)]
    print(f"[data] {len(trajs)} trajs -> {len(windows)} prediction windows")

    mse_by = {m: {s: [] for s in EVENTS} for m in MAT_NAMES}
    with torch.no_grad():
        for s0 in range(0, len(windows), args.batch):
            chunk = windows[s0:s0 + args.batch]
            ctx = np.stack([trajs[ti][0][t - ctx_len:t] for ti, t in chunk])
            ctx_t = torch.from_numpy(ctx).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
            pred = rollout(ae, dit, ctx_t, n_steps=1, num_steps=args.num_steps)
            pred = pred[:, 0].permute(0, 2, 3, 1).numpy()  # (B, H, W, 3) uint8
            for (ti, t), pr in zip(chunk, pred):
                frames, pos, radii, mats, hit_sets, pair_sets = trajs[ti]
                p = pos[t]
                for i in range(len(radii)):
                    x, y, r = p[i, 0], p[i, 1], radii[i]
                    col, row = x - 0.5, 63.5 - y
                    c0, c1 = int(max(col - r - 2, 0)), int(min(col + r + 3, 64))
                    r0, r1 = int(max(row - r - 2, 0)), int(min(row + r + 3, 64))
                    if c1 <= c0 or r1 <= r0:
                        continue
                    gt = frames[t][r0:r1, c0:c1].astype(np.float64) / 255.0
                    pd = pr[r0:r1, c0:c1].astype(np.float64) / 255.0
                    ev = _ball_event(i, t, hit_sets, pair_sets)
                    mse_by[mats[i]][ev].append(((gt - pd) ** 2).mean())
            if (s0 // args.batch) % 20 == 0:
                print(f"  {s0 + len(chunk)}/{len(windows)} windows")

    out = {"run": args.run, "data": args.data, "n_traj": len(trajs),
           "stride": args.stride, "by_material": {}, "events": {}}
    for m in MAT_NAMES:
        cells = {s: _agg(mse_by[m][s]) for s in EVENTS}
        cells["overall"] = _agg([v for s in EVENTS for v in mse_by[m][s]])
        out["by_material"][m] = cells
    for s in EVENTS:
        out["events"][s] = _agg([v for m in MAT_NAMES for v in mse_by[m][s]])

    print("\nBall-region px error (mean), material x event:")
    print(f"{'material':10s} | " + " | ".join(f"{s:>22s}" for s in EVENTS) + " |  overall")
    for m in MAT_NAMES:
        c = out["by_material"][m]
        print(f"{m:10s} | " + " | ".join(
            f"{c[s]['px_mean']:6.2f} n={c[s]['n']:<6d}"[:22].rjust(22) for s in EVENTS)
            + f" | {c['overall']['px_mean']:6.2f}")
    print("events margin: " + ", ".join(
        f"{s} {out['events'][s]['px_mean']:.2f}px n={out['events'][s]['n']}" for s in EVENTS))

    os.makedirs(os.path.join(args.run, "experiments"), exist_ok=True)
    with open(os.path.join(args.run, "experiments", "material_error.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.run}/experiments/material_error.json")


if __name__ == "__main__":
    main()
