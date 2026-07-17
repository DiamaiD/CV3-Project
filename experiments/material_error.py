"""Per-material 1-step prediction error, ball-region pixel PSNR.

For each target frame in held-out benchmark trajectories: predict frame t from
the GT context [t-CTX, t), decode to pixels, and score each ball's region
(box r+2, as in latent_analysis part A) against the ground-truth frame --
grouped by the ball's MATERIAL x SITUATION (free flight / wall / bb contact).

Answers "which material is hardest for the model" at the pixel level;
complements probes.py, which answers it at the physical-constants level.

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

MAT_NAMES = ["Superball", "Rubber", "Steel", "Sponge"]
SITS = ["free", "wall", "bb contact"]


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
    print(f"Material error | run {os.path.basename(args.run.rstrip('/'))} | "
          f"data {args.data} | {len(test)} held-out trajs | 1-step, ctx {ctx_len}")

    # load frames + meta
    trajs = []
    for td in test:
        pos = np.load(os.path.join(td, "positions.npy"))
        meta = load_traj_meta(td, pos[0])
        if meta is None:
            continue
        radii, mats = meta
        frames = np.stack([
            cv2.cvtColor(cv2.imread(os.path.join(td, f"frame_{t:03d}.png")), cv2.COLOR_BGR2RGB)
            for t in range(pos.shape[0])])
        trajs.append((frames, pos, radii, mats))
    windows = [(ti, t) for ti, (f, *_ ) in enumerate(trajs)
               for t in range(ctx_len, f.shape[0], args.stride)]
    print(f"[data] {len(trajs)} trajs -> {len(windows)} prediction windows")

    # batched 1-step predictions
    mse_by = {m: {s: [] for s in SITS} for m in MAT_NAMES}
    with torch.no_grad():
        for s0 in range(0, len(windows), args.batch):
            chunk = windows[s0:s0 + args.batch]
            ctx = np.stack([trajs[ti][0][t - ctx_len:t] for ti, t in chunk])
            ctx_t = torch.from_numpy(ctx).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
            pred = rollout(ae, dit, ctx_t, n_steps=1, num_steps=args.num_steps)
            pred = pred[:, 0].permute(0, 2, 3, 1).numpy()  # (B, H, W, 3) uint8
            for (ti, t), pr in zip(chunk, pred):
                frames, pos, radii, mats = trajs[ti]
                p = pos[t]
                N = len(radii)
                for i in range(N):
                    x, y, r = p[i, 0], p[i, 1], radii[i]
                    near_wall = (x - r < 1.0) or (x + r > 63.0) or (y - r < 1.0) or (y + r > 63.0)
                    in_contact = any(np.linalg.norm(p[i] - p[j]) < radii[i] + radii[j] + 1.0
                                     for j in range(N) if j != i)
                    col, row = x - 0.5, 63.5 - y
                    c0, c1 = int(max(col - r - 2, 0)), int(min(col + r + 3, 64))
                    r0, r1 = int(max(row - r - 2, 0)), int(min(row + r + 3, 64))
                    if c1 <= c0 or r1 <= r0:
                        continue
                    gt = frames[t][r0:r1, c0:c1].astype(np.float64)
                    pd = pr[r0:r1, c0:c1].astype(np.float64)
                    sit = "bb contact" if in_contact else ("wall" if near_wall else "free")
                    mse_by[mats[i]][sit].append(((gt - pd) ** 2).mean())
            if (s0 // args.batch) % 20 == 0:
                print(f"  {s0 + len(chunk)}/{len(windows)} windows")

    def psnr(v):
        return 10 * np.log10(255.0 ** 2 / np.mean(v)) if v else float("nan")

    out = {"run": args.run, "data": args.data, "n_traj": len(trajs), "stride": args.stride,
           "by_material": {}}
    print(f"\n1-step prediction, ball-region PSNR (dB) by material x situation:")
    hdr = f"{'material':10s} | " + " | ".join(f"{s:>12s}" for s in SITS) + " |      overall"
    print(hdr)
    for m in MAT_NAMES:
        cells = {s: {"psnr": psnr(mse_by[m][s]), "n": len(mse_by[m][s])} for s in SITS}
        allv = [v for s in SITS for v in mse_by[m][s]]
        cells["overall"] = {"psnr": psnr(allv), "n": len(allv)}
        out["by_material"][m] = cells
        print(f"{m:10s} | " + " | ".join(f"{cells[s]['psnr']:6.2f} n={cells[s]['n']:<5d}"[:12].rjust(12) for s in SITS)
              + f" | {cells['overall']['psnr']:6.2f} n={cells['overall']['n']}")

    os.makedirs(os.path.join(args.run, "experiments"), exist_ok=True)
    with open(os.path.join(args.run, "experiments", "material_error.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.run}/experiments/material_error.json")


if __name__ == "__main__":
    main()
