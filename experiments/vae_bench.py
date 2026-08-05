"""VAE reconstruction bench for the pymunk-era datasets (balls / shapes / tower).

Round-trips held-out frames through an autoencoder (deterministic mu path) and
scores, per dataset:

  frame PSNR / LPIPS   -- whole-frame fidelity
  object PSNR          -- pooled over each object's own box (r+2), background-free
  contact / free PSNR  -- object PSNR split by bounding-circle contact
  dot PSNR             -- pooled over small windows at the EXACT orientation-dot
                          positions computed from ground-truth state (positions +
                          angles + objects.json), the same formulas _draw uses.
                          This is the rotation-era state-precision metric: a VAE
                          that blurs the dots erases orientation from the latent.

Ground truth comes from the recorded state, so no extractor in the loop.

  python -m experiments.vae_bench --ae runs/<run> \
      [--datasets data/balls_20k,data/shapes_20k,data/tower_20k] [--n-traj 150]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from experiments.rollout import test_split
from experiments.surrogate import load_ae
from src.frameio import load_frames

DOT_BALL = ((0.0, 0.55, 1.4), (2.0 * np.pi / 3.0, 0.38, 1.0))   # (dtheta, frac, r)
DOT_POLY = ((0, 0.55, 1.3), (1, 0.35, 0.9))                     # (vertex, frac, r)


def _psnr(mse):
    return float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")


def _dot_centers(obj, x, y, theta):
    """World-space orientation-dot centers for one object, mirroring _draw."""
    if obj["verts"] is None:
        if obj["kind"] != "ball":
            return []
        return [(x + np.cos(theta + dt) * f * obj["size"],
                 y + np.sin(theta + dt) * f * obj["size"], r)
                for dt, f, r in DOT_BALL]
    if obj["kind"] == "halfdisc":
        return []
    v = np.asarray(obj["verts"])
    c, s = np.cos(theta), np.sin(theta)
    world = v @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    out = []
    for vi, f, r in DOT_POLY:
        vx, vy = world[vi % len(world)]
        out.append((x + f * (vx - x), y + f * (vy - y), r))
    return out


def _box(cx, cy, r, H=64, W=64):
    col, row = cx - 0.5, (H - 0.5) - cy
    c0, c1 = int(max(col - r, 0)), int(min(col + r + 1, W))
    r0, r1 = int(max(row - r, 0)), int(min(row + r + 1, H))
    return (r0, r1, c0, c1) if (c1 > c0 and r1 > r0) else None


def bench_dataset(ae, data_dir, n_traj, device, lpips_fn, batch=256):
    _, test = test_split(data_dir)
    test = sorted(test)[:n_traj]
    sums = {k: [0.0, 0] for k in ["frame", "object", "contact", "free", "dot"]}
    lp_sum, lp_n = 0.0, 0

    for td in test:
        pos = np.load(os.path.join(td, "positions.npy"))
        ang = np.load(os.path.join(td, "angles.npy"))[:, :, 0]
        objs = json.load(open(os.path.join(td, "objects.json")))
        T = pos.shape[0]
        frames = load_frames(td)[:T]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        rec = torch.empty_like(x)
        with torch.no_grad():
            for i in range(0, T, batch):
                xb = x[i:i + batch].to(device)
                rb = ae.decode(ae.encode(xb))
                rec[i:i + batch] = rb.cpu()
                if lpips_fn is not None:
                    lp_sum += float(lpips_fn(xb * 2 - 1, rb * 2 - 1).sum())
                    lp_n += xb.shape[0]
        err = ((x - rec) ** 2).numpy()                       # (T, 3, H, W)
        err_hw = err.mean(axis=1)                            # (T, H, W)

        sums["frame"][0] += float(err.mean()) * T
        sums["frame"][1] += T

        sizes = [o["size"] for o in objs]
        for t in range(T):
            touching = set()
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    if np.hypot(*(pos[t, i] - pos[t, j])) < sizes[i] + sizes[j] + 1.0:
                        touching.update((i, j))
            for i, o in enumerate(objs):
                b = _box(pos[t, i, 0], pos[t, i, 1], o["size"] + 2)
                if b is None:
                    continue
                r0, r1, c0, c1 = b
                m = float(err_hw[t, r0:r1, c0:c1].mean())
                sums["object"][0] += m; sums["object"][1] += 1
                key = "contact" if i in touching else "free"
                sums[key][0] += m; sums[key][1] += 1
                for dx, dy, dr in _dot_centers(o, pos[t, i, 0], pos[t, i, 1], ang[t, i]):
                    db = _box(dx, dy, dr + 1.0)
                    if db is None:
                        continue
                    q0, q1, p0, p1 = db
                    sums["dot"][0] += float(err_hw[t, q0:q1, p0:p1].mean())
                    sums["dot"][1] += 1

    out = {k: {"psnr": _psnr(v[0] / v[1]) if v[1] else float("nan"), "n": v[1]}
           for k, v in sums.items()}
    out["lpips"] = lp_sum / lp_n if lp_n else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True, help="run dir with run_config.json + autoencoder.pth")
    ap.add_argument("--datasets", default="data/balls_20k,data/shapes_20k,data/tower_20k")
    ap.add_argument("--n-traj", type=int, default=150)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, cfg = load_ae(args.ae, device)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    except Exception:
        lpips_fn = None
        print("[warn] lpips unavailable, skipping")

    results = {"ae": args.ae, "n_traj": args.n_traj, "datasets": {}}
    hdr = f"{'dataset':12s} | {'frame':>6s} | {'object':>6s} | {'contact':>7s} | {'free':>6s} | {'dot':>6s} | {'lpips':>6s}"
    print(f"\nVAE bench | {os.path.basename(args.ae.rstrip('/'))} | mu round-trip, PSNR dB\n{hdr}")
    for d in args.datasets.split(","):
        d = d.strip()
        r = bench_dataset(ae, d, args.n_traj, device, lpips_fn)
        results["datasets"][os.path.basename(d)] = r
        print(f"{os.path.basename(d):12s} | {r['frame']['psnr']:6.2f} | {r['object']['psnr']:6.2f} | "
              f"{r['contact']['psnr']:7.2f} | {r['free']['psnr']:6.2f} | {r['dot']['psnr']:6.2f} | "
              f"{r['lpips']:6.4f}")

    tag = args.tag or os.path.basename(args.ae.rstrip("/"))
    os.makedirs("experiments/results", exist_ok=True)
    out = f"experiments/results/vae_bench_{tag}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
