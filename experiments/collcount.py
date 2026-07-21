"""Per-ball-count collision eval: 1-step pixel error at collision frames, by ball count.

For each ball count, spawn fresh simkit scenes and take every frame the simulator
flags as a contact (ball-ball OR wall) as a "collision" target. Predict that frame
1-step from its GT context, decode to pixels, and score full-frame pixel MSE. Report
the error distribution (mean / median / variance) and PSNR per count, with a
non-collision ("free") reference. 1-ball scenes have no ball-ball contact, so their
"collision" bucket is wall bounces only.

  python -m experiments.collcount --run runs/<run> --counts 1,2,3,4,5,6,7,8
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from experiments.rollout import load_run, rollout
from experiments.simkit import simulate
from experiments.ood import spawn_balls


def _psnr(mse):
    return float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")


def _stats(errs):
    if len(errs) == 0:
        return None
    e = np.asarray(errs, np.float64)          # per-window full-frame MSE (0-1 scale)
    px = np.sqrt(e) * 255.0                    # per-window RMS pixel error (0-255)
    return {"n": int(e.size), "psnr": _psnr(float(e.mean())),
            "px_mean": float(px.mean()), "px_median": float(np.median(px)),
            "px_var": float(px.var())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_19__13_35_51__bouncing_36k_v2_cont5d_flow")
    ap.add_argument("--counts", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    ctx_len = cfg["context_len"]
    counts = [int(c) for c in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)
    T_end = ctx_len + args.steps
    print(f"Per-count collision eval | counts {counts} | {args.n} scenes x {args.steps} steps")

    CATS = ["free", "wall", "post", "bb"]
    results = {"run": args.run, "n_per_count": args.n, "sim_steps": args.steps, "counts": {}}
    for n_balls in counts:
        t0 = time.time()
        errs = {c: [] for c in CATS}
        made = 0
        while made < args.n:
            balls = spawn_balls(rng, n_balls)
            if balls is None:
                continue
            _, _, frames, bb_f, wall_f = simulate(balls, T_end, render=True)
            bb, wall = set(bb_f), set(wall_f)
            post = {t + d for t in bb for d in (1, 2, 3)} - bb
            ts = list(range(ctx_len, T_end))
            ctx = np.stack([frames[t - ctx_len:t] for t in ts])
            ctx_t = torch.from_numpy(ctx).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
            with torch.no_grad():
                pred = rollout(ae, dit, ctx_t, n_steps=1, num_steps=args.num_steps)
            pr = pred[:, 0].permute(0, 2, 3, 1).numpy().astype(np.float32) / 255.0
            for k, t in enumerate(ts):
                gt = frames[t].astype(np.float32) / 255.0
                e = float(((pr[k] - gt) ** 2).mean())
                cat = "bb" if t in bb else "post" if t in post else "wall" if t in wall else "free"
                errs[cat].append(e)
            made += 1

        rec = {c: _stats(errs[c]) for c in CATS}
        rec["all"] = _stats(sum(errs.values(), []))
        results["counts"][str(n_balls)] = rec
        parts = " ".join(f"{c}:{rec[c]['px_mean']:.2f}px" if rec[c] else f"{c}:-" for c in CATS)
        print(f"  {n_balls} balls: {parts} | {time.time() - t0:.0f}s")

    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "collcount.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir}/collcount.json")


if __name__ == "__main__":
    main()
