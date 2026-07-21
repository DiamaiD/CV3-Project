"""Per-material gravity & drag recovery from the model's rollouts.

Same free-flight second-difference fit as gravity.py, but the free-flight samples are
bucketed by the ball's material, so each material gets its own (g, drag) fit with a
standard error. Answers "does the model apply the same gravity/drag to every material".

  python -m experiments.gravity_per_material --run runs/<run>
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_states, GRAVITY, AIR_DRAG_COEFF, MAT_NAMES
from experiments.rollout import load_run, test_split, traj_frames_np, rollout_frames_np
from experiments.gravity import free_flight_samples, fit_gravity
from src.dataset import FrameCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_19__13_35_51__bouncing_36k_v2_cont5d_flow")
    ap.add_argument("--n-traj", type=int, default=300)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    ctx = cfg["context_len"]
    all_trajs, test = test_split(cfg["data_dir"], args.seed)
    test = test[:args.n_traj]
    fc = FrameCache(all_trajs, cache_device="cpu",
                    disk_cache_path=os.path.join(cfg["data_dir"], "frames_cache.pt"))
    real = [traj_frames_np(fc, td) for td in test]
    states = [extract_states(f) for f in real]
    t0 = time.time()
    model = rollout_frames_np(ae, dit, real, ctx, args.steps, num_steps=args.num_steps, device=device)
    print(f"rolled out {len(model)} trajs ({time.time()-t0:.0f}s) | true g {GRAVITY}, drag {AIR_DRAG_COEFF}")

    Sm = {m: [] for m in MAT_NAMES}
    for frames, mf, st_r in zip(real, model, states):
        seq = np.concatenate([frames[:ctx], mf], axis=0)
        st = extract_states(seq, inventory=st_r["inventory"])
        if st["valid"][ctx:].mean() < 0.2:
            continue
        mats = np.array(st["mats"])
        for m in MAT_NAMES:
            cm = (mats == m)
            if not cm.any():
                continue
            S, _ = free_flight_samples(st["pos"], st["valid"] & cm[None, :], st["radius"], st["mass"], t_min=ctx + 1)
            if len(S):
                Sm[m].append(S)

    out = {"run": args.run, "true_g": GRAVITY, "true_drag": AIR_DRAG_COEFF, "per_material": {}}
    print(f"\n{'material':10s} | {'g fit':>16s} | {'drag fit':>16s} | samples")
    for m in MAT_NAMES:
        S = np.concatenate(Sm[m]) if Sm[m] else np.empty((0, 6))
        if len(S) < 8:
            continue
        g_fit, c_fit, rms, inlier, g_se, c_se = fit_gravity(S)
        out["per_material"][m] = {"g_fit": g_fit, "g_se": g_se, "drag_fit": c_fit, "drag_se": c_se,
                                  "g_mean": float(S[:, 1].mean()), "g_std": float(np.std(S[:, 1], ddof=1)),
                                  "n": int(len(S))}
        print(f"{m:10s} | {g_fit:+.4f} ± {g_se:.4f} | {c_fit:.4f} ± {c_se:.4f} | {len(S)}")

    os.makedirs(os.path.join(args.run, "experiments"), exist_ok=True)
    with open(os.path.join(args.run, "experiments", "gravity_per_material.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.run}/experiments/gravity_per_material.json")


if __name__ == "__main__":
    main()
