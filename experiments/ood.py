import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_states, extract_frame, MAT_NAMES
from experiments.rollout import load_run, rollout_frames_np
from experiments.simkit import make_ball, simulate
from experiments.gravity import free_flight_samples

CTX = 5
ROLL = 40
TOTAL = CTX + ROLL


def spawn_balls(rng, n):
    balls = []
    for _ in range(n):
        placed = False
        for _ in range(300):
            mat = MAT_NAMES[int(rng.integers(len(MAT_NAMES)))]
            r = int(rng.integers(5, 9))
            speed = rng.uniform(3.0, 8.0)
            x = rng.uniform(r, 64 - r)
            y = rng.uniform(0.3 * 64, 64 - r)
            if all((x - b["x"]) ** 2 + (y - b["y"]) ** 2 >= (r + b["radius"]) ** 2
                   for b in balls):
                balls.append(make_ball(mat, r, x, y, rng.normal() * speed, rng.normal() * speed))
                placed = True
                break
        if not placed:
            return None
    return balls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_11__08_36_56__bouncing_20k_b25_flow")
    ap.add_argument("--counts", default="1,3,6,8")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    import torch
    from scipy.optimize import linear_sum_assignment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    counts = [int(c) for c in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)
    print(f"OOD ball counts {counts} (training: 2-5) | {args.n} scenarios each, {ROLL}-step rollouts")

    results = {"run": args.run, "counts": {}, "n_per_count": args.n, "rollout_steps": ROLL}
    curves = {}
    for n_balls in counts:
        t0 = time.time()
        sims = []
        while len(sims) < args.n:
            balls = spawn_balls(rng, n_balls)
            if balls is None:
                continue
            spec = [(b["mat_name"], b["radius"]) for b in balls]
            pos, vel, frames, _, _ = simulate(balls, TOTAL, render=True)
            sims.append({"pos": pos, "frames": frames, "spec": spec})

        torch.manual_seed(99)
        mf = rollout_frames_np(ae, dit, [s["frames"] for s in sims], CTX, ROLL, device=device)

        integ = np.zeros(TOTAL)
        integ_n = np.zeros(TOTAL)
        err10, err20 = [], []
        extra_frames = 0
        n_extra_checked = 0
        S_g = []
        for s, m in zip(sims, mf):
            seq = np.concatenate([s["frames"][:CTX], m], axis=0)
            inv = {}
            for mat, _ in s["spec"]:
                inv[mat] = inv.get(mat, 0) + 1
            st = extract_states(seq, inventory=inv)
            present = st["valid"] | st["merged"]
            integ[:present.shape[0]] += present.sum(1)
            integ_n[:present.shape[0]] += present.shape[1]

            for t in range(CTX, TOTAL, 3):
                dets = extract_frame(seq[t])
                extra_frames += int(len(dets) > n_balls)
                n_extra_checked += 1

            N = st["pos"].shape[1]
            cost = np.full((N, n_balls), 1e6)
            for i in range(N):
                d = np.linalg.norm(st["pos"][:CTX, i, None] - s["pos"][:CTX], axis=2)
                md = np.nanmean(np.where(st["valid"][:CTX, i, None], d, np.nan), axis=0)
                cost[i] = np.where(np.isfinite(md), md, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for tq, sink in ((10, err10), (20, err20)):
                t = CTX + tq
                es = [np.linalg.norm(st["pos"][t, i] - s["pos"][t, j])
                      for i, j in zip(rows, cols)
                      if cost[i, j] < 1e6 and st["valid"][t, i]]
                if es:
                    sink.append(float(np.mean(es)))

            S, _ = free_flight_samples(st["pos"], st["valid"], st["radius"], st["mass"],
                                       t_min=CTX + 1)
            if len(S):
                S_g.append(S)

        g_med = float(np.median(np.concatenate(S_g)[:, 1])) if S_g else float("nan")
        n_g = int(sum(len(s) for s in S_g))
        icurve = integ / np.maximum(integ_n, 1e-9)
        curves[n_balls] = icurve
        results["counts"][str(n_balls)] = {
            "integrity_end": float(icurve[TOTAL - 1]),
            "integrity_curve": icurve.tolist(),
            "pos_err_t10_median": float(np.median(err10)) if err10 else None,
            "pos_err_t20_median": float(np.median(err20)) if err20 else None,
            "g_median": g_med, "g_n_samples": n_g,
            "hallucination_rate": extra_frames / max(n_extra_checked, 1),
        }
        r = results["counts"][str(n_balls)]
        print(f"  {n_balls} balls: integrity@{ROLL} {r['integrity_end']:.1%} | "
              f"pos err t+10 {r['pos_err_t10_median']:.2f} px, t+20 {r['pos_err_t20_median']:.2f} px | "
              f"g {g_med:+.4f} ({n_g} samples) | extra-ball frames {r['hallucination_rate']:.1%} "
              f"| {time.time() - t0:.0f}s")

    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ood.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    make_plots(results, curves, plot_dir)
    print(f"Saved to {out_dir}/ood.json + plots/")


def make_plots(results, curves, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cmap = {1: "#7c3aed", 3: "#0b8fa8", 6: "#d97706", 8: "#dc2626"}

    ax = axes[0]
    for n, c in curves.items():
        style = "-" if 2 <= n <= 5 else "--"
        ax.plot(np.arange(len(c)), c, style, color=cmap.get(n, "gray"),
                label=f"{n} balls" + (" (in-dist)" if 2 <= n <= 5 else " (OOD)"))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("frame")
    ax.set_ylabel("tracked fraction")
    ax.set_title("Ball integrity over rollout", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    ns = sorted(int(k) for k in results["counts"])
    ax = axes[1]
    g = [results["counts"][str(n)]["g_median"] for n in ns]
    bars = ax.bar([str(n) for n in ns], g,
                  color=[cmap.get(n, "gray") for n in ns], alpha=0.8)
    ax.axhline(-0.5, color="black", lw=1, ls="--", label="true g = -0.5")
    ax.set_xlabel("ball count")
    ax.set_ylabel("recovered g (median d2y)")
    ax.set_title("Gravity in OOD rollouts", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    e10 = [results["counts"][str(n)]["pos_err_t10_median"] for n in ns]
    e20 = [results["counts"][str(n)]["pos_err_t20_median"] for n in ns]
    x = np.arange(len(ns))
    ax.bar(x - 0.18, e10, width=0.36, color="#0b8fa8", alpha=0.8, label="t = +10")
    ax.bar(x + 0.18, e20, width=0.36, color="#c2410c", alpha=0.8, label="t = +20")
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlabel("ball count")
    ax.set_ylabel("median pos error vs sim [px]")
    ax.set_title("Rollout agreement with simulator", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Out-of-distribution ball counts (trained on 2-5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "ood_counts.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
