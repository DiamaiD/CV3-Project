import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import combinations

from experiments.extractor import extract_states, central_velocity, _alpha_res, MAT_NAMES
from experiments.rollout import load_run, rollout_frames_np
from experiments.simkit import simulate
from experiments.longhorizon import ball_stats, pool_curves
from experiments.ood import spawn_balls

CTX = 5
PSNR_STEPS = 80


def material_masses(frame):
    """Per-material pixel mass (sum of alpha coverage) — sub-pixel ball area,
    immune to deformation, contact merging and tracking; matches the renderer's AA."""
    alpha, res = _alpha_res(frame.astype(np.float32))
    best = res.argmin(0)
    out = np.zeros(len(MAT_NAMES))
    for mi in range(len(MAT_NAMES)):
        sel = (best == mi) & (alpha[mi] > 0.15)
        out[mi] = alpha[mi][sel].sum()
    return out


def count_from_mass(A, areas):
    """How many of this material's balls does pixel mass A account for?
    Best subset-sum fit against the calibrated per-ball areas."""
    best_j, best_err = 0, abs(A)
    for j in range(1, len(areas) + 1):
        err = min(abs(sum(c) - A) for c in combinations(areas, j))
        if err < best_err:
            best_j, best_err = j, err
    return best_j


def survival_curve(frames, mat_ids, radii, cal_frames):
    """Fraction of balls still existing per frame, monotone non-increasing.
    A ball is gone only from the frame after which its material's pixel mass
    never again accounts for it (suffix max of the per-frame count)."""
    T = len(frames)
    k_m = {}
    r2_m = {}
    for mi, r in zip(mat_ids, radii):
        k_m[mi] = k_m.get(mi, 0) + 1
        r2_m.setdefault(mi, []).append(r * r)
    cal = np.mean([material_masses(f) for f in cal_frames], axis=0)
    areas_m = {mi: [cal[mi] * r2 / sum(r2s) for r2 in r2s] for mi, r2s in r2_m.items()}

    n_tot = len(mat_ids)
    counts = np.zeros((T, len(k_m)), int)
    mats = sorted(k_m)
    for t in range(T):
        mm = material_masses(frames[t])
        for j, mi in enumerate(mats):
            counts[t, j] = min(count_from_mass(mm[mi], areas_m[mi]), k_m[mi])
    alive = np.maximum.accumulate(counts[::-1], axis=0)[::-1]  # suffix max per material
    return alive.sum(1) / n_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_19__13_35_51__bouncing_36k_v2_cont5d_flow")
    ap.add_argument("--counts", default="1,2,3,4,5,6")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    counts = [int(c) for c in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)
    T_end = CTX + args.steps
    print(f"Per-count fidelity | counts {counts} | {args.n} scenes each x {args.steps} steps")

    results = {"run": args.run, "n_per_count": args.n, "rollout_steps": args.steps,
               "psnr_steps": PSNR_STEPS, "counts": {}}
    for n_balls in counts:
        t0 = time.time()
        sims = []
        while len(sims) < args.n:
            balls = spawn_balls(rng, n_balls)
            if balls is None:
                continue
            pos, vel, frames, _, _ = simulate(balls, T_end, render=True)
            radius = np.array([b["radius"] for b in balls], float)
            mass = np.array([b["mass"] for b in balls])
            sims.append({"pos": pos, "vel": vel, "frames": frames,
                         "radius": radius, "mass": mass,
                         "mat_ids": [MAT_NAMES.index(b["mat_name"]) for b in balls],
                         "inv": {b["mat_name"]: sum(1 for c in balls if c["mat_name"] == b["mat_name"])
                                 for b in balls}})

        torch.manual_seed(99)
        mf = rollout_frames_np(ae, dit, [s["frames"] for s in sims], CTX, args.steps,
                               device=device, batch=64)

        # per-step pixel PSNR vs the simulator render (pooled MSE, eval convention)
        mse_sum = np.zeros(PSNR_STEPS)
        for s, m in zip(sims, mf):
            gt = s["frames"][CTX:CTX + PSNR_STEPS].astype(np.float32) / 255.0
            pr = m[:PSNR_STEPS].astype(np.float32) / 255.0
            mse_sum += ((gt - pr) ** 2).mean(axis=(1, 2, 3))
        mse = mse_sum / len(sims)
        psnr = (10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))).tolist()

        # integrity + energy from extracted states of the model rollout; the same
        # extraction on the sim render is the instrument control (fast/overlapping
        # balls drop the flag even in perfect frames -- the control shows how often)
        integ = np.zeros(T_end)
        integ_n = np.zeros(T_end)
        integ_c = np.zeros(T_end)
        integ_cn = np.zeros(T_end)
        surv = np.zeros(T_end)
        surv_c = np.zeros(T_end)
        n_surv = 0
        sim_pt, mod_pt = [], []
        for s, m in zip(sims, mf):
            vok = np.ones(s["vel"].shape[:2], bool)
            E, H, rest = ball_stats(s["pos"], s["vel"], vok, s["radius"], s["mass"], T_end)
            E0 = E[CTX]
            sim_pt.append((E, H, rest, E0))

            st_c = extract_states(s["frames"], inventory=s["inv"])
            pres_c = st_c["valid"] | st_c["merged"]
            integ_c[:pres_c.shape[0]] += pres_c.sum(1)
            integ_cn[:pres_c.shape[0]] += pres_c.shape[1]

            seq = np.concatenate([s["frames"][:CTX], m], axis=0)
            st = extract_states(seq, inventory=s["inv"])
            present = st["valid"] | st["merged"]
            integ[:present.shape[0]] += present.sum(1)
            integ_n[:present.shape[0]] += present.shape[1]

            nb = len(s["mat_ids"])
            surv += survival_curve(seq, s["mat_ids"], s["radius"], s["frames"][:CTX]) * nb
            surv_c += survival_curve(s["frames"], s["mat_ids"], s["radius"], s["frames"][:CTX]) * nb
            n_surv += nb

            mvel, mok = central_velocity(st["pos"], present)
            Em, Hm, restm = ball_stats(st["pos"], mvel, mok, st["radius"], st["mass"], T_end)
            # normalize model energy by the SIM E0 of the matched scene (extractor tracks
            # may be ordered differently, but pooling sums over balls so order cancels)
            mod_pt.append((Em, Hm, restm, E0))

        e_sim, _, _, _ = pool_curves(sim_pt, T_end)
        e_mod, _, _, _ = pool_curves(mod_pt, T_end)
        icurve = integ / np.maximum(integ_n, 1e-9)
        ccurve = integ_c / np.maximum(integ_cn, 1e-9)
        scurve = surv / max(n_surv, 1)
        sccurve = surv_c / max(n_surv, 1)
        results["counts"][str(n_balls)] = {
            "psnr_per_step": psnr,
            "survival_curve": scurve.tolist(),
            "survival_curve_control": sccurve.tolist(),
            "measurable_curve": icurve.tolist(),
            "measurable_curve_control": ccurve.tolist(),
            "energy_sim": e_sim.tolist(),
            "energy_model": e_mod.tolist(),
            "survival_end": float(scurve[-1]),
            "integrity_end": float(icurve[-1]),
        }
        print(f"  {n_balls} balls: psnr s1 {psnr[0]:.2f} s20 {psnr[19]:.2f} s80 {psnr[79]:.2f} | "
              f"survival@{args.steps} {scurve[-1]:.1%} (control {sccurve[-1]:.1%}) | "
              f"measurable@{args.steps} {icurve[-1]:.1%} | {time.time() - t0:.0f}s")

    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "percount.json"), "w") as f:
        json.dump(results, f, indent=2)

    make_plots(results, os.path.join(out_dir, "plots"))
    print(f"Saved to {out_dir}/percount.json + plots/percount.png")


def make_plots(results, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    CMAP = {1: "#7c3aed", 2: "#0b8fa8", 3: "#047857", 4: "#ca8a04", 5: "#c2410c", 6: "#9f1239",
            7: "#3730a3", 8: "#831843"}
    counts = sorted(int(k) for k in results["counts"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    for n in counts:
        p = results["counts"][str(n)]["psnr_per_step"]
        ax.plot(np.arange(1, len(p) + 1), p, color=CMAP[n], lw=1.5, label=f"{n} balls")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("pixel PSNR vs simulator (dB)")
    ax.set_title("Rollout PSNR by ball count (horizon 80)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for n in counts:
        d = results["counts"][str(n)]
        c = d["survival_curve"]
        ax.plot(np.arange(CTX, len(c)), c[CTX:], color=CMAP[n], lw=1.5, label=f"{n} balls")
        cc = d.get("survival_curve_control")
        if cc:
            ax.plot(np.arange(CTX, len(cc)), cc[CTX:], color=CMAP[n], lw=0.9, ls=":", alpha=0.7)
    ax.set_ylim(0.975, 1.002)
    ax.set_xlabel("rollout frame (dotted = same measurement on the sim render)")
    ax.set_ylabel("fraction of balls still existing")
    ax.set_title("Ball survival vs rollout length (monotone by construction)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    for n in counts:
        d = results["counts"][str(n)]
        s = np.array(d["energy_sim"])
        mo = np.array(d["energy_model"])
        t = np.arange(len(s))
        # stop where the sim retains <1% of initial energy: below that both sides
        # are ~0 (settled scene) and the ratio is extractor noise over nothing
        keep = (t >= CTX + 5) & (s > 0.01)
        er = mo[keep] / s[keep]
        if len(er) >= 11:
            er = np.convolve(er, np.ones(11) / 11, mode="valid")
            tt = t[keep][5:-5]
        else:
            tt = t[keep]
        ax.plot(tt, er, color=CMAP[n], lw=1.4, label=f"{n} balls")
    ax.axhline(1.0, color="#555555", lw=0.9, ls="--")
    ax.set_ylim(0.6, 1.25)
    ax.set_xlabel("rollout frame (line ends when sim energy < 1% of start)")
    ax.set_ylabel("pooled energy, model / sim (11-frame mean)")
    ax.set_title("Energy ratio vs rollout length", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.25)

    fig.suptitle("Per-ball-count fidelity (fresh simkit scenes, identical spawn law per count)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "percount.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
