import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.simkit import make_ball, simulate, GRAVITY
from experiments.extractor import extract_states, MATERIALS
from experiments.rollout import load_run, rollout_frames_np

SWAP = ("Sponge", "Steel")
TARGET_MAT = "Rubber"
CTX = 5
ROLL = 25
TOTAL = CTX + ROLL


def sample_scenario(rng):
    rt, rp = int(rng.integers(5, 9)), int(rng.integers(5, 9))
    tx, ty = rng.uniform(24, 40), rng.uniform(30, 46)
    px = rng.uniform(rp + 3, 64 - rp - 3)
    py = rng.uniform(14.0, 64 - rp - 4)
    dist = np.hypot(tx - px, ty - py)
    gap = dist - (rt + rp)
    if gap < 11.0:
        return None
    f_star = rng.uniform(8.0, 13.0)
    speed = gap / f_star
    if speed < 1.2:
        return None
    T = dist / speed
    vx = (tx - px) / T
    vy = (ty - py) / T - 0.5 * GRAVITY * T
    if np.hypot(vx, vy) > 9.0:
        return None
    vtx, vty = rng.normal(0, 0.5), rng.uniform(1.0, 3.0)

    out = {}
    for mat in SWAP:
        balls = [make_ball(mat, rp, px, py, vx, vy),
                 make_ball(TARGET_MAT, rt, tx, ty, vtx, vty)]
        pos, vel, frames, bb, wall = simulate(balls, TOTAL, render=True)
        if not bb:
            return None
        out[mat] = {"pos": pos, "vel": vel, "frames": frames,
                    "f": bb[0], "wall": wall}

    fA, fB = out[SWAP[0]]["f"], out[SWAP[1]]["f"]
    if not (7 <= fA <= 16) or abs(fA - fB) > 2:
        return None
    f_end = max(fA, fB) + 6
    if f_end >= TOTAL:
        return None

    def near_wall(pos, i, r, t0, t1):
        p = pos[t0:t1 + 1, i]
        return bool(((p[:, 0] < r + 1.5) | (p[:, 0] > 64 - r - 1.5) |
                     (p[:, 1] < r + 1.5) | (p[:, 1] > 64 - r - 1.5)).any())

    for mat in SWAP:
        if near_wall(out[mat]["pos"], 0, rp, 0, f_end):
            return None
        if near_wall(out[mat]["pos"], 1, rt, max(0, out[mat]["f"] - 2), f_end):
            return None
    va, vb = out[SWAP[0]]["vel"][fA], out[SWAP[1]]["vel"][fB]
    n = out[SWAP[0]]["pos"][fA, 1] - out[SWAP[0]]["pos"][fA, 0]
    n = n / (np.linalg.norm(n) + 1e-9)
    if (va[0] - va[1]) @ n < 2.0:
        return None
    effect = np.linalg.norm(out[SWAP[0]]["pos"][fA + 4] - out[SWAP[1]]["pos"][fA + 4], axis=1)
    if effect[1] < 2.5:
        return None
    out["meta"] = {"fA": int(fA), "fB": int(fB), "rp": rp, "rt": rt,
                   "effect_target_px": float(effect[1]), "effect_proj_px": float(effect[0])}
    return out


def first_valid(pos, valid, i, ts):
    for t in ts:
        if 0 <= t < pos.shape[0] and valid[t, i] and np.isfinite(pos[t, i]).all():
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_11__08_36_56__bouncing_20k_b25_flow")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)

    rng = np.random.default_rng(args.seed)
    scenarios = []
    tries = 0
    t0 = time.time()
    while len(scenarios) < args.n and tries < 400 * args.n:
        tries += 1
        s = sample_scenario(rng)
        if s is not None:
            scenarios.append(s)
    print(f"[1/3] Sampled {len(scenarios)} paired scenarios ({tries} tries, {time.time() - t0:.0f}s)")

    t0 = time.time()
    states = {m: [] for m in SWAP}
    for mat in SWAP:
        real = [s[mat]["frames"] for s in scenarios]
        torch.manual_seed(1234)
        mf = rollout_frames_np(ae, dit, real, CTX, ROLL, device=device)
        for s, m in zip(scenarios, mf):
            seq = np.concatenate([s[mat]["frames"][:CTX], m], axis=0)
            st = extract_states(seq, inventory={mat: 1, TARGET_MAT: 1})
            st["_frames"] = seq
            states[mat].append(st)
    print(f"[2/3] Rolled out + extracted {2 * len(scenarios)} rollouts ({time.time() - t0:.0f}s)")

    rel_lo, rel_hi = -6, 10
    rels = np.arange(rel_lo, rel_hi + 1)
    div_sim = {0: [], 1: []}
    div_model = {0: [], 1: []}
    outcomes = []
    for k, s in enumerate(scenarios):
        f = s["meta"]["fA"]
        stA, stB = states[SWAP[0]][k], states[SWAP[1]][k]
        for i in (0, 1):
            ds, dm = np.full(len(rels), np.nan), np.full(len(rels), np.nan)
            for q, rel in enumerate(rels):
                t = f + rel
                if 0 <= t < TOTAL:
                    ds[q] = np.linalg.norm(s[SWAP[0]]["pos"][t, i] - s[SWAP[1]]["pos"][t, i])
                    if stA["valid"][t, i] and stB["valid"][t, i]:
                        dm[q] = np.linalg.norm(stA["pos"][t, i] - stB["pos"][t, i])
            div_sim[i].append(ds)
            div_model[i].append(dm)

        for mat, st in ((SWAP[0], stA), (SWAP[1], stB)):
            other = SWAP[1] if mat == SWAP[0] else SWAP[0]
            fv = s[mat]["f"]
            t = first_valid(st["pos"], st["valid"], 1, [fv + 4, fv + 5, fv + 3, fv + 6])
            if t is None:
                continue
            em = float(np.linalg.norm(st["pos"][t, 1] - s[mat]["pos"][t, 1]))
            ec = float(np.linalg.norm(st["pos"][t, 1] - s[other]["pos"][t, 1]))
            outcomes.append({"scenario": k, "variant": mat, "t": int(t),
                             "err_match": em, "err_cross": ec, "win": em < ec})

    wins = sum(o["win"] for o in outcomes)
    em = np.array([o["err_match"] for o in outcomes])
    ec = np.array([o["err_cross"] for o in outcomes])
    print(f"[3/3] Outcome test: model matches its own counterfactual in "
          f"{wins}/{len(outcomes)} rollouts ({wins / len(outcomes):.1%})")
    print(f"      target-ball pos err vs correct sim: median {np.median(em):.2f} px | "
          f"vs wrong sim: median {np.median(ec):.2f} px")

    curves = {}
    for i, name in ((0, "projectile"), (1, "target")):
        curves[name] = {
            "rel_frames": rels.tolist(),
            "sim_mean": np.nanmean(np.stack(div_sim[i]), axis=0).tolist(),
            "model_mean": np.nanmean(np.stack(div_model[i]), axis=0).tolist(),
            "model_n": np.isfinite(np.stack(div_model[i])).sum(axis=0).tolist(),
        }

    results = {"run": args.run, "n_scenarios": len(scenarios), "swap": list(SWAP),
               "target_mat": TARGET_MAT,
               "outcome": {"n": len(outcomes), "wins": wins,
                           "win_rate": wins / len(outcomes),
                           "err_match_median": float(np.median(em)),
                           "err_cross_median": float(np.median(ec))},
               "divergence": curves,
               "outcomes": outcomes,
               "effect_target_median": float(np.median([s["meta"]["effect_target_px"]
                                                        for s in scenarios]))}
    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "counterfactual.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    make_plots(results, scenarios, states, plot_dir)
    print(f"Saved to {out_dir}/counterfactual.json + plots/")


def make_plots(results, scenarios, states, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ("projectile (swapped ball)", "target (struck ball)")):
        c = results["divergence"][name.split(" ")[0]]
        r = np.array(c["rel_frames"])
        ax.plot(r, c["sim_mean"], "-o", ms=3.5, color="#0b8fa8", label="simulator pair")
        ax.plot(r, c["model_mean"], "-s", ms=3.5, color="#c2410c", label="model pair")
        ax.axvline(0, color="gray", lw=1, ls=":")
        ax.annotate("contact", (0, ax.get_ylim()[1] * 0.02), color="gray", fontsize=8,
                    xytext=(3, 4), textcoords="offset points")
        ax.set_xlabel("frames since first contact")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("|pos(Sponge run) − pos(Steel run)|  [px]")
    axes[0].legend(frameon=False)
    fig.suptitle("Counterfactual mass swap: same start, projectile color Sponge vs Steel", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "counterfactual_divergence.png"), dpi=140)
    plt.close(fig)

    em = np.array([o["err_match"] for o in results["outcomes"]])
    ec = np.array([o["err_cross"] for o in results["outcomes"]])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = max(1.0, np.percentile(np.concatenate([em, ec]), 98)) * 1.1
    ax.scatter(em, ec, s=22, alpha=0.7, color="#0b8fa8", edgecolors="none")
    ax.plot([0, lim], [0, lim], color="gray", lw=1, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("struck-ball error vs CORRECT counterfactual sim [px]")
    ax.set_ylabel("error vs WRONG counterfactual sim [px]")
    wr = results["outcome"]["win_rate"]
    ax.set_title(f"Post-collision outcome: {wr:.0%} above diagonal (n={len(em)})", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "counterfactual_outcome.png"), dpi=140)
    plt.close(fig)

    best = max(range(len(scenarios)),
               key=lambda k: scenarios[k]["meta"]["effect_target_px"]
               * all(states[m][k]["valid"][scenarios[k]["meta"]["fA"] + 4].all() for m in SWAP))
    s = scenarios[best]
    f = s["meta"]["fA"]
    cols = [f - 3, f - 1, f + 1, f + 3, f + 5, f + 7]
    rows, labels = [], []
    for mat in SWAP:
        rows.append([s[mat]["frames"][t] for t in cols])
        labels.append(f"sim   {mat}")
        rows.append([states[mat][best]["_frames"][t] for t in cols])
        labels.append(f"model {mat}")

    scale = 4
    pad = 2
    cell = 64 * scale
    W = len(cols) * (cell + pad) + pad + 110
    H = len(rows) * (cell + pad) + pad + 20
    canvas = np.full((H, W, 3), 255, np.uint8)
    for ri, row in enumerate(rows):
        for ci, img in enumerate(row):
            big = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST)
            y0 = 20 + pad + ri * (cell + pad)
            x0 = 110 + pad + ci * (cell + pad)
            canvas[y0:y0 + cell, x0:x0 + cell] = big
        cv2.putText(canvas, labels[ri], (4, 20 + pad + ri * (cell + pad) + cell // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    for ci, t in enumerate(cols):
        cv2.putText(canvas, f"t=f{t - f:+d}", (110 + pad + ci * (cell + pad) + 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(plot_dir, "counterfactual_example.png"),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
