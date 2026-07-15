import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_states, MATERIALS, WORLD, GRAVITY, AIR_DRAG_COEFF
from experiments.rollout import load_run, test_split, traj_frames_np, rollout_frames_np
from src.dataset import FrameCache

G = GRAVITY
MASS_MODELS = {
    "true": {m: MATERIALS[m]["density"] for m in MATERIALS},
    "uniform": {m: 1.0 for m in MATERIALS},
    "shuffled": {"Superball": 5.0, "Rubber": 0.2, "Steel": 0.5, "Sponge": 1.2},
}


def _speeds(pos):
    T = pos.shape[0]
    vel = np.full_like(pos, np.nan)
    if T >= 3:
        vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0
    if T >= 2:
        vel[0] = pos[1] - pos[0]
        vel[-1] = pos[-1] - pos[-2]
    return vel


def v_backward(pos, i, t):
    v = pos[t, i] - pos[t - 1, i]
    return np.array([v[0], v[1] + 0.375 * G])


def v_forward(pos, i, t):
    v = pos[t + 1, i] - pos[t, i]
    return np.array([v[0], v[1] - 0.625 * G])


def _zones(pos, valid, radius):
    T, N = pos.shape[:2]
    vel = _speeds(pos)
    speed = np.nan_to_num(np.linalg.norm(vel, axis=2), nan=99.0)

    wall = np.zeros((T, N, 4), bool)
    for i in range(N):
        r = radius[i]
        if not np.isfinite(r):
            wall[:, i] = True
            continue
        m = r + 0.5 + speed[:, i]
        x, y = pos[:, i, 0], pos[:, i, 1]
        with np.errstate(invalid="ignore"):
            fin = np.isfinite(x) & np.isfinite(y)
            wall[:, i, 0] = np.where(fin, x < m, True)
            wall[:, i, 1] = np.where(fin, x > WORLD - m, True)
            wall[:, i, 2] = np.where(fin, y < m, True)
            wall[:, i, 3] = np.where(fin, y > WORLD - m, True)

    near = np.zeros((T, N, N), bool)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
            rs = np.nan_to_num(np.linalg.norm(vel[:, i] - vel[:, j], axis=1), nan=99.0)
            thr = radius[i] + radius[j] + 0.5 + rs
            n = np.where(np.isfinite(d), d < thr, True)
            near[:, i, j] = near[:, j, i] = n
    return wall, near


def _clean(t, i, valid, wall, near):
    return valid[t, i] and not wall[t, i].any() and not near[t, i].any()


def _intervals(mask):
    out, t = [], 0
    T = len(mask)
    while t < T:
        if not mask[t]:
            t += 1
            continue
        a = t
        while t + 1 < T and mask[t + 1]:
            t += 1
        out.append((a, t))
        t += 1
    return out


def pair_events(pos, valid, radius, t_min=0, max_len=5):
    T, N = pos.shape[:2]
    wall, near = _zones(pos, valid, radius)
    events = []
    for i in range(N):
        for j in range(i + 1, N):
            for a, b in _intervals(near[:, i, j]):
                t_in, t_out = a - 1, b + 1
                if b - a + 1 > max_len or t_in < t_min + 1 or t_out + 1 >= T:
                    continue
                near_ex = near.copy()
                near_ex[:, i, j] = near_ex[:, j, i] = False
                flanks = all(
                    valid[t, k] and not wall[t, k].any() and not near_ex[t, k].any()
                    for k in (i, j) for t in (t_in - 1, t_in, t_out, t_out + 1))
                if not flanks:
                    continue
                if wall[a:b + 1, i].any() or wall[a:b + 1, j].any():
                    continue
                events.append((i, j, t_in, t_out))
    return events


def wall_events(pos, valid, radius, t_min=0, max_len=6):
    T, N = pos.shape[:2]
    wall, near = _zones(pos, valid, radius)
    events = []
    for i in range(N):
        anyw = wall[:, i].any(axis=1)
        for a, b in _intervals(anyw):
            t_in, t_out = a - 1, b + 1
            if b - a + 1 > max_len or t_in < t_min + 1 or t_out + 1 >= T:
                continue
            walls = np.where(wall[a:b + 1, i].any(axis=0))[0]
            if len(walls) != 1:
                continue
            if near[max(0, t_in - 1):t_out + 2, i].any():
                continue
            flanks = all(valid[t, i] and not wall[t, i].any()
                         for t in (t_in - 1, t_in, t_out, t_out + 1))
            if not flanks:
                continue
            events.append((i, int(walls[0]), t_in, t_out))
    return events


def simulate_ball(x, y, vx, vy, r, mass, rest, fric, n_frames, n_substeps=4):
    dt = 1.0 / n_substeps
    out = []
    for _ in range(n_frames):
        for _ in range(n_substeps):
            drag_x = -(AIR_DRAG_COEFF * vx * abs(vx) * r) / mass
            drag_y = -(AIR_DRAG_COEFF * vy * abs(vy) * r) / mass
            vx += drag_x * dt
            vy += (G + drag_y) * dt
            x += vx * dt
            y += vy * dt
            if y - r <= 0:
                y, vy = r, vy * -rest
                vx *= fric
                if abs(vy) < abs(G):
                    vy = 0.0
            if y + r >= WORLD:
                y, vy = WORLD - r, vy * -rest
                vx *= fric
            if x - r <= 0:
                x, vx = r, vx * -rest
                vy *= fric
            if x + r >= WORLD:
                x, vx = WORLD - r, vx * -rest
                vy *= fric
        out.append((x, y))
    return np.array(out)


def fit_wall_bounce(pos, i, t_in, t_out, r, mass):
    v0 = v_backward(pos, i, t_in)
    n_frames = t_out + 1 - t_in
    obs = np.stack([pos[t_out, i], pos[t_out + 1, i]])

    def err(rest, fric):
        sim = simulate_ball(pos[t_in, i, 0], pos[t_in, i, 1], v0[0], v0[1],
                            r, mass, rest, fric, n_frames)
        return float(((sim[-2:] - obs) ** 2).sum())

    best, be = None, np.inf
    for rest in np.arange(0.05, 1.01, 0.05):
        for fric in np.arange(0.5, 1.01, 0.05):
            e = err(rest, fric)
            if e < be:
                be, best = e, (rest, fric)
    r0, f0 = best
    for rest in np.arange(max(0.01, r0 - 0.06), min(1.05, r0 + 0.061), 0.01):
        for fric in np.arange(max(0.4, f0 - 0.06), min(1.02, f0 + 0.061), 0.01):
            e = err(rest, fric)
            if e < be:
                be, best = e, (rest, fric)
    return best[0], best[1], be


def analyze(pos, valid, radius, mats, t_min=0):
    mom_rows, rest_rows, wall_rows = [], [], []

    for i, j, t_in, t_out in pair_events(pos, valid, radius, t_min=t_min):
        vi_in, vj_in = v_backward(pos, i, t_in), v_backward(pos, j, t_in)
        vi_out, vj_out = v_forward(pos, i, t_out), v_forward(pos, j, t_out)
        dt_g = t_out - t_in
        gvec = np.array([0.0, G * dt_g])
        dvi, dvj = vi_out - vi_in - gvec, vj_out - vj_in - gvec

        d = pos[t_in:t_out + 1, i] - pos[t_in:t_out + 1, j]
        tc = t_in + int(np.argmin(np.linalg.norm(d, axis=1)))
        nvec = pos[tc, j] - pos[tc, i]
        nn = np.linalg.norm(nvec)
        if nn < 1e-6:
            continue
        nvec = nvec / nn
        appr_in = float((vi_in - vj_in) @ nvec)
        appr_out = float((vi_out - vj_out) @ nvec)
        if appr_in < 0.5:
            continue
        e_meas = -appr_out / appr_in

        mom_rows.append({"mi_mat": mats[i], "mj_mat": mats[j],
                         "ri": radius[i], "rj": radius[j],
                         "dvi": dvi.tolist(), "dvj": dvj.tolist(),
                         "e_meas": e_meas,
                         "e_true": min(MATERIALS[mats[i]]["restitution"],
                                       MATERIALS[mats[j]]["restitution"])})

    for i, w, t_in, t_out in wall_events(pos, valid, radius, t_min=t_min):
        r = radius[i]
        if not np.isfinite(r):
            continue
        mass = np.pi * r ** 2 * MATERIALS[mats[i]]["density"]
        v0 = v_backward(pos, i, t_in)
        axis = 0 if w < 2 else 1
        if abs(v0[axis]) < 0.8:
            continue
        rest, fric, be = fit_wall_bounce(pos, i, t_in, t_out, r, mass)
        if be > 1.0:
            continue
        wall_rows.append({"mat": mats[i], "wall": int(w), "rest": rest,
                          "fric": fric, "err": be,
                          "v_impact": float(abs(v0[axis]))})
    return mom_rows, wall_rows


def momentum_rel(mom_rows, dens, cross_only=False):
    rel = []
    for ev in mom_rows:
        if cross_only and ev["mi_mat"] == ev["mj_mat"]:
            continue
        mi = np.pi * ev["ri"] ** 2 * dens[ev["mi_mat"]]
        mj = np.pi * ev["rj"] ** 2 * dens[ev["mj_mat"]]
        dP = mi * np.array(ev["dvi"]) + mj * np.array(ev["dvj"])
        J = 0.5 * (mi * np.linalg.norm(ev["dvi"]) + mj * np.linalg.norm(ev["dvj"]))
        if J < 1e-6:
            continue
        rel.append(np.linalg.norm(dP) / J)
    return np.array(rel)


def momentum_stats(mom_rows, cross_only=False):
    out = {}
    for name, dens in MASS_MODELS.items():
        rel = momentum_rel(mom_rows, dens, cross_only)
        out[name] = {"n": int(rel.size), "median": float(np.median(rel)),
                     "p90": float(np.percentile(rel, 90))} if rel.size else None
    return out


def restitution_stats(mom_rows):
    groups = {}
    for ev in mom_rows:
        groups.setdefault(round(ev["e_true"], 2), []).append(ev["e_meas"])
    return {str(k): {"n": len(v), "median": float(np.median(v)),
                     "mad": float(np.median(np.abs(np.array(v) - np.median(v)))),
                     "mean": float(np.mean(v)),
                     "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                     "values": [float(x) for x in v]}
            for k, v in sorted(groups.items())}


def wall_stats(wall_rows):
    out = {}
    for m in MATERIALS:
        rows = [r for r in wall_rows if r["mat"] == m]
        if not rows:
            continue
        rest = np.array([r["rest"] for r in rows])
        fric = np.array([r["fric"] for r in rows])
        out[m] = {"n": len(rows),
                  "rest_median": float(np.median(rest)), "rest_true": MATERIALS[m]["restitution"],
                  "rest_mean": float(rest.mean()),
                  "rest_std": float(rest.std(ddof=1)) if len(rows) > 1 else 0.0,
                  "fric_median": float(np.median(fric)), "fric_true": MATERIALS[m]["friction"],
                  "fric_mean": float(fric.mean()),
                  "fric_std": float(fric.std(ddof=1)) if len(rows) > 1 else 0.0,
                  "rest_values": [float(x) for x in rest], "fric_values": [float(x) for x in fric]}
    return out


def print_report(tag, mom_rows, wall_rows):
    print(f"\n=== {tag} ===")
    ms = momentum_stats(mom_rows)
    ms_x = momentum_stats(mom_rows, cross_only=True)
    print(f"Momentum conservation over {len(mom_rows)} ball-ball collisions "
          f"(|dP| / impulse J, lower = better conserved):")
    for name, s in ms.items():
        sx = ms_x.get(name)
        xtra = f" | cross-material only: median {sx['median']:.4f} (n={sx['n']})" if sx else ""
        if s:
            print(f"  mass model {name:>9}: median {s['median']:.4f} | p90 {s['p90']:.4f}{xtra}")
    rs = restitution_stats(mom_rows)
    print("Ball-ball restitution (measured vs min-rule):")
    for k, v in rs.items():
        print(f"  e_true {k}: e_meas median {v['median']:.3f} (MAD {v['mad']:.3f}, n={v['n']})")
    ws = wall_stats(wall_rows)
    print(f"Wall bounces ({len(wall_rows)} events, sim-matched):")
    for m, v in ws.items():
        print(f"  {m:>9}: restitution {v['rest_median']:.3f} (true {v['rest_true']}) | "
              f"friction {v['fric_median']:.3f} (true {v['fric_true']}) | n={v['n']}")
    return {"momentum": ms, "momentum_cross_material": ms_x, "bb_restitution": rs,
            "wall": ws, "n_bb_events": len(mom_rows), "n_wall_events": len(wall_rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_11__08_36_56__bouncing_20k_b25_flow")
    ap.add_argument("--n-traj", type=int, default=300)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from scipy.optimize import linear_sum_assignment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    ctx_len = cfg["context_len"]
    all_trajs, test_trajs = test_split(cfg["data_dir"], args.seed)
    test_trajs = test_trajs[:args.n_traj]
    print(f"Collision physics | run {os.path.basename(args.run)} | {len(test_trajs)} test trajs")

    fc = FrameCache(all_trajs, cache_device="cpu",
                    disk_cache_path=os.path.join(cfg["data_dir"], "frames_cache.pt"))
    real_frames = [traj_frames_np(fc, td) for td in test_trajs]

    t0 = time.time()
    acc = {"gt": ([], []), "real": ([], []), "model": ([], [])}
    real_states = []
    for td, frames in zip(test_trajs, real_frames):
        st = extract_states(frames)
        real_states.append(st)
        m, w = analyze(st["pos"], st["valid"], st["radius"], st["mats"])
        acc["real"][0].extend(m)
        acc["real"][1].extend(w)

        gt_pos = np.load(os.path.join(td, "positions.npy"))
        if gt_pos.shape[1] == st["pos"].shape[1]:
            cost = np.full((st["pos"].shape[1], gt_pos.shape[1]), 1e6)
            for i in range(st["pos"].shape[1]):
                d = np.linalg.norm(st["pos"][:, i, None] - gt_pos, axis=2)
                md = np.nanmean(np.where(st["valid"][:, i, None], d, np.nan), axis=0)
                cost[i] = np.where(np.isfinite(md), md, 1e6)
            rows, cols = linear_sum_assignment(cost)
            gt_matched = gt_pos[:, cols[np.argsort(rows)]]
            m, w = analyze(gt_matched, np.ones(gt_matched.shape[:2], bool),
                           st["radius"], st["mats"])
            acc["gt"][0].extend(m)
            acc["gt"][1].extend(w)
    print(f"[1/2] Real + GT controls analyzed ({time.time() - t0:.0f}s)")

    t0 = time.time()
    model_frames = rollout_frames_np(ae, dit, real_frames, ctx_len, args.steps,
                                     num_steps=args.num_steps, device=device)
    for frames, mf, st_r in zip(real_frames, model_frames, real_states):
        seq = np.concatenate([frames[:ctx_len], mf], axis=0)
        st = extract_states(seq, inventory=st_r["inventory"])
        if st["valid"][ctx_len:].mean() < 0.2:
            continue
        m, w = analyze(st["pos"], st["valid"], st["radius"], st["mats"], t_min=ctx_len)
        acc["model"][0].extend(m)
        acc["model"][1].extend(w)
    print(f"[2/2] Model rollouts analyzed ({time.time() - t0:.0f}s)")

    results = {"run": args.run, "n_traj": len(test_trajs), "rollout_steps": args.steps}
    for tag, label in [("gt", "GT positions.npy"), ("real", "extractor on real frames"),
                       ("model", "MODEL rollouts")]:
        results[tag] = print_report(label, acc[tag][0], acc[tag][1])

    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "collisions.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    make_plots(acc, plot_dir)
    print(f"\nSaved to {out} + plots/")


MAT_COLORS = {"Superball": "#dc2626", "Rubber": "#2563eb",
              "Steel": "#6b7280", "Sponge": "#16a34a"}


def make_plots(acc, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mom_rows, wall_rows = acc["model"]
    mom_gt = acc["gt"][0]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    for m in MATERIALS:
        rows = [r for r in wall_rows if r["mat"] == m]
        if not rows:
            continue
        for key, marker, label in (("rest", "o", "restitution"), ("fric", "s", "friction")):
            vals = np.array([r[key] for r in rows])
            true = MATERIALS[m]["restitution" if key == "rest" else "friction"]
            med = np.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            ax.errorbar(true, med, yerr=[[med - q1], [q3 - med]], fmt=marker, ms=8,
                        color=MAT_COLORS[m], capsize=3,
                        markeredgecolor="white", markeredgewidth=0.8)
    bb = {}
    for ev in mom_rows:
        bb.setdefault(ev["e_true"], []).append(ev["e_meas"])
    for et, vals in bb.items():
        med = np.median(vals)
        ax.plot(et, med, "D", ms=7, color="#7c3aed", markeredgecolor="white",
                markeredgewidth=0.8)
    ax.plot([0, 1.05], [0, 1.05], color="gray", lw=1, ls="--")
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(0.1, 1.05)
    ax.set_xlabel("true constant")
    ax.set_ylabel("measured from model rollouts (median, IQR)")
    ax.set_title("Material constants read out of generated video", fontsize=10)
    handles = [plt.Line2D([], [], color=MAT_COLORS[m], marker="o", ls="", label=m)
               for m in MATERIALS]
    handles += [plt.Line2D([], [], color="gray", marker="o", ls="", label="wall restitution"),
                plt.Line2D([], [], color="gray", marker="s", ls="", label="wall friction"),
                plt.Line2D([], [], color="#7c3aed", marker="D", ls="", label="ball-ball e (min rule)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels, data, colors = [], [], []
    for name in MASS_MODELS:
        for rows, src, alpha in ((mom_gt, "sim", 0.45), (mom_rows, "model", 0.9)):
            rel = momentum_rel(rows, MASS_MODELS[name], cross_only=True)
            if rel.size:
                labels.append(f"{name}\n({src})")
                data.append(rel)
                colors.append({"true": "#0b8fa8", "uniform": "#d97706",
                               "shuffled": "#dc2626"}[name])
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                    medianprops={"color": "black"})
    for patch, c, lab in zip(bp["boxes"], colors, labels):
        patch.set_facecolor(c)
        patch.set_alpha(0.5 if "(sim)" in lab else 0.85)
    for d, x in zip(data, range(1, len(data) + 1)):
        ax.scatter(np.full(len(d), x) + np.random.uniform(-0.08, 0.08, len(d)),
                   d, s=14, color="black", alpha=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("|momentum change| / impulse exchanged")
    ax.set_title("Cross-material momentum conservation by assumed mass model", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Collision physics from model rollouts", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "collision_constants.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
