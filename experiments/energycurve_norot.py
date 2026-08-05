"""Energy curves for the norot era (v2: 500 scenes, GPU-batched 96, parallel
CPU tracking overlapped with the GPU): fresh 305-frame simulations (same physics
seed for both looks -> identical scenes), 300-step model rollouts, per-frame
total energy (kinetic + potential; rotation frozen) pooled as a fraction of
context-time energy. Outputs energy_norot.json + energy_norot.png."""
import json
import os
import zlib

import numpy as np

from environments.env_bouncing import MATERIALS, GRAVITY
from environments.env_pymunk import generate_shapes_pymunk
from environments.env_shapes import (EASY_KIND_WEIGHTS, EASY_SIZE_RANGES,
                                     EASY_SIZE_RANGES_V3, _poly_props)
from experiments.deformation import _batched_rollout, detect_outline, score_frames
from experiments.rollout import load_run
from src.frameio import load_frames

CTX = 5
STEPS = 300
N_SCENES = 500
GROUP = 96
CONFIGS = [
    ("v3, 10 epochs", "runs/2026_07_28__21_54_10__shapes_easy20k_norot_v3",
     dict(outline=True), "scratchpad/examples/energy_sim500_v3", "#2563EB"),
    ("v4 no outline, 10 epochs", "runs/2026_07_28__17_04_41__shapes_easy20k_norot_v4",
     dict(outline=False), "scratchpad/examples/energy_sim500_v4", "#E8710A"),
    # 60-epoch model trained on the BORDERED norot-v1 look: border insets the
    # walls, so its scenes are a separate (statistically matched) set
    ("norot-v1, 60 epochs", "runs/2026_07_28__02_54_07__shapes_easy20k_norot_cont",
     dict(outline=True, border=1.5, size_ranges_override="v2"),
     "scratchpad/examples/energy_sim500_v1n", "#7C3AED"),
    ("v3 60k + noise, 23 epochs (final)",
     "runs/2026_07_29__03_57_19__shapes_easy60k_norot_v3",
     dict(outline=True), "scratchpad/examples/energy_sim500_v3", "#059669"),
]


def obj_mass_low(obj, theta):
    rho = MATERIALS[obj["material"]]["density"]
    if obj["verts"] is None:
        return np.pi * obj["size"] ** 2 * rho, obj["size"]
    v = np.asarray(obj["verts"])
    _, mass, _ = _poly_props(v, rho)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    low = -float((v @ R.T)[:, 1].min())
    return mass, low


def energy_series(pos_xy, vel_xy, masses, lows):
    y = pos_xy[:, :, 1] - np.asarray(lows)[None, :]
    v2 = (vel_xy ** 2).sum(axis=2)
    m = np.asarray(masses)[None, :]
    return 0.5 * m * v2 + m * abs(GRAVITY) * np.maximum(y, 0.0)


def _track_energy(payload):
    """Worker: template-track a rollout, return (E_mod (T,n), E0 (n,))."""
    pred, objs, thetas, init, masses, lows, E0, outline = payload
    _, _, tracks = score_frames(pred, objs, thetas, init, outline=outline)
    p = np.stack([tracks[:, :, 1] + 0.5, 64.0 - tracks[:, :, 0] - 0.5], axis=2)
    v = np.zeros_like(p)
    v[1:-1] = (p[2:] - p[:-2]) / 2.0
    v[0], v[-1] = p[1] - p[0], p[-1] - p[-2]
    return energy_series(p, v, masses, lows), E0


def pool_energy(pairs, steps):
    num = np.zeros(steps)
    den = np.zeros(steps)
    for E, E0 in pairs:
        for i in range(E.shape[1]):
            if not np.isfinite(E0[i]) or E0[i] <= 0:
                continue
            T = E.shape[0]
            num[:T] += E[:, i]
            den[:T] += E0[i]
    return num / np.maximum(den, 1e-9)


def main():
    import time
    import torch
    from multiprocessing import get_context

    out = {}
    for label, run_dir, look, simdir, _ in CONFIGS:
        if not os.path.isdir(simdir):
            np.random.seed(501)          # same base seed everywhere
            look = dict(look)
            sr = (EASY_SIZE_RANGES if look.pop("size_ranges_override", None) == "v2"
                  else EASY_SIZE_RANGES_V3)
            generate_shapes_pymunk(
                data_dir=simdir, n_trajectories=N_SCENES, max_frames=CTX + STEPS,
                n_objects_min=2, n_objects_max=5, kind_weights=EASY_KIND_WEIGHTS,
                vertex_jitter=0.0, size_min=5, size_max=8,
                size_ranges=sr, markers="none",
                spin_max=0.0, frozen_rotation=True,
                **{"border": 0.0, **look})
        look.pop("size_ranges_override", None) if isinstance(look, dict) else None
        ae, dit, cfg = load_run(run_dir, "cuda")
        t0 = time.time()
        sim_pairs, mod_pairs = [], []
        trajs = [os.path.join(simdir, f"traj-{t}") for t in range(N_SCENES)]
        pool = get_context("spawn").Pool(12)
        pending = None
        for gi in range(0, N_SCENES, GROUP):
            group = trajs[gi:gi + GROUP]
            ctxs, seeds, metas = [], [], []
            for td in group:
                frames_np = load_frames(td)
                objs = json.load(open(os.path.join(td, "objects.json")))
                ang = np.load(os.path.join(td, "angles.npy"))
                pos = np.load(os.path.join(td, "positions.npy"))
                vel = np.load(os.path.join(td, "velocities.npy"))
                thetas = ang[0, :, 0]
                ml = [obj_mass_low(o, thetas[i]) for i, o in enumerate(objs)]
                masses = [m for m, _ in ml]
                lows = [l for _, l in ml]
                E_sim = energy_series(pos, vel, masses, lows)
                E0 = E_sim[CTX]
                sim_pairs.append((E_sim[CTX:CTX + STEPS], E0))
                ctxs.append(frames_np[:CTX])
                seeds.append(zlib.crc32(os.path.basename(td).encode()) & 0x7FFFFFFF)
                metas.append((objs, thetas, pos[CTX - 1], masses, lows, E0,
                              detect_outline(frames_np[0], objs, pos[0], thetas)))
            preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS)
            payloads = [(pred, *meta) for pred, meta in zip(preds, metas)]
            if pending is not None:
                mod_pairs.extend(pending.get())
            pending = pool.map_async(_track_energy, payloads)
            print(f"{label}: {min(gi + GROUP, N_SCENES)}/{N_SCENES} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        mod_pairs.extend(pending.get())
        pool.close()
        pool.join()
        out[label] = {"sim": pool_energy(sim_pairs, STEPS).tolist(),
                      "model": pool_energy(mod_pairs, STEPS).tolist()}
        print(f"{label}: sim@300 {out[label]['sim'][-1]:.3f} "
              f"model@300 {out[label]['model'][-1]:.3f} | "
              f"{time.time() - t0:.0f}s total", flush=True)

    json.dump(out, open("scratchpad/examples/energy_norot.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    # trim boundary frames: the one-sided velocity estimate at the ends passes
    # tracker jitter through unaveraged (the "uptick" artifact)
    lo, hi = 1, STEPS - 1
    x = np.arange(1, STEPS + 1)[lo:hi]
    sim_main = np.array(out["v3, 10 epochs"]["sim"])[lo:hi]
    sim_v1n = np.array(out["norot-v1, 60 epochs"]["sim"])[lo:hi]
    ax.plot(x, sim_main, color="#444444", linewidth=2, linestyle="--",
            label="simulator (borderless scenes)")
    if np.abs(sim_v1n - sim_main).max() > 0.005:
        ax.plot(x, sim_v1n, color="#999999", linewidth=1.5, linestyle="--",
                label="simulator (bordered scenes)")
    for label, _, _, _, color in CONFIGS:
        m = np.array(out[label]["model"])[lo:hi]
        ax.plot(x, m, color=color, linewidth=2, label=f"model {label}")
        ax.annotate(f"model {label}", (x[-1], m[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", fontweight="bold")
    ax.annotate("simulator", (x[-1], sim_main[-1]), xytext=(6, -8),
                textcoords="offset points", color="#444444", fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 1.1)
    ax.set_yticks([1.0, 0.5, 0.2, 0.1, 0.05])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("rollout frame")
    ax.set_ylabel("total energy (fraction of context-time energy, log)")
    ax.set_title("Energy over long rollouts: models vs simulator", fontsize=12, pad=10)
    ax.set_xlim(1, STEPS + 65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.text(0.01, 0.005, f"{N_SCENES} fresh 300-frame scenes per model; model state "
             "via template tracking, velocity by central difference; boundary frames "
             "trimmed (one-sided velocity artifact)", fontsize=7.5, color="#666666")
    fig.tight_layout()
    fig.savefig("scratchpad/examples/energy_norot.png", bbox_inches="tight")
    print("saved scratchpad/examples/energy_norot.png")


if __name__ == "__main__":
    main()
