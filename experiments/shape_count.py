"""Count-based integrity check: per frame, count shapes present
per material (connected components of the color mask, merged blobs counted by
mass) and compare against how many SHOULD be there. Identity-free -- immune to
tracking/swap artifacts by construction. Outputs shape_count.json + png."""
import json
import os
import zlib

import cv2
import numpy as np

from experiments.deformation import (_batched_rollout, _color_mask,
                                     _fill_outline_rgb, detect_border_px,
                                     detect_outline, render_template)
from experiments.rollout import load_run
from src.frameio import load_frames

MODELS = [("v3-10ep", "runs/2026_07_28__21_54_10__shapes_easy20k_norot_v3", "#2563EB"),
          ("60ep norot-v1", "runs/2026_07_28__02_54_07__shapes_easy20k_norot_cont", "#7C3AED"),
          ("final (60k v3 + noise, 23ep)", "runs/2026_07_29__03_57_19__shapes_easy60k_norot_v3", "#059669")]
SIMDIRS = {"v3-10ep": "scratchpad/examples/energy_sim500_v3",
           "60ep norot-v1": "scratchpad/examples/energy_sim500_v1n",
           "final (60k v3 + noise, 23ep)": "scratchpad/examples/energy_sim500_v3"}
CTX = 5
N_STEPS = 300
N_SCENES = 500
GROUP = 96


def _match_counts(masses, areas):
    """Explain the observed component soft-masses with the objects' TRUE
    template areas: exact search over DISJOINT subset assignments (one subset
    of objects per component), minimizing band error + 100 per miscount in
    either direction. Returns (deficit, surplus).

    Mass bands, not point sums: contact junctions destroy 0-60px each (the
    upper shape's outline overdraws the lower one; blended seam pixels fail
    both color masks), while a light resting touch destroys nothing and AA
    bridging can even ADD a few px. Singles get the same slack (neighbors of
    another material erode them too). Components no subset explains feed a
    fragment pool for still-missing objects (draw-order occlusion splits
    shapes); with nothing missing, shape-sized leftovers count as SURPLUS
    phantoms. Merge slack is capped at 28% of the subset mass and any
    explanation missing its band by >25px is invalid outright."""
    # areas entries may be scalars (exact expected mass, the normal case)
    # or (lo, hi) intervals -- used by the size-OOD sets, where the model
    # renormalizes balls toward the trained radius band and a resized-but-
    # present ball must not be declared dead
    lohi = [(a if isinstance(a, tuple) else (a, a)) for a in areas]
    los = [a for a, _ in lohi]
    his = [b for _, b in lohi]
    mids = [(a + b) / 2.0 for a, b in lohi]
    n = len(areas)
    full = (1 << n) - 1
    sums_lo = [0.0] * (1 << n)
    sums_hi = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        i = (mask & -mask).bit_length() - 1
        sums_lo[mask] = sums_lo[mask & (mask - 1)] + los[i]
        sums_hi[mask] = sums_hi[mask & (mask - 1)] + his[i]
    cnt = [bin(mask).count("1") for mask in range(1 << n)]
    areas = mids                             # pool/surplus heuristics
    amin, amed = min(areas), float(np.median(areas))
    masses = masses[:6]                      # sanity cap on shattered frames

    def band_err(ms, mask):
        a, r = sums_lo[mask], cnt[mask]
        low = a - min(60.0 * (r - 1), 0.28 * a) if r > 1 else 0.75 * a
        high = 1.04 * sums_hi[mask] + 4.0 + 8.0 * (r - 1)
        return low - ms if ms < low else max(ms - high, 0.0)

    best = [float("inf"), 0, 0]              # cost, deficit, surplus

    def rec(ci, used, err_acc, leftover):
        if err_acc >= best[0]:
            return
        if ci == len(masses):
            rem = sorted(areas[i] for i in range(n) if not (used >> i) & 1)
            surplus = 0
            if rem and leftover:
                # pool covers at most ONE object per leftover comp: it exists
                # to reassemble an occlusion-split object from its fragments,
                # not to let a single swallowed-pair blob pay for two objects
                pool, pops = sum(leftover), 0
                while rem and pops < len(leftover) and pool >= 0.4 * rem[0]:
                    pool -= rem.pop(0)
                    pops += 1
            elif not rem:
                surplus = sum(max(1, round(ms / amed)) for ms in leftover
                              if ms >= 0.5 * amin)
            cost = err_acc + 100.0 * (len(rem) + surplus)
            if cost < best[0]:
                best[:] = [cost, len(rem), surplus]
            return
        ms = masses[ci]
        rec(ci + 1, used, err_acc, leftover + [ms])   # unexplained blob
        free = full & ~used
        sub = free
        while sub:
            e = band_err(ms, sub)
            if e <= 25.0:                    # beyond that the story is wrong
                rec(ci + 1, used | sub, err_acc + e, leftover)
            sub = (sub - 1) & free

    rec(0, 0, 0.0, [])
    return best[1], best[2]


def _count_pair(payload):
    pred, objs, thetas, outline = payload
    # env var (not a module flag: it must survive into spawned pool
    # workers): widen every BALL's expected mass into an interval spanning
    # its true size and the trained radius band r5-8
    widen = os.environ.get("CV3_WIDEN_BALL_BANDS") == "1"
    trained = {}
    mats = {}
    for i, o in enumerate(objs):
        tpl, _, _ = render_template(o, thetas[i], outline=outline)
        a = float(tpl.sum())
        if widen and o["kind"] == "ball":
            key = o["material"]
            if key not in trained:
                lo_t = float(render_template(
                    {"kind": "ball", "size": 5.0, "material": key,
                     "verts": None}, 0.0, outline=outline)[0].sum())
                hi_t = float(render_template(
                    {"kind": "ball", "size": 8.0, "material": key,
                     "verts": None}, 0.0, outline=outline)[0].sum())
                trained[key] = (lo_t, hi_t)
            lo_t, hi_t = trained[key]
            mats.setdefault(key, []).append(
                (min(a, lo_t), max(a, hi_t)))
        else:
            mats.setdefault(o["material"], []).append(a)
    # per material, not per frame: each call renders a calibration image
    colors = {mat: _fill_outline_rgb(mat) for mat in mats}
    # bordered looks: the gray frame is within COLOR_TOL of Steel's fill
    # and merges with every object resting on it; exclude it (the
    # deformation scorer does too)
    bpx = detect_border_px(pred[0])
    deficit = np.zeros(pred.shape[0])
    surplus = np.zeros(pred.shape[0])
    for t in range(pred.shape[0]):
        f = pred[t].astype(np.float64)
        for mat, areas in mats.items():
            w = _color_mask(f, *colors[mat])
            if bpx:
                k = bpx + 1                  # +1 for the AA blend row
                w[:k] = 0.0
                w[-k:] = 0.0
                w[:, :k] = 0.0
                w[:, -k:] = 0.0
            m = (w >= 0.5).astype(np.uint8)
            n_comp, lab = cv2.connectedComponents(m)
            # comp mass must be the SOFT coverage sum, same as the template
            # areas -- binary-counting the AA skirt inflates small objects
            # and mints phantom shapes
            soft = np.bincount(lab.ravel(), weights=w.astype(np.float64).ravel(),
                               minlength=n_comp)
            masses = sorted((float(s) for s in soft[1:] if s >= 8.0),
                            reverse=True)
            d, s = _match_counts(masses, areas)
            deficit[t] += d
            surplus[t] += s
    return deficit, surplus


def main():
    import torch
    from multiprocessing import get_context
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}
    for label, run_dir, color in MODELS:
        print(label, flush=True)
        simdir = SIMDIRS[label]
        ae, dit, cfg = load_run(run_dir, "cuda")
        trajs = [os.path.join(simdir, f"traj-{t}") for t in range(N_SCENES)]
        curves = []
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
                metas.append((objs, ang[0, :, 0],
                              detect_outline(frames_np[0], objs, pos[0],
                                             ang[0, :, 0])))
                ctxs.append(frames_np[:CTX])
                seeds.append(zlib.crc32(os.path.basename(td).encode()) & 0x7FFFFFFF)
            preds = _batched_rollout(ae, dit, ctxs, seeds, N_STEPS)
            payloads = [(pred, *meta) for pred, meta in zip(preds, metas)]
            if pending is not None:
                curves.extend(pending.get())
            pending = pool.map_async(_count_pair, payloads)
            print(f"  {min(gi + GROUP, N_SCENES)}/{N_SCENES}", flush=True)
        curves.extend(pending.get())
        pool.close()
        pool.join()
        m = np.stack([c[0] for c in curves])      # (scenes, T) missing count
        sp = np.stack([c[1] for c in curves])     # (scenes, T) surplus count
        out[label] = {"mean_missing": m.mean(axis=0).tolist(),
                      "mean_surplus": sp.mean(axis=0).tolist(),
                      "scenes_missing_f300": float((m[:, -1] >= 1).mean()),
                      "scenes_missing_sustained": float(
                          (m[:, -60:].min(axis=1) >= 1).mean()),
                      "scenes_surplus_sustained": float(
                          (sp[:, -60:].min(axis=1) >= 1).mean())}
        print(f"{label}: mean missing f80 {m[:, 79].mean():.3f} | "
              f"f300 {m[:, -1].mean():.3f} | scenes w/ sustained loss "
              f"{out[label]['scenes_missing_sustained']*100:.1f}% | surplus "
              f"f300 {sp[:, -1].mean():.3f} (sustained "
              f"{out[label]['scenes_surplus_sustained']*100:.1f}%)", flush=True)
    json.dump(out, open("scratchpad/examples/shape_count.json", "w"))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    x = np.arange(1, N_STEPS + 1)
    gt_path = "scratchpad/examples/gt_count.json"
    if os.path.exists(gt_path):
        gt = json.load(open(gt_path))
        g = np.array(gt["mean_missing"])
        ax.plot(x, g[:N_STEPS], color="#94A3B8", linewidth=1.6,
                linestyle="--", label="real sim frames (tool floor)")
    for label, _, color in MODELS:
        m = np.array(out[label]["mean_missing"])
        ax.plot(x, m, color=color, linewidth=2, label=label)
        ax.annotate(label.split(" (")[0], (x[-1], m[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", fontweight="bold")
    ax.set_xlabel("rollout frame")
    ax.set_ylabel("missing shapes per scene (count deficit)")
    ax.set_title("Identity-free shape counting: how many objects are missing?",
                 fontsize=12, pad=10)
    ax.set_xlim(1, N_STEPS + 60)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(0.01, 0.005, "500 scenes; detected = connected color regions per "
             "material, merged blobs counted by mass; transient dips = "
             "same-material contacts", fontsize=7.5, color="#666666")
    fig.tight_layout()
    fig.savefig("scratchpad/examples/shape_count.png", bbox_inches="tight")
    print("saved scratchpad/examples/shape_count.png")


if __name__ == "__main__":
    main()
