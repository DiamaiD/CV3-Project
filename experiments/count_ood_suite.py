"""Count-OOD suite for the balls-v3 world (train dist 2-5 balls): scenes with
exactly 3/5/7/9 balls, 300-frame rollouts of the balls-60k model, measured
with the validated stack: identity-free counting survival (zero-floor
counter), metric-v5 deformation excess over the VAE-roundtrip floor
(mean +- std across object-frame samples), and tracked energy vs simulator
(per-scene ratio, mean +- std across scenes). GT counting control per set
gates everything. Outputs count_ood.json + 3 graphs."""
import json
import os
import zlib

import numpy as np

from experiments.deformation import _batched_rollout, detect_outline, score_frames
from experiments.rollout import load_run
from experiments.energycurve_norot import energy_series, obj_mass_low
from experiments.shape_count import _count_pair
from src.frameio import load_frames

RUN = "runs/2026_07_30__02_46_32__balls_easy60k_norot_v3"
COUNTS = [3, 5, 7, 9]
COLORS = {3: "#2563EB", 5: "#059669", 7: "#E8710A", 9: "#DC2626"}
CTX = 5
STEPS = 300
N_SCENES = 500
GROUP = 128


def _score(payload):
    pred, rec, objs, thetas, init, outline, masses, lows, E0, E_sim = payload
    deficit, surplus = _count_pair((pred, objs, thetas, outline))
    ms, mc, tracks = score_frames(pred, objs, thetas, init, outline=outline)
    gs, gc, _ = score_frames(rec, objs, thetas, init, outline=outline)
    ok = ~mc & ~gc                                # clean object-frame samples
    d = np.where(ok, ms - gs, np.nan)             # (T, n) excess over floor
    p = np.stack([tracks[:, :, 1] + 0.5, 64.0 - tracks[:, :, 0] - 0.5], axis=2)
    v = np.zeros_like(p)
    v[1:-1] = (p[2:] - p[:-2]) / 2.0
    v[0], v[-1] = p[1] - p[0], p[-1] - p[-2]
    E_mod = energy_series(p, v, masses, lows)
    e0 = max(float(np.sum(E0)), 1e-9)
    return (deficit, int((surplus > 0).sum()),
            np.nansum(d, axis=1), np.nansum(d * d, axis=1),
            (~np.isnan(d)).sum(axis=1),
            E_mod.sum(axis=1) / e0, E_sim.sum(axis=1) / e0)


def _gt_count(td):
    frames = load_frames(td)
    objs = json.load(open(os.path.join(td, "objects.json")))
    ang = np.load(os.path.join(td, "angles.npy"))
    pos = np.load(os.path.join(td, "positions.npy"))
    outline = detect_outline(frames[0], objs, pos[0], ang[0, :, 0])
    d, s = _count_pair((frames[CTX:CTX + STEPS], objs, ang[0, :, 0], outline))
    return int((d > 0).sum()), int((s > 0).sum())


def main():
    import time
    import torch
    from multiprocessing import get_context

    ae, dit, cfg = load_run(RUN, "cuda")
    out = {}
    for cnt in COUNTS:
        simdir = f"scratchpad/examples/ood_count_c{cnt}"
        trajs = [os.path.join(simdir, f"traj-{t}") for t in range(N_SCENES)]
        ext = simdir + "b"
        if os.path.isdir(ext):
            trajs += [os.path.join(ext, f"traj-{t}") for t in range(N_SCENES)]
        with get_context("spawn").Pool(24) as pool:
            g = pool.map(_gt_count, trajs)
        print(f"c{cnt}: GT counting control deficit-frames "
              f"{sum(x[0] for x in g)} surplus-frames {sum(x[1] for x in g)} "
              f"(of {len(trajs) * STEPS})", flush=True)

        t0 = time.time()
        pool = get_context("spawn").Pool(20)
        pending, results = None, []
        for gi in range(0, len(trajs), GROUP):
            group = trajs[gi:gi + GROUP]
            ctxs, seeds, metas, gts = [], [], [], []
            for td in group:
                frames_np = load_frames(td)
                objs = json.load(open(os.path.join(td, "objects.json")))
                ang = np.load(os.path.join(td, "angles.npy"))
                pos = np.load(os.path.join(td, "positions.npy"))
                vel = np.load(os.path.join(td, "velocities.npy"))
                thetas = ang[0, :, 0]
                ml = [obj_mass_low(o, thetas[i]) for i, o in enumerate(objs)]
                masses, lows = [m for m, _ in ml], [l for _, l in ml]
                E_sim = energy_series(pos, vel, masses, lows)
                metas.append((objs, thetas, pos[CTX - 1],
                              detect_outline(frames_np[0], objs, pos[0], thetas),
                              masses, lows, E_sim[CTX],
                              E_sim[CTX:CTX + STEPS]))
                ctxs.append(frames_np[:CTX])
                seeds.append(zlib.crc32(os.path.basename(td).encode()) & 0x7FFFFFFF)
                gts.append(frames_np[CTX:CTX + STEPS])
            preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS)
            flat = np.concatenate(gts)
            with torch.no_grad():
                xg = (torch.from_numpy(flat).permute(0, 3, 1, 2).float()
                      .div_(255.0).cuda())
                rec = torch.cat([ae.decode(ae.encode(xg[i:i + 1024])).clamp(0, 1)
                                 for i in range(0, xg.shape[0], 1024)])
            rec = (rec.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
            payloads, ofs = [], 0
            for pred, meta, gt in zip(preds, metas, gts):
                payloads.append((pred[:len(gt)], rec[ofs:ofs + len(gt)], *meta))
                ofs += len(gt)
            if pending is not None:
                results.extend(pending.get())
            pending = pool.map_async(_score, payloads)
            print(f"  c{cnt} {min(gi + GROUP, len(trajs))}/{len(trajs)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        results.extend(pending.get())
        pool.close()
        pool.join()

        deficit = np.stack([r[0] for r in results])          # (S, T)
        surplus_frames = sum(r[1] for r in results)
        dsum = np.stack([r[2] for r in results]).sum(axis=0)
        dsq = np.stack([r[3] for r in results]).sum(axis=0)
        dn = np.stack([r[4] for r in results]).sum(axis=0).astype(float)
        dmean = dsum / np.maximum(dn, 1)
        dstd = np.sqrt(np.maximum(dsq / np.maximum(dn, 1) - dmean ** 2, 0.0))
        e_mod = np.stack([r[5] for r in results])             # (S, T)
        e_sim = np.stack([r[6] for r in results])
        out[cnt] = {
            "survival": (1.0 - deficit.mean(axis=0) / cnt).tolist(),
            "survival_f300": float(1.0 - deficit[:, -1].mean() / cnt),
            "surplus_frames": surplus_frames,
            "deform_mean": dmean.tolist(), "deform_std": dstd.tolist(),
            "deform_n_f300": int(dn[-1]),
            "e_mod_mean": e_mod.mean(axis=0).tolist(),
            "e_mod_std": e_mod.std(axis=0).tolist(),
            "e_sim_mean": e_sim.mean(axis=0).tolist(),
            "gap_f300": float((e_mod[:, -1] - e_sim[:, -1]).mean())}
        print(f"c{cnt}: survival f300 {out[cnt]['survival_f300']*100:.1f}% | "
              f"surplus frames {surplus_frames} | deform f80 {dmean[79]:.3f} "
              f"f300 {dmean[-1]:.3f} (std {dstd[-1]:.3f}) | energy gap f300 "
              f"{out[cnt]['gap_f300']:+.3f} | {time.time() - t0:.0f}s",
              flush=True)
    json.dump(out, open("scratchpad/examples/count_ood.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(1, STEPS + 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    for cnt in COUNTS:
        tag = " (OOD)" if cnt > 5 else ""
        ax.plot(x, np.array(out[cnt]["survival"]) * 100, color=COLORS[cnt],
                linewidth=2, label=f"{cnt} balls{tag}")
    ax.axhline(100, color="#94A3B8", linewidth=1.2, linestyle="--")
    ax.set_xlabel("rollout frame")
    ax.set_ylabel("balls present (%)")
    ax.set_title("Count OOD: survival (identity-free count) over 300-frame "
                 "rollouts", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig("scratchpad/examples/ood_survival.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    for cnt in COUNTS:
        m = np.array(out[cnt]["deform_mean"])
        s = np.array(out[cnt]["deform_std"])
        tag = " (OOD)" if cnt > 5 else ""
        ax.plot(x, m, color=COLORS[cnt], linewidth=2, label=f"{cnt} balls{tag}")
        ax.fill_between(x, m - s, m + s, color=COLORS[cnt], alpha=0.13,
                        linewidth=0)
    ax.set_xlabel("rollout frame")
    ax.set_ylabel("deformation excess over VAE floor (mass fraction)")
    ax.set_title("Count OOD: ball deformation over 300-frame rollouts "
                 "(mean ± std)", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig("scratchpad/examples/ood_deform.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    for cnt in COUNTS:
        m = np.array(out[cnt]["e_mod_mean"])
        s = np.array(out[cnt]["e_mod_std"])
        tag = " (OOD)" if cnt > 5 else ""
        ax.plot(x, m, color=COLORS[cnt], linewidth=2,
                label=f"model, {cnt} balls{tag}")
        ax.fill_between(x, m - s, m + s, color=COLORS[cnt], alpha=0.12,
                        linewidth=0)
        ax.plot(x, np.array(out[cnt]["e_sim_mean"]), color=COLORS[cnt],
                linewidth=1.1, linestyle="--", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("rollout frame")
    ax.set_ylabel("total energy (fraction of context energy, log)")
    ax.set_title("Count OOD: energy vs simulator (dashed) over 300-frame "
                 "rollouts", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig("scratchpad/examples/ood_energy.png", bbox_inches="tight")
    print("saved ood_survival.png + ood_deform.png + ood_energy.png")


if __name__ == "__main__":
    main()
