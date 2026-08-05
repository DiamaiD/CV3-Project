"""Held-out-kind evaluation: the no-ball shapes model vs the regular shapes
model (in-distribution control) on the SAME 500 ball-guaranteed scenes
(300-frame rollouts, ns3, identical per-scene seeds). Metrics split balls
vs other shapes:
  - survival: per-object template presence (> 0.5 aligned overlap), with the
    VAE-roundtrip control tracked identically as the instrument ceiling
  - deformation: metric v5 excess over the roundtrip floor
  - energy: tracked total energy vs matched simulator (global)
  - identity-free count deficit (global gate + curve)
Ball free-flight arcs (balls only; isolation against ALL objects) feed the
whole-arc fit for drag (ceiling-calibrated) and free-g gravity.
Saves heldout_eval.json."""
import json
import os
import zlib

import numpy as np

from experiments.deformation import (_batched_rollout, detect_outline,
                                     score_frames)
from experiments.rollout import load_run
from experiments import drag_arcfit
from experiments.energycurve_norot import energy_series, obj_mass_low
from experiments.shape_count import _count_pair
from src.frameio import load_frames

SIMDIR = "scratchpad/examples/heldout_mix500"   # extended past 500 scenes
CTX = 5
STEPS = 300
GROUP = 128
RUNS = [
    ("control", "runs/2026_07_29__03_57_19__shapes_easy60k_norot_v3"),
    ("noball", "runs/2026_08_02__12_14_57__shapes_noball_60k"),
]


def _ball_segments(pos, objs, cont, min_len=12, max_len=40):
    """segments_from_tracks, but segments only for BALL objects (the drag
    model's mass/area formula is ball-specific); isolation still respects
    every object in the scene."""
    T, n = pos.shape[0], pos.shape[1]
    radii = np.array([float(o["size"]) for o in objs])
    segs = []
    for i in range(n):
        if objs[i]["kind"] != "ball":
            continue
        r = radii[i]
        ok = np.zeros(T, dtype=bool)
        for t in range(T):
            if cont is not None and cont[t, i]:
                continue
            x, y = pos[t, i]
            if not (r + 2.5 < x < 64 - r - 2.5 and r + 2.5 < y < 64 - r - 2.5):
                continue
            d = np.linalg.norm(pos[t] - pos[t, i], axis=1)
            d[i] = 99
            if d.min() < r + radii.max() + 6.0:
                continue
            ok[t] = True
        t = 0
        while t < T:
            if ok[t]:
                e = t
                while e < T and ok[e] and e - t < max_len:
                    e += 1
                if e - t >= min_len:
                    P = pos[t:e, i]
                    sp = np.linalg.norm(np.diff(P, axis=0), axis=1).mean()
                    if sp > 1.0:
                        segs.append((P.copy(), float(r),
                                     float(np.pi * r * r
                                           * drag_arcfit.MATERIALS[
                                               objs[i]["material"]]
                                           ["density"])))
                t = e
            else:
                t += 1
    return segs


def _to_world(tracks):
    return np.stack([tracks[:, :, 1] + 0.5, 64.0 - tracks[:, :, 0] - 0.5],
                    axis=2)


def _debounce(pr):
    """Hysteresis on the presence fraction: an object flips to lost only
    below 0.35 and back to present only above 0.65, so objects hovering at
    the plain 0.5 threshold cannot flicker the survival curve."""
    T, n = pr.shape
    st = np.ones((T, n), dtype=bool)
    alive = np.ones(n, dtype=bool)
    for t in range(T):
        alive = np.where(pr[t] < 0.35, False,
                         np.where(pr[t] > 0.65, True, alive))
        st[t] = alive
    return st


def _score(payload):
    pred, rec, objs, thetas, init, outline, masses, lows, E0, E_sim = payload
    ball = np.array([o["kind"] == "ball" for o in objs])
    deficit, surplus = _count_pair((pred, objs, thetas, outline))
    ms, mc, mtr, mpr = score_frames(pred, objs, thetas, init, outline=outline,
                                    with_presence=True)
    gs, gc, gtr, gpr = score_frames(rec, objs, thetas, init, outline=outline,
                                    with_presence=True)
    ok = ~mc & ~gc
    d = np.where(ok, ms - gs, np.nan)                       # (T, n)
    out = {"deficit": deficit, "surplus_frames": int((surplus > 0).sum())}
    for cls, mask in (("ball", ball), ("other", ~ball)):
        dm = d[:, mask]
        out[f"d_sum_{cls}"] = np.nansum(dm, axis=1)
        out[f"d_sq_{cls}"] = np.nansum(dm * dm, axis=1)
        out[f"d_n_{cls}"] = (~np.isnan(dm)).sum(axis=1)
        out[f"pres_m_{cls}"] = _debounce(mpr[:, mask]).sum(axis=1)
        out[f"pres_g_{cls}"] = _debounce(gpr[:, mask]).sum(axis=1)
        out[f"pres_raw_m_{cls}"] = (mpr[:, mask] > 0.5).sum(axis=1)
        out[f"n_{cls}"] = int(mask.sum())
    p = _to_world(mtr)
    v = np.zeros_like(p)
    v[1:-1] = (p[2:] - p[:-2]) / 2.0
    v[0], v[-1] = p[1] - p[0], p[-1] - p[-2]
    e0 = max(float(np.sum(E0)), 1e-9)
    out["e_mod"] = energy_series(p, v, masses, lows).sum(axis=1) / e0
    out["e_sim"] = E_sim.sum(axis=1) / e0
    out["segs_m"] = _ball_segments(p, objs, mc)
    out["segs_g"] = _ball_segments(_to_world(gtr), objs, gc)
    return out


def _gt_count(td):
    frames = load_frames(td)
    objs = json.load(open(os.path.join(td, "objects.json")))
    ang = np.load(os.path.join(td, "angles.npy"))
    pos = np.load(os.path.join(td, "positions.npy"))
    outline = detect_outline(frames[0], objs, pos[0], ang[0, :, 0])
    d, s = _count_pair((frames[CTX:CTX + STEPS], objs, ang[0, :, 0], outline))
    return int((d > 0).sum()), int((s > 0).sum())


def _fit_arcs(segs, workers, free_g):
    from multiprocessing import get_context
    args = [(P, r, m, free_g) for P, r, m in segs]
    with get_context("spawn").Pool(workers) as pool:
        got = [g for g in pool.map(drag_arcfit._work, args) if g is not None]
    return (np.array([g[0] for g in got]),
            np.array([g[3] for g in got]))


def _med_se(cs):
    return float(np.median(cs)), float(1.253 * cs.std() / np.sqrt(len(cs)))


def main():
    import time

    import torch
    from multiprocessing import get_context

    n_scenes = sum(1 for d in os.listdir(SIMDIR) if d.startswith("traj-"))
    trajs = [os.path.join(SIMDIR, f"traj-{t}") for t in range(n_scenes)]
    with get_context("spawn").Pool(24) as pool:
        g = pool.map(_gt_count, trajs)
    print(f"GT counting control: deficit-frames {sum(x[0] for x in g)} "
          f"surplus-frames {sum(x[1] for x in g)} "
          f"(of {len(trajs) * STEPS})", flush=True)

    out = {}
    for label, run in RUNS:
        ae, dit, cfg = load_run(run, "cuda")
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
                              detect_outline(frames_np[0], objs, pos[0],
                                             thetas),
                              masses, lows, E_sim[CTX],
                              E_sim[CTX:CTX + STEPS]))
                ctxs.append(frames_np[:CTX])
                seeds.append(zlib.crc32(os.path.basename(td).encode())
                             & 0x7FFFFFFF)
                gts.append(frames_np[CTX:CTX + STEPS])
            preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS)
            flat = np.concatenate(gts)
            with torch.no_grad():
                xg = (torch.from_numpy(flat).permute(0, 3, 1, 2).float()
                      .div_(255.0).cuda())
                rec = torch.cat([ae.decode(ae.encode(xg[i:i + 1024]))
                                 .clamp(0, 1)
                                 for i in range(0, xg.shape[0], 1024)])
            rec = (rec.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
            payloads, ofs = [], 0
            for pred, meta, gt in zip(preds, metas, gts):
                payloads.append((pred[:len(gt)], rec[ofs:ofs + len(gt)],
                                 *meta))
                ofs += len(gt)
            if pending is not None:
                results.extend(pending.get())
            pending = pool.map_async(_score, payloads)
            print(f"  {label} {min(gi + GROUP, len(trajs))}/{len(trajs)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        results.extend(pending.get())
        pool.close()
        pool.join()

        o = {"deficit_mean": np.stack([r["deficit"] for r in results])
             .mean(axis=0).tolist(),
             "surplus_frames": sum(r["surplus_frames"] for r in results)}
        for cls in ("ball", "other"):
            n_tot = sum(r[f"n_{cls}"] for r in results)
            pm = np.stack([r[f"pres_m_{cls}"] for r in results]).sum(axis=0)
            pg = np.stack([r[f"pres_g_{cls}"] for r in results]).sum(axis=0)
            pr = np.stack([r[f"pres_raw_m_{cls}"] for r in results]) \
                .sum(axis=0)
            o[f"n_{cls}"] = n_tot
            o[f"survival_{cls}"] = (pm / n_tot).tolist()
            o[f"survival_raw_{cls}"] = (pr / n_tot).tolist()
            o[f"survival_rt_{cls}"] = (pg / n_tot).tolist()
            dsum = np.stack([r[f"d_sum_{cls}"] for r in results]).sum(axis=0)
            dsq = np.stack([r[f"d_sq_{cls}"] for r in results]).sum(axis=0)
            dn = np.stack([r[f"d_n_{cls}"] for r in results]) \
                .sum(axis=0).astype(float)
            dmean = dsum / np.maximum(dn, 1)
            o[f"deform_mean_{cls}"] = dmean.tolist()
            o[f"deform_std_{cls}"] = np.sqrt(np.maximum(
                dsq / np.maximum(dn, 1) - dmean ** 2, 0.0)).tolist()
            o[f"deform_n_{cls}"] = dn.tolist()
        e_mod = np.stack([r["e_mod"] for r in results])
        e_sim = np.stack([r["e_sim"] for r in results])
        o["e_mod_mean"] = e_mod.mean(axis=0).tolist()
        o["e_mod_std"] = e_mod.std(axis=0).tolist()
        o["e_sim_mean"] = e_sim.mean(axis=0).tolist()

        segs_m = [s for r in results for s in r["segs_m"]]
        segs_g = [s for r in results for s in r["segs_g"]]
        o["arcs"] = {}
        if len(segs_m) >= 20 and len(segs_g) >= 20:
            cs_m, _ = _fit_arcs(segs_m, 24, False)
            cs_g, _ = _fit_arcs(segs_g, 24, False)
            _, gfree_m = _fit_arcs(segs_m, 24, True)
            m_med, m_se = _med_se(cs_m)
            c_med, c_se = _med_se(cs_g)
            cal = m_med / c_med * 0.02
            cal_se = cal * np.sqrt((m_se / m_med) ** 2 + (c_se / c_med) ** 2)
            o["arcs"] = {"model": [m_med, m_se, len(cs_m)],
                         "ceiling": [c_med, c_se, len(cs_g)],
                         "calibrated": [float(cal), float(cal_se)],
                         "freeg_g": float(np.median(gfree_m)),
                         "cs_model": cs_m.tolist(),
                         "cs_ceiling": cs_g.tolist()}
            print(f"{label} ARCS: model {m_med:.5f}±{m_se:.5f} "
                  f"({len(cs_m)}) | ceiling {c_med:.5f}±{c_se:.5f} "
                  f"({len(cs_g)}) | CAL {cal:.5f}±{cal_se:.5f} | "
                  f"g {np.median(gfree_m):+.4f}", flush=True)
        else:
            print(f"{label} ARCS: too few segments "
                  f"(model {len(segs_m)}, ceiling {len(segs_g)})", flush=True)
        sb = o["survival_ball"][-1] * 100
        so = o["survival_other"][-1] * 100
        rb = o["survival_rt_ball"][-1] * 100
        print(f"{label}: f300 survival balls {sb:.1f}% others {so:.1f}% "
              f"(roundtrip ctrl balls {rb:.1f}%) | deficit f300 "
              f"{o['deficit_mean'][-1]:.3f} | surplus frames "
              f"{o['surplus_frames']} | {time.time() - t0:.0f}s", flush=True)
        out[label] = o
    json.dump(out, open("scratchpad/examples/heldout_eval.json", "w"))
    print("saved heldout_eval.json")


if __name__ == "__main__":
    main()
