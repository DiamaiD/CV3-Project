"""Eccentricity-OOD eval (train e<=0.25, test e 0.3-0.6): does the model keep
Kepler's SECOND law -- equal areas in equal times -- on orbits more oval than
anything it saw? Per surviving planet, from tracked positions:
- swept-area rate L(t)/2 with L = x*vy - y*vx: constant on ANY Kepler orbit;
  metric = std(L)/|mean(L)| over f10-250 (GT-control defines the tracker
  floor of the same statistic);
- speed modulation: v_peri/v_apo vs the true (1+e)/(1-e);
- plus the standard radius/phase split via orbit_errors.
One rollout pass on solar_ecc. Prints the comparison; saves solar_ecc.json."""
import json
import os
import zlib

import numpy as np

from experiments.deformation import _batched_rollout
from experiments.rollout import load_run
from experiments.solar_metrics import orbit_errors, track_bodies
from src.frameio import load_frames

RUN = "runs/2026_08_01__17_35_56__solar_200k_s2_cont"
SIMDIR = "scratchpad/examples/solar_ecc"
CTX = 5
STEPS = 300
N_SCENES = 500
GROUP = 96
W0, W1 = 10, 250


def _kepler2(tracked_world, n_star, periods):
    """Per-planet swept-rate constancy + peri/apo speed ratio from tracks."""
    out = []
    for j in range(tracked_world.shape[1] - n_star):
        p = tracked_world[W0:W1, n_star + j] - 32.0
        v = np.zeros_like(p)
        v[1:-1] = (p[2:] - p[:-2]) / 2.0
        p, v = p[1:-1], v[1:-1]
        L = p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]
        if abs(np.mean(L)) < 1e-6:
            continue
        cv = float(np.std(L) / abs(np.mean(L)))
        r = np.linalg.norm(p, axis=1)
        sp = np.linalg.norm(v, axis=1)
        # sample speeds at the radius extremes actually visited
        lo_i, hi_i = np.argsort(r)[:8], np.argsort(r)[-8:]
        ratio = float(sp[lo_i].mean() / max(sp[hi_i].mean(), 1e-6))
        out.append((cv, ratio, j))
    return out


def _score(payload):
    pred, objs, sol, init, gt_pos = payload
    tracked, presence = track_bodies(pred, objs, init)
    n_star = len(sol["stars"])
    periods = [p["T"] for p in sol["planets"]]
    t_arr = np.arange(CTX, CTX + pred.shape[0], dtype=np.float64)
    err = orbit_errors(tracked, sol, t_arr)
    k2_m = _kepler2(tracked, n_star, periods)
    k2_g = _kepler2(gt_pos, n_star, periods)
    true_ratio = [(1 + p["e"]) / (1 - p["e"]) for p in sol["planets"]]
    alive = [presence[W0:W1, n_star + j].min() >= 0.5
             for j in range(len(periods))]
    return (err["pos_err"][[49, 149, -1]].mean(axis=1),
            np.abs(err["radius_err"][-1]).mean(),
            k2_m, k2_g, true_ratio, alive,
            err["pos_err"].mean(axis=1), np.abs(err["radius_err"]).mean(axis=1))


def main():
    import time
    from multiprocessing import get_context

    ae, dit, cfg = load_run(RUN, "cuda")
    trajs = ([os.path.join(SIMDIR, f"traj-{t}") for t in range(N_SCENES)]
             + [os.path.join(SIMDIR + "2", f"traj-{t}")
                for t in range(N_SCENES)])
    t0 = time.time()
    pool = get_context("spawn").Pool(12)
    pending, results = None, []
    for gi in range(0, len(trajs), GROUP):
        group = trajs[gi:gi + GROUP]
        ctxs, seeds, metas = [], [], []
        for td in group:
            frames_np = load_frames(td)
            objs = json.load(open(os.path.join(td, "objects.json")))
            pos = np.load(os.path.join(td, "positions.npy"))
            sol = json.load(open(os.path.join(td, "solar.json")))
            ctxs.append(frames_np[:CTX])
            seeds.append(zlib.crc32(os.path.basename(td).encode()) & 0x7FFFFFFF)
            metas.append((objs, sol, pos[CTX], pos[CTX:CTX + STEPS]))
        preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS)
        payloads = [(pred, o, s, ip, gp)
                    for pred, (o, s, ip, gp) in zip(preds, metas)]
        if pending is not None:
            results.extend(pending.get())
        pending = pool.map_async(_score, payloads)
        print(f"  {min(gi + GROUP, N_SCENES)}/{N_SCENES} "
              f"({time.time() - t0:.0f}s)", flush=True)
    results.extend(pending.get())
    pool.close()
    pool.join()

    pos = np.stack([r[0] for r in results])
    rad = np.array([r[1] for r in results])
    cv_m = [c for r in results for (c, _, j) in r[2] if r[5][j]]
    cv_g = [c for r in results for (c, _, j) in r[3]]
    ratios = [(m[1], t) for r in results for m, t in
              zip(sorted(r[2], key=lambda x: x[2]),
                  [r[4][x[2]] for x in sorted(r[2], key=lambda x: x[2])])
              if r[5][m[2]]]
    ratio_err = [abs(m - t) / t for m, t in ratios]
    out = {"pos_f50": float(pos[:, 0].mean()), "pos_f150": float(pos[:, 1].mean()),
           "pos_f300": float(pos[:, 2].mean()), "rad_f300": float(rad.mean()),
           "kepler2_cv_model": float(np.median(cv_m)),
           "kepler2_cv_control": float(np.median(cv_g)),
           "speed_ratio_relerr_med": float(np.median(ratio_err)),
           "n_planets": len(cv_m),
           "pos_curve": np.stack([r[6] for r in results]).mean(axis=0).tolist(),
           "pos_std": np.stack([r[6] for r in results]).std(axis=0).tolist(),
           "rad_curve": np.stack([r[7] for r in results]).mean(axis=0).tolist(),
           "rad_std": np.stack([r[7] for r in results]).std(axis=0).tolist()}
    json.dump(out, open("scratchpad/examples/solar_ecc.json", "w"))
    print(f"\nECC OOD (e 0.3-0.6, trained <=0.25), {len(cv_m)} surviving "
          f"planets:")
    print(f"pos err f50 {out['pos_f50']:.2f} | f150 {out['pos_f150']:.2f} | "
          f"f300 {out['pos_f300']:.2f} px | radius |drift| f300 "
          f"{out['rad_f300']:.2f}")
    print(f"Kepler II swept-rate CV: model {out['kepler2_cv_model']:.3f} vs "
          f"tracker control {out['kepler2_cv_control']:.3f}")
    print(f"peri/apo speed-ratio rel err median "
          f"{out['speed_ratio_relerr_med']*100:.1f}%")
    print(f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
