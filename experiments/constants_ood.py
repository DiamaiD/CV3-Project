"""Material-constants extraction under count OOD: does the balls-60k model
KEEP the physics constants at 7 and 9 balls (never seen in training)?

From the same deterministic rollouts as the count-OOD suite (identical crc32
seeds), tracked with the v5 tracker:
- GRAVITY + DRAG: free-flight second differences. env_pymunk: dv = g*dt -
  (C/m)*v|v|*r_eff*dt with C=AIR_DRAG_COEFF, 16 substeps -> per-frame
  d2y ~= g - (C r/m) vy|v| to first order. Joint (g, c_hat) least squares on
  airborne segments (far from walls and every other ball), MAD-trimmed.
- WALL RESTITUTION per material: vertical floor bounces; pymunk product rule
  (wall e=1.0) -> ratio |vy_out/vy_in| = material restitution. Events: track
  crosses y < r+1.5 with vy<0 then vy>0; ratio from velocities 2 frames off
  the impact (avoids the contact frames themselves).
Controls: the same extraction on the GT sims of the same scenes (the fits
must recover g=-0.5 / restitutions exactly there, or the instrument is
biased). Per count: 3/5/7/9. Outputs constants_ood.json + printed table."""
import json
import os
import zlib

import numpy as np

from environments.env_bouncing import AIR_DRAG_COEFF, GRAVITY, MATERIALS
from experiments.deformation import _batched_rollout, detect_outline, score_frames
from experiments.rollout import load_run
from src.frameio import load_frames

RUN = "runs/2026_07_30__02_46_32__balls_easy60k_norot_v3"
COUNTS = [3, 5, 7, 9]
CTX = 5
STEPS = 300
N_SCENES = 500
GROUP = 96
MARGIN_WALL = 3.0        # extra clearance beyond radius for "free flight"
MARGIN_BALL = 2.5


def _tracks_world(pred, objs, thetas, init, outline):
    _, mc, tracks = score_frames(pred, objs, thetas, init, outline=outline)
    p = np.stack([tracks[:, :, 1] + 0.5, 64.0 - tracks[:, :, 0] - 0.5], axis=2)
    return p, mc


def _collect(payload):
    """Free-flight second-difference rows + floor-bounce events from tracks.
    payload = (pos, objs, contaminated[, stencil_h]) -- h widens the second
    difference (p[t+h]-2p[t]+p[t-h])/h^2: a low-pass filter against the
    sub-pixel phase ripple of tracked positions, whose speed-dependent
    frequency rectifies into fake drag at h=1."""
    pos, objs, contaminated = payload[:3]
    h = payload[3] if len(payload) > 3 else 1
    T, n = pos.shape[0], pos.shape[1]
    radii = np.array([float(o["size"]) for o in objs])
    rows = []                                  # (d2y, vy|v|*r/m) for g,c fit
    bounces = []                               # (material, ratio)
    v = np.zeros_like(pos)
    v[1:-1] = (pos[2:] - pos[:-2]) / 2.0
    masses = np.array([np.pi * float(o["size"]) ** 2
                       * MATERIALS[o["material"]]["density"] for o in objs])
    for i in range(n):
        r = radii[i]
        for t in range(h + 1, T - h - 1):
            # isolation, wall clearance and tracker health must hold on the
            # WHOLE stencil (t-h..t+h), or a ball one frame out of a pile
            # injects a contact-contaminated position (speed-correlated bias)
            clean = True
            for tt in range(t - h, t + h + 1):
                if contaminated is not None and contaminated[tt, i]:
                    clean = False
                    break
                x, y = pos[tt, i]
                if not (r + MARGIN_WALL < x < 64 - r - MARGIN_WALL
                        and r + MARGIN_WALL < y < 64 - r - MARGIN_WALL):
                    clean = False
                    break
                d = np.linalg.norm(pos[tt] - pos[tt, i], axis=1)
                d[i] = 99
                if d.min() < r + radii.max() + MARGIN_BALL:
                    clean = False
                    break
            if not clean:
                continue
            sp = np.linalg.norm(v[t, i])
            # motion gate: resting balls give exact (0, 0) rows; at 7-9 balls
            # they dominate and the MAD trim collapses the fit onto them
            # (g = +0.0000 artifact). Free-FLIGHT means moving.
            if sp < 0.5:
                continue
            d2y = (pos[t + h, i, 1] - 2 * pos[t, i, 1]
                   + pos[t - h, i, 1]) / (h * h)
            d2x = (pos[t + h, i, 0] - 2 * pos[t, i, 0]
                   + pos[t - h, i, 0]) / (h * h)
            # y-row carries gravity + drag; x-row is a PURE drag channel
            # (no gravity horizontally) -- flag 1/0 selects the g column
            rows.append((d2y, 1.0, v[t, i, 1] * sp * r / masses[i], sp))
            rows.append((d2x, 0.0, v[t, i, 0] * sp * r / masses[i], sp))
        # floor bounces, estimator v2: GRAVITY-PROJECTED impact speeds.
        # One-frame differences give the exact parabola speed at the interval
        # midpoint; projecting both to the impact frame removes the ~1 px/f
        # gravity haircut that biased ratios low by ~10% and produced the fake
        # count-trend (slow settled-scene bounces are hit hardest). Events
        # must be isolated (no other ball nearby t-3..t+3) -- crowded rebounds
        # get clipped mid-window.
        y = pos[:, i, 1]
        G = -0.5
        for t in range(3, T - 3):
            if y[t] <= y[t - 1] and y[t] < y[t + 1] and y[t] < r + 2.5:
                # REAL-FALL gate: the model settles soft materials with a
                # hover-wobble whose sub-px local minima masquerade as
                # bounces at ratio ~1 (inflated Sponge to 0.47; caught by
                # Viktor + frame audit). A genuine impact is preceded by a
                # monotonic multi-px approach.
                if not (y[t - 3] > y[t - 2] > y[t - 1]
                        and y[t - 3] - y[t] >= 2.5):
                    continue
                iso = True
                for tt in range(max(t - 3, 0), min(t + 4, T)):
                    d = np.linalg.norm(pos[tt] - pos[tt, i], axis=1)
                    d[i] = 99
                    if d.min() < r + radii.max() + 1.5:
                        iso = False
                        break
                if not iso:
                    continue
                # v4: per-event sub-frame contact time with DRAG-CORRECTED
                # branch accelerations (drag ~ v^2 is 7% of g at Superball
                # speeds -- ignoring it biased fast bounces ~1.3% low) and
                # the crossing target lowered by the mean sub-substep
                # penetration v/32 (the low-e materials' bias). All physics,
                # no fitted constants.
                if min(y[t - 2], y[t - 1], y[t + 1], y[t + 2]) < r - 0.5:
                    continue
                from environments.env_bouncing import AIR_DRAG_COEFF as _C
                m_i = masses[i]
                v_hat_in = abs(y[t - 1] - y[t - 2])
                v_hat_out = abs(y[t + 2] - y[t + 1])
                d_in = _C * v_hat_in ** 2 * r / m_i     # drag decel, up
                d_out = _C * v_hat_out ** 2 * r / m_i   # drag decel, down
                A_in = G + d_in          # falling: drag opposes gravity
                A_out = G - d_out        # rising: drag adds to gravity
                y_star = r - v_hat_in / 32.0
                v_tm1 = (y[t - 1] - y[t - 2]) + A_in * 0.5
                disc = v_tm1 * v_tm1 - 2.0 * A_in * (y[t - 1] - y_star)
                if disc < 0:
                    continue
                sq = np.sqrt(disc)
                cands = [x for x in ((-v_tm1 - sq) / A_in,
                                     (-v_tm1 + sq) / A_in)
                         if 0.0 <= x <= 1.6]
                if not cands:
                    continue
                s_in = min(cands)
                v_in = v_tm1 + A_in * s_in
                v_tp1 = (y[t + 2] - y[t + 1]) - A_out * 0.5
                disc = v_tp1 * v_tp1 - 2.0 * A_out * (y[t + 1] - y_star)
                if disc < 0:
                    continue
                sq = np.sqrt(disc)
                cands = [x for x in ((v_tp1 - sq) / A_out,
                                     (v_tp1 + sq) / A_out)
                         if 0.0 <= x <= 1.6]
                if not cands:
                    continue
                s_out = min(cands)
                v_out = v_tp1 - A_out * s_out
                if v_in < -1.0 and v_out > 0.05:
                    bounces.append((objs[i]["material"], -v_out / v_in))
    return rows, bounces


def _fit(rows):
    """Hybrid per-quantity fit. GRAVITY: unweighted OLS on y-rows only (the
    configuration whose tracked ceiling is proven flat at -0.500 -- speed
    weighting couples g to fast rows where tracking noise rectifies).
    DRAG: WLS over y-rows + pure-drag x-rows (precision estimator,
    SE ~1-2%; its residual transfer is absorbed by per-count ceiling
    calibration)."""
    a = np.array(rows)
    if len(a) < 200:
        return dict(g=float("nan"), c=float("nan"), g_se=float("nan"),
                    c_se=float("nan"), n=len(a))

    def _trimmed_ols(X, y, w=None):
        w = np.ones(len(y)) if w is None else w
        beta = None
        for _ in range(3):
            sw = np.sqrt(w)
            beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            res = y - X @ beta
            mad = np.median(np.abs(res - np.median(res))) + 1e-12
            keep = np.abs(res - np.median(res)) < 6 * 1.4826 * mad
            X, y, w = X[keep], y[keep], w[keep]
        sw = np.sqrt(w)
        Xw, yw = X * sw[:, None], y * sw
        res = yw - Xw @ beta
        sigma2 = float(res @ res) / max(len(yw) - 2, 1)
        cov = sigma2 * np.linalg.pinv(Xw.T @ Xw)
        return beta, np.sqrt(np.diag(cov)), len(yw)

    ymask = a[:, 1] == 1.0
    if ymask.sum() < 100 or (~ymask).sum() < 100:
        return dict(g=float("nan"), c=float("nan"), g_se=float("nan"),
                    c_se=float("nan"), n=len(a))
    Xg = np.stack([np.ones(int(ymask.sum())), -a[ymask, 2]], axis=1)
    bg, sg, ng = _trimmed_ols(Xg, a[ymask, 0])
    Xc = np.stack([a[:, 1], -a[:, 2]], axis=1)
    bc, sc, nc = _trimmed_ols(Xc, a[:, 0], 1.0 + a[:, 3] ** 2)
    return dict(g=float(bg[0]), g_se=float(sg[0]),
                c=float(bc[1]), c_se=float(sc[1]), n=nc)


def _boot_ci(vals, iters=1000, seed=0):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(iters, len(v)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    import time
    from multiprocessing import get_context

    ae, dit, cfg = load_run(RUN, "cuda")
    out = {}
    for cnt in COUNTS:
        trajs = [os.path.join(f"scratchpad/examples/ood_count_c{cnt}",
                              f"traj-{t}") for t in range(N_SCENES)]
        for extra, n_e in ((f"scratchpad/examples/ood_count_c{cnt}b", 500),
                           (f"scratchpad/examples/ood_hot_c{cnt}", 400)):
            if os.path.isdir(extra):
                trajs += [os.path.join(extra, f"traj-{t}")
                          for t in range(n_e)]
        t0 = time.time()
        model_payloads, gt_payloads = [], []
        for gi in range(0, len(trajs), GROUP):
            group = trajs[gi:gi + GROUP]
            ctxs, seeds, metas = [], [], []
            for td in group:
                frames_np = load_frames(td)
                objs = json.load(open(os.path.join(td, "objects.json")))
                ang = np.load(os.path.join(td, "angles.npy"))
                pos = np.load(os.path.join(td, "positions.npy"))
                metas.append((objs, ang[0, :, 0], pos[CTX - 1],
                              detect_outline(frames_np[0], objs, pos[0],
                                             ang[0, :, 0]), pos))
                ctxs.append(frames_np[:CTX])
                seeds.append(zlib.crc32(os.path.basename(td).encode()) & 0x7FFFFFFF)
            preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS)
            from experiments.centroid_track import track_balls_centroid
            for pred, (objs, thetas, init, outline, pos) in zip(preds, metas):
                p, mc = track_balls_centroid(pred, objs, init, refine=True)
                model_payloads.append((p, objs, mc))
                gt_payloads.append((pos[CTX:CTX + STEPS], objs, None))
        with get_context("spawn").Pool(12) as pool:
            got_m = pool.map(_collect, model_payloads)
            got_g = pool.map(_collect, gt_payloads)
            got_m2 = pool.map(_collect, [(p_, o_, m_, 2)
                                         for p_, o_, m_ in model_payloads])
            got_g2 = pool.map(_collect, [(p_, o_, m_, 2)
                                         for p_, o_, m_ in gt_payloads])
        res = {}
        for tag, got, got2 in (("model", got_m, got_m2),
                               ("gt", got_g, got_g2)):
            rows = [r for g_ in got for r in g_[0]]
            bounces = [b for g_ in got for b in g_[1]]
            fit = _fit(rows)
            fit_h2 = _fit([r for g_ in got2 for r in g_[0]])
            per_mat = {}
            for mat, ratio in bounces:
                per_mat.setdefault(mat, []).append(ratio)
            rest = {}
            for m, v in per_mat.items():
                if len(v) < 50:               # unusable cell (Sponge)
                    continue
                lo, hi = _boot_ci(v)
                rest[m] = {"med": float(np.median(v)), "n": len(v),
                           "ci": [lo, hi]}
            res[tag] = {"fit": fit, "fit_h2": fit_h2, "restitution": rest}
        out[cnt] = res
        fm, fg = res["model"]["fit"], res["gt"]["fit"]
        print(f"c{cnt}: MODEL g {fm['g']:+.4f}±{fm['g_se']:.4f} c "
              f"{fm['c']:.4f}±{fm['c_se']:.4f} (n={fm['n']}) | GT g "
              f"{fg['g']:+.4f}±{fg['g_se']:.4f} c {fg['c']:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)
        for m in sorted(res["model"]["restitution"]):
            mv = res["model"]["restitution"][m]
            gv = res["gt"]["restitution"].get(m)
            gtxt = (f"{gv['med']:.3f} [{gv['ci'][0]:.3f},{gv['ci'][1]:.3f}] "
                    f"(n={gv['n']})") if gv else "n<50"
            print(f"    {m:10s} rest: model {mv['med']:.3f} "
                  f"[{mv['ci'][0]:.3f},{mv['ci'][1]:.3f}] (n={mv['n']}) | "
                  f"control {gtxt}", flush=True)
    json.dump(out, open("scratchpad/examples/constants_ood.json", "w"))
    print(f"true g {GRAVITY} | drag C {AIR_DRAG_COEFF}")


if __name__ == "__main__":
    main()
