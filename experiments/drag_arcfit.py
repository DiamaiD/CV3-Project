"""Whole-arc drag estimator: fit each free-flight segment by integrating the
ENGINE'S EXACT map (32 substeps, semi-implicit, per-axis quadratic drag --
vx|vx| / vy|vy|, exactly as _vel_func does; a vector-speed c*v*|v| form
is a bias source). Drag's cumulative displacement over a 15-40
frame arc is pixels -- hundreds of times above tracking noise -- and nothing
is differentiated, so sub-pixel ripple cannot rectify. Per-segment c via
least squares over (x0, y0, vx0, vy0, c); robust median across segments.

Validation main: exact-state c3 + c9 (target: c = 0.0200)."""
import json
import os

import numpy as np
from scipy.optimize import least_squares

from environments.env_bouncing import GRAVITY, MATERIALS

SUB = 32
DT = 1.0 / SUB


def integrate_arc(theta, r_eff, m, n_frames, g=None):
    x, y, vx, vy = theta[0], theta[1], theta[2], theta[3]
    c = theta[4]
    gg = GRAVITY if g is None else g
    k = c * r_eff / m
    out = np.empty((n_frames, 2))
    out[0] = (x, y)
    for f in range(1, n_frames):
        for _ in range(SUB):
            vy += gg * DT
            vx -= k * vx * abs(vx) * DT
            vy -= k * vy * abs(vy) * DT
            x += vx * DT
            y += vy * DT
        out[f] = (x, y)
    return out


def fit_segment(P, r_eff, m, free_g=False):
    n = len(P)
    v0 = (P[1] - P[0]) - np.array([0.0, GRAVITY * 0.5])
    if free_g:
        th0 = np.array([P[0, 0], P[0, 1], v0[0], v0[1], 0.02, GRAVITY])

        def resid(th):
            return (integrate_arc(th[:5], r_eff, m, n, g=th[5]) - P).ravel()

        bounds = ([0, 0, -12, -12, 0.001, -0.7],
                  [64, 64, 12, 12, 0.06, -0.3])
    else:
        th0 = np.array([P[0, 0], P[0, 1], v0[0], v0[1], 0.02])

        def resid(th):
            return (integrate_arc(th, r_eff, m, n) - P).ravel()

        bounds = ([0, 0, -12, -12, 0.001], [64, 64, 12, 12, 0.06])
    try:
        sol = least_squares(resid, th0, bounds=bounds,
                            xtol=1e-10, ftol=1e-12, max_nfev=150)
    except Exception:
        return None
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    sp = float(np.hypot(th0[2], th0[3]))
    g_out = float(sol.x[5]) if free_g else GRAVITY
    return sol.x[4], rms, sp, n, g_out


def segments_from_tracks(pos, objs, contaminated, min_len=12, max_len=40):
    """Maximal clean free-flight runs per ball (isolation, bounds, motion)."""
    T, n = pos.shape[0], pos.shape[1]
    radii = np.array([float(o["size"]) for o in objs])
    segs = []
    for i in range(n):
        r = radii[i]
        ok = np.zeros(T, dtype=bool)
        for t in range(T):
            if contaminated is not None and contaminated[t, i]:
                continue
            x, y = pos[t, i]
            if not (r + 2.5 < x < 64 - r - 2.5 and r + 2.5 < y < 64 - r - 2.5):
                continue
            d = np.linalg.norm(pos[t] - pos[t, i], axis=1)
            d[i] = 99
            # measurement margin: decoder softness near clutter displaces
            # the tracked centroid several px out -- wider than contact
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
                        segs.append((P.copy(), float(radii[i]),
                                     float(np.pi * radii[i] ** 2
                                           * MATERIALS[objs[i]["material"]]
                                           ["density"])))
                t = e
            else:
                t += 1
    return segs


def _work(args):
    P, r_eff, m = args[:3]
    free_g = args[3] if len(args) > 3 else False
    got = fit_segment(P, r_eff, m, free_g=free_g)
    if got is None:
        return None
    c, rms, sp, n, g_out = got
    if rms > 0.35:                     # distorted arc (render clutter etc.)
        return None
    return c, sp, n, g_out


def collect_c(all_tracks, workers=24, free_g=False):
    from multiprocessing import get_context
    segs = []
    for pos, objs, cont in all_tracks:
        segs += segments_from_tracks(pos, objs, cont)
    args = [(P, r, m, free_g) for P, r, m in segs]
    with get_context("spawn").Pool(workers) as pool:
        got = [g for g in pool.map(_work, args) if g is not None]
    cs = np.array([g[0] for g in got])
    gs = np.array([g[3] for g in got])
    return (cs, gs, len(segs)) if free_g else (cs, len(segs))


def main():
    for cnt in (3, 9):
        tracks = []
        for src in (f"scratchpad/examples/ood_count_c{cnt}",
                    f"scratchpad/examples/ood_hot_c{cnt}"):
            if not os.path.isdir(src):
                continue
            for t in range(300):
                td = os.path.join(src, f"traj-{t}")
                pos = np.load(os.path.join(td, "positions.npy"))[5:305]
                objs = json.load(open(os.path.join(td, "objects.json")))
                tracks.append((pos, objs, None))
        cs, nseg = collect_c(tracks)
        print(f"c{cnt} exact-state ARC FIT: c = {np.median(cs):.5f} "
              f"(IQR {np.percentile(cs, 25):.5f}-{np.percentile(cs, 75):.5f}, "
              f"{len(cs)}/{nseg} segments) | true {0.02}", flush=True)


if __name__ == "__main__":
    main()
