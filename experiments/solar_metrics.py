"""Orbit metrics for the solar env: sub-pixel body tracking on space-black
frames + position/radius/phase error of model rollouts against the CLOSED-FORM
ground truth (solar.json orbit elements -- exact, no integration error).

The white-background _color_mask of experiments/deformation.py does NOT apply
here (it measures blending toward white); _solar_mask projects onto the
black->fill axis instead. Bodies are circles, so the soft-mask centroid IS the
sub-pixel center -- no template machinery needed.

Validation gate (run this module directly): tracking real sim frames must
recover positions.npy to ~0.05 px before any model claim."""
import argparse
import json
import os

import numpy as np

from environments.env_bouncing import MATERIALS
from environments.env_solar import _kepler_E

COLOR_TOL = 60.0
CENTER = 32.0


def _fill_rgb(mat_name):
    b, g, r = MATERIALS[mat_name]["color"]     # MATERIALS stores BGR
    return np.array([r, g, b], dtype=np.float64)


def _solar_mask(frame_f64, fill):
    """Continuous [0,1] coverage toward BLACK background: alpha is the
    projection onto the black->fill axis, residual gates off other colors."""
    n2 = float((fill ** 2).sum())
    alpha = (frame_f64 @ fill) / n2
    resid = frame_f64 - alpha[..., None] * fill
    d = np.sqrt((resid ** 2).sum(axis=2))
    return (np.clip(alpha, 0.0, 1.0)
            * np.clip(1.0 - d / COLOR_TOL, 0.0, 1.0)).astype(np.float32)


_TPL_CACHE = {}
_TPL_OFFS = [(a / 8.0, b / 8.0) for a in range(-4, 5) for b in range(-4, 5)]


def _disc_templates(mat_name, size):
    """9x9 grid of exactly-rendered disc templates at 1/8-px offsets on the
    space-black background: (81, S, S, 3) float RGB + support mask + half."""
    key = (mat_name, round(float(size), 3))
    if key in _TPL_CACHE:
        return _TPL_CACHE[key]
    import cv2

    from environments.env_shapes import _draw
    S = 2 * int(np.ceil(size)) + 7
    tpls = np.zeros((81, S, S, 3), np.float64)
    for k, (qr, qc) in enumerate(_TPL_OFFS):
        obj = {"kind": "ball", "mat": MATERIALS[mat_name],
               "mat_name": mat_name, "size": float(size), "theta": 0.0,
               "verts": None, "x": S / 2.0 + qc, "y": S / 2.0 - qr}
        img = _draw([obj], S, S, 4, markers="none", outline=False, bg=0)
        tpls[k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    support = tpls.max(axis=(0, 3)) > 2.0
    _TPL_CACHE[key] = (tpls, support, (S - 1) // 2)
    return _TPL_CACHE[key]


def _para(m, z, p):
    den = m - 2.0 * z + p
    return 0.0 if den <= 1e-12 else max(-0.5, min(0.5, 0.5 * (m - p) / den))


def track_bodies(frames, objs, init_pos_world, height=64, win_pad=6,
                 refine=False):
    """Per-frame sub-pixel centers of every body (soft-mask centroid inside a
    window that follows the track). Returns (T, n, 2) world x,y + presence
    (T, n) mask-mass fraction vs the first frame's.

    refine=True: after the centroid, match 81 exactly-rendered 1/8-px disc
    templates by weighted SSD (own-color pixels only) and interpolate the
    SSD minimum parabolically -- cuts frame-to-frame jitter for
    derivative-based metrics (energy)."""
    T = frames.shape[0]
    n = len(objs)
    fills = [_fill_rgb(o["material"]) for o in objs]
    radii = [float(o["size"]) for o in objs]
    same_mat = [[j for j in range(n)
                 if j != i and objs[j]["material"] == objs[i]["material"]]
                for i in range(n)]
    uniq = {o["material"]: _fill_rgb(o["material"]) for o in objs}
    other_fills = [[v for k, v in uniq.items() if k != objs[i]["material"]]
                   for i in range(n)]
    pos = np.array([[height - y - 0.5, x - 0.5] for x, y in init_pos_world])
    out = np.zeros((T, n, 2))
    presence = np.zeros((T, n))
    m0 = np.zeros(n)
    for t in range(T):
        f = frames[t].astype(np.float64)
        for i in range(n):
            half = int(np.ceil(radii[i])) + win_pad
            r0 = int(round(pos[i][0])) - half
            c0 = int(round(pos[i][1])) - half
            rr0, cc0 = max(r0, 0), max(c0, 0)
            rr1 = min(r0 + 2 * half, frames.shape[1])
            cc1 = min(c0 + 2 * half, frames.shape[2])
            fwin = f[rr0:rr1, cc0:cc1]
            w = _solar_mask(fwin, fills[i])
            # winner-take-all across materials: Rocky brown is close to dark
            # gold, so a YellowStar's rim scores ~0.14 on the Rocky axis and
            # drags innermost-planet centroids sunward by ~3px (constant!).
            # A pixel counts for us only if no other material matches it better.
            for of in other_fills[i]:
                wo = _solar_mask(fwin, of)
                w = np.where(wo > w, 0.0, w).astype(np.float32)
            if same_mat[i]:
                # Voronoi gate: a same-color sibling inside the window (binary
                # stars sit 5.5-8.5 px apart!) pulls the centroid between the
                # two bodies -- keep only pixels closer to us than to it
                ys, xs = np.mgrid[rr0:rr1, cc0:cc1]
                di = (ys - pos[i][0]) ** 2 + (xs - pos[i][1]) ** 2
                for j in same_mat[i]:
                    dj = (ys - pos[j][0]) ** 2 + (xs - pos[j][1]) ** 2
                    w = np.where(dj < di, 0.0, w).astype(np.float32)
            mass = float(w.sum())
            if t == 0:
                m0[i] = max(mass, 1e-6)
            presence[t, i] = mass / m0[i]
            if mass > 0.15 * m0[i]:
                ys, xs = np.mgrid[rr0:rr1, cc0:cc1]
                cy = float((w * ys).sum() / mass)
                cx = float((w * xs).sum() / mass)
                pos[i] = (cy, cx)
                if refine:
                    tpls, support, B = _disc_templates(objs[i]["material"],
                                                       radii[i])
                    ar, ac = int(round(pos[i][0])), int(round(pos[i][1]))
                    if (ar - B >= 0 and ac - B >= 0
                            and ar + B + 1 <= frames.shape[1]
                            and ac + B + 1 <= frames.shape[2]):
                        crop = f[ar - B:ar + B + 1, ac - B:ac + B + 1]
                        wown = _solar_mask(crop, fills[i])
                        ok = support.copy()
                        for of in other_fills[i]:
                            ok &= ~(_solar_mask(crop, of) > wown)
                        if same_mat[i]:
                            ys2, xs2 = np.mgrid[ar - B:ar + B + 1,
                                                ac - B:ac + B + 1]
                            di = ((ys2 - pos[i][0]) ** 2
                                  + (xs2 - pos[i][1]) ** 2)
                            for j in same_mat[i]:
                                dj = ((ys2 - pos[j][0]) ** 2
                                      + (xs2 - pos[j][1]) ** 2)
                                ok &= ~(dj < di)
                        ssd = (((tpls - crop) ** 2).sum(axis=3)
                               * ok).sum(axis=(1, 2))
                        k = int(np.argmin(ssd))
                        g = ssd.reshape(9, 9)
                        kr, kc = k // 9, k % 9
                        dr = (_para(g[kr - 1, kc], g[kr, kc],
                                    g[kr + 1, kc]) / 8.0
                              if 0 < kr < 8 else 0.0)
                        dc = (_para(g[kr, kc - 1], g[kr, kc],
                                    g[kr, kc + 1]) / 8.0
                              if 0 < kc < 8 else 0.0)
                        qr, qc = _TPL_OFFS[k]
                        cand = (ar + qr + dr, ac + qc + dc)
                        if (abs(cand[0] - pos[i][0]) < 0.75
                                and abs(cand[1] - pos[i][1]) < 0.75):
                            pos[i] = cand
            out[t, i] = (pos[i][1] + 0.5, height - pos[i][0] - 0.5)
    return out, presence


def gt_positions(sol, t_arr):
    """Closed-form GT positions for all bodies at the given frame times."""
    d = sol["direction"]
    cols = []
    for s in sol["stars"]:
        w = d * 2.0 * np.pi / s["T"]
        phi = s["phi0"] + w * t_arr
        cols.append(np.stack([CENTER + s["orb_r"] * np.cos(phi),
                              CENTER + s["orb_r"] * np.sin(phi)], axis=1))
    for p in sol["planets"]:
        n_mot = d * 2.0 * np.pi / p["T"]
        E = _kepler_E(p["M0"] + n_mot * t_arr, p["e"])
        b = p["a"] * np.sqrt(1.0 - p["e"] ** 2)
        xo, yo = p["a"] * (np.cos(E) - p["e"]), b * np.sin(E)
        co, so = np.cos(p["omega"]), np.sin(p["omega"])
        cols.append(np.stack([CENTER + co * xo - so * yo,
                              CENTER + so * xo + co * yo], axis=1))
    return np.stack(cols, axis=1)          # (T, n, 2) world xy


def orbit_errors(tracked, sol, t_arr):
    """Split planet error into radius drift and phase (along-track) drift.
    Returns dict of (T, n_planets) arrays: pos_err, radius_err, phase_err."""
    gt = gt_positions(sol, t_arr)
    n_star = len(sol["stars"])
    tr = tracked[:, n_star:] - CENTER
    gtp = gt[:, n_star:] - CENTER
    pos_err = np.linalg.norm(tracked[:, n_star:] - gt[:, n_star:], axis=2)
    r_tr = np.linalg.norm(tr, axis=2)
    r_gt = np.linalg.norm(gtp, axis=2)
    radius_err = r_tr - r_gt
    ang = (np.arctan2(tr[..., 1], tr[..., 0])
           - np.arctan2(gtp[..., 1], gtp[..., 0]))
    phase_err = (ang + np.pi) % (2 * np.pi) - np.pi
    return {"pos_err": pos_err, "radius_err": radius_err,
            "phase_err": phase_err}


def _validate_traj(td):
    from src.frameio import load_frames
    frames = load_frames(td)
    objs = json.load(open(os.path.join(td, "objects.json")))
    pos = np.load(os.path.join(td, "positions.npy"))
    tracked, presence = track_bodies(frames, objs, pos[0])
    err = np.linalg.norm(tracked - pos, axis=2)
    return float(err.mean()), float(err.max()), float(presence.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/solar_s1")
    ap.add_argument("--n-traj", type=int, default=100)
    args = ap.parse_args()
    from multiprocessing import get_context
    trajs = [os.path.join(args.data, f"traj-{t}") for t in range(args.n_traj)]
    with get_context("spawn").Pool(24) as pool:
        res = pool.map(_validate_traj, trajs)
    mean_err = float(np.mean([r[0] for r in res]))
    max_err = float(np.max([r[1] for r in res]))
    min_pres = float(np.min([r[2] for r in res]))
    print(f"tracker validation on {args.data} ({args.n_traj} trajs): "
          f"mean position error {mean_err:.3f} px | worst {max_err:.3f} px | "
          f"min presence {min_pres:.2f}")
    print("TRACKER OK" if mean_err < 0.08 and max_err < 0.5
          else "TRACKER NEEDS WORK")


if __name__ == "__main__":
    main()
