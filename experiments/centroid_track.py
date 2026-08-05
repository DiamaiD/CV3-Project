"""Centroid ball tracker for the constants pipeline: mask-weighted centroid
instead of template-matching peaks -- TM peaks lag on the soft renders of
crowded scenes, and that lag reads as acceleration damping (fake drag and
gravity dilution).

Balls only, white-bg world: per ball, per frame -- window at the running
track, soft color mask (fill+outline axes, deformation-metric machinery),
Voronoi gate against same-material siblings, centroid = sub-pixel center.

Validation main: track GT RENDERED frames of c3/c9 scenes, compare against
positions.npy, and run the gravity fit on the tracks. Gate: mean err small
AND fitted g ~= -0.5 at BOTH counts."""
import json
import os

import numpy as np

from experiments.deformation import (_color_mask, _fill_outline_rgb,
                                     render_template)
from experiments.constants_ood import _collect, _fit
from src.frameio import load_frames

_ref_cache = {}


def _ball_variants(radius, material, outline=True):
    """Stack of exact-rendered 1/8-px template variants for one ball radius
    (quantized 0.25): (81, th, tw) masks + their sub-pixel center offsets."""
    rq = round(radius * 4) / 4.0
    key = (rq, material, outline)
    if key in _ref_cache:
        return _ref_cache[key]
    obj = {"kind": "ball", "material": material, "size": rq, "verts": None}
    base, (cy, cx), box = render_template(obj, 0.0, outline=outline)
    tpls, offs = [], []
    for qy in np.arange(-0.5, 0.501, 0.125):
        for qx in np.arange(-0.5, 0.501, 0.125):
            t, _, _ = render_template(obj, 0.0, outline=outline,
                                      frac=(qy, qx), box=box)
            tpls.append(t)
            offs.append((qy, qx))
    _ref_cache[key] = (np.stack(tpls), np.array(offs), (cy, cx))
    return _ref_cache[key]


def _refine(win_mask, anchor_rc, tpl_stack, offs, center):
    """Best sub-pixel placement of the ball template inside the window:
    SSD over the 81 exact-rendered 1/8-px variants."""
    th, tw = tpl_stack.shape[1:]
    r0, c0 = anchor_rc
    if r0 < 0 or c0 < 0 or r0 + th > win_mask.shape[0]             or c0 + tw > win_mask.shape[1]:
        return None
    sub = win_mask[r0:r0 + th, c0:c0 + tw]
    ssd = ((tpl_stack - sub[None]) ** 2).sum(axis=(1, 2))
    k = int(np.argmin(ssd))
    return offs[k]


def track_balls_centroid(frames, objs, init_pos_world, height=64, pad=12,
                         refine=False, outline=True):
    T = frames.shape[0]
    n = len(objs)
    fills = [_fill_outline_rgb(o["material"]) for o in objs]
    radii = [float(o["size"]) for o in objs]
    same = [[j for j in range(n)
             if j != i and objs[j]["material"] == objs[i]["material"]]
            for i in range(n)]
    pos = np.array([[height - y - 0.5, x - 0.5] for x, y in init_pos_world])
    vel = np.zeros((n, 2))
    out = np.zeros((T, n, 2))
    ok = np.ones((T, n), dtype=bool)
    for t in range(T):
        f = frames[t].astype(np.float64)
        for i in range(n):
            half = int(np.ceil(radii[i])) + pad
            # window centered at the VELOCITY-PREDICTED position: centering
            # on the previous position clips fast balls at the window edge,
            # dragging the centroid backward -- a speed-proportional lag the
            # gravity fit reads as fake drag
            pred_r = pos[i][0] + vel[i][0]
            pred_c = pos[i][1] + vel[i][1]
            r0 = int(round(pred_r)) - half
            c0 = int(round(pred_c)) - half
            rr0, cc0 = max(r0, 0), max(c0, 0)
            rr1 = min(r0 + 2 * half, frames.shape[1])
            cc1 = min(c0 + 2 * half, frames.shape[2])
            if rr1 <= rr0 or cc1 <= cc0:
                ok[t, i] = False
                out[t, i] = (pos[i][1] + 0.5, height - pos[i][0] - 0.5)
                continue
            w = _color_mask(f[rr0:rr1, cc0:cc1], *fills[i]).astype(np.float64)
            ys, xs = np.mgrid[rr0:rr1, cc0:cc1]
            if same[i]:
                di = (ys - pos[i][0]) ** 2 + (xs - pos[i][1]) ** 2
                for j in same[i]:
                    dj = (ys - pos[j][0]) ** 2 + (xs - pos[j][1]) ** 2
                    w = np.where(dj < di, 0.0, w)
            mass = w.sum()
            if mass > 5.0:
                new = np.array([float((w * ys).sum() / mass),
                                float((w * xs).sum() / mass)])
                if refine:
                    tpls, offs, (tcy, tcx) = _ball_variants(
                        radii[i], objs[i]["material"], outline)
                    # anchor the template box so its center sits at the
                    # integer-rounded centroid; the variant argmin supplies
                    # the sub-pixel remainder ripple-free
                    ar = int(round(new[0] - tcy)) - rr0
                    ac = int(round(new[1] - tcx)) - cc0
                    got = _refine(w, (ar, ac), tpls, offs, (tcy, tcx))
                    if got is not None:
                        base_r = int(round(new[0] - tcy)) + tcy
                        base_c = int(round(new[1] - tcx)) + tcx
                        new = np.array([base_r + got[0], base_c + got[1]])
                vel[i] = np.clip(new - pos[i], -9, 9) if t > 0 else vel[i]
                pos[i] = new
            else:
                ok[t, i] = False
            out[t, i] = (pos[i][1] + 0.5, height - pos[i][0] - 0.5)
    return out, ~ok                          # world xy, contaminated-style


def main():
    from multiprocessing import get_context

    for cnt in (3, 9):
        base = f"scratchpad/examples/ood_count_c{cnt}"
        for refine in (False, True):
            payloads, errs = [], []
            for t in range(150):
                td = os.path.join(base, f"traj-{t}")
                frames = load_frames(td)[5:305]
                objs = json.load(open(os.path.join(td, "objects.json")))
                pos = np.load(os.path.join(td, "positions.npy"))
                tw, bad = track_balls_centroid(frames, objs, pos[5],
                                               refine=refine)
                errs.append(np.linalg.norm(tw - pos[5:305], axis=2))
                payloads.append((tw, objs, bad))
            e = np.concatenate([x.ravel() for x in errs])
            with get_context("spawn").Pool(12) as pool:
                got = pool.map(_collect, payloads)
            fit = _fit([r for g_ in got for r in g_[0]])
            print(f"c{cnt} refine={refine}: pos err mean {e.mean():.3f} "
                  f"p99 {np.percentile(e, 99):.3f} px | g {fit['g']:+.4f}"
                  f"±{fit['g_se']:.4f} c {fit['c']:.4f}±{fit['c_se']:.4f} "
                  f"(n={fit['n']})", flush=True)


if __name__ == "__main__":
    main()
