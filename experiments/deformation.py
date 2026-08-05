"""Deformation metric for the frozen-rotation shape worlds.

With rotation frozen, every object has ONE rigid silhouette for its whole
trajectory. A frame deforms an object exactly to the degree its rendered color
blob differs from that rigid template at the best translation -- there is no
rotation ambiguity, so template matching measures deformation directly.

Per object per frame:
  1. M = color mask: pixels within tolerance of the object's fill OR outline
     color (both calibrated by rendering through the real pipeline, so channel
     conventions cannot bite), inside a search window around the tracked position
  2. align the rigid template T over the window (translation-only cross
     correlation, cv2.matchTemplate)
  3. at the best offset:
       miss   = |T and not M| / |T|   (template pixels not covered: erosion,
                                       swallowing, displaced mass)
       excess = |M and not T, within 2px of T| / |T|   (mass bulging out)
       score  = miss + excess
  4. the peak position becomes the next frame's search center (tracking)

Anti-aliasing and outline blending give clean frames a nonzero floor
(~0.05-0.10 depending on kind/size); always compare model scores against the
GT floor measured on ground-truth frames of the same trajectories.

Flags: when another object of the SAME material comes within the search halo,
`contaminated` is set for that (object, frame) -- masks can merge and inflate
`excess`; exclude flagged frames for the strictest numbers.
"""
import argparse
import glob
import json
import os
import zlib

import cv2
import numpy as np

from environments.env_bouncing import MATERIALS
from environments.env_shapes import _draw

PAD = 14          # search halo around the tracked position (px)
COLOR_TOL = 60.0  # euclidean RGB distance for the color masks
NEAR_PX = 2       # ring around the template that counts as "bulge"


def _fill_outline_rgb(mat_name):
    """Fill + outline color of a material in STORED-FRAME RGB space, calibrated
    by rendering one ball through the same cv2 path the datasets use."""
    mat = MATERIALS[mat_name]
    o = {"kind": "ball", "mat": mat, "mat_name": mat_name, "size": 8.0,
         "theta": 0.0, "x": 16.0, "y": 16.0, "verts": None}
    img = cv2.cvtColor(_draw([o], 32, 32, 4, markers="none", outline=False),
                       cv2.COLOR_BGR2RGB)
    fill = img[16, 16].astype(np.float64)
    outline = np.round(fill * 0.45)
    return fill, outline


def _obj_for_draw(obj_json, theta, x, y):
    v = obj_json.get("verts")
    return {"kind": obj_json["kind"], "mat": MATERIALS[obj_json["material"]],
            "mat_name": obj_json["material"], "size": float(obj_json["size"]),
            "theta": float(theta), "x": float(x), "y": float(y),
            "verts": None if v is None else np.asarray(v, dtype=np.float64)}


def render_template(obj_json, theta, canvas=48, ss=4, outline=True,
                    frac=(0.0, 0.0), box=None):
    """Binary silhouette mask of one object, tightly cropped. CRITICAL: the
    template mask must use the SAME color-distance criterion as the frame masks
    (a non-white threshold would include the anti-aliased boundary ring that the
    color mask excludes, putting a permanent ~0.2-0.3 'miss' floor on every
    score)."""
    fy, fx = frac
    # sub-pixel placement is native to the renderer: shifting the object inside
    # the canvas gives an EXACT fractional-offset template (no interpolation)
    o = _obj_for_draw(obj_json, theta, canvas / 2 + fx, canvas / 2 - fy)
    img = cv2.cvtColor(_draw([o], canvas, canvas, ss, markers="none", outline=outline),
                       cv2.COLOR_BGR2RGB)
    fill, outline_c = _fill_outline_rgb(obj_json["material"])
    mask = _color_mask(img.astype(np.float64), fill, outline_c)
    if box is None:
        ys, xs = np.where(mask > 0.02)
        box = (max(int(ys.min()) - 2, 0), min(int(ys.max()) + 3, canvas),
               max(int(xs.min()) - 2, 0), min(int(xs.max()) + 3, canvas))
    y0, y1, x0, x1 = box
    # the object's TRUE base center inside the fixed crop (frac excluded -- the
    # caller adds its measured fractional offset)
    cy, cx = (canvas - canvas / 2) - 0.5 - y0, canvas / 2 - 0.5 - x0
    return mask[y0:y1, x0:x1], (cy, cx), box


def _color_mask(frame_f64, fill, outline):
    """Continuous [0,1] coverage map. Each pixel is projected onto the
    white->fill and white->outline color axes: a boundary pixel that is 30%%
    object by anti-aliasing projects to alpha 0.3 with tiny residual, so the
    mask recovers FRACTIONAL coverage instead of a hard threshold -- the key to
    sub-pixel scoring."""
    white = np.array([255.0, 255.0, 255.0])
    out = np.zeros(frame_f64.shape[:2], np.float32)
    for proto in (fill, outline):
        axis = proto - white
        n2 = float((axis ** 2).sum())
        if n2 < 1e-6:
            continue
        alpha = ((frame_f64 - white) @ axis) / n2
        resid = frame_f64 - (white + alpha[..., None] * axis)
        d = np.sqrt((resid ** 2).sum(axis=2))
        w = np.clip(alpha, 0.0, 1.0) * np.clip(1.0 - d / COLOR_TOL, 0.0, 1.0)
        out = np.maximum(out, w.astype(np.float32))
    return out


def detect_outline(frame0, objs_json, pos0, thetas, height=64):
    """A dataset has outlines iff dark-rim pixels (0.45 x fill) appear around the
    objects of the first frame. Checked once per trajectory."""
    f = frame0.astype(np.float64)
    hits = total = 0
    for i, o in enumerate(objs_json):
        _, outline_c = _fill_outline_rgb(o["material"])
        d = np.linalg.norm(f - outline_c, axis=2)
        r = int(o["size"]) + 3
        row, col = int(round(height - pos0[i][1])), int(round(pos0[i][0]))
        win = d[max(0, row - r):row + r, max(0, col - r):col + r]
        hits += int((win < 40).sum())
        total += 1
    return hits / max(total, 1) > 3


def detect_border_px(frame):
    """Bordered datasets frame the image in gray (90,90,90) -- close enough to
    Steel's fill to contaminate color masks, so those pixels must be excluded."""
    c = frame[0, 0].astype(int)
    return 2 if (abs(c - 90) < 12).all() else 0


def score_frames(frames, objs_json, thetas, init_pos_world, height=64,
                 border_px=None, outline=True, with_presence=False,
                 reacquire=False):
    """Score every object in every frame.

    frames: (T, H, W, 3) uint8 RGB (model rollout or ground truth)
    objs_json: the trajectory's objects.json list
    thetas: per-object frozen angle (angles.npy[0, :, 0])
    init_pos_world: (n_obj, 2) world x,y at frames[0] time

    Returns: scores (T, n_obj) float, contaminated (T, n_obj) bool,
             tracks (T, n_obj, 2) pixel row,col of the template center.
    """
    T_frames = frames.shape[0]
    n = len(objs_json)
    if border_px is None:
        border_px = detect_border_px(frames[0])
    valid = np.zeros(frames.shape[1:3], np.uint8)
    b = border_px
    valid[b:frames.shape[1] - b if b else frames.shape[1],
          b:frames.shape[2] - b if b else frames.shape[2]] = 1
    tpl_data = [render_template(o, thetas[i], outline=outline)
                for i, o in enumerate(objs_json)]
    templates = [t for t, _, _ in tpl_data]
    tpl_centers = [c for _, c, _ in tpl_data]
    tpl_boxes = [b for _, _, b in tpl_data]
    tpl_cache = {}

    def tpl_variant(i, dy, dx):
        """Exactly-rendered template at the fractional offset, 1/8-px cached."""
        qy, qx = round(dy * 8) / 8.0, round(dx * 8) / 8.0
        key = (i, qy, qx)
        if key not in tpl_cache:
            tpl_cache[key] = render_template(
                objs_json[i], thetas[i], outline=outline, frac=(qy, qx),
                box=tpl_boxes[i])[0]
        return tpl_cache[key], qy, qx
    colors = [_fill_outline_rgb(o["material"]) for o in objs_json]
    mat_fills = {o["material"]: _fill_outline_rgb(o["material"])
                 for o in objs_json}
    mat_objs = {}
    for i, o in enumerate(objs_json):
        mat_objs.setdefault(o["material"], []).append(i)
    mats_multi = [m for m, idx in mat_objs.items() if len(idx) > 1]
    pos = np.zeros((n, 2))          # ARRAY-INDEX row, col of the object center
    # world -> index convention: pixel centers sit at half-integers (the
    # renderer maps world x to index x - 0.5)
    for i, (wx, wy) in enumerate(init_pos_world):
        pos[i] = (height - wy - 0.5, wx - 0.5)
    scores = np.zeros((T_frames, n))
    contaminated = np.zeros((T_frames, n), dtype=bool)
    tracks = np.zeros((T_frames, n, 2))
    presence = np.zeros((T_frames, n))   # aligned overlap fraction (1 - miss)
    _full_masks = {}                     # (t, material) -> full-frame soft mask
    kernel = np.ones((2 * NEAR_PX + 1, 2 * NEAR_PX + 1), np.uint8)

    for t in range(T_frames):
        f = frames[t].astype(np.float64)
        _full_masks = {k: v for k, v in _full_masks.items() if k[0] == t}
        foots = [None] * n              # aligned template footprint this frame
        for i, tpl in enumerate(templates):
            th, tw = tpl.shape
            half = max(th, tw) // 2 + PAD
            r0, c0 = int(round(pos[i][0])), int(round(pos[i][1]))
            top, left = r0 - half, c0 - half
            win = np.zeros((2 * half, 2 * half), np.float32)
            fr0, fc0 = max(top, 0), max(left, 0)
            fr1 = min(top + 2 * half, frames.shape[1])
            fc1 = min(left + 2 * half, frames.shape[2])
            if fr1 <= fr0 or fc1 <= fc0:
                scores[t, i] = 1.0
                tracks[t, i] = pos[i]
                continue
            m = _color_mask(f[fr0:fr1, fc0:fc1], *colors[i])
            m = m * valid[fr0:fr1, fc0:fc1]
            win[fr0 - top:fr1 - top, fc0 - left:fc1 - left] = m
            if win.shape[0] < th or win.shape[1] < tw:
                scores[t, i] = 1.0
                tracks[t, i] = pos[i]
                continue
            sibs_w = [j for j in mat_objs[objs_json[i]["material"]]
                      if j != i]
            if sibs_w:
                # Voronoi-gate the window BEFORE alignment: a same-material
                # sibling's mass inside the window can pull the correlation
                # peak a pixel off, and a misaligned template fakes a large
                # "missing" band -- the dominant floor tail in crowded scenes
                ys_w, xs_w = np.mgrid[top:top + win.shape[0],
                                      left:left + win.shape[1]]
                di_w = (ys_w - pos[i][0]) ** 2 + (xs_w - pos[i][1]) ** 2
                for j in sibs_w:
                    dj_w = (ys_w - pos[j][0]) ** 2 + (xs_w - pos[j][1]) ** 2
                    win = np.where(dj_w < di_w, 0.0, win).astype(np.float32)
            cc = cv2.matchTemplate(win, tpl, cv2.TM_CCORR)
            _, _, _, (mx, my) = cv2.minMaxLoc(cc)
            # sub-pixel refinement: 1D parabola through the correlation peak per
            # axis (~0.1px accuracy), then evaluate at the fractional offset by
            # shifting the window with linear interpolation
            def _para(cm, c0v, cp):
                den = cm - 2 * c0v + cp
                return 0.0 if abs(den) < 1e-9 else max(-0.5, min(0.5, 0.5 * (cm - cp) / den))
            dx = dy = 0.0
            if 0 < mx < cc.shape[1] - 1:
                dx = _para(cc[my, mx - 1], cc[my, mx], cc[my, mx + 1])
            if 0 < my < cc.shape[0] - 1:
                dy = _para(cc[my - 1, mx], cc[my, mx], cc[my + 1, mx])
            tv, qy, qx = tpl_variant(i, dy, dx)
            sub = win[my:my + th, mx:mx + tw]
            area_t = float(tv.sum())
            inter = float(np.minimum(sub, tv).sum())
            if reacquire and inter <= 0.2 * area_t:
                # lost lock: global search over the whole frame -- distinguishes
                # "moved out of the local window" from "truly gone"
                key = (t, objs_json[i]["material"])
                if key not in _full_masks:
                    fm = _color_mask(f, *colors[i]).astype(np.float32)
                    fm *= valid
                    _full_masks[key] = fm
                ccg = cv2.matchTemplate(_full_masks[key], tpl, cv2.TM_CCORR)
                _, _, _, (gx, gy) = cv2.minMaxLoc(ccg)
                gsub = _full_masks[key][gy:gy + th, gx:gx + tw]
                ginter = float(np.minimum(gsub, tpl).sum())
                if ginter > max(inter * 1.5, 0.25 * area_t):
                    sub, inter = gsub, ginter
                    tv2 = tpl
                    top, left, my, mx, dy, dx = gy, gx, 0, 0, 0.0, 0.0
                    tcy, tcx = tpl_centers[i]
                    pos[i] = (gy + tcy, gx + tcx)
            presence[t, i] = inter / area_t
            # the SCORE counts only CLEAR
            # pixel disagreement (coverage difference > 0.5), so the
            # decoder's soft anti-aliased rim -- invisible gray
            # disagreement that dominated the soft score -- does not count
            # as deformation. A pixel visibly claimed by another object is
            # not "missing" (contact/occlusion is not deformation), and
            # excess pixels closer to a same-material sibling are not
            # ours. Tracking, alignment, presence and contamination still
            # use the soft masks unchanged.
            tb = tv >= 0.5
            area_b = max(float(tb.sum()), 1.0)
            near = cv2.dilate(tb.astype(np.uint8), kernel).astype(bool)
            d = sub - tv
            om_acc = np.zeros_like(win)
            for mname, fo in mat_fills.items():
                if mname == objs_json[i]["material"]:
                    continue
                omask = _color_mask(f[fr0:fr1, fc0:fc1], *fo)
                om_acc[fr0 - top:fr1 - top, fc0 - left:fc1 - left] = \
                    np.maximum(om_acc[fr0 - top:fr1 - top,
                                      fc0 - left:fc1 - left], omask)
            other = om_acc[my:my + th, mx:mx + tw]
            miss_m = ((-d) > 0.5) & (other < 0.5)
            if miss_m.any() and n > 1:
                # a "missing" pixel sitting inside another object's disc
                # (plus a 1px blend margin) is contact/occlusion, not
                # deformation -- two touching outlines blend into a color
                # neither object's mask claims
                ys, xs = np.nonzero(miss_m)
                rr = top + my + ys
                cc = left + mx + xs
                keep = np.ones(len(ys), bool)
                for j in range(n):
                    if j == i:
                        continue
                    rj = float(objs_json[j]["size"]) + 1.0
                    keep &= ((rr - pos[j][0]) ** 2
                             + (cc - pos[j][1]) ** 2) > rj * rj
                miss_m = np.zeros_like(miss_m)
                miss_m[ys[keep], xs[keep]] = True
            excess_m = (d > 0.5) & near
            sibs = [j for j in mat_objs[objs_json[i]["material"]] if j != i]
            if sibs and excess_m.any():
                ys, xs = np.nonzero(excess_m)
                rr = top + my + ys
                cc = left + mx + xs
                di = (rr - pos[i][0]) ** 2 + (cc - pos[i][1]) ** 2
                keep = np.ones(len(ys), bool)
                for j in sibs:
                    dj = (rr - pos[j][0]) ** 2 + (cc - pos[j][1]) ** 2
                    keep &= di <= dj
                n_exc = float(keep.sum())
            else:
                n_exc = float(excess_m.sum())
            scores[t, i] = (float(miss_m.sum()) + n_exc) / area_b
            if inter > 0.2 * area_t:
                tcy, tcx = tpl_centers[i]
                pos[i] = (top + my + qy + tcy, left + mx + qx + tcx)
            tracks[t, i] = pos[i]
            foots[i] = (top + my, left + mx, tv > 0.5)
        # blob-based contamination: same-material objects poison each
        # other's scores only when their pixels actually CONNECT (shared
        # component within the NEAR_PX bulge ring) -- a plain radius test
        # discards most contact frames, exactly where deformation lives.
        # A tracking swap still flags, because both footprints land on the
        # same blob the moment it happens.
        for mat in mats_multi:
            key = (t, mat)
            if key not in _full_masks:
                fm = _color_mask(f, *colors[mat_objs[mat][0]]).astype(np.float32)
                fm *= valid
                _full_masks[key] = fm
            mb = cv2.dilate((_full_masks[key] >= 0.5).astype(np.uint8), kernel)
            _, lab = cv2.connectedComponents(mb)
            claimed = {}
            for i in mat_objs[mat]:
                if foots[i] is None:
                    continue
                r0f, c0f, tb = foots[i]
                rr0, cc0 = max(r0f, 0), max(c0f, 0)
                rr1 = min(r0f + tb.shape[0], lab.shape[0])
                cc1 = min(c0f + tb.shape[1], lab.shape[1])
                if rr1 <= rr0 or cc1 <= cc0:
                    continue
                sub_lab = lab[rr0:rr1, cc0:cc1]
                sub_tb = tb[rr0 - r0f:rr1 - r0f, cc0 - c0f:cc1 - c0f]
                for L in set(np.unique(sub_lab[sub_tb])) - {0}:
                    if L in claimed:
                        contaminated[t, i] = True
                        contaminated[t, claimed[L]] = True
                    else:
                        claimed[L] = i
    if with_presence:
        return scores, contaminated, tracks, presence
    return scores, contaminated, tracks


def score_gt_trajectory(traj_dir, max_frames=None):
    """Ground-truth floor: score the dataset's own frames against the templates."""
    from src.frameio import load_frames
    objs = json.load(open(os.path.join(traj_dir, "objects.json")))
    ang = np.load(os.path.join(traj_dir, "angles.npy"))
    pos = np.load(os.path.join(traj_dir, "positions.npy"))
    frames = load_frames(traj_dir)
    if max_frames:
        frames, pos = frames[:max_frames], pos[:max_frames]
    return score_frames(frames, objs, ang[0, :, 0], pos[0]), objs


def rollout_and_score(run_dir, traj_dir, n_steps=80, device="cuda", loaded=None):
    """Free-run the DiT from the trajectory's context and score the PREDICTED
    frames; also scores the matching GT frames as the floor. The rollout keeps
    its context in LATENT space (z_run feedback), exactly like the eval
    pipeline's _chunk_rollout -- decoding is only a tap for the metric.
    Pass loaded=(ae, dit, cfg) to reuse models across trajectories."""
    import torch
    from experiments.rollout import load_run
    from src.eval import flow_sample
    from src.frameio import load_frames

    ae, dit, cfg = loaded if loaded is not None else load_run(run_dir, device)
    ctx_len = cfg["context_len"]
    ns = 3
    objs = json.load(open(os.path.join(traj_dir, "objects.json")))
    ang = np.load(os.path.join(traj_dir, "angles.npy"))
    pos = np.load(os.path.join(traj_dir, "positions.npy"))
    frames_np = load_frames(traj_dir)
    n_steps = min(n_steps, frames_np.shape[0] - ctx_len)

    x = (torch.from_numpy(frames_np[:ctx_len]).permute(0, 3, 1, 2)
         .float().div_(255.0).to(device))
    # deterministic rollout noise per trajectory: without this, every scoring
    # pass draws fresh flow-matching noise and per-kind excesses wobble by
    # ~0.02 between identical invocations
    torch.manual_seed(zlib.crc32(os.path.basename(traj_dir).encode()) & 0x7FFFFFFF)
    preds = []
    with torch.no_grad():
        z = ae.encode(x) / dit.latent_scale
        z_run = z.unsqueeze(0)                          # (1, T, C, g, g)
        while len(preds) < n_steps:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_chunk = flow_sample(dit, z_run, ns)
            z_chunk = z_chunk.float()
            dec = ae.decode((z_chunk.reshape(-1, *z_chunk.shape[2:])
                             * dit.latent_scale)).clamp(0, 1)
            for j in range(dec.shape[0]):
                if len(preds) < n_steps:
                    preds.append(dec[j])
            z_run = torch.cat([z_run, z_chunk], dim=1)[:, -ctx_len:]
    pred_frames = (torch.stack(preds).permute(0, 2, 3, 1) * 255).byte().cpu().numpy()

    thetas = ang[0, :, 0]
    init = pos[ctx_len - 1]
    outline = detect_outline(frames_np[0], objs, pos[0], thetas)
    model = score_frames(pred_frames, objs, thetas, init, outline=outline)
    # floor = GT through the SAME VAE round-trip: model frames are decoder
    # outputs, and the decoder smooths seam/AA artifacts that raw GT frames
    # carry -- a raw-GT floor over-subtracts (negative excesses,
    # look-dependent bias)
    gt_slice = frames_np[ctx_len:ctx_len + n_steps]
    with torch.no_grad():
        xg = (torch.from_numpy(gt_slice).permute(0, 3, 1, 2)
              .float().div_(255.0).to(device))
        rec = torch.cat([ae.decode(ae.encode(xg[i:i + 256])).clamp(0, 1)
                         for i in range(0, xg.shape[0], 256)])
    rec_frames = (rec.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
    gt = score_frames(rec_frames, objs, thetas, init, outline=outline)
    return model, gt, objs


def _score_pair(payload):
    """Worker: score one trajectory's predicted + roundtrip-GT frames.
    Returns [(kind, model_mean, floor_mean), ...] for clean objects."""
    pred, rec, objs, thetas, init, outline = payload
    ms, mc, _ = score_frames(pred, objs, thetas, init, outline=outline)
    gs, gc, _ = score_frames(rec, objs, thetas, init, outline=outline)
    out = []
    for i, o in enumerate(objs):
        ok = ~mc[:, i] & ~gc[:, i]
        if ok.sum() < 10:
            continue
        out.append((o["kind"], float(ms[ok, i].mean()), float(gs[ok, i].mean())))
    return out


def _batched_rollout(ae, dit, ctx_frames_list, seeds, n_steps, ns=3, device="cuda"):
    """Roll out a GROUP of trajectories in one batch. Noise is drawn from a
    per-trajectory CPU generator (seeded), so each trajectory's rollout is
    deterministic regardless of how the group is composed."""
    import torch
    B = len(ctx_frames_list)
    x = torch.stack([torch.from_numpy(f).permute(0, 3, 1, 2).float().div_(255.0)
                     for f in ctx_frames_list]).to(device)
    Bv, T, C, H, W = x.shape
    chunk, lc, g = dit.chunk_len, dit.latent_ch, dit.grid
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    preds = [[] for _ in range(Bv)]
    dt = 1.0 / ns
    with torch.no_grad():
        z = ae.encode(x.reshape(-1, C, H, W)) / dit.latent_scale
        z_run = z.view(Bv, T, *z.shape[1:])
        while len(preds[0]) < n_steps:
            z_t = torch.stack([torch.randn((chunk, lc, g, g), generator=gen)
                               for gen in gens]).to(device)
            for i in range(ns):
                t = torch.full((Bv,), i * dt, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v = dit(z_t, t, z_run)
                z_t = z_t + v.float() * dt
            dec = ae.decode((z_t.reshape(-1, lc, g, g) * dit.latent_scale)).clamp(0, 1)
            dec = dec.view(Bv, chunk, C, H, W)
            for b in range(Bv):
                for j in range(chunk):
                    if len(preds[b]) < n_steps:
                        preds[b].append(dec[b, j])
            z_run = torch.cat([z_run, z_t], dim=1)[:, -T:]
    return [(torch.stack(p).permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
            for p in preds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-traj", type=int, default=30)
    ap.add_argument("--n-steps", type=int, default=80)
    ap.add_argument("--group", type=int, default=24, help="trajectories per GPU batch")
    ap.add_argument("--workers", type=int, default=12, help="CPU scoring processes")
    args = ap.parse_args()

    import time
    from multiprocessing import get_context
    import torch
    from experiments.rollout import load_run, test_split
    from src.frameio import load_frames

    _, test = test_split(args.data)
    trajs = sorted(test)[:args.n_traj]
    ae, dit, cfg = load_run(args.run, "cuda")
    ctx_len = cfg["context_len"]
    t0 = time.time()
    n_done = 0
    per_kind = {}   # kind -> list of (model_mean, floor_mean) per object
    pool = get_context("spawn").Pool(args.workers)
    pending = None

    def build_group(group):
        """GPU part for one group: batched rollouts + batched roundtrip floors.
        Returns worker payloads."""
        metas, ctxs, seeds, gts = [], [], [], []
        for td in group:
            frames_np = load_frames(td)
            objs = json.load(open(os.path.join(td, "objects.json")))
            ang = np.load(os.path.join(td, "angles.npy"))
            pos = np.load(os.path.join(td, "positions.npy"))
            n_steps = min(args.n_steps, frames_np.shape[0] - ctx_len)
            metas.append((objs, ang[0, :, 0], pos[ctx_len - 1],
                          detect_outline(frames_np[0], objs, pos[0], ang[0, :, 0])))
            ctxs.append(frames_np[:ctx_len])
            seeds.append(zlib.crc32(os.path.basename(td).encode()) & 0x7FFFFFFF)
            gts.append(frames_np[ctx_len:ctx_len + n_steps])
        preds = _batched_rollout(ae, dit, ctxs, seeds, args.n_steps)
        # roundtrip floors for the whole group in chunked batches
        flat = np.concatenate(gts)
        with torch.no_grad():
            xg = (torch.from_numpy(flat).permute(0, 3, 1, 2).float()
                  .div_(255.0).cuda())
            rec = torch.cat([ae.decode(ae.encode(xg[i:i + 512])).clamp(0, 1)
                             for i in range(0, xg.shape[0], 512)])
        rec = (rec.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        payloads, ofs = [], 0
        for (objs, thetas, init, outline), gt, pred in zip(metas, gts, preds):
            payloads.append((pred[:len(gt)], rec[ofs:ofs + len(gt)],
                             objs, thetas, init, outline))
            ofs += len(gt)
        return payloads

    groups = [trajs[i:i + args.group] for i in range(0, len(trajs), args.group)]
    for gi, group in enumerate(groups):
        payloads = build_group(group)
        if pending is not None:
            for out in pending.get():
                for kind, m, f in out:
                    per_kind.setdefault(kind, []).append((m, f))
            n_done += args.group
        pending = pool.map_async(_score_pair, payloads)
        if n_done and n_done % 48 == 0:
            el = time.time() - t0
            print(f"[{n_done}/{len(trajs)}] {el:.0f}s elapsed, "
                  f"{el / n_done:.2f}s/traj", flush=True)
    for out in pending.get():
        for kind, m, f in out:
            per_kind.setdefault(kind, []).append((m, f))
    n_done = len(trajs)
    pool.close()
    pool.join()
    el = time.time() - t0
    print(f"\nDeformation by kind | run {os.path.basename(args.run)} | "
          f"{args.n_traj} trajs x {args.n_steps} steps | clean frames only | "
          f"{el:.0f}s ({el / max(n_done, 1):.2f}s/traj)")
    print(f"{'kind':10s} {'model raw':>10s} {'GT floor':>9s} "
          f"{'excess':>8s} {'sem':>8s} {'n':>5s}")
    for k in sorted(per_kind):
        v = np.array(per_kind[k])
        d = v[:, 0] - v[:, 1]
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"{k:10s} {v[:, 0].mean():10.4f} {v[:, 1].mean():9.4f} "
              f"{d.mean():8.4f} {sem:8.4f} {len(d):5d}")


if __name__ == "__main__":
    main()
