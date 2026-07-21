import math
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

MATERIALS = {
    "Superball": {"rgb": (255, 0, 0), "density": 0.5, "restitution": 0.95, "friction": 0.98},
    "Rubber":    {"rgb": (0, 0, 255), "density": 1.2, "restitution": 0.80, "friction": 0.90},
    "Steel":     {"rgb": (100, 100, 100), "density": 5.0, "restitution": 0.40, "friction": 0.80},
    "Sponge":    {"rgb": (0, 255, 0), "density": 0.2, "restitution": 0.20, "friction": 0.60},
}
MAT_NAMES = list(MATERIALS.keys())
WORLD = 64
GRAVITY = -0.5
AIR_DRAG_COEFF = 0.02

_MAT_RGB = np.array([MATERIALS[m]["rgb"] for m in MAT_NAMES], np.float32)
_WC = 255.0 - _MAT_RGB
_WC_N2 = (_WC ** 2).sum(1)
_KERN3 = np.ones((3, 3), np.uint8)


def snap_radius(r):
    ri = int(round(r))
    return float(ri) if 5 <= ri <= 8 and abs(r - ri) < 0.4 else float(r)


def ball_mass(mat, radius):
    return math.pi * radius ** 2 * MATERIALS[mat]["density"]


def frames_from_torch(t):
    import torch
    x = t.detach().cpu()
    if x.is_floating_point():
        x = x.clamp(0, 1).mul(255).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().numpy()


def _alpha_res(img_f):
    d = 255.0 - img_f
    alpha = np.clip(np.einsum("hwc,mc->mhw", d, _WC) / _WC_N2[:, None, None], 0.0, 1.0)
    pred = alpha[..., None] * _WC[:, None, None, :]
    res = np.sqrt(((d[None] - pred) ** 2).sum(-1))
    return alpha, res


def extract_frame(img, res_tol=80.0, a_seed=0.3, min_area=15.0, return_pixels=False):
    alpha, res = _alpha_res(img.astype(np.float32))
    best = res.argmin(0)
    dets = []
    for m, name in enumerate(MAT_NAMES):
        seed = ((best == m) & (alpha[m] >= a_seed) & (res[m] <= res_tol)).astype(np.uint8)
        if not seed.any():
            continue
        n, lab = cv2.connectedComponents(seed, connectivity=8)
        w_full = np.where(res[m] <= res_tol, alpha[m], 0.0).astype(np.float32)
        for k in range(1, n):
            comp = (lab == k).astype(np.uint8)
            w = w_full * cv2.dilate(comp, _KERN3)
            area = float(w.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(w)
            ww = w[ys, xs]
            cx = float((xs * ww).sum() / area)
            cy = float((ys * ww).sum() / area)
            det = {"mat": name, "x": cx + 0.5, "y": (WORLD - 0.5) - cy,
                   "area": area, "radius": math.sqrt(area / math.pi), "merged": False}
            if return_pixels:
                det["px"] = (xs, ys, ww)
            dets.append(det)
    return dets


def _weighted_kmeans(pts, ww, k, centers=None):
    """Lloyd iterations on weighted pixels (farthest-point init); returns per-part
    (points, weights) subsets. Shared by detection splitting and ball counting."""
    if centers is None:
        i0 = int(np.argmax(ww))
        centers = [pts[i0]]
        while len(centers) < k:
            d2 = np.min([((pts - c) ** 2).sum(1) for c in centers], axis=0)
            centers.append(pts[int(np.argmax(d2 * ww))])
        centers = np.array(centers, np.float64)
    assign = None
    for _ in range(15):
        d2 = ((pts[:, None] - centers[None]) ** 2).sum(-1)
        assign = d2.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = (pts[m] * ww[m, None]).sum(0) / ww[m].sum()
    return [(pts[assign == j], ww[assign == j]) for j in range(k)], centers


def _radial_extent_ratio(pts, ww):
    """Weighted p99.5 pixel distance from the centroid, over the equivalent-area circle
    radius. ~1.0 for a single rendered ball; >>1 for same-material balls in contact."""
    area = float(ww.sum())
    if area <= 0:
        return 0.0, 0.0
    c = (pts * ww[:, None]).sum(0) / area
    d = np.sqrt(((pts - c) ** 2).sum(1))
    order = np.argsort(d)
    cw = np.cumsum(ww[order])
    d99 = d[order[min(np.searchsorted(cw, 0.995 * cw[-1]), len(order) - 1)]]
    r_eq = math.sqrt(area / math.pi)
    return d99 / max(r_eq, 1e-6), r_eq


def _count_balls(det, max_k=4):
    """How many balls does this detection cover? Same-material balls that stay in contact
    form ONE connected component, so counting components undercounts the inventory. Area
    cannot discriminate (two r=5 balls ~ one r=7 ball), but shape can: a single ball is a
    filled circle (radial extent ~ its equivalent-area radius), while k tangent balls are
    elongated. If elongated, accept the smallest k whose k-means parts each look like a
    plausible ball; otherwise count 1 (never invent balls a split cannot justify)."""
    if "px" not in det:
        return 1
    xs, ys, ww = det["px"]
    pts = np.stack([xs, ys], 1).astype(np.float64)
    ratio, _ = _radial_extent_ratio(pts, ww)
    if ratio <= 1.15:
        return 1
    for k in range(2, max_k + 1):
        if det["area"] < k * math.pi * 3.5 ** 2:
            break
        parts, _ = _weighted_kmeans(pts, ww, k)
        if len(parts) != k:
            continue
        ok = True
        for pp, pw in parts:
            rr, re = _radial_extent_ratio(pp, pw)
            if not (3.8 <= re <= 9.2 and rr <= 1.3):
                ok = False
                break
        if ok:
            return k
    return 1


def _split_detection(det, k, seeds_world=None):
    xs, ys, ww = det["px"]
    pts = np.stack([xs, ys], 1).astype(np.float64)
    centers = None
    if seeds_world is not None and len(seeds_world) == k:
        centers = np.array([[sx - 0.5, (WORLD - 0.5) - sy] for sx, sy in seeds_world])
    part_px, centers = _weighted_kmeans(pts, ww, k, centers=centers)
    parts = []
    for j, (pp, pw) in enumerate(part_px):
        area = float(pw.sum())
        if area <= 0:
            continue
        parts.append({"mat": det["mat"], "x": centers[j, 0] + 0.5,
                      "y": (WORLD - 0.5) - centers[j, 1], "area": area,
                      "radius": math.sqrt(area / math.pi), "merged": True})
    return parts


def _track_material(dets_seq, k, gate, coast_gate):
    T = len(dets_seq)
    pos = np.full((T, k, 2), np.nan)
    valid = np.zeros((T, k), bool)
    merged = np.zeros((T, k), bool)
    rad = np.full((T, k), np.nan)

    t0 = next((t for t, d in enumerate(dets_seq) if len(d) >= k), None)
    if t0 is None:
        # never k separate detections (same-material balls in contact the whole trajectory):
        # split the largest detection at the best frame instead of dropping tracks
        t0 = int(np.argmax([len(d) for d in dets_seq]))
        dets0 = list(dets_seq[t0])
        big = max(dets0, key=lambda d: d["area"]) if dets0 else None
        if big is not None and "px" in big:
            dets0.remove(big)
            dets0.extend(_split_detection(big, k - len(dets0)))
            dets_seq = list(dets_seq)
            dets_seq[t0] = dets0
        else:
            k = len(dets_seq[t0])
            pos, valid, merged, rad = pos[:, :k], valid[:, :k], merged[:, :k], rad[:, :k]
    init = sorted(dets_seq[t0], key=lambda d: -d["area"])[:k]
    for i, d in enumerate(init):
        pos[t0, i] = (d["x"], d["y"])
        valid[t0, i] = not d["merged"]
        merged[t0, i] = d["merged"]
        rad[t0, i] = d["radius"]

    last_pos = pos[t0].copy()
    last_real = pos[t0].copy()
    last_real_t = np.full(k, t0)
    vel = np.zeros((k, 2))
    vel_known = np.zeros(k, bool)
    coast = np.zeros(k, np.int32)
    BIG = 1e6

    for t in range(t0 + 1, T):
        pred = np.clip(last_pos + np.where(coast[:, None] > 3, 0.0, vel), 0.5, WORLD - 0.5)
        dets = list(dets_seq[t])
        deficit = k - len(dets)
        if deficit > 0 and dets:
            big = max(dets, key=lambda d: d["area"])
            if "px" in big:
                d2 = ((pred - np.array([big["x"], big["y"]])) ** 2).sum(1)
                seeds = pred[np.argsort(d2)[:deficit + 1]]
                dets.remove(big)
                dets.extend(_split_detection(big, deficit + 1, seeds_world=seeds))

        cost = np.full((k, max(len(dets), k)), BIG)
        for i in range(k):
            g = (gate + 2.0 * math.hypot(*vel[i]) + coast_gate * coast[i]) if vel_known[i] \
                else 32.0 + coast_gate * coast[i]
            for j, d in enumerate(dets):
                dist = min(math.hypot(pred[i, 0] - d["x"], pred[i, 1] - d["y"]),
                           math.hypot(last_pos[i, 0] - d["x"], last_pos[i, 1] - d["y"]))
                if dist <= g:
                    cost[i, j] = dist
        rows, cols = linear_sum_assignment(cost)
        matched = np.zeros(k, bool)
        for i, j in zip(rows, cols):
            if cost[i, j] >= BIG or j >= len(dets):
                continue
            d = dets[j]
            pos[t, i] = (d["x"], d["y"])
            valid[t, i] = not d["merged"]
            merged[t, i] = d["merged"]
            rad[t, i] = d["radius"]
            vel[i] = (pos[t, i] - last_real[i]) / max(t - last_real_t[i], 1)
            vel_known[i] = True
            last_real[i] = pos[t, i]
            last_real_t[i] = t
            last_pos[i] = pos[t, i]
            coast[i] = 0
            matched[i] = True
        for i in range(k):
            if not matched[i]:
                pos[t, i] = pred[i]
                last_pos[i] = pred[i]
                coast[i] += 1
    return pos, valid, merged, rad


def extract_states(frames, inventory=None, gate=10.0, coast_gate=4.0,
                   res_tol=80.0, a_seed=0.3, min_area=15.0):
    T = len(frames)
    dets_all = [extract_frame(f, res_tol=res_tol, a_seed=a_seed,
                              min_area=min_area, return_pixels=True) for f in frames]

    if inventory is None:
        inventory = {}
        min_persist = max(3, T // 20)
        for m in MAT_NAMES:
            counts = np.array([sum(_count_balls(d) for d in dets if d["mat"] == m)
                               for dets in dets_all])
            km = 0
            for c in range(1, int(counts.max(initial=0)) + 1):
                if (counts >= c).sum() >= min_persist:
                    km = c
            if km > 0:
                inventory[m] = km

    pos_l, valid_l, merged_l, rad_l, mats = [], [], [], [], []
    for m, k in inventory.items():
        seq = [[d for d in dets if d["mat"] == m] for dets in dets_all]
        p, v, mg, r = _track_material(seq, k, gate, coast_gate)
        pos_l.append(p); valid_l.append(v); merged_l.append(mg); rad_l.append(r)
        mats.extend([m] * p.shape[1])

    if not pos_l:
        return {"pos": np.zeros((T, 0, 2)), "valid": np.zeros((T, 0), bool),
                "merged": np.zeros((T, 0), bool), "mats": [], "radius": np.zeros(0),
                "mass": np.zeros(0), "inventory": inventory}

    pos = np.concatenate(pos_l, axis=1)
    valid = np.concatenate(valid_l, axis=1)
    merged = np.concatenate(merged_l, axis=1)
    rad_t = np.concatenate(rad_l, axis=1)

    for i in range(pos.shape[1]):
        if not valid[:, i].any():
            continue
        med = np.nanmedian(np.where(valid[:, i], rad_t[:, i], np.nan))
        ratio = rad_t[:, i] / med
        valid[:, i] &= (ratio >= 0.96) & (ratio <= 1.04)

    radius = np.array([snap_radius(np.nanmedian(np.where(valid[:, i], rad_t[:, i], np.nan)))
                       if valid[:, i].any() else np.nan for i in range(pos.shape[1])])
    mass = np.array([ball_mass(mats[i], radius[i]) if np.isfinite(radius[i]) else np.nan
                     for i in range(pos.shape[1])])
    return {"pos": pos, "valid": valid, "merged": merged, "mats": mats,
            "radius": radius, "mass": mass, "inventory": inventory}


def central_velocity(pos, valid):
    vel = np.full_like(pos, np.nan)
    ok = np.zeros(valid.shape, bool)
    if pos.shape[0] >= 3:
        vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0
        ok[1:-1] = valid[2:] & valid[:-2] & valid[1:-1]
    return vel, ok
