import os
import json
import cv2
import numpy as np
from tqdm import tqdm

from environments.env_bouncing import MATERIALS, WIDTH, HEIGHT, GRAVITY, AIR_DRAG_COEFF, \
    SUBPIX_BITS, _SUBPIX, REST_VELOCITY

# Rigid bodies with rotation in the same world as env_bouncing (gravity, drag,
# materials, 64x64 render path). One contact law for every body: restitution
# with a low-speed cutoff plus Coulomb friction from the materials. Kinds:
#   ball     -- disc with a diameter line showing rotation; friction makes it
#               spin and roll
#   halfdisc -- half circle; orientation visible from geometry alone
#   triangle / square / plank / hexagon -- convex polygons
# Material "friction" (tangential retention) maps to a Coulomb coefficient:
# low retention = grippy.
# Positional-correction energy control. Two mechanisms, both disabled by
# CV3_SHAPES_LEGACY_SETTLE=1 (which reproduces shapes_20k / tower_10k exactly):
#
# PEN_SLOP: corrections leave this much overlap unresolved. Without it, every substep of
# a gravity-loaded stack cycles sink -> impulse zeroes the approach speed (dissipative,
# correct) -> positional pass lifts the body fully back up, injecting free potential
# energy (up to +2% of scene energy per frame in tower collapses). 0.08 px is
# invisible at 64x64 under AA.
#
# PCORR_CAP: per-body, per-SUBSTEP total positional-correction budget (Box2D's
# maxLinearCorrection idea). During active collapse, bodies interpenetrate by ~their
# approach speed x dt each substep -- deeper than any invisible slop -- and lifting that
# out in full every substep is the remaining energy pump. The budget bounds the
# injection rate; leftover overlap resolves over the next substeps as motion slows.
# Budgets exist only in the dynamic path (_contact_step resets them); the build-time
# _settle in _verified_tower / spawning has no budget keys and settles fully.
_LEGACY_SETTLE = os.environ.get("CV3_SHAPES_LEGACY_SETTLE", "0") == "1"
PEN_SLOP = 0.0 if _LEGACY_SETTLE else 0.08
PCORR_CAP = float("inf") if _LEGACY_SETTLE else 0.15
# Deep-penetration escape hatch: a fixed budget alone lets a violent collapse accumulate
# multi-px overlap (3px transients -- a thin plank visibly inside another) because
# the sink rate outpaces PCORR_CAP for several substeps. Box2D-style proportional boost:
# when a contact is over-penetrated, the involved bodies' budgets are topped up to
# PCORR_FRAC x the excess, so a 3px overlap resolves within ~a frame while ordinary
# contacts keep the gentle energy-safe cap.
PCORR_FRAC = 0.3
# Continuous-collision guard: after integrating a substep, if any pair (or wall) is
# penetrated deeper than CCD_TOL, the substep is rolled back and re-run as CCD_SPLIT
# micro-steps, so contacts fire before bodies sink visibly into each other. This is
# time-of-impact detection by bisection-in-time: it only costs extra in the few
# impact substeps where it triggers. Off under the legacy flag.
CCD_TOL = 0.2
CCD_SPLIT = 4
# SKIN stays 0: a physical contact skin makes bodies pogo on the skin surface and
# never sleep. No render-side inset or edge outline either: markers="on" renders
# flat bodies with orientation dots only.
SKIN = 0.0


def _rot_drag(o, dt):
    """Quadratic air drag on rotation, same physics and coefficient as the linear drag:
    the rim moves at omega*r_eff, feels the standard v|v| drag, and that force acts at
    lever arm r_eff -- so torque ~ AIR_DRAG_COEFF * omega|omega| * r_eff^4. Without this,
    spin never decays in free flight (a ball in the air spins forever), and rare contact
    kicks can exceed the pi/frame orientation-readability limit; quadratic drag bleeds
    those extreme spins fast while barely touching ordinary ones. Off under the legacy
    flag (reproduces pre-fix datasets)."""
    if _LEGACY_SETTLE or o["omega"] == 0.0:
        return
    w = o["omega"]
    o["omega"] = w - (AIR_DRAG_COEFF * w * abs(w) * o["r_eff"] ** 4 / o["inertia"]) * dt

KINDS = ["ball", "halfdisc", "triangle", "square", "plank", "hexagon", "plus"]
DEFAULT_KIND_WEIGHTS = {"ball": 0.25, "halfdisc": 0.15, "triangle": 0.15,
                        "square": 0.15, "plank": 0.15, "hexagon": 0.15}
# The easy-mode roster: four maximally distinct silhouettes.
EASY_KIND_WEIGHTS = {"ball": 0.25, "triangle": 0.25, "plus": 0.25, "plank": 0.25}
PLUS_ARM_FRAC = 0.34    # plus arm half-thickness as a fraction of its half-length
# Per-kind size ranges: triangle and plank have the smallest AREA
# per unit size (~1.3r^2 / 1.6r^2 vs the ball's 3.1r^2), which made their small
# instances the DiT's contact victims (deformed then swallowed). Their ranges
# rise mostly from the bottom -- rough area parity with the other kinds at the
# low end, max nudged only slightly.
EASY_SIZE_RANGES = {"triangle": (7.0, 9.0), "plank": (6.5, 8.5)}
# v3: plus joined the enlarged kinds (it became the deformation victim once
# triangles grew). EASY_SIZE_RANGES stays frozen for v2 reproducibility;
# v3 datasets use this.
EASY_SIZE_RANGES_V3 = {"triangle": (7.0, 9.0), "plank": (6.5, 8.5),
                       "plus": (6.5, 8.5)}
REST_OMEGA = 0.05
HALFDISC_ARC = 9
# Sleeping: a body whose speeds stay under these thresholds for SLEEP_AFTER
# consecutive substeps is frozen exactly (v = omega = 0) until a contact
# impulse above ~WAKE_FACTOR x its gravity-support impulse arrives. This is how
# resting stacks become *pixel-static* -- an iterative solver alone always
# leaves micro-creep.
SLEEP_V = 0.02
SLEEP_W = 0.003
SLEEP_AFTER = 8
WAKE_FACTOR = 2.0
WAKE_SPEED = 0.1   # contact with a body moving faster than this wakes a sleeper
                   # (you cannot sleep on a conveyor belt -- a beam sliding under
                   # frozen columns looks exactly as wrong as it is)


def _mu_pair(a, b):
    return 0.5 * ((1.0 - a["mat"]["friction"]) + (1.0 - b["mat"]["friction"]))


def _local_verts(kind, size):
    if kind == "triangle":
        ang = np.pi / 2 + np.arange(3) * (2 * np.pi / 3)
    elif kind == "square":
        ang = np.pi / 4 + np.arange(4) * (np.pi / 2)
    elif kind == "hexagon":
        ang = np.arange(6) * (np.pi / 3)
    elif kind == "plank":
        b = size / np.sqrt(1 + 2.0 ** 2)
        a = 2.0 * b
        return np.array([(-a, -b), (a, -b), (a, b), (-a, b)])
    elif kind == "halfdisc":
        ang = np.linspace(0, np.pi, HALFDISC_ARC)
        verts = np.stack([size * np.cos(ang), size * np.sin(ang)], axis=1)
        verts[:, 1] -= 4 * size / (3 * np.pi)  # shift so the centroid is the origin
        return verts
    else:
        raise ValueError(kind)
    return np.stack([size * np.cos(ang), size * np.sin(ang)], axis=1)


def _jitter_verts(verts, jitter, jitter_y=None):
    """Perturb a polygon's vertices by up to +-jitter px, fixed for the object's
    lifetime: visible shape imperfection (the anti-aliased render carries the
    sub-pixel silhouette), never hidden state. jitter_y caps the local-y
    perturbation separately (tower planks need nearly-flat contact faces or
    stacks stand on randomly slanted feet). Convex hull keeps the SAT math
    valid; re-centering keeps rotation about the true centroid."""
    if jitter <= 0:
        return verts
    jy = jitter if jitter_y is None else jitter_y
    v = verts + np.column_stack([np.random.uniform(-jitter, jitter, len(verts)),
                                 np.random.uniform(-jy, jy, len(verts))])
    hull = cv2.convexHull(v.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    x, y = hull[:, 0], hull[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    area = cr.sum() / 2
    cx = ((x + xn) * cr).sum() / (6 * area)
    cy = ((y + yn) * cr).sum() / (6 * area)
    return hull - np.array([cx, cy])


def _poly_props(verts, density):
    x, y = verts[:, 0], verts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = float(abs(cross.sum()) / 2)
    num = (cross * (x * x + x * xn + xn * xn + y * y + y * yn + yn * yn)).sum()
    mass = area * density
    inertia = float(abs(num) / (6 * abs(cross.sum()))) * mass
    return area, mass, inertia


def _plus_geometry(size, density):
    """A plus sign is concave, which SAT and pymunk.Poly cannot take whole: the
    PHYSICS body is a COMPOUND of two axis-aligned rectangles (the pymunk backend
    attaches both as fixtures), while the RENDER polygon is the single 12-vertex
    union outline (cv2.fillPoly handles concave), so fills, outlines and the dot
    markers use the ordinary polygon path. Exact area/inertia by
    inclusion-exclusion: the bars overlap in the central square, subtract it once.
    Returns (render_verts, parts, area, mass, inertia, bound, r_eff)."""
    a, b = float(size), float(size) * PLUS_ARM_FRAC
    horiz = np.array([(-a, -b), (a, -b), (a, b), (-a, b)])
    vert = np.array([(-b, -a), (b, -a), (b, a), (-b, a)])
    union = np.array([(a, b), (b, b), (b, a), (-b, a), (-b, b), (-a, b),
                      (-a, -b), (-b, -b), (-b, -a), (b, -a), (b, -b), (a, -b)])
    area = 2 * (2 * a) * (2 * b) - (2 * b) ** 2
    mass = area * density
    m_bar = (2 * a) * (2 * b) * density
    m_sq = (2 * b) ** 2 * density
    i_bar = m_bar * ((2 * a) ** 2 + (2 * b) ** 2) / 12.0
    i_sq = m_sq * (2 * (2 * b) ** 2) / 12.0
    inertia = 2 * i_bar - i_sq
    return union, [horiz, vert], area, mass, float(inertia), \
        float(np.hypot(a, b)), float(np.sqrt(area / np.pi))


CLASSIC_MATERIALS = ["Superball", "Rubber", "Steel", "Sponge"]


def _make_shape(width, height, speed_min, speed_max, kind_weights, spin_max,
                y_frac_min=0.3, y_frac_max=1.0, vertex_jitter=0.0,
                size_min=5, size_max=8, size_ranges=None):
    # sample from the CLASSIC four, never the whole dict: MATERIALS also
    # holds the env_solar bodies, and `list(MATERIALS.keys())` would roll
    # star/planet colors into shapes/balls scenes. Every pre-solar dataset
    # used exactly these four, in this order (np.random stream compatible).
    mat_name = np.random.choice(CLASSIC_MATERIALS)
    mat = MATERIALS[mat_name]
    kinds = list(kind_weights.keys())
    probs = np.array([kind_weights[k] for k in kinds], dtype=np.float64)
    kind = str(np.random.choice(kinds, p=probs / probs.sum()))
    if size_ranges is not None and kind in size_ranges:
        lo, hi = size_ranges[kind]
        size = float(np.random.uniform(lo, hi))
    else:
        size = float(np.random.randint(int(size_min), int(size_max) + 1))
    obj = {"kind": kind, "mat": mat, "mat_name": mat_name, "size": size,
           "theta": float(np.random.uniform(0, 2 * np.pi)),
           "omega": float(np.random.uniform(-spin_max, spin_max))}
    if kind == "ball":
        obj["verts"] = None
        obj["mass"] = (np.pi * size ** 2) * mat["density"]
        obj["inertia"] = 0.5 * obj["mass"] * size ** 2
        obj["r_eff"] = size
    elif kind == "plus":
        # verts = the concave union outline: valid for rendering and objects.json,
        # NOT for the old engine's convex SAT -- plus is pymunk-backend only,
        # where _add_obj attaches obj["parts"] as the physical fixtures.
        union, parts, area, obj["mass"], obj["inertia"], bound, obj["r_eff"] = \
            _plus_geometry(size, mat["density"])
        obj["verts"] = union
        obj["parts"] = parts
        obj["size"] = bound
    else:
        verts = _jitter_verts(_local_verts(kind, size), vertex_jitter)
        area, obj["mass"], obj["inertia"] = _poly_props(verts, mat["density"])
        obj["verts"] = verts
        obj["size"] = float(np.hypot(verts[:, 0], verts[:, 1]).max())
        obj["r_eff"] = float(np.sqrt(area / np.pi))
    speed = np.random.uniform(speed_min, speed_max)
    y_lo = max(size, int(y_frac_min * height))
    y_hi = max(y_lo + 1, min(height - size, int(y_frac_max * height)))
    obj["x"] = float(np.random.randint(int(size), int(width - size)))
    obj["y"] = float(np.random.randint(int(y_lo), int(y_hi)))
    obj["vx"] = np.random.randn() * speed
    obj["vy"] = np.random.randn() * speed
    obj["hit_wall"] = False
    obj["hit_pair"] = False
    obj["asleep"] = False
    obj["still"] = 0
    return obj


def _spawn_shapes(n, width, height, speed_min, speed_max, kind_weights, spin_max,
                  max_tries=100, y_frac_min=0.3, y_frac_max=1.0, vertex_jitter=0.0,
                  size_min=5, size_max=8, size_ranges=None):
    objs = []
    for _ in range(n):
        for _ in range(max_tries):
            cand = _make_shape(width, height, speed_min, speed_max, kind_weights,
                               spin_max, y_frac_min, y_frac_max, vertex_jitter,
                               size_min, size_max, size_ranges)
            ok = True
            for o in objs:
                min_dist = cand["size"] + o["size"]
                if (cand["x"] - o["x"]) ** 2 + (cand["y"] - o["y"]) ** 2 < min_dist ** 2:
                    ok = False
                    break
            if ok:
                objs.append(cand)
                break
    return objs


def _world_geo(o):
    # pose-keyed cache: bodies are queried many times per pose (settle sweeps,
    # velocity iterations, rendering); values are identical, only recomputed
    # when the body actually moved. Callers must not mutate the results.
    # Cached alongside the vertices: edge vectors (verts rolled by one minus
    # verts -- a pure permutation and subtraction, identical to the previous
    # per-call np.roll computation).
    key = (o["x"], o["y"], o["theta"])
    cached = o.get("_wv")
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    c, s = np.cos(o["theta"]), np.sin(o["theta"])
    R = np.array([[c, -s], [s, c]])
    w = o["verts"] @ R.T + np.array([o["x"], o["y"]])
    e = np.concatenate([w[1:], w[:1]]) - w
    o["_wv"] = (key, w, e)
    return w, e


def _world_verts(o):
    return _world_geo(o)[0]


def _vel_at(o, px, py):
    rx, ry = px - o["x"], py - o["y"]
    return o["vx"] - o["omega"] * ry, o["vy"] + o["omega"] * rx


def _apply_impulse(o, jx, jy, px, py):
    o["vx"] += jx / o["mass"]
    o["vy"] += jy / o["mass"]
    rx, ry = px - o["x"], py - o["y"]
    o["omega"] += (rx * jy - ry * jx) / o["inertia"]


def _project_interval(verts, nx, ny):
    d = verts[:, 0] * nx + verts[:, 1] * ny
    return d.min(), d.max()


def _poly_poly_contact(A, B):
    # Vectorized SAT over all edge normals at once. Arithmetic is elementwise
    # only (no matmul/FMA reassociation) and np.argmin takes the FIRST
    # minimum, so every value stays bit-identical to a per-edge loop with
    # strictly-smaller updates (dataset reproducibility).
    (va, ea), (vb, eb) = _world_geo(A), _world_geo(B)
    E = np.concatenate([ea, eb])
    ln = np.hypot(E[:, 0], E[:, 1])
    valid = ln >= 1e-12
    n_a_valid = int(valid[:len(ea)].sum())
    if not valid.all():
        E, ln = E[valid], ln[valid]
    nxs = -E[:, 1] / ln
    nys = E[:, 0] / ln
    pa = va[:, 0][:, None] * nxs[None, :] + va[:, 1][:, None] * nys[None, :]
    pb = vb[:, 0][:, None] * nxs[None, :] + vb[:, 1][:, None] * nys[None, :]
    pens = (np.minimum(pa.max(axis=0), pb.max(axis=0))
            - np.maximum(pa.min(axis=0), pb.min(axis=0)))
    if (pens <= -SKIN).any():
        return None
    k = int(np.argmin(pens))
    pen = pens[k] + SKIN
    nx, ny = nxs[k], nys[k]
    owner = "A" if k < n_a_valid else "B"
    d = (B["x"] - A["x"]) * nx + (B["y"] - A["y"]) * ny
    if d < 0:
        nx, ny = -nx, -ny
    # Up to two contact points from the incident polygon: a face resting on a
    # face must be supported at both ends or it rocks about a single point.
    if owner == "A":
        ref, inc = va, vb
        d = vb[:, 0] * nx + vb[:, 1] * ny
        order = np.argsort(d)
        keep = d[order[1]] < d[order[0]] + 0.15 if len(order) > 1 else False
    else:
        ref, inc = vb, va
        d = va[:, 0] * nx + va[:, 1] * ny
        order = np.argsort(-d)
        keep = d[order[1]] > d[order[0]] - 0.15 if len(order) > 1 else False
    pts = [inc[order[0]]] + ([inc[order[1]]] if keep else [])
    # Clamp the points into the reference body's span along the face direction:
    # the incident body's deepest corners can lie far outside the overlap (a
    # long beam resting on a narrow column), and support applied there torques
    # the pair apart.
    tx, ty = -ny, nx
    rt = ref[:, 0] * tx + ref[:, 1] * ty
    rt0, rt1 = float(rt.min()), float(rt.max())
    out = []
    for p in pts:
        s = float(p[0] * tx + p[1] * ty)
        sc = min(max(s, rt0), rt1)
        q = (float(p[0] + (sc - s) * tx), float(p[1] + (sc - s) * ty))
        if not any(abs(q[0] - o[0]) + abs(q[1] - o[1]) < 0.05 for o in out):
            out.append(q)
    return nx, ny, pen, out


def _circle_poly_contact(c, P):
    """Contact with the normal pointing circle -> polygon."""
    verts, edges = _world_geo(P)
    px, py = c["x"], c["y"]
    rel = np.array([px, py]) - verts
    t = np.clip((rel[:, 0] * edges[:, 0] + rel[:, 1] * edges[:, 1]) /
                np.maximum((edges ** 2).sum(axis=1), 1e-12), 0, 1)
    closest = verts + edges * t[:, None]
    d2 = ((closest - np.array([px, py])) ** 2).sum(axis=1)
    k = np.argmin(d2)
    dist = np.sqrt(d2[k])
    inside = cv2.pointPolygonTest(verts.astype(np.float32), (float(px), float(py)), False) >= 0
    if not inside and dist >= c["size"] + SKIN:
        return None
    if dist < 1e-9:
        nx, ny = 0.0, -1.0
    else:
        nx = (closest[k, 0] - px) / dist
        ny = (closest[k, 1] - py) / dist
    if inside:
        nx, ny = -nx, -ny
        pen = c["size"] + SKIN + dist
    else:
        pen = c["size"] + SKIN - dist
    return nx, ny, pen, [(float(closest[k, 0]), float(closest[k, 1]))]


def _axis_clamped_move(o, dx, dy, width, height):
    """Move o by (dx, dy), clamped per axis so it never leaves the box. Returns
    the applied displacement. Keeping objects inside here (instead of letting a
    later wall clamp teleport them back) is what keeps piles from re-penetrating."""
    if o["verts"] is None:
        lo_x, lo_y = o["x"] - o["size"], o["y"] - o["size"]
        hi_x, hi_y = width - o["x"] - o["size"], height - o["y"] - o["size"]
    else:
        v = _world_verts(o)
        lo_x, lo_y = v[:, 0].min(), v[:, 1].min()
        hi_x, hi_y = width - v[:, 0].max(), height - v[:, 1].max()
    mdx = float(np.clip(dx, -max(lo_x, 0.0), max(hi_x, 0.0)))
    mdy = float(np.clip(dy, -max(lo_y, 0.0), max(hi_y, 0.0)))
    o["x"] += mdx
    o["y"] += mdy
    return mdx, mdy


def _separate(a, b, nx, ny, pen, width, height):
    """Positional separation, wall-aware (a moves along -n, b along +n): mass
    shares first, each move clamped per axis at the walls (tangential sliding
    survives), and separation a wall-pinned partner could not absorb is handed
    to the other object. Sleeping bodies are positionally immovable -- their
    share goes to the awake partner (a genuine squeeze wakes them through the
    impulse threshold instead)."""
    if a["asleep"] and b["asleep"]:
        return
    pen = pen - PEN_SLOP
    if pen <= 0.0:
        return
    inf = float("inf")
    a_cap = a.get("_pcorr", inf)
    b_cap = b.get("_pcorr", inf)
    if a_cap <= 0.0 and b_cap <= 0.0:
        return
    if a["asleep"]:
        a_share, b_share = 0.0, pen
    elif b["asleep"]:
        a_share, b_share = pen, 0.0
    else:
        a_share = pen * (b["mass"] / (a["mass"] + b["mass"]))
        b_share = pen - a_share
    a_share = min(a_share, a_cap)
    b_share = min(b_share, b_cap)
    adx, ady = _axis_clamped_move(a, -a_share * nx, -a_share * ny, width, height)
    bdx, bdy = _axis_clamped_move(b, b_share * nx, b_share * ny, width, height)
    if "_pcorr" in a:
        a["_pcorr"] = max(0.0, a["_pcorr"] + (adx * nx + ady * ny))
    if "_pcorr" in b:
        b["_pcorr"] = max(0.0, b["_pcorr"] - (bdx * nx + bdy * ny))
    rem = pen + (adx * nx + ady * ny) - (bdx * nx + bdy * ny)
    if rem > 1e-9 and not b["asleep"]:
        extra = min(rem, b.get("_pcorr", inf))
        if extra > 0.0:
            bdx, bdy = _axis_clamped_move(b, extra * nx, extra * ny, width, height)
            if "_pcorr" in b:
                b["_pcorr"] = max(0.0, b["_pcorr"] - (bdx * nx + bdy * ny))
            rem -= bdx * nx + bdy * ny
    if rem > 1e-9 and not a["asleep"]:
        extra = min(rem, a.get("_pcorr", inf))
        if extra > 0.0:
            adx, ady = _axis_clamped_move(a, -extra * nx, -extra * ny, width, height)
            if "_pcorr" in a:
                a["_pcorr"] = max(0.0, a["_pcorr"] + (adx * nx + ady * ny))


def _pair_contact(a, b):
    """Contact normal points a -> b."""
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    reach = a["size"] + b["size"] + SKIN
    if dx * dx + dy * dy >= reach * reach:
        return None
    if a["verts"] is None and b["verts"] is None:
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        dist = np.hypot(dx, dy)
        min_dist = a["size"] + b["size"] + SKIN
        if dist == 0 or dist >= min_dist:
            return None
        nx, ny = dx / dist, dy / dist
        cx = a["x"] + nx * (a["size"] - (min_dist - dist) / 2)
        cy = a["y"] + ny * (a["size"] - (min_dist - dist) / 2)
        return nx, ny, min_dist - dist, [(cx, cy)]
    if a["verts"] is None:
        return _circle_poly_contact(a, b)
    if b["verts"] is None:
        r = _circle_poly_contact(b, a)
        if r is None:
            return None
        nx, ny, pen, pts = r
        return -nx, -ny, pen, pts
    return _poly_poly_contact(a, b)


def _wake_on(o, j, dt):
    if o["asleep"] and abs(j) > WAKE_FACTOR * o["mass"] * abs(GRAVITY) * dt:
        o["asleep"] = False
        o["still"] = 0


def _pair_vel_pass(a, b, contact, dt, first, reverse=False):
    nx, ny, pen, points = contact
    m_eff = 1.0 / (1.0 / a["mass"] + 1.0 / b["mass"])
    mu = _mu_pair(a, b)
    for cx, cy in (points[::-1] if reverse else points):
        rax, ray = cx - a["x"], cy - a["y"]
        rbx, rby = cx - b["x"], cy - b["y"]
        avx, avy = _vel_at(a, cx, cy)
        bvx, bvy = _vel_at(b, cx, cy)
        vrel = (avx - bvx) * nx + (avy - bvy) * ny
        j = 0.0
        if vrel > 0:
            ta = (rax * ny - ray * nx) ** 2 / a["inertia"]
            tb = (rbx * ny - rby * nx) ** 2 / b["inertia"]
            kn = 1.0 / a["mass"] + 1.0 / b["mass"] + ta + tb
            # restitution only on the first pass and only above the rest-speed
            # cutoff: later passes just drive residual approach speed to zero,
            # or piles rattle forever
            e = min(a["mat"]["restitution"], b["mat"]["restitution"]) \
                if (first and vrel > REST_VELOCITY) else 0.0
            j = (1 + e) * vrel / kn
            _apply_impulse(a, -j * nx, -j * ny, cx, cy)
            _apply_impulse(b, j * nx, j * ny, cx, cy)
            _wake_on(a, j, dt)
            _wake_on(b, j, dt)
        if mu > 0:
            tx, ty = -ny, nx
            avx, avy = _vel_at(a, cx, cy)
            bvx, bvy = _vel_at(b, cx, cy)
            vt = (avx - bvx) * tx + (avy - bvy) * ty
            ta = (rax * ty - ray * tx) ** 2 / a["inertia"]
            tb = (rbx * ty - rby * tx) ** 2 / b["inertia"]
            kt = 1.0 / a["mass"] + 1.0 / b["mass"] + ta + tb
            # friction acts on every contact (not just approaching ones); the cap
            # includes the gravity support impulse so resting objects can't skate
            cap = mu * (abs(j) + m_eff * abs(GRAVITY) * dt / len(points))
            jt = np.clip(vt / kt, -cap, cap)
            _apply_impulse(a, -jt * tx, -jt * ty, cx, cy)
            _apply_impulse(b, jt * tx, jt * ty, cx, cy)


def _wall_contacts(o, width, height):
    """Gather wall contacts for one body, clamping it back into the box. The
    contact set is computed once per substep and reused across velocity passes.
    The clamp is slop- and budget-limited like _separate: lifting a body fully
    out of the floor every substep is the same free-potential-energy pump as in
    pair separation (under the legacy gate slop is 0 and the budget infinite,
    which reduces exactly to the original full clamp)."""
    cs = []
    for nx, ny, wall_c in ((0.0, 1.0, 0.0), (0.0, -1.0, float(height)),
                           (1.0, 0.0, 0.0), (-1.0, 0.0, float(width))):
        pts = _support_points(o, nx, ny, wall_c)
        depth = min(p[2] for p in pts)
        if depth >= 0:
            continue
        corr = max(0.0, -depth - PEN_SLOP)
        corr = min(corr, o.get("_pcorr", float("inf")))
        if corr > 0.0:
            o["x"] += corr * nx
            o["y"] += corr * ny
            if "_pcorr" in o:
                o["_pcorr"] = max(0.0, o["_pcorr"] - corr)
        o["hit_wall"] = True
        touching = [(px + corr * nx, py + corr * ny) for px, py, d in pts if d < 0.25]
        cs.append((o, nx, ny, touching))
    return cs


def _wall_vel_pass(o, nx, ny, touching, dt, first, reverse=False):
    e, mu = o["mat"]["restitution"], 1.0 - o["mat"]["friction"]
    for px, py in (touching[::-1] if reverse else touching):
        rx, ry = px - o["x"], py - o["y"]
        pvx, pvy = _vel_at(o, px, py)
        vn = pvx * nx + pvy * ny
        j = 0.0
        if vn < 0:
            kn = 1.0 / o["mass"] + (rx * ny - ry * nx) ** 2 / o["inertia"]
            rest = e if (first and abs(vn) > REST_VELOCITY) else 0.0
            j = -(1 + rest) * vn / kn
            _apply_impulse(o, j * nx, j * ny, px, py)
        if mu > 0:
            tx, ty = -ny, nx
            pvx, pvy = _vel_at(o, px, py)
            vt = pvx * tx + pvy * ty
            kt = 1.0 / o["mass"] + (rx * ty - ry * tx) ** 2 / o["inertia"]
            cap = mu * (abs(j) + o["mass"] * abs(GRAVITY) * dt / len(touching))
            jt = np.clip(-vt / kt, -cap, cap)
            _apply_impulse(o, jt * tx, jt * ty, px, py)


def _find_supporters(o, objs, height):
    """What holds o up right now: (supporter, contact_x) pairs -- 'floor' and/or
    bodies it would press against when nudged down. Probed with a small downward
    shift so exactly-touching resting contacts (pen = 0) register."""
    probe = dict(o)
    probe["y"] = o["y"] - 0.15
    sup = []
    if o["verts"] is None:
        if o["y"] - o["size"] <= 0.2:
            sup.append(("floor", o["x"]))
    else:
        v = _world_verts(o)
        for vx, vy in v:
            if vy <= 0.2:
                sup.append(("floor", float(vx)))
    for b in objs:
        if b is o:
            continue
        c = _pair_contact(probe, b)
        if c is not None and c[1] < -0.5:
            sup.extend((b, x) for x, y in c[3])
    return sup


def _audit_sleepers(objs, height):
    """Wake sleeping bodies whose support has changed: when a supporter is
    awake (moving), re-probe with the same stability rule the sleep gate uses
    (center of mass over the support span). Without this, knocking a tower's
    lower storey out leaves the upper storeys frozen in mid-air -- or balanced
    on a single surviving column they should topple off. Sleepers whose
    supporters are all still asleep (or the floor) are skipped."""
    for o in objs:
        if not o["asleep"]:
            continue
        if not any(s != "floor" and not s["asleep"] for s in o.get("sup", [])):
            continue
        sup = _find_supporters(o, objs, height)
        xs = [x for _, x in sup]
        if sup and min(xs) - 0.05 <= o["x"] <= max(xs) + 0.05:
            o["sup"] = [s for s, _ in sup]
        else:
            o["asleep"] = False
            o["still"] = 0


def _contact_step(objs, width, height, dt, vel_iters=8):
    """One substep of contact handling: wake sleepers that lost their support,
    gather all pair and wall contacts, run iterated velocity passes over the
    full set (a single pass cannot balance a stack -- solving one contact
    re-violates its neighbor and the residual rocks the pile apart), then
    positional cleanup."""
    _audit_sleepers(objs, height)
    for o in objs:
        o["_pcorr"] = PCORR_CAP
    pair_cs = []
    support = {}
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if objs[i]["asleep"] and objs[j]["asleep"]:
                continue
            c = _pair_contact(objs[i], objs[j])
            if c is not None:
                objs[i]["hit_pair"] = True
                objs[j]["hit_pair"] = True
                pair_cs.append((objs[i], objs[j], c))
                nx, ny, _, pts = c
                for a, b in ((objs[i], objs[j]), (objs[j], objs[i])):
                    if a["asleep"] and not b["asleep"]:
                        cx, cy = pts[0]
                        bvx, bvy = _vel_at(b, cx, cy)
                        if np.hypot(bvx, bvy) > WAKE_SPEED:
                            a["asleep"] = False
                            a["still"] = 0
                for cx, cy in pts:
                    if ny > 0.5:
                        support.setdefault(id(objs[j]), []).append((cx, objs[i]))
                    elif ny < -0.5:
                        support.setdefault(id(objs[i]), []).append((cx, objs[j]))
    for a, b, c in pair_cs:
        over = c[2] - PEN_SLOP
        if over > 0.0:
            boost = PCORR_FRAC * over
            if not a["asleep"] and boost > a["_pcorr"]:
                a["_pcorr"] = boost
            if not b["asleep"] and boost > b["_pcorr"]:
                b["_pcorr"] = boost
    wall_cs = []
    for o in objs:
        if not o["asleep"]:
            wall_cs.extend(_wall_contacts(o, width, height))
    for o, nx, ny, touching in wall_cs:
        if ny == 1.0:
            support.setdefault(id(o), []).extend((px, "floor") for px, py in touching)

    for it in range(vel_iters):
        first = it == 0
        rev = it % 2 == 1
        # alternate sweep and point order every pass: a fixed order leaves a
        # systematic residual that creeps piles sideways frame after frame
        for a, b, c in (pair_cs[::-1] if rev else pair_cs):
            _pair_vel_pass(a, b, c, dt, first, reverse=rev)
        for o, nx, ny, touching in (wall_cs[::-1] if rev else wall_cs):
            _wall_vel_pass(o, nx, ny, touching, dt, first, reverse=rev)

    for a, b, c in pair_cs:
        nx, ny, pen, _ = c
        _separate(a, b, nx, ny, pen, width, height)
    for o, nx, ny, touching in wall_cs:
        if ny == 1.0 and len(touching) > 1 and abs(o["vy"]) < REST_VELOCITY:
            # resting flat on the floor: kill the residual creep
            o["vy"] = 0.0
            if abs(o["omega"]) < REST_OMEGA:
                o["omega"] = 0.0
                if abs(o["vx"]) < REST_VELOCITY * (1.0 - o["mat"]["friction"]):
                    o["vx"] = 0.0
    _settle(objs, width, height)
    for o in objs:
        if o["asleep"]:
            # sub-threshold micro-impulses must not accumulate in a sleeper
            o["vx"] = o["vy"] = o["omega"] = 0.0
            continue
        sup = support.get(id(o))
        # sleep requires quiet velocities AND static stability: the center of
        # mass over the support span. A square balanced on its corner is quiet
        # but must stay awake so gravity can topple it.
        xs = [x for x, s in sup] if sup else None
        stable = xs is not None and min(xs) - 0.05 <= o["x"] <= max(xs) + 0.05
        if stable and abs(o["vx"]) < SLEEP_V and abs(o["vy"]) < SLEEP_V and abs(o["omega"]) < SLEEP_W:
            o["still"] += 1
            if o["still"] >= SLEEP_AFTER:
                o["asleep"] = True
                o["vx"] = o["vy"] = o["omega"] = 0.0
                o["sup"] = [s for x, s in sup]
        else:
            o["still"] = 0


def _integrate_body(o, dt):
    """One substep of free motion for an awake body: quadratic air drag (linear and
    rotational), gravity, position and angle update."""
    drag_x = -(AIR_DRAG_COEFF * o["vx"] * abs(o["vx"]) * o["r_eff"]) / o["mass"]
    drag_y = -(AIR_DRAG_COEFF * o["vy"] * abs(o["vy"]) * o["r_eff"]) / o["mass"]
    o["vx"] += drag_x * dt
    o["vy"] += (GRAVITY + drag_y) * dt
    o["x"] += o["vx"] * dt
    o["y"] += o["vy"] * dt
    o["theta"] = (o["theta"] + o["omega"] * dt) % (2 * np.pi)
    _rot_drag(o, dt)


def _max_pen(objs, width, height):
    """Deepest current penetration: pair overlaps (pairs with at least one awake body,
    bounding-circle prefiltered) and wall overlaps of awake bodies."""
    mx = 0.0
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if a["asleep"] and b["asleep"]:
                continue
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            reach = a["size"] + b["size"]
            if dx * dx + dy * dy >= reach * reach:
                continue
            c = _pair_contact(a, b)
            if c is not None and c[2] - SKIN > mx:   # c[2] includes the contact skin;
                mx = c[2] - SKIN                      # CCD cares about REAL overlap only
    for o in objs:
        if o["asleep"]:
            continue
        s = o["size"]
        for nx, ny, wall_c in ((0.0, 1.0, 0.0), (0.0, -1.0, float(height)),
                               (1.0, 0.0, 0.0), (-1.0, 0.0, float(width))):
            near = (o["y"] - s < 0.5 if ny == 1.0 else o["y"] + s > height - 0.5 if ny == -1.0
                    else o["x"] - s < 0.5 if nx == 1.0 else o["x"] + s > width - 0.5)
            if not near:
                continue
            depth = min(p[2] for p in _support_points(o, nx, ny, wall_c))
            if -depth > mx:
                mx = -depth
    return mx


def _sim_substeps(objs, width, height, n_substeps):
    """One rendered frame of physics. Each substep integrates the awake bodies and runs
    the contact solver; if integration sank anything deeper than CCD_TOL, the substep is
    rolled back and re-run as CCD_SPLIT micro-steps (time-of-impact refinement), so fast
    impacts are caught before visible interpenetration forms."""
    dt = 1.0 / n_substeps
    keys = ("x", "y", "vx", "vy", "theta", "omega")
    for _ in range(n_substeps):
        awake = [o for o in objs if not o["asleep"]]
        snap = [tuple(o[k] for k in keys) for o in awake]
        for o in awake:
            _integrate_body(o, dt)
        if not _LEGACY_SETTLE and _max_pen(objs, width, height) > CCD_TOL:
            for o, sv in zip(awake, snap):
                for k, v in zip(keys, sv):
                    o[k] = v
            mdt = dt / CCD_SPLIT
            for _ in range(CCD_SPLIT):
                for o in objs:
                    if not o["asleep"]:
                        _integrate_body(o, mdt)
                _contact_step(objs, width, height, mdt)
        else:
            _contact_step(objs, width, height, dt)


def _support_points(o, nx, ny, wall_c):
    """Contact points of o against wall (n, c): the deepest vertex plus a second
    one at nearly the same depth (a flat face resting on the wall), or the single
    lowest point for round bodies."""
    if o["verts"] is None:
        return [(o["x"] - nx * o["size"], o["y"] - ny * o["size"],
                 o["x"] * nx + o["y"] * ny + wall_c - o["size"])]
    v = _world_verts(o)
    d = v[:, 0] * nx + v[:, 1] * ny + wall_c
    order = np.argsort(d)
    pts = [(float(v[order[0], 0]), float(v[order[0], 1]), float(d[order[0]]))]
    if len(order) > 1 and d[order[1]] < d[order[0]] + 0.1:
        pts.append((float(v[order[1], 0]), float(v[order[1], 1]), float(d[order[1]])))
    return pts


def _settle(objs, width, height, iters=64):
    """Gauss-Seidel positional relaxation of residual interpenetration. The
    sweep direction alternates each pass -- a fixed order systematically favors
    late pairs in contact chains and stalls convergence in piles.

    Candidate prefilter (result-identical): pairs whose bounding circles are
    more than MARGIN apart cannot come into contact while every body has moved
    less than MARGIN/2 - REBUILD since the snapshot, so the SAT test would
    return None for them anyway. The list is rebuilt if any body drifts past
    REBUILD, keeping the skip provably conservative."""
    live = [(i, j) for i in range(len(objs)) for j in range(i + 1, len(objs))
            if not (objs[i]["asleep"] and objs[j]["asleep"])]
    if not live:
        return
    MARGIN = 4.0
    REBUILD = 1.5

    def build():
        snap = [(o["x"], o["y"]) for o in objs]
        cand = []
        for i, j in live:
            dx = snap[i][0] - snap[j][0]
            dy = snap[i][1] - snap[j][1]
            r = objs[i]["size"] + objs[j]["size"] + MARGIN
            if dx * dx + dy * dy < r * r:
                cand.append((i, j))
        return cand, snap

    pairs, snap = build()
    # pose-keyed contact memo: a pair whose bodies have not moved since its
    # last test returns the identical cached result instead of re-running SAT.
    # Converged parts of a pile stop costing anything while the rest relaxes.
    memo = {}
    for it in range(iters):
        moved = False
        for i, j in (pairs if it % 2 == 0 else pairs[::-1]):
            a, b = objs[i], objs[j]
            key = (a["x"], a["y"], a["theta"], b["x"], b["y"], b["theta"])
            hit = memo.get((i, j))
            if hit is not None and hit[0] == key:
                contact = hit[1]
            else:
                contact = _pair_contact(a, b)
                memo[(i, j)] = (key, contact)
            if contact is None:
                continue
            nx, ny, pen, _ = contact
            if pen < 0.001:
                continue
            _separate(a, b, nx, ny, pen, width, height)
            moved = True
        if not moved:
            break
        for k, o in enumerate(objs):
            dx = o["x"] - snap[k][0]
            dy = o["y"] - snap[k][1]
            if dx * dx + dy * dy > REBUILD * REBUILD:
                pairs, snap = build()
                break


def _pt(x, y, ss, height):
    return (round((x * ss - 0.5) * _SUBPIX), round(((height - y) * ss - 0.5) * _SUBPIX))


def _draw(objs, width, height, ss, markers="off", outline=False, border=0.0,
          dot_scale=1.0, bg=255):
    """markers="on": every rotatable body carries an asymmetric marker constellation so its
    orientation is unique over the full 360 deg (a symmetric texture is unreadable modulo the
    body's symmetry -- and what the reader can't see, the model can't know either). Ball: two
    unequal dots at unequal radii, 120 deg apart (a diameter line repeats every 180 deg).
    Polygons: dots toward vertex 0 and vertex 1 (breaks the 60-180 deg near-symmetries).
    Halfdisc: none needed, its outline is already asymmetric. Default (outline=False,
    border=0) = the legacy flat render, byte-identical for old datasets.
    outline=True: every body gets a dark edge stroke (easy-mode identity cue).
    border > 0: a gray frame of that many world-px around the image (pair with the
    matching physics-wall inset in the pymunk backend).
    markers="none": no orientation markings at all -- no dots AND no legacy ball
    line (for frozen-rotation datasets where orientation never changes).
    bg: canvas gray level, default 255 = the legacy white (0 = space-black for
    env_solar)."""
    img = np.ones((height * ss, width * ss, 3), dtype=np.uint8) * bg
    ow = max(1, round(1.25 * ss))
    for o in objs:
        color = o["mat"]["color"]
        dark = tuple(int(c * 0.45) for c in color)
        draw_dots = markers in ("on", "dots")
        if o["verts"] is None:
            center = _pt(o["x"], o["y"], ss, height)
            cv2.circle(img, center, round(o["size"] * ss * _SUBPIX), color, -1,
                       lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
            if outline:
                cv2.circle(img, center, round(o["size"] * ss * _SUBPIX), dark, ow,
                           lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
            if o["kind"] == "ball" and markers != "none":
                if draw_dots:
                    # two unequal dots at unequal radii, 120 deg apart: an asymmetric
                    # constellation readable over the full 360 deg, without the visual
                    # weight of a line
                    for ang, frac, mr in ((o["theta"], 0.55, 1.4),
                                          (o["theta"] + 2.0 * np.pi / 3.0, 0.38, 1.0)):
                        md = frac * o["size"]
                        cv2.circle(img, _pt(o["x"] + np.cos(ang) * md, o["y"] + np.sin(ang) * md,
                                            ss, height),
                                   round(mr * dot_scale * ss * _SUBPIX), dark, -1,
                                   lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
                else:
                    dx, dy = np.cos(o["theta"]), np.sin(o["theta"])
                    r = o["size"] * 0.9
                    cv2.line(img, _pt(o["x"] - dx * r, o["y"] - dy * r, ss, height),
                             _pt(o["x"] + dx * r, o["y"] + dy * r, ss, height),
                             dark, max(1, round(1.5 * ss)), lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
        else:
            verts = _world_verts(o)
            pts = np.stack([(verts[:, 0] * ss - 0.5) * _SUBPIX,
                            ((height - verts[:, 1]) * ss - 0.5) * _SUBPIX], axis=1)
            ipts = np.round(pts).astype(np.int32)
            cv2.fillPoly(img, [ipts], color, lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
            if outline:
                cv2.polylines(img, [ipts], True, dark, ow,
                              lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
            if draw_dots and o["kind"] != "halfdisc":
                for vi, frac, mr in ((0, 0.55, 1.3), (1, 0.35, 0.9)):
                    vx, vy = verts[vi % len(verts)]
                    mx = o["x"] + frac * (vx - o["x"])
                    my = o["y"] + frac * (vy - o["y"])
                    cv2.circle(img, _pt(mx, my, ss, height),
                               round(mr * dot_scale * ss * _SUBPIX), dark, -1,
                               lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
    if border > 0.0:
        t = max(1, round(border * ss))
        bc = (90, 90, 90)
        img[:t, :] = bc
        img[-t:, :] = bc
        img[:, :t] = bc
        img[:, -t:] = bc
    if ss > 1:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def generate_shapes_data(data_dir="data/shapes", n_trajectories=5000, max_frames=100,
                         width=WIDTH, height=HEIGHT, n_objects_min=1, n_objects_max=5,
                         speed_min=3.0, speed_max=8.0, kind_weights=None, spin_max=0.25,
                         vertex_jitter=0.4, n_substeps=16, supersample=1, start_idx=0,
                         y_frac_min=0.3, y_frac_max=1.0, size_min=5, size_max=8,
                         markers="off", progress_cb=None):
    ss = max(1, int(supersample))
    kw = dict(DEFAULT_KIND_WEIGHTS if kind_weights is None else kind_weights)
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    dt = 1.0 / n_substeps

    for i in tqdm(range(n_trajectories), desc="Generating rotating-shapes envs"):
        traj_dir = os.path.join(data_dir, f'traj-{start_idx + i}')
        os.makedirs(traj_dir, exist_ok=True)

        n = np.random.randint(n_objects_min, n_objects_max + 1)
        objs = _spawn_shapes(n, width, height, speed_min, speed_max, kw,
                             spin_max, y_frac_min=y_frac_min, y_frac_max=y_frac_max,
                             vertex_jitter=vertex_jitter,
                             size_min=size_min, size_max=size_max)

        positions, velocities, angles, events = [], [], [], []

        for frame in range(max_frames):
            cv2.imwrite(os.path.join(traj_dir, f'frame_{frame:03d}.png'),
                        _draw(objs, width, height, ss, markers=markers))
            positions.append([(o["x"], o["y"]) for o in objs])
            velocities.append([(o["vx"], o["vy"]) for o in objs])
            angles.append([(o["theta"], o["omega"]) for o in objs])

            for o in objs:
                o["hit_wall"] = False
                o["hit_pair"] = False
            _sim_substeps(objs, width, height, n_substeps)
            # events over the step that PRODUCED the next frame: [hit_wall, hit_pair]
            events.append([(o["hit_wall"], o["hit_pair"]) for o in objs])

        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))
        np.save(os.path.join(traj_dir, "angles.npy"), np.array(angles))
        np.save(os.path.join(traj_dir, "events.npy"), np.array(events, dtype=np.uint8))
        with open(os.path.join(traj_dir, "objects.json"), "w") as f:
            json.dump([{"kind": o["kind"], "size": o["size"], "material": o["mat_name"],
                        "verts": None if o["verts"] is None else np.round(o["verts"], 3).tolist()}
                       for o in objs], f)

        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


def generate_balls2_data(data_dir="data/balls2", n_trajectories=5000, max_frames=100,
                         width=WIDTH, height=HEIGHT, n_balls_min=1, n_balls_max=6,
                         speed_min=3.0, speed_max=8.0, spin_max=0.25,
                         n_substeps=16, supersample=1, start_idx=0,
                         y_frac_min=0.3, y_frac_max=1.0, radius_min=5, radius_max=8,
                         markers="on", progress_cb=None):
    """Bouncing-balls scenes on the SHAPES engine: rotation, ball-ball Coulomb friction
    (rolling/spin transfer) and stability-aware sleeping -- the successor to
    env_bouncing's frictionless point-mass balls. Same spawn law as
    generate_bouncing_data (counts, speeds, radii, floor bias); markers default ON so
    every ball's orientation is readable from pixels (two-dot constellation).
    For new datasets use env_pymunk.generate_balls2_pymunk."""
    return generate_shapes_data(
        data_dir=data_dir, n_trajectories=n_trajectories, max_frames=max_frames,
        width=width, height=height, n_objects_min=n_balls_min, n_objects_max=n_balls_max,
        speed_min=speed_min, speed_max=speed_max, kind_weights={"ball": 1.0},
        spin_max=spin_max, n_substeps=n_substeps, supersample=supersample,
        start_idx=start_idx, y_frac_min=y_frac_min, y_frac_max=y_frac_max,
        size_min=radius_min, size_max=radius_max, markers=markers,
        progress_cb=progress_cb)


if __name__ == "__main__":
    from environments.parallel import generate_shapes_parallel
    generate_shapes_parallel(data_dir="data/shapes", n_trajectories=5000)
