import os
import json
import cv2
import numpy as np
from tqdm import tqdm

from environments.env_bouncing import MATERIALS, WIDTH, HEIGHT, GRAVITY, AIR_DRAG_COEFF
from environments.env_shapes import (_poly_props, _local_verts, _jitter_verts,
                                     _contact_step, _find_supporters, _draw, _rot_drag,
                                     _sim_substeps, SKIN)

TRIAL_FRAMES = 12          # awake trial proving a tower is a real equilibrium
TRIAL_DRIFT = 2.0          # px; stable towers stay under ~1.1 in the trial,
TRIAL_TILT = 0.15          # rad; unstable ones reach 3px+ / 0.9rad+

# Card-house towers: storeys of two vertical plank columns carrying a horizontal
# beam, randomized sizes and materials. The tower stands perfectly still while a
# projectile (already in flight from frame 0, so every context window is
# predictable) crosses the screen and hits it. The whole sim runs honest physics
# from frame 0 -- the static phase is real stability, not a freeze.
PROJECTILE_KINDS = {"ball": 0.40, "square": 0.15, "triangle": 0.15, "halfdisc": 0.15,
                    "hexagon": 0.15}


def _make_plank(half_len, half_thick, x, y, theta, mat_name, vertex_jitter=0.0):
    mat = MATERIALS[mat_name]
    verts = np.array([(-half_len, -half_thick), (half_len, -half_thick),
                      (half_len, half_thick), (-half_len, half_thick)])
    # planks in a card house need near-flat contact faces on BOTH axes (columns
    # stand on their short ends) -- heavy per-vertex jitter makes towers that
    # physically cannot stand, so tower planks get only a light raggedness
    j = min(vertex_jitter, 0.1)
    verts = _jitter_verts(verts, j, jitter_y=j)
    area, mass, inertia = _poly_props(verts, mat["density"])
    return {"kind": "plank", "mat": mat, "mat_name": mat_name,
            "size": float(np.hypot(verts[:, 0], verts[:, 1]).max()),
            "half_len": float(half_len), "half_thick": float(half_thick),
            "verts": verts, "mass": mass, "inertia": inertia,
            "r_eff": float(np.sqrt(area / np.pi)),
            "x": float(x), "y": float(y), "theta": float(theta),
            "vx": 0.0, "vy": 0.0, "omega": 0.0,
            "hit_wall": False, "hit_pair": False, "role": "tower",
            "asleep": True, "still": 99}


TOPPER_KINDS = ["ball", "square", "hexagon", "halfdisc"]


def _make_topper(cx, beam_top, vertex_jitter):
    """A big shape resting on the tower's top beam: extra weight pressing down
    through the stack, and something heavy to come crashing down."""
    kind = str(np.random.choice(TOPPER_KINDS))
    mat_name = str(np.random.choice(list(MATERIALS.keys())))
    mat = MATERIALS[mat_name]
    size = float(np.random.uniform(5.5, 8.0))
    obj = {"kind": kind, "mat": mat, "mat_name": mat_name, "size": size,
           "theta": 0.0, "omega": 0.0, "vx": 0.0, "vy": 0.0,
           "hit_wall": False, "hit_pair": False, "role": "topper",
           "asleep": True, "still": 99}
    if kind == "ball":
        obj["verts"] = None
        obj["mass"] = (np.pi * size ** 2) * mat["density"]
        obj["inertia"] = 0.5 * obj["mass"] * size ** 2
        obj["r_eff"] = size
        obj["theta"] = float(np.random.uniform(0, 2 * np.pi))
        low = size
    else:
        verts = _jitter_verts(_local_verts(kind, size), vertex_jitter)
        area, obj["mass"], obj["inertia"] = _poly_props(verts, mat["density"])
        obj["verts"] = verts
        obj["size"] = float(np.hypot(verts[:, 0], verts[:, 1]).max())
        obj["r_eff"] = float(np.sqrt(area / np.pi))
        low = -float(verts[:, 1].min())
    obj["x"] = float(cx + np.random.uniform(-2.0, 2.0))
    obj["y"] = float(beam_top + low)
    return obj


def _build_tower(width, n_storeys, vertex_jitter=0.0):
    """Stack storeys of [column, column, beam] plus a heavy topper shape; every
    body exactly touching its support. Returns the body list, the tower's top y,
    center x and half-width."""
    objs = []
    cx = width / 2 + np.random.uniform(-6, 6)
    y = 0.0
    span = np.random.uniform(13, 19)
    half_w = 0.0
    budget = 42.0                                     # storey height budget; the topper needs headroom
    for s in range(n_storeys):
        beam_ht = np.random.uniform(1.1, 1.6)
        col_max = (budget / n_storeys - 2 * beam_ht) / 2
        col_hl = float(np.clip(np.random.uniform(0.75, 1.0) * col_max, 4.5, 9.0))
        col_ht = np.random.uniform(1.2, 1.7)          # column half-thickness
        beam_hl = span / 2 + col_ht + np.random.uniform(1.0, 2.5)
        half_w = max(half_w, beam_hl)
        mats = list(MATERIALS.keys())
        for sx in (-1, 1):
            objs.append(_make_plank(col_hl, col_ht, cx + sx * span / 2, y + col_hl,
                                    np.pi / 2, np.random.choice(mats), vertex_jitter))
        # each pair interface is built with the contact skin already open, so the
        # settle pass does not have to push the tower apart (which would fail the
        # stability trial's drift check)
        objs.append(_make_plank(beam_hl, beam_ht, cx, y + 2 * col_hl + SKIN + beam_ht,
                                0.0, np.random.choice(mats), vertex_jitter))
        y += 2 * col_hl + 2 * beam_ht + 2 * SKIN
        span *= np.random.uniform(0.8, 0.95)          # upper storeys narrower
    topper = _make_topper(cx, y, vertex_jitter)
    objs.append(topper)
    y += 2 * topper["size"]
    return objs, y, cx, half_w


def _verified_tower(width, height, n_storeys, n_substeps, vertex_jitter=0.0):
    """Build towers until one proves itself: simulated AWAKE for TRIAL_FRAMES,
    it must hold position and angle. Only then is it put to sleep (at its exact
    constructed coordinates) -- sleep represents a verified equilibrium, it is
    never allowed to freeze an unstable structure in place."""
    dt = 1.0 / n_substeps
    while True:
        objs, top, cx, half_w = _build_tower(width, n_storeys, vertex_jitter)
        trial = [dict(o) for o in objs]
        for o in trial:
            o["asleep"] = False
            o["still"] = 0
        for _ in range(TRIAL_FRAMES * n_substeps):
            for o in trial:
                if o["asleep"]:
                    continue
                o["vy"] += GRAVITY * dt
                o["x"] += o["vx"] * dt
                o["y"] += o["vy"] * dt
                o["theta"] = (o["theta"] + o["omega"] * dt) % (2 * np.pi)
            _contact_step(trial, width, height, dt)
        ok = True
        for o, t in zip(objs, trial):
            dth = abs(t["theta"] - o["theta"]) % (2 * np.pi)
            if np.hypot(t["x"] - o["x"], t["y"] - o["y"]) > TRIAL_DRIFT \
                    or min(dth, 2 * np.pi - dth) > TRIAL_TILT:
                ok = False
                break
        if ok:
            for o in objs:
                o["sup"] = [s for s, _ in _find_supporters(o, objs, height)]
            return objs, top, cx, half_w


def _make_projectile(kind, tower_cx, tower_top, tower_half_w, plank_ys, width,
                     arrive_min, arrive_max, vertex_jitter=0.0):
    """Projectile in flight from frame 0, aimed to reach the tower face at a
    sampled arrival frame (> context length, so the static phase is long enough
    and every context window sees it coming)."""
    mats = ["Steel", "Rubber", "Superball", "Sponge"]
    mat_name = str(np.random.choice(mats, p=[0.45, 0.30, 0.15, 0.10]))
    mat = MATERIALS[mat_name]
    size = float(np.random.uniform(5.0, 7.0))
    side = np.random.choice((-1, 1))
    x0 = (size + 1.0) if side > 0 else (width - size - 1.0)
    # aim at an actual body (uniform aim mostly sails through the storey
    # window between the columns or over the top); bias toward the topper --
    # column centers sit low, so plain uniform choice clusters shots at the base
    if np.random.rand() < 0.4:
        y_target = float(max(plank_ys)) + float(np.random.uniform(-1.5, 1.5))
    else:
        y_target = float(np.random.choice(plank_ys)) + float(np.random.uniform(-1.5, 1.5))
    # half the shots launch flat from near the target height, half plunge in
    # from high up
    if np.random.rand() < 0.5:
        y0 = float(np.clip(y_target + np.random.uniform(-6.0, 6.0), size + 1.0, 58.0))
    else:
        y0 = float(np.random.uniform(max(y_target, 34.0), 58.0))
    t_arrive = np.random.uniform(arrive_min, arrive_max)
    face_x = tower_cx - side * (tower_half_w + size)
    vx = (face_x - x0) / t_arrive
    # loft against gravity so the shot arrives near its aim height (drag makes
    # this approximate, which adds natural aim variety)
    vy = (y_target - y0) / t_arrive + 0.5 * abs(GRAVITY) * t_arrive
    obj = {"kind": kind, "mat": mat, "mat_name": mat_name, "size": size,
           "theta": float(np.random.uniform(0, 2 * np.pi)),
           "omega": float(np.random.uniform(-0.3, 0.3)),
           "x": float(x0), "y": y0,
           "vx": float(vx), "vy": float(vy),
           "hit_wall": False, "hit_pair": False, "role": "projectile",
           "asleep": False, "still": 0}
    if kind == "ball":
        obj["verts"] = None
        obj["mass"] = (np.pi * size ** 2) * mat["density"]
        obj["inertia"] = 0.5 * obj["mass"] * size ** 2
        obj["r_eff"] = size
    else:
        verts = _jitter_verts(_local_verts(kind, size), vertex_jitter)
        area, obj["mass"], obj["inertia"] = _poly_props(verts, mat["density"])
        obj["verts"] = verts
        obj["size"] = float(np.hypot(verts[:, 0], verts[:, 1]).max())
        obj["r_eff"] = float(np.sqrt(area / np.pi))
    return obj


def generate_tower_data(data_dir="data/tower", n_trajectories=5000, max_frames=100,
                        width=WIDTH, height=HEIGHT, n_storeys_min=1, n_storeys_max=3,
                        arrive_min=6.0, arrive_max=8.0, projectile_kinds=None, vertex_jitter=0.4,
                        n_substeps=16, supersample=1, start_idx=0, markers="off", progress_cb=None):
    ss = max(1, int(supersample))
    pk = dict(PROJECTILE_KINDS if projectile_kinds is None else projectile_kinds)
    kinds = list(pk.keys())
    probs = np.array([pk[k] for k in kinds], dtype=np.float64)
    probs = probs / probs.sum()
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    dt = 1.0 / n_substeps

    for i in tqdm(range(n_trajectories), desc="Generating plank-tower envs"):
        traj_dir = os.path.join(data_dir, f'traj-{start_idx + i}')
        os.makedirs(traj_dir, exist_ok=True)

        n_storeys = np.random.randint(n_storeys_min, n_storeys_max + 1)
        objs, top, cx, half_w = _verified_tower(width, height, n_storeys, n_substeps, vertex_jitter)
        kind = str(np.random.choice(kinds, p=probs))
        plank_ys = [o["y"] for o in objs]
        objs.append(_make_projectile(kind, cx, top, half_w, plank_ys, width,
                                     arrive_min, arrive_max, vertex_jitter))

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
            events.append([(o["hit_wall"], o["hit_pair"]) for o in objs])

        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))
        np.save(os.path.join(traj_dir, "angles.npy"), np.array(angles))
        np.save(os.path.join(traj_dir, "events.npy"), np.array(events, dtype=np.uint8))
        with open(os.path.join(traj_dir, "objects.json"), "w") as f:
            json.dump([{"kind": o["kind"], "size": o["size"], "material": o["mat_name"],
                        "role": o["role"],
                        "verts": None if o["verts"] is None else np.round(o["verts"], 3).tolist()}
                       for o in objs], f)

        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


if __name__ == "__main__":
    from environments.parallel import generate_parallel
    generate_parallel("environments.env_tower", "generate_tower_data",
                      data_dir="data/tower", n_trajectories=5000)
