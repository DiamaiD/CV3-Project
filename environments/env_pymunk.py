"""Chipmunk2D (pymunk) backend for the shapes/tower worlds.

Same world as env_shapes: 64x64 px, y-up, gravity -0.5 px/frame^2, quadratic air drag
on velocity AND rotation (same AIR_DRAG_COEFF), our exact masses and inertias, our
materials. Bodies, spawn laws, renderer and the output format (frame PNGs +
positions/velocities/angles npys + objects.json) are shared with env_shapes, so every
audit / gif / extraction tool runs unchanged on both backends.

THE PRODUCTION BACKEND for shapes / tower / balls2. Chipmunk's contact conventions
are the defining physics; the old engine's rules are kept only to reproduce
pre-migration datasets:
  - restitution of a contact = product of the two shapes' elasticity (old engine: min).
  - friction of a contact = product of the two shapes' friction. Per-material friction
    is sqrt(1 - retention), so a same-material pair reproduces the old Coulomb mu
    exactly; cross-material pairs get the geometric mean (old: arithmetic mean).
  - walls have elasticity = friction = 1.0, so a wall contact uses the body's OWN
    Chipmunk constants: restitution e_i (same as old), friction sqrt(1 - retention)
    (old used 1 - retention -- walls are somewhat grippier now, one consistent
    product rule everywhere instead of a special wall case).
  - contact resolution: warm-started sequential impulses with Baumgarte bias + slop.
"""
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import pymunk

from src.frameio import save_frames
from environments.env_bouncing import MATERIALS, WIDTH, HEIGHT, GRAVITY, AIR_DRAG_COEFF
from environments.env_shapes import (_draw, _make_shape, _spawn_shapes, _poly_props,
                                     DEFAULT_KIND_WEIGHTS)
from environments.env_tower import (_build_tower, _build_tower_uniform, _make_projectile,
                                    PROJECTILE_KINDS)

# Production solver settings: 32 substeps halve the per-step travel (fast
# projectiles move <=0.25px/substep -> no tunneling class of error), 30 iterations
# converge stacks, slop 0.05 is the resting overlap.
N_SUBSTEPS = 32
ITERATIONS = 30
SLOP = 0.05


def _vel_func(r_eff):
    C = AIR_DRAG_COEFF

    def f(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        vx, vy = body.velocity
        m = body.mass
        body.velocity = (vx - (C * vx * abs(vx) * r_eff / m) * dt,
                         vy - (C * vy * abs(vy) * r_eff / m) * dt)
        w = body.angular_velocity
        body.angular_velocity = w - (C * w * abs(w) * r_eff ** 4 / body.moment) * dt
    return f


def _make_space(width, height, iterations=ITERATIONS, slop=SLOP, bias=None, inset=0.0):
    """inset > 0 pulls all four walls that many world-px into the image: pair it
    with _draw(border=inset) so bodies bounce off the visible frame's inner edge."""
    sp = pymunk.Space()
    sp.gravity = (0.0, GRAVITY)
    sp.iterations = iterations
    sp.collision_slop = slop
    if bias is not None:
        sp.collision_bias = bias
    sp.sleep_time_threshold = 0.6      # frames of quiet before sleeping
    sp.idle_speed_threshold = 0.05
    t = float(inset)
    lo_x, lo_y, hi_x, hi_y = t, t, width - t, height - t
    walls = [((lo_x, lo_y), (hi_x, lo_y)), ((lo_x, hi_y), (hi_x, hi_y)),
             ((lo_x, lo_y), (lo_x, hi_y)), ((hi_x, lo_y), (hi_x, hi_y))]
    for a, b in walls:
        seg = pymunk.Segment(sp.static_body, a, b, 0.0)
        seg.elasticity = 1.0           # product-combine -> pair value = the body's own
        seg.friction = 1.0
        sp.add(seg)
    return sp


def _add_obj(sp, o):
    """Mirror one of our object dicts into the space; returns the pymunk body."""
    mat = o["mat"]
    body = pymunk.Body(o["mass"], o["inertia"])
    body.position = (o["x"], o["y"])
    body.angle = o["theta"]
    body.velocity = (o["vx"], o["vy"])
    body.angular_velocity = o["omega"]
    body.velocity_func = _vel_func(o["r_eff"])
    if o.get("parts") is not None:
        # concave body (plus): its convex parts attach as separate fixtures of
        # the ONE body -- o["verts"] holds the concave render outline, which
        # pymunk.Poly must never see
        shapes = [pymunk.Poly(body, [tuple(v) for v in part]) for part in o["parts"]]
    elif o["verts"] is None:
        shapes = [pymunk.Circle(body, o["size"])]
    else:
        shapes = [pymunk.Poly(body, [tuple(v) for v in o["verts"]])]
    for shape in shapes:
        shape.elasticity = float(mat["restitution"])
        shape.friction = float(np.sqrt(max(1.0 - mat["friction"], 0.0)))
    sp.add(body, *shapes)
    return body


def _record_and_render(objs, bodies, traj_dir, max_frames, sp, width, height, ss, markers,
                       n_substeps=N_SUBSTEPS, outline=False, border=0.0, dot_scale=1.0):
    positions, velocities, angles, frames = [], [], [], []
    dt = 1.0 / n_substeps
    for frame in range(max_frames):
        for o, b in zip(objs, bodies):
            o["x"], o["y"] = b.position
            o["theta"] = float(b.angle) % (2 * np.pi)
        # _draw returns BGR (for cv2); store RGB so packed == the old PIL(png) read
        frames.append(cv2.cvtColor(_draw(objs, width, height, ss, markers=markers,
                                         outline=outline, border=border,
                                         dot_scale=dot_scale),
                                   cv2.COLOR_BGR2RGB))
        positions.append([(float(b.position[0]), float(b.position[1])) for b in bodies])
        velocities.append([(float(b.velocity[0]), float(b.velocity[1])) for b in bodies])
        angles.append([(float(b.angle) % (2 * np.pi), float(b.angular_velocity)) for b in bodies])
        for _ in range(n_substeps):
            sp.step(dt)
    save_frames(traj_dir, np.stack(frames))
    np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
    np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))
    np.save(os.path.join(traj_dir, "angles.npy"), np.array(angles))
    with open(os.path.join(traj_dir, "objects.json"), "w") as f:
        json.dump([{"kind": o["kind"], "size": o["size"], "material": o["mat_name"],
                    "verts": None if o["verts"] is None else np.round(o["verts"], 3).tolist()}
                   for o in objs], f)


def generate_shapes_pymunk(data_dir="data/shapes_pm", n_trajectories=100, max_frames=100,
                           width=WIDTH, height=HEIGHT, n_objects_min=1, n_objects_max=5,
                           speed_min=3.0, speed_max=8.0, kind_weights=None, spin_max=0.25,
                           vertex_jitter=0.4, supersample=4, start_idx=0,
                           y_frac_min=0.3, y_frac_max=1.0, size_min=5, size_max=8,
                           markers="on", n_substeps=N_SUBSTEPS, iterations=ITERATIONS, slop=SLOP,
                           bias=None, outline=False, border=0.0, size_ranges=None,
                           dot_scale=1.0, frozen_rotation=False, progress_cb=None):
    ss = max(1, int(supersample))
    kw = dict(DEFAULT_KIND_WEIGHTS if kind_weights is None else kind_weights)
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    for i in tqdm(range(n_trajectories), desc="Generating pymunk shapes"):
        traj_dir = os.path.join(data_dir, f'traj-{start_idx + i}')
        os.makedirs(traj_dir, exist_ok=True)
        n = np.random.randint(n_objects_min, n_objects_max + 1)
        # with a border, spawn in the inner (wall-inset) box and shift into place
        iw, ih = width - 2 * border, height - 2 * border
        objs = _spawn_shapes(n, iw, ih, speed_min, speed_max, kw, spin_max,
                             y_frac_min=y_frac_min, y_frac_max=y_frac_max,
                             vertex_jitter=vertex_jitter, size_min=size_min, size_max=size_max,
                             size_ranges=size_ranges)
        for o in objs:
            o["x"] += border
            o["y"] += border
        sp = _make_space(width, height, iterations, slop, bias, inset=border)
        bodies = [_add_obj(sp, o) for o in objs]
        if frozen_rotation:
            # random INITIAL angle stays forever: infinite rotational inertia
            # means contacts and friction can never spin the body
            for b in bodies:
                b.moment = float("inf")
                b.angular_velocity = 0.0
        _record_and_render(objs, bodies, traj_dir, max_frames, sp, width, height, ss, markers,
                           n_substeps, outline=outline, border=border, dot_scale=dot_scale)
        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


def generate_tower_pymunk(data_dir="data/tower_pm", n_trajectories=100, max_frames=100,
                          width=WIDTH, height=HEIGHT, n_storeys_min=1, n_storeys_max=3,
                          arrive_min=6.0, arrive_max=8.0, projectile_kinds=None,
                          vertex_jitter=0.4, supersample=4, start_idx=0, markers="on",
                          n_substeps=N_SUBSTEPS, iterations=ITERATIONS, slop=SLOP,
                          bias=None, uniform_planks=False, topper_kinds=None,
                          outline=False, progress_cb=None):
    """Card-house tower + projectile on the pymunk backend. The tower is built with our
    exact geometry, dropped into the space, given a short settle warmup (Chipmunk's own
    solver + sleeping hold it up -- no hand-verification loop needed), then the
    projectile is launched. uniform_planks=True switches to the easy-mode builder:
    one plank size everywhere, fixed span, distinct-silhouette topper roster."""
    ss = max(1, int(supersample))
    pk = dict(PROJECTILE_KINDS if projectile_kinds is None else projectile_kinds)
    kinds = list(pk.keys())
    probs = np.array([pk[k] for k in kinds], dtype=np.float64)
    probs = probs / probs.sum()
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    dt = 1.0 / n_substeps
    for i in tqdm(range(n_trajectories), desc="Generating pymunk towers"):
        traj_dir = os.path.join(data_dir, f'traj-{start_idx + i}')
        os.makedirs(traj_dir, exist_ok=True)
        n_storeys = np.random.randint(n_storeys_min, n_storeys_max + 1)
        if uniform_planks:
            objs, top, cx, half_w = _build_tower_uniform(width, n_storeys,
                                                         topper_kinds=topper_kinds)
        else:
            objs, top, cx, half_w = _build_tower(width, n_storeys, vertex_jitter)
        sp = _make_space(width, height, iterations, slop, bias)
        bodies = [_add_obj(sp, o) for o in objs]
        for _ in range(12 * n_substeps):          # settle warmup before the projectile
            sp.step(dt)
        plank_ys = [o["y"] for o in objs if o["kind"] == "plank"]
        kind = str(np.random.choice(kinds, p=probs))
        # easy mode: heavy FAST shots, mostly plunging from high up
        # (flat_prob keeps a flat minority).
        # Slow lobs (the arrival-derived speeds, ~2 px/frame) lean on the tower
        # instead of toppling it, so speed is set directly and the launch side
        # maximizes the runway. All 4 materials stay in play (material variety
        # over a few extra surviving towers).
        aim = dict(flat_prob=0.3, y0_jitter=4.0, aim_jitter=1.0,
                   size_min=6.0, size_max=8.0,
                   speed_min=3.5, speed_max=5.0) if uniform_planks else {}
        proj = _make_projectile(kind, cx, top, half_w, plank_ys, width,
                                arrive_min, arrive_max, vertex_jitter, **aim)
        objs.append(proj)
        bodies.append(_add_obj(sp, proj))
        _record_and_render(objs, bodies, traj_dir, max_frames, sp, width, height, ss, markers,
                           n_substeps, outline=outline)
        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


def generate_balls2_pymunk(data_dir="data/balls2_pm", n_trajectories=5000, max_frames=100,
                           width=WIDTH, height=HEIGHT, n_balls_min=1, n_balls_max=6,
                           speed_min=3.0, speed_max=8.0, spin_max=0.25,
                           supersample=4, start_idx=0,
                           y_frac_min=0.3, y_frac_max=1.0, radius_min=5, radius_max=8,
                           markers="on", n_substeps=N_SUBSTEPS, iterations=ITERATIONS,
                           slop=SLOP, bias=None, outline=False, border=0.0, progress_cb=None):
    """Bouncing-balls scenes on the pymunk backend: rotation, ball-ball friction and
    sleeping, same spawn law as generate_bouncing_data (counts, speeds, radii, floor
    bias). Successor to env_shapes.generate_balls2_data."""
    return generate_shapes_pymunk(
        data_dir=data_dir, n_trajectories=n_trajectories, max_frames=max_frames,
        width=width, height=height, n_objects_min=n_balls_min, n_objects_max=n_balls_max,
        speed_min=speed_min, speed_max=speed_max, kind_weights={"ball": 1.0},
        spin_max=spin_max, supersample=supersample, start_idx=start_idx,
        y_frac_min=y_frac_min, y_frac_max=y_frac_max,
        size_min=radius_min, size_max=radius_max, markers=markers,
        n_substeps=n_substeps, iterations=iterations, slop=slop, bias=bias,
        outline=outline, border=border, progress_cb=progress_cb)
