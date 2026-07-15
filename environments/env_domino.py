import os
import cv2
import numpy as np
from tqdm import tqdm


WIDTH, HEIGHT = 64, 64
GRAVITY = -0.5
SUBPIX_BITS = 4
_SUBPIX = 1 << SUBPIX_BITS

# Cosmetic only -- color has no effect on physics/material. Each domino picks a
# random color from this palette (BGR, matching the balls' color convention).
DOMINO_COLORS = [
    (60, 60, 220),   # red
    (220, 120, 40),  # blue
    (50, 160, 50),   # green
    (30, 170, 230),  # orange
    (200, 60, 180),  # purple
    (40, 190, 190),  # yellow
    (180, 180, 60),  # teal
    (120, 80, 200),  # pink
]
DOMINO_SIZE = (4.0, 22.0)  # (width, height) -- height ~= WIDTH/3, matches canvas scale
DOMINO_WIDTH = DOMINO_SIZE[0]
DOMINO_HEIGHT = DOMINO_SIZE[1]
DOMINO_HALF_WIDTH = DOMINO_WIDTH / 2.0
DOMINO_HALF_HEIGHT = DOMINO_HEIGHT / 2.0

# A domino is modeled as a rigid rod of length DOMINO_HEIGHT pivoting about a
# fixed base point on the floor (it doesn't translate, only rotates -- unlike the
# bouncing balls, dominoes stay planted and topple in place). `angle` is the tilt
# away from vertical (0 = standing upright, +-90deg = lying flat).
#
# An upright domino is an *inverted* pendulum: gravity acting through its center
# of mass produces zero torque at angle=0 (unstable equilibrium) and a torque that
# GROWS the tilt once angle != 0. For a uniform rod of length L pivoted at one end,
# with I = (1/3) m L^2 about the pivot and lever arm (L/2) sin(theta):
#   alpha = torque / I = [m g (L/2) sin(theta)] / [(1/3) m L^2] = (3g)/(2L) * sin(theta)
# Mass cancels out, so (like the balls' gravity term) this needs no per-object mass.
GRAVITY_TORQUE_COEFF = (3.0 * abs(GRAVITY)) / (2.0 * DOMINO_HEIGHT)

# Quadratic angular drag -- the rotational analogue of AIR_DRAG_COEFF in the
# bouncing env, included for the same reason: keeps motion from feeling weightless
# and prevents runaway spin.
ANGULAR_DRAG_COEFF = 0.05

# A domino is "down" once it reaches horizontal; clamp there and stop, mirroring
# REST_VELOCITY in the bouncing env (below-threshold motion is treated as settled).
DOMINO_FALLEN_ANGLE = np.deg2rad(90.0)

# When a toppling domino's tip reaches a still-upright neighbor, some of its
# angular velocity is transferred as a kick, plus a small nudge off angle=0 so the
# neighbor's own gravity torque (which is exactly zero at angle=0) can take over.
TIP_TRANSFER_EFFICIENCY = 0.6
TIP_IMPACT_DAMPING = 0.5
TIP_NUDGE_ANGLE = np.deg2rad(3.0)


def _make_domino(x):
    """Create one domino, standing upright at floor position x."""
    color = DOMINO_COLORS[np.random.randint(0, len(DOMINO_COLORS))]
    return {
        "x": float(x),
        "angle": 0.0,
        "ang_vel": 0.0,
        "color": tuple(int(c) for c in color),
    }


def _spawn_dominoes(n_dominoes, width, tilt_deg=50.0):
    """Place n_dominoes evenly along the floor and tilt one random domino to
    kick off the chain (an upright domino at angle=0 never falls on its own,
    since gravity torque there is exactly zero).

    Dominoes can be packed closer together than DOMINO_HEIGHT, so tilting one
    the full tilt_deg could swing its body past a neighbor before any physics
    has run. To avoid spawning already-clipped into that neighbor, the tilt
    is capped with the same exact polygon-overlap bisection used during the
    simulation (see _max_safe_angle) -- if there's room for the full
    tilt_deg it's used as-is, otherwise it settles for whatever angle just
    grazes the neighbor."""
    dominoes = []
    if n_dominoes <= 0:
        return dominoes

    if n_dominoes == 1:
        x_positions = [width / 2.0]
    else:
        x_positions = np.linspace(DOMINO_HALF_WIDTH + 2, width - DOMINO_HALF_WIDTH - 2, n_dominoes)

    for x in x_positions:
        dominoes.append(_make_domino(x))

    tilt_idx = int(np.random.randint(0, n_dominoes))
    tilt_dir = float(np.random.choice([-1.0, 1.0]))
    target_angle = tilt_dir * np.deg2rad(tilt_deg)
    tilt_x = dominoes[tilt_idx]["x"]

    neighbors = [d for d in dominoes if d is not dominoes[tilt_idx] and np.sign(d["x"] - tilt_x) == tilt_dir]
    if neighbors:
        nearest = min(neighbors, key=lambda d: abs(d["x"] - tilt_x))
        dominoes[tilt_idx]["angle"] = _max_safe_angle(tilt_x, 0.0, target_angle, nearest)
    else:
        dominoes[tilt_idx]["angle"] = target_angle

    return dominoes


def _domino_polygon(x, angle):
    """World-space corners of a domino's rectangle, base pinned at (x, 0),
    rotated by angle. This is the single source of truth for a domino's
    footprint -- both the collision code and the renderer build off it, so
    what gets drawn can never disagree with what physics considered clear."""
    ux, uy = np.sin(angle), np.cos(angle)          # along the long axis
    perp_x, perp_y = np.cos(angle), -np.sin(angle)  # along the short axis
    base_x, base_y = x, 0.0
    tip_x, tip_y = base_x + DOMINO_HEIGHT * ux, base_y + DOMINO_HEIGHT * uy
    off_x, off_y = DOMINO_HALF_WIDTH * perp_x, DOMINO_HALF_WIDTH * perp_y
    return [
        (base_x - off_x, base_y - off_y),
        (base_x + off_x, base_y + off_y),
        (tip_x + off_x, tip_y + off_y),
        (tip_x - off_x, tip_y - off_y),
    ]


def _polygons_overlap(poly_a, poly_b):
    """Separating Axis Theorem test -- exact for convex polygons, which the
    rotated-rectangle dominoes always are. Checking the full body (not just
    the tip point) is what catches a domino's SIDE edge swinging into a
    neighbor before its tip gets there."""
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            axis_x, axis_y = -(y2 - y1), (x2 - x1)
            norm = np.hypot(axis_x, axis_y)
            if norm == 0:
                continue
            axis_x, axis_y = axis_x / norm, axis_y / norm
            proj_a = [px * axis_x + py * axis_y for px, py in poly_a]
            proj_b = [px * axis_x + py * axis_y for px, py in poly_b]
            if max(proj_a) < min(proj_b) or max(proj_b) < min(proj_a):
                return False  # found a separating axis -- no overlap
    return True


def _domino_corners(domino, height):
    """Fixed-point pixel corners for cv2.fillPoly (same sub-pixel shift trick
    used for the balls' cv2.circle calls, so rotation looks smooth rather
    than staircased), derived from the same polygon the physics uses."""
    pts = []
    for wx, wy in _domino_polygon(domino["x"], domino["angle"]):
        pixel_x = wx
        pixel_y = height - wy  # flip, same convention as the balls' rendering
        pts.append([round(pixel_x * _SUBPIX), round(pixel_y * _SUBPIX)])
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def _domino_com(domino):
    """Center-of-mass position, for logging (analogous to a ball's x, y)."""
    x = domino["x"] + DOMINO_HALF_HEIGHT * np.sin(domino["angle"])
    y = DOMINO_HALF_HEIGHT * np.cos(domino["angle"])
    return (x, y)


def _domino_com_velocity(domino):
    """Tangential velocity of the center of mass as it swings about the base
    pivot, for logging (analogous to a ball's vx, vy)."""
    vx = DOMINO_HALF_HEIGHT * domino["ang_vel"] * np.cos(domino["angle"])
    vy = -DOMINO_HALF_HEIGHT * domino["ang_vel"] * np.sin(domino["angle"])
    return (vx, vy)


def _max_safe_angle(x, safe_angle, unsafe_angle, target, iterations=16):
    """Bisect between a known-clear angle and a penetrating one to find the
    angle at which `x`'s rectangle just grazes target's rectangle -- full
    polygon-vs-polygon, so this is exact regardless of which part of the two
    bodies makes contact first."""
    target_poly = _domino_polygon(target["x"], target["angle"])
    lo, hi = safe_angle, unsafe_angle
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if _polygons_overlap(_domino_polygon(x, mid), target_poly):
            hi = mid
        else:
            lo = mid
    return lo


def _resolve_domino_penetration(dominoes, old_angles, iterations=6):
    """Stop each domino's rotation the instant its body would overlap its
    next neighbor in the lean direction -- so it rests against/on that
    neighbor instead of clipping through it -- and pass along a kick so the
    neighbor (asleep at angle=0, mid-fall, or already down) keeps the chain
    reaction moving.

    A single left-to-right pass isn't enough: domino[1] might check out safe
    against domino[2]'s angle, only for domino[2] to get pulled back a moment
    later in that same pass (because *it* overlaps domino[3]) -- silently
    invalidating domino[1]'s already-accepted result. Iterating a few times,
    always re-bisecting from the pre-substep `old_angles` (the last known
    fully-consistent state), lets corrections like that propagate back down
    the chain until nothing moves anymore.
    """
    for _ in range(iterations):
        settled = True
        for i, a in enumerate(dominoes):
            if a["angle"] == 0:
                continue
            direction = np.sign(a["angle"])

            candidates = [b for b in dominoes if b is not a and np.sign(b["x"] - a["x"]) == direction]
            if not candidates:
                continue
            target = min(candidates, key=lambda d: abs(d["x"] - a["x"]))

            a_poly = _domino_polygon(a["x"], a["angle"])
            target_poly = _domino_polygon(target["x"], target["angle"])
            if not _polygons_overlap(a_poly, target_poly):
                continue  # not touching -- nothing to resolve

            safe_angle = _max_safe_angle(a["x"], old_angles[i], a["angle"], target)
            if safe_angle != a["angle"]:
                a["angle"] = safe_angle
                a["ang_vel"] *= TIP_IMPACT_DAMPING
                settled = False

            if abs(target["angle"]) < TIP_NUDGE_ANGLE:
                target["angle"] = direction * TIP_NUDGE_ANGLE
            target["ang_vel"] += direction * abs(a["ang_vel"]) * TIP_TRANSFER_EFFICIENCY
        if settled:
            break


def generate_domino_data(data_dir="data/domino", n_trajectories=5000, max_frames=100,
                         width=WIDTH, height=HEIGHT, n_dominoes_min=1, n_dominoes_max=6,
                         speed_min=0.0, speed_max=0.0, n_substeps=4, progress_cb=None):
    """Generate the domino dataset. Resolution must be a multiple of 8 to match
    the autoencoder's 3 stride-2 down/up-sampling stages. `progress_cb(done, total)` is
    called periodically (used by the GUI to report progress).

    speed_min/speed_max are accepted for interface parity with the bouncing-ball
    generator but unused here -- dominoes start at rest and are set in motion by
    the initial tilt, not an initial velocity.
    """
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    dt = 1.0 / n_substeps

    for i in tqdm(range(n_trajectories), desc="Generating RGB Physics Envs"):
        traj_dir = os.path.join(data_dir, f"traj-{i}")
        os.makedirs(traj_dir, exist_ok=True)

        n_dominoes = np.random.randint(n_dominoes_min, n_dominoes_max + 1)
        dominoes = _spawn_dominoes(n_dominoes, width)

        positions, velocities, angles = [], [], []

        for frame in range(max_frames):
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            for domino in dominoes:
                points = _domino_corners(domino, height)
                cv2.fillPoly(img, [points], domino["color"], lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
            cv2.imwrite(os.path.join(traj_dir, f"frame_{frame:03d}.png"), img)

            positions.append([_domino_com(d) for d in dominoes])
            velocities.append([_domino_com_velocity(d) for d in dominoes])
            angles.append([d["angle"] for d in dominoes])

            for _ in range(n_substeps):
                old_angles = [domino["angle"] for domino in dominoes]

                for domino in dominoes:
                    angle = domino["angle"]
                    gravity_alpha = GRAVITY_TORQUE_COEFF * np.sin(angle)
                    drag_alpha = -ANGULAR_DRAG_COEFF * domino["ang_vel"] * abs(domino["ang_vel"])
                    domino["ang_vel"] += (gravity_alpha + drag_alpha) * dt
                    domino["angle"] += domino["ang_vel"] * dt

                    # Clamp here, before penetration resolution runs, so a domino
                    # lying flat never briefly reports an over-rotated (past
                    # horizontal) angle -- gravity torque alone doesn't know to
                    # stop at the floor and would otherwise keep pushing it past
                    # +-90deg for one substep, corrupting the geometry that
                    # neighbors are checked against.
                    if domino["angle"] >= DOMINO_FALLEN_ANGLE:
                        domino["angle"], domino["ang_vel"] = DOMINO_FALLEN_ANGLE, 0.0
                    elif domino["angle"] <= -DOMINO_FALLEN_ANGLE:
                        domino["angle"], domino["ang_vel"] = -DOMINO_FALLEN_ANGLE, 0.0

                _resolve_domino_penetration(dominoes, old_angles)

        # Shape: (frames, n_dominoes, 2) / (frames, n_dominoes). n_dominoes can vary per trajectory.
        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))
        np.save(os.path.join(traj_dir, "angles.npy"), np.array(angles))

        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


if __name__ == "__main__":
    generate_domino_data()