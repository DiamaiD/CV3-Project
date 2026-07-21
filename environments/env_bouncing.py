import os
import cv2
import numpy as np
from tqdm import tqdm

MATERIALS = {
    "Superball": {"color": (0, 0, 255), "density": 0.5, "restitution": 0.95, "friction": 0.98},
    "Rubber":    {"color": (255, 0, 0), "density": 1.2, "restitution": 0.80, "friction": 0.90},
    "Steel":     {"color": (100, 100, 100), "density": 5.0, "restitution": 0.40, "friction": 0.80},
    "Sponge":    {"color": (0, 255, 0), "density": 0.2, "restitution": 0.20, "friction": 0.60}
}

WIDTH, HEIGHT = 64, 64
GRAVITY = -0.5
AIR_DRAG_COEFF = 0.02
SUBPIX_BITS = 4
_SUBPIX = 1 << SUBPIX_BITS
REST_VELOCITY = abs(GRAVITY)

# CV3_LEGACY_CONTACTS=1 reproduces the pre-2026-07-15 contact behavior (positional
# correction skipped for resting/separating pairs -> interpenetrating piles).
# Needed to bit-exactly re-simulate datasets/models from before the fix.
LEGACY_CONTACTS = os.environ.get("CV3_LEGACY_CONTACTS", "0") == "1"

# Sleeping: a ball that stays slow AND supported (on the floor or on a sleeping ball)
# for SLEEP_AFTER substeps is frozen (v=0), so the positional correction can no longer
# pump energy into resting piles. A sleeper wakes the instant a moving ball collides with
# it (the impulse is applied as usual and propagates through the pile over substeps).
# Default OFF: floor-only sleeping stops the 6-ball pile energy-pump but traps balls perched
# on a now-immovable floor sleeper (3-ball energy plateaus at ~3e-3 instead of settling to 0),
# so it is not correct enough to be default yet -- needs static-stability sleep (COM over
# support, like env_shapes) that won't freeze metastable wedges. With it off, no ball is ever
# marked asleep, every asleep-guard below is a no-op, and the physics is byte-identical to the
# pre-fix code (what bouncing_36k_v2 and the champion were trained on). Opt in with CV3_BALL_SLEEP=1.
BALL_SLEEP = os.environ.get("CV3_BALL_SLEEP", "0") == "1"
SLEEP_V = 0.4
SLEEP_AFTER = 6
WAKE_V = 0.5      # a sleeper only wakes on an approach faster than this; gentle jostles
                  # (e.g. one pile ball nudging another) are absorbed -- the sleeper acts
                  # as an immovable anchor, so a settled pile is not shaken awake frame by frame.


def _make_ball(width, height, speed_min, speed_max, y_frac_min=0.3, y_frac_max=1.0,
               radius_min=5, radius_max=8):
    mat_name = np.random.choice(list(MATERIALS.keys()))
    mat = MATERIALS[mat_name]
    radius = np.random.randint(radius_min, radius_max + 1)
    speed = np.random.uniform(speed_min, speed_max)
    y_lo = max(radius, int(y_frac_min * height))
    y_hi = max(y_lo + 1, min(height - radius, int(y_frac_max * height)))
    return {
        "mat": mat,
        "radius": radius,
        "mass": (np.pi * radius ** 2) * mat["density"],
        "x": float(np.random.randint(radius, width - radius)),
        "y": float(np.random.randint(y_lo, y_hi)),
        "vx": np.random.randn() * speed,
        "vy": np.random.randn() * speed,
        "asleep": False,
        "still": 0,
    }


def _spawn_balls(n_balls, width, height, speed_min, speed_max, max_tries=100,
                 y_frac_min=0.3, y_frac_max=1.0, radius_min=5, radius_max=8):
    balls = []
    for _ in range(n_balls):
        for _ in range(max_tries):
            cand = _make_ball(width, height, speed_min, speed_max, y_frac_min, y_frac_max,
                              radius_min, radius_max)
            ok = True
            for b in balls:
                min_dist = cand["radius"] + b["radius"]
                if (cand["x"] - b["x"]) ** 2 + (cand["y"] - b["y"]) ** 2 < min_dist ** 2:
                    ok = False
                    break
            if ok:
                balls.append(cand)
                break
    return balls


def _wall_blocked(b, dx, dy, width, height):
    """True if moving b along (dx, dy) is blocked by a wall it already touches."""
    eps, r = 0.01, b["radius"]
    return ((b["x"] - r <= eps and dx < -1e-9) or (b["x"] + r >= width - eps and dx > 1e-9)
            or (b["y"] - r <= eps and dy < -1e-9) or (b["y"] + r >= height - eps and dy > 1e-9))


def _separate_pair(a, b, nx, ny, overlap, width, height):
    """Positional separation with wall-aware shares: a wall-pinned ball would be
    clamped right back, so its partner takes the full displacement instead. A sleeping
    ball is immovable (its partner takes the full share); two sleepers are left alone so
    the correction cannot pump energy into a settled pile."""
    if a["asleep"] and b["asleep"]:
        return
    a_blk = a["asleep"] or _wall_blocked(a, -nx, -ny, width, height)
    b_blk = b["asleep"] or _wall_blocked(b, nx, ny, width, height)
    total_mass = a["mass"] + b["mass"]
    a_share = overlap * (b["mass"] / total_mass)
    b_share = overlap * (a["mass"] / total_mass)
    if a_blk and not b_blk:
        a_share, b_share = 0.0, overlap
    elif b_blk and not a_blk:
        a_share, b_share = overlap, 0.0
    a["x"] -= a_share * nx
    a["y"] -= a_share * ny
    b["x"] += b_share * nx
    b["y"] += b_share * ny


def _wall_project(b, width, height):
    """Position-only wall clamp (no bounce physics) for relaxation passes."""
    r = b["radius"]
    if b["y"] - r < 0:
        b["y"] = r
    if b["y"] + r > height:
        b["y"] = height - r
    if b["x"] - r < 0:
        b["x"] = r
    if b["x"] + r > width:
        b["x"] = width - r


def _positional_pass(balls, width, height):
    moved = False
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            a, b = balls[i], balls[j]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            dist = np.hypot(dx, dy)
            min_dist = a["radius"] + b["radius"]
            if dist == 0 or dist >= min_dist - 1e-9:
                continue
            _separate_pair(a, b, dx / dist, dy / dist, min_dist - dist, width, height)
            moved = True
    if moved:
        for b in balls:
            _wall_project(b, width, height)
    return moved


def _settle_contacts(balls, width=WIDTH, height=HEIGHT, iters=8):
    """Relax residual interpenetration in multi-contact piles (position only --
    bounce physics has already been applied this substep)."""
    if LEGACY_CONTACTS:
        return
    for _ in range(iters):
        if not _positional_pass(balls, width, height):
            break


def _update_sleep(balls, height=HEIGHT):
    """Freeze balls that have stayed slow while resting ON THE FLOOR, so the positional
    correction can no longer pump energy into them. A floor ball's settled height is 0, so
    freezing it traps no potential energy; elevated balls keep simulating until they roll or
    fall to the floor, so nothing is frozen mid-air. No effect under LEGACY_CONTACTS or with
    CV3_BALL_SLEEP=0. (Dense stacks that never reach the floor are out of scope for now.)"""
    if LEGACY_CONTACTS or not BALL_SLEEP:
        return
    for b in balls:
        if b["asleep"]:
            b["vx"] = b["vy"] = 0.0          # micro-impulses must not accumulate in a sleeper
            continue
        on_floor = b["y"] - b["radius"] <= 0.6
        if on_floor and abs(b["vx"]) < SLEEP_V and abs(b["vy"]) < SLEEP_V:
            b["still"] += 1
            if b["still"] >= SLEEP_AFTER:
                b["asleep"] = True
                b["vx"] = b["vy"] = 0.0
        else:
            b["still"] = 0


def _resolve_ball_collisions_legacy(balls):
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            a, b = balls[i], balls[j]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            dist = np.hypot(dx, dy)
            min_dist = a["radius"] + b["radius"]
            if dist == 0 or dist >= min_dist:
                continue
            nx, ny = dx / dist, dy / dist
            rvx, rvy = a["vx"] - b["vx"], a["vy"] - b["vy"]
            vrel = rvx * nx + rvy * ny
            if vrel <= 0:
                continue
            e = min(a["mat"]["restitution"], b["mat"]["restitution"])
            imp = -(1 + e) * vrel / (1.0 / a["mass"] + 1.0 / b["mass"])
            a["vx"] += (imp / a["mass"]) * nx
            a["vy"] += (imp / a["mass"]) * ny
            b["vx"] -= (imp / b["mass"]) * nx
            b["vy"] -= (imp / b["mass"]) * ny
            overlap = min_dist - dist
            total_mass = a["mass"] + b["mass"]
            a_share = overlap * (b["mass"] / total_mass)
            b_share = overlap * (a["mass"] / total_mass)
            a["x"] -= a_share * nx
            a["y"] -= a_share * ny
            b["x"] += b_share * nx
            b["y"] += b_share * ny


def _resolve_ball_collisions(balls, width=WIDTH, height=HEIGHT):
    if LEGACY_CONTACTS:
        return _resolve_ball_collisions_legacy(balls)
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            a, b = balls[i], balls[j]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            dist = np.hypot(dx, dy)
            min_dist = a["radius"] + b["radius"]
            if dist == 0 or dist >= min_dist:
                continue

            nx, ny = dx / dist, dy / dist
            rvx, rvy = a["vx"] - b["vx"], a["vy"] - b["vy"]
            vrel = rvx * nx + rvy * ny
            if vrel > 0:
                a_sl, b_sl = a["asleep"], b["asleep"]
                if (a_sl or b_sl) and vrel < WAKE_V:
                    # gentle contact: the sleeper stays asleep and acts as an immovable
                    # anchor (inverse mass 0); the awake partner just bounces off it
                    inv_a = 0.0 if a_sl else 1.0 / a["mass"]
                    inv_b = 0.0 if b_sl else 1.0 / b["mass"]
                    if inv_a + inv_b > 0:
                        e = min(a["mat"]["restitution"], b["mat"]["restitution"])
                        imp = -(1 + e) * vrel / (inv_a + inv_b)
                        a["vx"] += imp * inv_a * nx; a["vy"] += imp * inv_a * ny
                        b["vx"] -= imp * inv_b * nx; b["vy"] -= imp * inv_b * ny
                else:
                    # a real hit wakes either sleeper so it takes the impulse and rejoins
                    if a_sl or b_sl:
                        a["asleep"] = b["asleep"] = False
                        a["still"] = b["still"] = 0
                    e = min(a["mat"]["restitution"], b["mat"]["restitution"])
                    imp = -(1 + e) * vrel / (1.0 / a["mass"] + 1.0 / b["mass"])
                    a["vx"] += (imp / a["mass"]) * nx
                    a["vy"] += (imp / a["mass"]) * ny
                    b["vx"] -= (imp / b["mass"]) * nx
                    b["vy"] -= (imp / b["mass"]) * ny

            # positional correction is applied regardless of approach speed --
            # skipping it for resting/separating pairs left piles interpenetrated
            _separate_pair(a, b, nx, ny, min_dist - dist, width, height)


def generate_bouncing_data(data_dir="data/bouncing", n_trajectories=5000, max_frames=100,
                           width=WIDTH, height=HEIGHT, n_balls_min=1, n_balls_max=5,
                           speed_min=3.0, speed_max=8.0, n_substeps=4, supersample=1,
                           start_idx=0, y_frac_min=0.3, y_frac_max=1.0,
                           radius_min=5, radius_max=8, progress_cb=None):
    ss = max(1, int(supersample))
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    dt = 1.0 / n_substeps

    for i in tqdm(range(n_trajectories), desc="Generating RGB Physics Envs"):
        traj_dir = os.path.join(data_dir, f'traj-{start_idx + i}')
        os.makedirs(traj_dir, exist_ok=True)

        n_balls = np.random.randint(n_balls_min, n_balls_max + 1)
        balls = _spawn_balls(n_balls, width, height, speed_min, speed_max,
                             y_frac_min=y_frac_min, y_frac_max=y_frac_max,
                             radius_min=radius_min, radius_max=radius_max)

        positions, velocities = [], []

        for frame in range(max_frames):
            img = np.ones((height * ss, width * ss, 3), dtype=np.uint8) * 255
            for b in balls:
                center = (round((b["x"] * ss - 0.5) * _SUBPIX), round(((height - b["y"]) * ss - 0.5) * _SUBPIX))
                cv2.circle(img, center, round(b["radius"] * ss * _SUBPIX), b["mat"]["color"], -1,
                           lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
            if ss > 1:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(traj_dir, f'frame_{frame:03d}.png'), img)

            positions.append([(b["x"], b["y"]) for b in balls])
            velocities.append([(b["vx"], b["vy"]) for b in balls])

            for _ in range(n_substeps):
                for b in balls:
                    if b["asleep"]:
                        continue
                    radius, mass = b["radius"], b["mass"]
                    drag_x = -(AIR_DRAG_COEFF * b["vx"] * abs(b["vx"]) * radius) / mass
                    drag_y = -(AIR_DRAG_COEFF * b["vy"] * abs(b["vy"]) * radius) / mass

                    b["vx"] += drag_x * dt
                    b["vy"] += (GRAVITY + drag_y) * dt
                    b["x"] += b["vx"] * dt
                    b["y"] += b["vy"] * dt

                _resolve_ball_collisions(balls, width, height)

                for b in balls:
                    if b["asleep"]:
                        continue
                    radius = b["radius"]
                    fric = b["mat"]["friction"]
                    rest = b["mat"]["restitution"]
                    if b["y"] - radius <= 0:
                        b["y"], b["vy"] = radius, b["vy"] * -rest
                        b["vx"] *= fric
                        if abs(b["vy"]) < REST_VELOCITY:
                            b["vy"] = 0.0
                    if b["y"] + radius >= height:
                        b["y"], b["vy"] = height - radius, b["vy"] * -rest
                        b["vx"] *= fric
                    if b["x"] - radius <= 0:
                        b["x"], b["vx"] = radius, b["vx"] * -rest
                        b["vy"] *= fric
                    if b["x"] + radius >= width:
                        b["x"], b["vx"] = width - radius, b["vx"] * -rest
                        b["vy"] *= fric

                _settle_contacts(balls, width, height)
                _update_sleep(balls, height)

        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))

        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


if __name__ == "__main__":
    generate_bouncing_data()