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


def _make_ball(width, height, speed_min, speed_max):
    mat_name = np.random.choice(list(MATERIALS.keys()))
    mat = MATERIALS[mat_name]
    radius = np.random.randint(5, 9)
    speed = np.random.uniform(speed_min, speed_max)
    return {
        "mat": mat,
        "radius": radius,
        "mass": (np.pi * radius ** 2) * mat["density"],
        "x": float(np.random.randint(radius, width - radius)),
        "y": float(np.random.randint(int(0.3 * height), height - radius)),
        "vx": np.random.randn() * speed,
        "vy": np.random.randn() * speed,
    }


def _spawn_balls(n_balls, width, height, speed_min, speed_max, max_tries=100):
    balls = []
    for _ in range(n_balls):
        for _ in range(max_tries):
            cand = _make_ball(width, height, speed_min, speed_max)
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


def _resolve_ball_collisions(balls):
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


def generate_bouncing_data(data_dir="data/bouncing", n_trajectories=5000, max_frames=100,
                           width=WIDTH, height=HEIGHT, n_balls_min=1, n_balls_max=5,
                           speed_min=3.0, speed_max=8.0, n_substeps=4, supersample=1,
                           progress_cb=None):
    ss = max(1, int(supersample))
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    dt = 1.0 / n_substeps

    for i in tqdm(range(n_trajectories), desc="Generating RGB Physics Envs"):
        traj_dir = os.path.join(data_dir, f'traj-{i}')
        os.makedirs(traj_dir, exist_ok=True)

        n_balls = np.random.randint(n_balls_min, n_balls_max + 1)
        balls = _spawn_balls(n_balls, width, height, speed_min, speed_max)

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
                    radius, mass = b["radius"], b["mass"]
                    drag_x = -(AIR_DRAG_COEFF * b["vx"] * abs(b["vx"]) * radius) / mass
                    drag_y = -(AIR_DRAG_COEFF * b["vy"] * abs(b["vy"]) * radius) / mass

                    b["vx"] += drag_x * dt
                    b["vy"] += (GRAVITY + drag_y) * dt
                    b["x"] += b["vx"] * dt
                    b["y"] += b["vy"] * dt

                _resolve_ball_collisions(balls)

                for b in balls:
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

        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))

        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


if __name__ == "__main__":
    generate_bouncing_data()