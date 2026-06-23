import os
import cv2
import numpy as np
from tqdm import tqdm

# Define materials: (Color BGR, Density, Restitution/Bounciness, Friction)
MATERIALS = {
    "Superball": {"color": (0, 0, 255), "density": 0.5, "restitution": 0.95, "friction": 0.98}, # Red, very light, super bouncy
    "Rubber":    {"color": (255, 0, 0), "density": 1.2, "restitution": 0.80, "friction": 0.90}, # Blue, medium weight, normal bounce
    "Steel":     {"color": (100, 100, 100), "density": 5.0, "restitution": 0.40, "friction": 0.80}, # Gray, very heavy, low bounce
    "Sponge":    {"color": (0, 255, 0), "density": 0.2, "restitution": 0.20, "friction": 0.60}  # Green, extremely light, dead bounce
}

WIDTH, HEIGHT = 64, 64
GRAVITY = -0.5
AIR_DRAG_COEFF = 0.02


def _make_ball(width, height, speed_min, speed_max):
    """Create one ball with a random material, size, position and velocity."""
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
    """Place balls so they don't start overlapping."""
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
    """Elastic 2D collision with restitution + mass between every pair of balls."""
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            a, b = balls[i], balls[j]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            dist = np.hypot(dx, dy)
            min_dist = a["radius"] + b["radius"]
            if dist == 0 or dist >= min_dist:
                continue

            # Collision normal (from a to b).
            nx, ny = dx / dist, dy / dist
            # Relative velocity along the normal.
            rvx, rvy = a["vx"] - b["vx"], a["vy"] - b["vy"]
            vrel = rvx * nx + rvy * ny
            if vrel <= 0:
                continue  # already separating

            e = min(a["mat"]["restitution"], b["mat"]["restitution"])
            imp = -(1 + e) * vrel / (1.0 / a["mass"] + 1.0 / b["mass"])
            a["vx"] += (imp / a["mass"]) * nx
            a["vy"] += (imp / a["mass"]) * ny
            b["vx"] -= (imp / b["mass"]) * nx
            b["vy"] -= (imp / b["mass"]) * ny

            # Positional correction so the balls don't sink into each other.
            overlap = (min_dist - dist) / 2.0
            a["x"] -= overlap * nx
            a["y"] -= overlap * ny
            b["x"] += overlap * nx
            b["y"] += overlap * ny


def generate_bouncing_data(data_dir="data/bouncing", n_trajectories=5000, max_frames=100,
                           width=WIDTH, height=HEIGHT, n_balls_min=1, n_balls_max=5,
                           speed_min=3.0, speed_max=8.0, progress_cb=None):
    """Generate the bouncing-ball dataset. Resolution must be a multiple of 8 to match
    the autoencoder's 3 stride-2 down/up-sampling stages. `progress_cb(done, total)` is
    called periodically (used by the GUI to report progress)."""
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)

    for i in tqdm(range(n_trajectories), desc="Generating RGB Physics Envs"):
        traj_dir = os.path.join(data_dir, f'traj-{i}')
        os.makedirs(traj_dir, exist_ok=True)

        n_balls = np.random.randint(n_balls_min, n_balls_max + 1)
        balls = _spawn_balls(n_balls, width, height, speed_min, speed_max)

        positions, velocities = [], []

        for frame in range(max_frames):
            # Render RGB Image (balls drawn over a white background).
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            for b in balls:
                cv2.circle(img, (int(b["x"]), height - int(b["y"])), b["radius"], b["mat"]["color"], -1)
            cv2.imwrite(os.path.join(traj_dir, f'frame_{frame:03d}.png'), img)

            positions.append([(b["x"], b["y"]) for b in balls])
            velocities.append([(b["vx"], b["vy"]) for b in balls])

            # Physics update per ball (gravity + quadratic air drag).
            for b in balls:
                radius, mass = b["radius"], b["mass"]
                drag_x = -(AIR_DRAG_COEFF * b["vx"] * abs(b["vx"]) * radius) / mass
                drag_y = -(AIR_DRAG_COEFF * b["vy"] * abs(b["vy"]) * radius) / mass

                b["vx"] += drag_x
                b["vy"] += GRAVITY + drag_y
                b["x"] += b["vx"]
                b["y"] += b["vy"]

                # Floor collision (uses material restitution + friction).
                if b["y"] - radius <= 0:
                    b["y"], b["vy"] = radius, b["vy"] * -b["mat"]["restitution"]
                    b["vx"] *= b["mat"]["friction"]
                # Ceiling collision: closes the box so balls never leave the frame.
                if b["y"] + radius >= height:
                    b["y"], b["vy"] = height - radius, b["vy"] * -b["mat"]["restitution"]
                    b["vx"] *= b["mat"]["friction"]
                # Wall collisions.
                if b["x"] - radius <= 0:
                    b["x"], b["vx"] = radius, b["vx"] * -b["mat"]["restitution"]
                if b["x"] + radius >= width:
                    b["x"], b["vx"] = width - radius, b["vx"] * -b["mat"]["restitution"]

            # Ball-ball collisions after the per-ball integration step.
            _resolve_ball_collisions(balls)

        # Shape: (frames, n_balls, 2). n_balls can vary per trajectory.
        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))

        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


if __name__ == "__main__":
    generate_bouncing_data()