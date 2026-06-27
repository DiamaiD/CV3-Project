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
# Sub-pixel rendering precision: cv2.circle interprets the center/radius as fixed-point numbers
# with this many fractional bits, so the ball's CONTINUOUS position is preserved instead of being
# truncated to a whole pixel. Combined with LINE_AA this is what keeps frame-to-frame motion smooth
# and predictable -- integer rendering turns a ball drifting 1.4 px/frame into a 1,1,2,1,2 staircase
# and injects unrecoverable quantization noise into the dynamics. 4 bits = 1/16 px.
SUBPIX_BITS = 4
_SUBPIX = 1 << SUBPIX_BITS
# A ball whose post-bounce vertical speed is below this can't escape gravity for a
# full frame, so we treat it as resting and zero its velocity to avoid sub-pixel jitter.
REST_VELOCITY = abs(GRAVITY)

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
def _make_plane(width,height):
    """Create a static plane (line segment / surface) with random position and orientation."""

    mat_name = np.random.choice(list(MATERIALS.keys()))
    mat = MATERIALS[mat_name]

    w = 1

    # Start links am Bildrand (oben oder mittig leicht variierbar)
    x1 = 0
    y1 = np.random.randint(height // 2, height)  # eher oben links
    # Ende am Boden, irgendwo rechts
    x2 = np.random.randint(width // 2, width )  # etwas weiter nach rechts gestreckt
    y2 = 0  # Nahe 0, damit es am Boden endet

    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        tx, ty = 1.0, 0.0
        nx, ny = 0.0, 1.0
    else:
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx

    # Normale (senkrecht)
    nx, ny = -ty, tx

    return {
        "mat": mat,
        "width": w,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "tx": tx,
        "ty": ty,
        "nx": nx,
        "ny": ny,
    }

def _spawn_balls(n_balls, plane, width, height, speed_min, speed_max, max_tries=100):
    """Place balls so they don't start overlapping."""
    balls = []

    # Sicherstellen, dass wir nicht durch 0 teilen, falls die Ebene vertikal wäre
    plane_dx = plane["x2"] - plane["x1"]
    if plane_dx == 0:
        plane_dx = 1.0

    for _ in range(n_balls):
        for _ in range(max_tries):
            # Erstellt das Standard-Ball-Objekt (inklusive Material, Radius, Masse)
            cand = _make_ball(width, height, speed_min, speed_max)

            # 1. Nur Gravitation wirken lassen -> Anfangsgeschwindigkeit komplett auf 0
            cand["vx"] = 0.0
            cand["vy"] = 0.0

            # 2. X-Position zufällig entlang der Ebene wählen (mit etwas Puffer zu den Enden)
            buffer_x = cand["radius"] + 5
            cand["x"] = np.random.uniform(plane["x1"] + buffer_x, plane["x2"] - buffer_x)

            # Berechne die Höhe (y) der Ebene an exakt dieser X-Position
            # Formel für die Gerade: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            x_percentage = (cand["x"] - plane["x1"]) / plane_dx
            y_on_plane = plane["y1"] + x_percentage * (plane["y2"] - plane["y1"])

            # Platziere den Ball ein Stück ÜBER diesem Punkt (Radius + kleiner zufälliger Abstand)
            random_air_gap = np.random.uniform(10, 40)  # 10 bis 40 Pixel Luft zur Ebene
            cand["y"] = y_on_plane + cand["radius"] + random_air_gap

            # Sicherheitscheck: Falls die Rampe sehr hoch liegt, nicht über die Decke spawnen
            if cand["y"] + cand["radius"] >= height:
                cand["y"] = height - cand["radius"] - 5

            # 3. Kollisions-Check mit bereits existierenden Bällen
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

def resolve_ball_plane_collisions(balls, plane):
    """Elastic 2D collision with restitution between balls and static planes (line segments)."""
    for ball in balls:

        # Vektor vom Startpunkt der Ebene (x1, y1) zum Ball
        dx = ball["x"] - plane["x1"]
        dy = ball["y"] - plane["y1"]

        # Da die Länge nicht im Plane-Dict gespeichert ist, berechnen wir sie kurz
        plane_len = np.hypot(plane["x2"] - plane["x1"], plane["y2"] - plane["y1"])

        # Projektion des Balls auf den Tangentenvektor der Ebene
        # t gibt an, wie weit der Ball entlang der Linie umherwandert
        t = dx * plane["tx"] + dy * plane["ty"]

        # "Clamping": Begrenzt den Punkt auf das tatsächliche Liniensegment
        t = max(0.0, min(plane_len, t))

        # Der exakte Punkt auf dem Segment, der dem Ball am nächsten ist
        closest_x = plane["x1"] + t * plane["tx"]
        closest_y = plane["y1"] + t * plane["ty"]

        # Vektor vom nächsten Punkt auf der Ebene zum Ball
        sub_dx = ball["x"] - closest_x
        sub_dy = ball["y"] - closest_y
        dist = np.hypot(sub_dx, sub_dy)

        # Wenn keine Berührung stattfindet, weitergehen
        if dist == 0 or dist >= ball["radius"]:
            continue

        # Kollisionsnormale (zeigt von der Ebene weg zum Ball)
        nx = sub_dx / dist
        ny = sub_dy / dist

        # Relative Geschwindigkeit entlang der Normale
        # Da die Ebene statisch ist (vx=0, vy=0), ist das einfach die Ballgeschwindigkeit
        vrel = ball["vx"] * nx + ball["vy"] * ny

        # Wenn sich der Ball bereits von der Ebene wegfolgt, ignorieren
        if vrel >= 0:
            continue

        # Elastizität (Restitution) zwischen Ball und Ebene ermitteln
        e = min(ball["mat"]["restitution"], plane["mat"]["restitution"])

        # Impuls berechnen (Ebene hat unendliche Masse, m2 fällt aus der Formel weg)
        # Reduziert sich zu: imp = -(1 + e) * vrel * ball["mass"]
        imp = -(1 + e) * vrel * ball["mass"]

        # Nur die Geschwindigkeit des Balls wird angepasst
        ball["vx"] += (imp / ball["mass"]) * nx
        ball["vy"] += (imp / ball["mass"]) * ny

        # Positionskorrektur (Wand bewegt sich nicht, Ball übernimmt 100% des Overlaps)
        overlap = ball["radius"] - dist
        ball["x"] += overlap * nx
        ball["y"] += overlap * ny


def generate_slope_data(data_dir="data/slope", n_trajectories=5000, max_frames=100,
                           width=WIDTH, height=HEIGHT, n_balls_min=1, n_balls_max=5,
                           speed_min=3.0, speed_max=8.0, n_substeps=4, progress_cb=None):
    """Generate the slope dataset. Resolution must be a multiple of 8 to match
    the autoencoder's 3 stride-2 down/up-sampling stages. `progress_cb(done, total)` is
    called periodically (used by the GUI to report progress)."""
    os.makedirs(data_dir, exist_ok=True)
    report_every = max(1, n_trajectories // 100)
    dt = 1.0 / n_substeps

    for i in tqdm(range(n_trajectories), desc="Generating RGB Physics Envs"):
        traj_dir = os.path.join(data_dir, f'traj-{i}')
        os.makedirs(traj_dir, exist_ok=True)

        n_balls = np.random.randint(n_balls_min, n_balls_max + 1)

        plane = _make_plane(width,height)
        balls = _spawn_balls(n_balls,plane, width, height, speed_min, speed_max)
        positions, velocities = [], []

        for frame in range(max_frames):
            # Render RGB Image (balls drawn over a white background). Fixed-point sub-pixel center +
            # anti-aliasing so the continuous physics position survives into the pixels (see SUBPIX_BITS).
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            for b in balls:
                center = (round(b["x"] * _SUBPIX), round((height - b["y"]) * _SUBPIX))
                cv2.circle(img, center, round(b["radius"] * _SUBPIX), b["mat"]["color"], -1,
                           lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
            cv2.line(
                img,
                (plane["x1"], height - plane["y1"]),
                (plane["x2"], height - plane["y2"]),
                (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.imwrite(os.path.join(traj_dir, f'frame_{frame:03d}.png'), img)

            positions.append([(b["x"], b["y"]) for b in balls])
            velocities.append([(b["vx"], b["vy"]) for b in balls])

            # Advance the physics in several small substeps per rendered frame.
            # Smaller per-step displacement keeps fast balls from passing through
            # each other or the walls before a collision can be detected.
            for _ in range(n_substeps):
                # Per-ball integration (gravity + quadratic air drag).
                for b in balls:
                    radius, mass = b["radius"], b["mass"]
                    drag_x = -(AIR_DRAG_COEFF * b["vx"] * abs(b["vx"]) * radius) / mass
                    drag_y = -(AIR_DRAG_COEFF * b["vy"] * abs(b["vy"]) * radius) / mass

                    b["vx"] += drag_x * dt
                    b["vy"] += (GRAVITY + drag_y) * dt
                    b["x"] += b["vx"] * dt
                    b["y"] += b["vy"] * dt

                #_resolve_ball_collisions(balls)
                resolve_ball_plane_collisions(balls,plane)
                for b in balls:
                    radius = b["radius"]
                    fric = b["mat"]["friction"]
                    rest = b["mat"]["restitution"]
                    # Floor collision (uses material restitution + friction).
                    if b["y"] - radius <= 0:
                        b["y"], b["vy"] = radius, b["vy"] * -rest
                        b["vx"] *= fric
                        # Let the ball settle instead of jittering forever once its
                        # rebound is too weak to lift it off the floor.
                        if abs(b["vy"]) < REST_VELOCITY:
                            b["vy"] = 0.0
                    # Ceiling collision: closes the box so balls never leave the frame.
                    if b["y"] + radius >= height:
                        b["y"], b["vy"] = height - radius, b["vy"] * -rest
                        b["vx"] *= fric
                    # Wall collisions (friction damps the tangential, i.e. vertical, speed).
                    ''' if b["x"] - radius <= 0:
                        b["x"], b["vx"] = radius, b["vx"] * -rest
                        b["vy"] *= fric
                    if b["x"] + radius >= width:
                        b["x"], b["vx"] = width - radius, b["vx"] * -rest
                        b["vy"] *= fric'''

        # Shape: (frames, n_balls, 2). n_balls can vary per trajectory.
        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))

        if progress_cb is not None and ((i + 1) % report_every == 0 or i + 1 == n_trajectories):
            progress_cb(i + 1, n_trajectories)


if __name__ == "__main__":
    generate_slope_data()