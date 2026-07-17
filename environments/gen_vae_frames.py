"""Synthesize a VAE-training frame set (no physics, no trajectories).

Static scenes rendered with the exact drawing path of env_bouncing (same AA,
supersampling, colors), packaged as traj-* dirs of 100 frames so all existing
tooling works: each "traj" keeps FIXED balls (count/materials/radii) and
resamples positions per frame. positions.npy/velocities.npy are saved
(velocities all zero). The in-run LatentProbe is meaningless on shuffled
frames -- score AEs trained here with experiments/surrogate.py on real data.

Batches (fractions of --n-traj):
  40%  natural    -- non-overlapping scenes, ~35% of balls resting on the floor
  35%  overlap    -- forced pair/triple overlaps, mostly shallow (f 0.55-1.05),
                     20% deep (f 0.25-0.55)
  15%  radii      -- continuous radii U(4.0, 8.5) instead of int 5-8
  10%  wall       -- balls pinned at walls/corners, incl. overlapping pairs

  python -m environments.gen_vae_frames --data-dir data/vaeset_v1 --n-traj 3000 --seed 301
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from environments.env_bouncing import MATERIALS, WIDTH, HEIGHT, SUBPIX_BITS, _SUBPIX


def render(balls, ss=4, width=WIDTH, height=HEIGHT):
    img = np.ones((height * ss, width * ss, 3), dtype=np.uint8) * 255
    for b in balls:
        center = (round((b["x"] * ss - 0.5) * _SUBPIX), round(((height - b["y"]) * ss - 0.5) * _SUBPIX))
        cv2.circle(img, center, round(b["radius"] * ss * _SUBPIX), b["mat"]["color"], -1,
                   lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
    if ss > 1:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def _clamp(b):
    r = b["radius"]
    b["x"] = float(np.clip(b["x"], r, WIDTH - r))
    b["y"] = float(np.clip(b["y"], r, HEIGHT - r))


def _place_free(balls, i, resting_p=0.0, tries=60):
    """Position ball i without overlapping balls[:i]."""
    b = balls[i]
    r = b["radius"]
    for _ in range(tries):
        b["x"] = float(np.random.uniform(r, WIDTH - r))
        if np.random.rand() < resting_p:
            b["y"] = float(r)
        else:
            b["y"] = float(np.random.uniform(r, HEIGHT - r))
        if all((b["x"] - o["x"]) ** 2 + (b["y"] - o["y"]) ** 2 >= (r + o["radius"]) ** 2
               for o in balls[:i]):
            return
    # give up on separation; leave last sample


def sample_frame(balls, regime):
    n = len(balls)
    if regime == "natural":
        for i in range(n):
            _place_free(balls, i, resting_p=0.35)
    elif regime == "overlap":
        for i in range(n):
            _place_free(balls, i)
        # force overlaps: chain consecutive pairs with sampled depth
        n_pairs = max(1, n // 2)
        for k in range(n_pairs):
            a, b = balls[2 * k % n], balls[(2 * k + 1) % n]
            if a is b:
                continue
            f = np.random.uniform(0.25, 0.55) if np.random.rand() < 0.2 else np.random.uniform(0.55, 1.05)
            d = f * (a["radius"] + b["radius"])
            ang = np.random.uniform(0, 2 * np.pi)
            b["x"], b["y"] = a["x"] + d * np.cos(ang), a["y"] + d * np.sin(ang)
            _clamp(b)
    elif regime == "radii":
        for i in range(n):
            _place_free(balls, i, resting_p=0.25)
    elif regime == "wall":
        for i, b in enumerate(balls):
            r = b["radius"]
            wall = np.random.randint(4)
            if wall == 0:   b["x"], b["y"] = float(r), float(np.random.uniform(r, HEIGHT - r))
            elif wall == 1: b["x"], b["y"] = float(WIDTH - r), float(np.random.uniform(r, HEIGHT - r))
            elif wall == 2: b["x"], b["y"] = float(np.random.uniform(r, WIDTH - r)), float(r)
            else:           b["x"], b["y"] = float(np.random.choice([r, WIDTH - r])), float(r)  # corners
            if i > 0 and np.random.rand() < 0.5:  # overlap a wall-pinned pair
                o = balls[i - 1]
                d = np.random.uniform(0.6, 1.05) * (r + o["radius"])
                b["x"] = o["x"] + d * np.random.choice([-1, 1])
                b["y"] = o["y"]
                _clamp(b)
    return balls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/vaeset_v1")
    ap.add_argument("--n-traj", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--supersample", type=int, default=4)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)
    cuts = np.cumsum([0.40, 0.35, 0.15, 0.10]) * args.n_traj
    for ti in tqdm(range(args.n_traj), desc="Synthesizing VAE frames"):
        regime = ("natural" if ti < cuts[0] else "overlap" if ti < cuts[1]
                  else "radii" if ti < cuts[2] else "wall")
        td = os.path.join(args.data_dir, f"traj-{ti}")
        os.makedirs(td, exist_ok=True)
        n = np.random.randint(2, 6) if regime != "wall" else np.random.randint(2, 5)
        balls = []
        for _ in range(n):
            mat = np.random.choice(list(MATERIALS.keys()))
            radius = float(np.random.uniform(4.0, 8.5)) if regime == "radii" else float(np.random.randint(5, 9))
            balls.append({"mat": MATERIALS[mat], "radius": radius, "x": 32.0, "y": 32.0})
        positions = []
        for fi in range(args.frames):
            sample_frame(balls, regime)
            img = render(balls, ss=args.supersample)
            cv2.imwrite(os.path.join(td, f"frame_{fi:03d}.png"), img)
            positions.append([(b["x"], b["y"]) for b in balls])
        pos = np.array(positions)
        np.save(os.path.join(td, "positions.npy"), pos)
        np.save(os.path.join(td, "velocities.npy"), np.zeros_like(pos))
    print(f"done: {args.n_traj} trajs x {args.frames} frames in {args.data_dir}")


if __name__ == "__main__":
    main()
