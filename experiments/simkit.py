import os
import sys
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.env_bouncing import (_resolve_ball_collisions, GRAVITY, AIR_DRAG_COEFF,
                                       WIDTH, HEIGHT, REST_VELOCITY, SUBPIX_BITS, _SUBPIX)
from experiments.extractor import MATERIALS


def make_ball(mat, radius, x, y, vx, vy):
    m = dict(MATERIALS[mat])
    return {"mat": {"color": m["rgb"], "density": m["density"],
                    "restitution": m["restitution"], "friction": m["friction"]},
            "mat_name": mat, "radius": radius,
            "mass": (np.pi * radius ** 2) * m["density"],
            "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy)}


def step_frame(balls, n_substeps=4):
    dt = 1.0 / n_substeps
    bb_hit = False
    wall_hit = False
    for _ in range(n_substeps):
        for b in balls:
            radius, mass = b["radius"], b["mass"]
            drag_x = -(AIR_DRAG_COEFF * b["vx"] * abs(b["vx"]) * radius) / mass
            drag_y = -(AIR_DRAG_COEFF * b["vy"] * abs(b["vy"]) * radius) / mass
            b["vx"] += drag_x * dt
            b["vy"] += (GRAVITY + drag_y) * dt
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt

        pre = [(b["vx"], b["vy"]) for b in balls]
        _resolve_ball_collisions(balls)
        if any((b["vx"], b["vy"]) != p for b, p in zip(balls, pre)):
            bb_hit = True

        for b in balls:
            radius = b["radius"]
            fric = b["mat"]["friction"]
            rest = b["mat"]["restitution"]
            if (b["y"] - radius <= 0 or b["y"] + radius >= HEIGHT
                    or b["x"] - radius <= 0 or b["x"] + radius >= WIDTH):
                wall_hit = True
            if b["y"] - radius <= 0:
                b["y"], b["vy"] = radius, b["vy"] * -rest
                b["vx"] *= fric
                if abs(b["vy"]) < REST_VELOCITY:
                    b["vy"] = 0.0
            if b["y"] + radius >= HEIGHT:
                b["y"], b["vy"] = HEIGHT - radius, b["vy"] * -rest
                b["vx"] *= fric
            if b["x"] - radius <= 0:
                b["x"], b["vx"] = radius, b["vx"] * -rest
                b["vy"] *= fric
            if b["x"] + radius >= WIDTH:
                b["x"], b["vx"] = WIDTH - radius, b["vx"] * -rest
                b["vy"] *= fric
    return bb_hit, wall_hit


def render_frame(balls, ss=4):
    img = np.ones((HEIGHT * ss, WIDTH * ss, 3), dtype=np.uint8) * 255
    for b in balls:
        center = (round((b["x"] * ss - 0.5) * _SUBPIX),
                  round(((HEIGHT - b["y"]) * ss - 0.5) * _SUBPIX))
        cv2.circle(img, center, round(b["radius"] * ss * _SUBPIX),
                   tuple(int(v) for v in b["mat"]["color"]), -1,
                   lineType=cv2.LINE_AA, shift=SUBPIX_BITS)
    if ss > 1:
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return img


def simulate(balls, n_frames, render=True, ss=4):
    N = len(balls)
    pos = np.zeros((n_frames, N, 2))
    vel = np.zeros((n_frames, N, 2))
    frames = np.zeros((n_frames, HEIGHT, WIDTH, 3), np.uint8) if render else None
    bb_frames, wall_frames = [], []
    for t in range(n_frames):
        if render:
            frames[t] = render_frame(balls, ss=ss)
        pos[t] = [(b["x"], b["y"]) for b in balls]
        vel[t] = [(b["vx"], b["vy"]) for b in balls]
        bb, wall = step_frame(balls)
        if bb:
            bb_frames.append(t)
        if wall:
            wall_frames.append(t)
    return pos, vel, frames, bb_frames, wall_frames
