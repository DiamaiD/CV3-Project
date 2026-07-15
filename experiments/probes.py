"""Balanced physics probe set.

The regular physics suite measures on the training distribution, where event
counts follow the physics itself (a Superball bounces all trajectory long, a
Sponge dies after one floor hit) -- so per-material statistics have wildly
uneven power (X's suite: 404 Superball wall events vs 14 Sponge; 13
cross-material collisions). This module builds *engineered* scenarios with
event quotas instead:

  - wall bank:  single-ball drop/shot scenarios per material, accepted until
                each material has `--wall-events` clean sim-side bounces.
                Dead materials get more (shorter) scenarios, not longer ones.
  - pair bank:  aimed two-ball collisions for every unordered material pair,
                accepted until each pair has `--pair-events` clean events
                (vs ~1-2 cross-material events per material pair before).

Scenarios are generated with simkit (bit-exact env clone), so the model sees
context frames indistinguishable from training data, and the sim states of
every accepted scenario double as the ground-truth control. A fixed --seed
defines the probe set: every certification run measures on identical scenarios.
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_states, MAT_NAMES, MATERIALS
from experiments.simkit import make_ball, simulate
from experiments.collisions import analyze, print_report

CTX = 5
ROLL = 40
TOTAL = CTX + ROLL
T_MIN = CTX + 1
G = -0.5
WORLD = 64
PROBE_VERSION = 1


def _sim_states(spec):
    balls = [make_ball(m, r, x, y, vx, vy) for (m, r, x, y, vx, vy) in spec]
    pos, vel, _, _, _ = simulate(balls, TOTAL, render=False)
    return pos


def _sim_frames(spec):
    balls = [make_ball(m, r, x, y, vx, vy) for (m, r, x, y, vx, vy) in spec]
    _, _, frames, _, _ = simulate(balls, TOTAL, render=True)
    return frames


def _gt_analyze(spec, pos):
    mats = [s[0] for s in spec]
    radius = np.array([float(s[1]) for s in spec])
    valid = np.ones(pos.shape[:2], bool)
    return analyze(pos, valid, radius, mats, t_min=T_MIN)


def _propose_wall(rng, mat):
    r = int(rng.integers(5, 9))
    if rng.random() < 0.5:  # floor drop with a horizontal component (rest + fric)
        x = rng.uniform(14, 50)
        y = rng.uniform(34, WORLD - r - 2)
        vx = rng.uniform(1.5, 3.5) * (1.0 if rng.random() < 0.5 else -1.0)
        vy = rng.uniform(-1.0, 1.0)
    else:  # side-wall shot
        x = rng.uniform(18, 46)
        y = rng.uniform(28, 52)
        vx = rng.uniform(3.5, 6.0) * (1.0 if rng.random() < 0.5 else -1.0)
        vy = rng.uniform(0.5, 2.5)
    return [(mat, r, x, y, vx, vy)]


def _propose_pair(rng, mat_a, mat_b):
    """Both balls are launched ballistically at a common meeting point, so the
    closing speed is split between them (each stays in-distribution) while the
    head-on approach is fast enough that even e=0.2 pairs separate out of the
    proximity zone within a few frames. Relative motion is gravity-free, so
    contact lands near f * (1 - r_sum / (2 d)), inside the model window."""
    ra, rb = int(rng.integers(5, 9)), int(rng.integers(5, 9))
    mx, my = rng.uniform(24, 40), rng.uniform(28, 44)
    ang = rng.uniform(0, 2 * math.pi)
    d = rng.uniform(16, 24)
    f = float(rng.integers(13, 19))
    ax, ay = mx + d * math.cos(ang), my + d * math.sin(ang)
    bx, by = mx - d * math.cos(ang), my - d * math.sin(ang)
    if not (ra + 1 <= ax <= WORLD - ra - 1 and ra + 3 <= ay <= WORLD - ra - 1):
        return None
    if not (rb + 1 <= bx <= WORLD - rb - 1 and rb + 3 <= by <= WORLD - rb - 1):
        return None
    # small perpendicular offset on one aim for impact-parameter diversity
    off = rng.uniform(-3.0, 3.0)
    tbx, tby = mx - off * math.sin(ang), my + off * math.cos(ang)
    avx, avy = (mx - ax) / f, (my - ay) / f - 0.5 * G * f
    bvx, bvy = (tbx - bx) / f, (tby - by) / f - 0.5 * G * f
    return [(mat_a, ra, ax, ay, avx, avy), (mat_b, rb, bx, by, bvx, bvy)]


def _fill_bank(rng, propose, count_events, target, max_tries, label):
    bank, n_events, tries = [], 0, 0
    while n_events < target and tries < max_tries:
        tries += 1
        spec = propose(rng)
        if spec is None:
            continue
        pos = _sim_states(spec)
        k = count_events(spec, pos)
        if k > 0:
            bank.append({"spec": spec, "gt_events": k})
            n_events += k
    if n_events < target:
        print(f"  [warn] {label}: only {n_events}/{target} events after {max_tries} tries")
    return bank, n_events


def build_banks(seed, wall_target, pair_target):
    rng = np.random.default_rng(seed)
    wall_bank, pair_bank = {}, {}

    for mat in MAT_NAMES:
        t0 = time.time()

        def count_wall(spec, pos):
            _, wall_rows = _gt_analyze(spec, pos)
            return len(wall_rows)

        bank, n = _fill_bank(rng, lambda r: _propose_wall(r, mat), count_wall,
                             wall_target, wall_target * 60, f"wall/{mat}")
        wall_bank[mat] = bank
        print(f"  wall {mat:10s}: {n} events from {len(bank)} scenarios ({time.time() - t0:.0f}s)")

    pairs = [(a, b) for i, a in enumerate(MAT_NAMES) for b in MAT_NAMES[i:]]
    for mat_a, mat_b in pairs:
        t0 = time.time()

        def count_pair(spec, pos):
            mom_rows, _ = _gt_analyze(spec, pos)
            return len(mom_rows)

        bank, n = _fill_bank(rng, lambda r: _propose_pair(r, mat_a, mat_b), count_pair,
                             pair_target, pair_target * 120, f"pair/{mat_a}-{mat_b}")
        pair_bank[f"{mat_a}|{mat_b}"] = bank
        print(f"  pair {mat_a:9s}-{mat_b:9s}: {n} events from {len(bank)} scenarios ({time.time() - t0:.0f}s)")
    return wall_bank, pair_bank


def _all_scenarios(wall_bank, pair_bank):
    # each bank contributes only its designed event type, so incidental wall
    # bounces inside pair scenarios can't re-imbalance the wall statistics
    for bank in wall_bank.values():
        for sc in bank:
            yield sc, "wall"
    for bank in pair_bank.values():
        for sc in bank:
            yield sc, "pair"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_13__07_38_22__bouncing_28k_b36_flow")
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--wall-events", type=int, default=100)
    ap.add_argument("--pair-events", type=int, default=40)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--bank-only", action="store_true",
                    help="build + validate the probe banks (GT + extractor controls) without GPU/model")
    args = ap.parse_args()

    print(f"Physics probe set v{PROBE_VERSION} | seed {args.seed} | "
          f"targets: {args.wall_events} wall events/material, {args.pair_events} bb events/pair")
    t0 = time.time()
    wall_bank, pair_bank = build_banks(args.seed, args.wall_events, args.pair_events)
    scenarios = list(_all_scenarios(wall_bank, pair_bank))
    print(f"[1/4] banks built: {len(scenarios)} scenarios ({time.time() - t0:.0f}s)")

    acc = {"gt": ([], []), "real": ([], []), "model": ([], [])}

    def _acc(tag, kind, mom_rows, wall_rows):
        if kind == "wall":
            acc[tag][1].extend(wall_rows)
        else:
            acc[tag][0].extend(mom_rows)

    t0 = time.time()
    for sc, kind in scenarios:
        pos = _sim_states(sc["spec"])
        m, w = _gt_analyze(sc["spec"], pos)
        _acc("gt", kind, m, w)
    print(f"[2/4] GT control analyzed ({time.time() - t0:.0f}s)")

    t0 = time.time()
    inventories = []
    for sc, kind in scenarios:
        inv = {}
        for s in sc["spec"]:
            inv[s[0]] = inv.get(s[0], 0) + 1
        inventories.append(inv)
        st = extract_states(_sim_frames(sc["spec"]), inventory=inv)
        m, w = analyze(st["pos"], st["valid"], st["radius"], st["mats"], t_min=T_MIN)
        _acc("real", kind, m, w)
    print(f"[3/4] extractor-on-real control analyzed ({time.time() - t0:.0f}s)")

    if not args.bank_only:
        import torch
        from experiments.rollout import load_run, rollout_frames_np
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ae, dit, cfg = load_run(args.run, device)
        t0 = time.time()
        real_frames = [_sim_frames(sc["spec"]) for sc, _ in scenarios]
        torch.manual_seed(99)
        model_frames = rollout_frames_np(ae, dit, real_frames, CTX, ROLL,
                                         num_steps=args.num_steps, device=device)
        for (sc, kind), frames, mf, inv in zip(scenarios, real_frames, model_frames, inventories):
            seq = np.concatenate([frames[:CTX], mf], axis=0)
            st = extract_states(seq, inventory=inv)
            if st["valid"][CTX:].mean() < 0.2:
                continue
            m, w = analyze(st["pos"], st["valid"], st["radius"], st["mats"], t_min=T_MIN)
            _acc("model", kind, m, w)
        print(f"[4/4] model rollouts analyzed ({time.time() - t0:.0f}s)")
    else:
        print("[4/4] skipped (--bank-only)")

    results = {"run": args.run, "probe_version": PROBE_VERSION, "seed": args.seed,
               "n_scenarios": len(scenarios), "rollout_steps": ROLL,
               "wall_scenarios": {m: len(b) for m, b in wall_bank.items()},
               "wall_gt_events": {m: sum(sc["gt_events"] for sc in b) for m, b in wall_bank.items()},
               "pair_scenarios": {p: len(b) for p, b in pair_bank.items()},
               "pair_gt_events": {p: sum(sc["gt_events"] for sc in b) for p, b in pair_bank.items()}}
    sources = [("gt", "GT probe states"), ("real", "extractor on real frames")]
    if not args.bank_only:
        sources.append(("model", "MODEL rollouts"))
    for tag, label in sources:
        results[tag] = print_report(label, acc[tag][0], acc[tag][1])

    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "probes.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
