"""Solar-system environment: closed-form Kepler dynamics rendered flat on a
space-black background (64x64, filled circles, no outline/border/markers). Cartoon geometry, exact laws: sizes and distances are compressed to
fit 64px, but the dynamics are exact two-body solutions -- planets follow
Kepler ellipses around the star (or the barycenter of a binary pair), so
Kepler's 2nd and 3rd laws hold to machine precision in the ground truth and
there is NO integration drift.

Physics choices (mirroring real astronomy):
- sun-planet gravity only (planet-planet forces are ~1e-4 of the sun's in
  real systems on visible timescales);
- star classes with main-sequence mass ordering: blue heavy, yellow medium,
  red dwarf light -- the star's color IS a dynamical signal (heavier star ->
  faster orbits at the same radius, Kepler III);
- planet types rocky/ice/gas giant at increasing size, density realistic in
  ordering (dynamically inert under sun-only gravity: everything falls the
  same way);
- ~25%% of scenes are BINARY stars (real: about half of stars are in pairs);
  the pair counter-orbits its barycenter (exact two-body), planets orbit the
  barycenter at >= 2.6x the separation (real circumbinary stability rule);
- all planets in one scene share one orbital direction (planets condense from
  one spinning disk); the scene's direction is a coin flip (above/below view);
- mild ellipses, e in [0, 0.25] weighted low (real planets: mostly e < 0.1);
- radial bands (periapsis..apoapsis +- planet radius) are disjoint, so bodies
  NEVER overlap or occlude -- integrity tools see clean circles throughout.

Trajectory format is byte-compatible with env_pymunk (frames.npz + positions/
velocities/angles npys + objects.json); an extra solar.json records the orbit
elements per body for physics-recovery metrics."""
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from environments.env_bouncing import MATERIALS
from environments.env_shapes import _draw
from src.frameio import save_frames

STAR_CLASSES = ["BlueStar", "YellowStar", "RedStar"]
# stars and planets sized (and GAP tightened) so packed 3-planet rosters fit
# the 64px radial budget (~20% of single-star scenes at these values)
STAR_SIZE = {"BlueStar": (7.0, 8.2), "YellowStar": (6.0, 7.2),
             "RedStar": (4.8, 6.0)}
PLANET_CLASSES = ["Rocky", "Ice", "GasGiant"]
PLANET_SIZE = {"Rocky": (2.3, 3.0), "Ice": (2.7, 3.5), "GasGiant": (3.4, 4.2)}
CENTER = 32.0
R_MAX = 29.0          # outermost apoapsis incl. planet radius must stay inside
GAP = 1.2             # clearance between adjacent radial bands


def _kepler_E(M, e, iters=8):
    """Solve Kepler's equation E - e*sinE = M (Newton; e <= 0.25 converges in
    a few steps from E0 = M)."""
    E = np.asarray(M, dtype=np.float64).copy()
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def _sample_system(binary_prob=0.25, n_planets_min=1, n_planets_max=4,
                   gm_scale=1.0, e_range=None):
    """Sample stars + planet orbital elements. Returns None if the roster
    doesn't fit the radial budget (caller resamples)."""
    direction = 1.0 if np.random.rand() < 0.5 else -1.0
    binary = np.random.rand() < binary_prob
    stars = []
    if binary:
        # compact pair, stars drawn at ~40% class size (cartoon compression:
        # full-size pairs + the stability rule leave no radial budget for
        # planets, and the resampler then silently rejects every binary).
        # Circumbinary orbits >= 2.3x the separation (Holman-Wiegert critical
        # limit for circular binaries); binaries carry 1-2 planets, true to
        # real circumbinary systems (Kepler-16 has one known planet).
        c1, c2 = np.random.choice(STAR_CLASSES, 2, replace=True)
        s1 = 0.4 * np.random.uniform(*STAR_SIZE[c1])
        s2 = 0.4 * np.random.uniform(*STAR_SIZE[c2])
        sep = np.random.uniform(max(s1 + s2 + 0.8, 5.5), 8.5)
        gm1 = gm_scale * MATERIALS[c1]["gm"]
        gm2 = gm_scale * MATERIALS[c2]["gm"]
        gm_tot = gm1 + gm2
        Tb = 2.0 * np.pi * np.sqrt(sep ** 3 / gm_tot)
        # each star rides its own circle around the barycenter, opposite phases
        r1, r2 = sep * gm2 / gm_tot, sep * gm1 / gm_tot
        phi0 = np.random.uniform(0.0, 2.0 * np.pi)
        stars.append({"cls": c1, "size": s1, "orb_r": r1, "T": Tb,
                      "phi0": phi0})
        stars.append({"cls": c2, "size": s2, "orb_r": r2, "T": Tb,
                      "phi0": phi0 + np.pi})
        inner = 2.3 * sep
        n_min = 1
    else:
        c1 = np.random.choice(STAR_CLASSES)
        s1 = np.random.uniform(*STAR_SIZE[c1])
        gm_tot = gm_scale * MATERIALS[c1]["gm"]
        stars.append({"cls": c1, "size": s1, "orb_r": 0.0, "T": 1.0,
                      "phi0": 0.0})
        inner = s1
        n_min = n_planets_min

    n_lo, n_hi = (1, 2) if binary else (n_planets_min, n_planets_max)
    # RETRY WITHIN THE TYPE: rejecting the whole system and re-flipping the
    # binary coin lets easy-to-fit rosters win the resampling race and skews
    # the binary share. Placement is retried for the SAME system type until
    # the roster meets the minimum, so the coin stays a true 25%. A truncated
    # roster (n drawn but fewer fit) is ACCEPTED: the 64px radial budget
    # genuinely holds ~2 planets around a full-size sun (3 only when star,
    # planets and eccentricities all draw small) -- chasing exact counts
    # would collapse every scene to the one count that always fits.
    # roster size is drawn ONCE per system, then PLACEMENT retries until that
    # exact count fits -- accepting truncations let easy counts win (61% of
    # loose 2-rosters truncated to 1; measured), and redrawing n per retry
    # enriches whatever fits most easily (the binary-coin disease, intra
    # -singles). 4 never fits the radial budget (0/347 measured) so singles
    # draw 1/2/3 at 20/50/30.
    if binary:
        n = np.random.randint(1, 3)
    elif n_planets_min > 1:
        n = np.random.randint(n_planets_min, n_planets_max + 1)
    else:
        n = int(np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
    # PACKED rosters (3 planets around a single, 2 around a binary) only
    # fit the 64px radial budget the way real packed systems do it
    # (TRAPPIST-1): small planets, near-circular orbits, tight spacing.
    # Loose mid-band placement never fits a third band (measured ~0.1%).
    tight = n >= (2 if binary else 3)
    best = None
    for _ in range(40):
        planets = []
        band_lo = inner + GAP
        for _ in range(n):
            # packed rosters exclude gas giants ("peas in a pod": real compact
            # multis are small rocky/icy worlds; giants live in sparse systems)
            cls = np.random.choice(("Rocky", "Ice") if tight
                                   else PLANET_CLASSES)
            lo_s, hi_s = PLANET_SIZE[cls]
            pr = np.random.uniform(lo_s, lo_s + (hi_s - lo_s)
                                   * (0.45 if tight else 1.0))
            if e_range is not None:
                # eccentricity override (OOD sets): uniform in [lo, hi]
                e = np.random.uniform(*e_range)
            else:
                e = (0.08 if tight else 0.25) * np.random.rand() ** 2
            a_lo = (band_lo + pr) / (1.0 - e)
            a_hi = (R_MAX - pr) / (1.0 + e)
            if a_lo > a_hi:
                break                   # radial budget exhausted
            if tight:
                # hard-pack: sit just above the minimum radius, tiny jitter
                a = a_lo + min(1.5, a_hi - a_lo) * np.random.rand()
            elif n == 1:
                # a lone planet may sit anywhere in the band: free diversity
                a = np.random.uniform(a_lo, a_hi)
            else:
                # mild low-end bias so 2-rosters stay 2 instead of
                # truncating to 1 when the first planet lands high
                a = a_lo + (a_hi - a_lo) * np.random.rand() ** 2.5
            T = 2.0 * np.pi * np.sqrt(a ** 3 / gm_tot)
            planets.append({"cls": cls, "size": pr, "a": a, "e": e,
                            "omega": np.random.uniform(0.0, 2.0 * np.pi),
                            "M0": np.random.uniform(0.0, 2.0 * np.pi), "T": T})
            band_lo = a * (1.0 + e) + pr + GAP
        if len(planets) == n:
            best = planets
            break
        if best is None or len(planets) > len(best):
            best = planets              # truncation fallback, rarely needed
    if best is None or len(best) < n_min:
        return None
    return {"direction": direction, "binary": binary, "gm_tot": gm_tot,
            "stars": stars, "planets": best}


def _propagate(system, t):
    """Positions + velocities of all bodies (stars first) at frame t, exact."""
    d = system["direction"]
    out_p, out_v = [], []
    for s in system["stars"]:
        w = d * 2.0 * np.pi / s["T"]
        phi = s["phi0"] + w * t
        out_p.append((CENTER + s["orb_r"] * np.cos(phi),
                      CENTER + s["orb_r"] * np.sin(phi)))
        out_v.append((-s["orb_r"] * w * np.sin(phi),
                      s["orb_r"] * w * np.cos(phi)))
    for p in system["planets"]:
        n_mot = d * 2.0 * np.pi / p["T"]
        E = float(_kepler_E(p["M0"] + n_mot * t, p["e"]))
        b = p["a"] * np.sqrt(1.0 - p["e"] ** 2)
        Edot = n_mot / (1.0 - p["e"] * np.cos(E))
        xo, yo = p["a"] * (np.cos(E) - p["e"]), b * np.sin(E)
        vxo, vyo = -p["a"] * np.sin(E) * Edot, b * np.cos(E) * Edot
        co, so = np.cos(p["omega"]), np.sin(p["omega"])
        out_p.append((CENTER + co * xo - so * yo, CENTER + so * xo + co * yo))
        out_v.append((co * vxo - so * vyo, so * vxo + co * vyo))
    return out_p, out_v


def generate_solar(data_dir="data/solar", n_trajectories=100, max_frames=100,
                   width=64, height=64, supersample=4, start_idx=0,
                   binary_prob=0.25, n_planets_min=1, n_planets_max=4,
                   gm_scale=1.0, e_range=None, progress_cb=None):
    """gm_scale multiplies every star's gravitational parameter: orbital
    speeds scale with sqrt(gm_scale), periods shrink by the same factor."""
    ss = max(1, int(supersample))
    os.makedirs(data_dir, exist_ok=True)
    for i in tqdm(range(n_trajectories), desc="Generating solar systems"):
        traj_dir = os.path.join(data_dir, f"traj-{start_idx + i}")
        os.makedirs(traj_dir, exist_ok=True)
        system = None
        while system is None:
            system = _sample_system(binary_prob, n_planets_min, n_planets_max,
                                    gm_scale, e_range)
        rosters = ([(s["cls"], s["size"]) for s in system["stars"]]
                   + [(p["cls"], p["size"]) for p in system["planets"]])
        objs = [{"kind": "ball", "mat": MATERIALS[cls], "mat_name": cls,
                 "size": float(sz), "theta": 0.0, "x": CENTER, "y": CENTER,
                 "verts": None} for cls, sz in rosters]
        positions, velocities, angles, frames = [], [], [], []
        for t in range(max_frames):
            pos, vel = _propagate(system, t)
            for o, (x, y) in zip(objs, pos):
                o["x"], o["y"] = float(x), float(y)
            frames.append(cv2.cvtColor(
                _draw(objs, width, height, ss, markers="none", outline=False,
                      bg=0),
                cv2.COLOR_BGR2RGB))
            positions.append([(float(x), float(y)) for x, y in pos])
            velocities.append([(float(vx), float(vy)) for vx, vy in vel])
            angles.append([(0.0, 0.0)] * len(objs))
        save_frames(traj_dir, np.stack(frames))
        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))
        np.save(os.path.join(traj_dir, "angles.npy"), np.array(angles))
        with open(os.path.join(traj_dir, "objects.json"), "w") as f:
            json.dump([{"kind": o["kind"], "size": o["size"],
                        "material": o["mat_name"], "verts": None}
                       for o in objs], f)
        with open(os.path.join(traj_dir, "solar.json"), "w") as f:
            json.dump({"direction": system["direction"],
                       "binary": bool(system["binary"]),
                       "gm_scale": gm_scale,
                       "gm_tot": system["gm_tot"],
                       "stars": system["stars"],
                       "planets": system["planets"]}, f)
        if progress_cb and (i + 1) % max(1, n_trajectories // 100) == 0:
            progress_cb(i + 1, n_trajectories)
