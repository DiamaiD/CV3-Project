"""Specific-orbital-energy instrument for the solar world.

eps(t) = |v|^2 / 2 - GM / r   per planet, per unit planet mass (masses
cancel). True value from solar.json elements: eps_true = -GM / (2a),
exactly constant on a Kepler orbit. Velocities are 4th-order central
differences on positions; optional Savitzky-Golay position smoothing
(quadratic, window 9) for tracked (noisy) positions.

Single-star scenes only (binaries would need a two-center potential).

Stages (run: python -m experiments.solar_energy <stage>):
  exact     -- exact simulator positions: formula + velocity-scheme floor
  roundtrip -- GT frames through VAE + solar tracker: instrument ceiling
  model     -- 300-frame rollouts of the finetuned solar model on the
               in-dist set (solar_eval_s2) and the eccentric sets
               (solar_ecc, solar_ecc2)
Each stage prints deviation stats |eps/eps_true - 1| and saves
solar_energy_<stage>.json with per-frame curves.
"""
import json
import os
import sys
import zlib

import numpy as np

RUN = "runs/2026_08_01__17_35_56__solar_200k_s2_cont"
SETS = {"indist": ["scratchpad/examples/solar_eval_s2"],
        "ecc": ["scratchpad/examples/solar_ecc",
                "scratchpad/examples/solar_ecc2"]}
CTX = 5
STEPS = 300
GROUP = 96
N_RT = 150          # roundtrip scenes per set (calibration)
TRIM = 3            # FD edge frames excluded from stats


def _fd_velocity(P):
    """4th-order central differences (2nd order at the edges)."""
    v = np.zeros_like(P)
    v[2:-2] = (8.0 * (P[3:-1] - P[1:-3]) - (P[4:] - P[:-4])) / 12.0
    v[1] = (P[2] - P[0]) / 2.0
    v[-2] = (P[-1] - P[-3]) / 2.0
    v[0] = P[1] - P[0]
    v[-1] = P[-1] - P[-2]
    return v


def _smooth(P):
    from scipy.signal import savgol_filter
    return savgol_filter(P, 9, 2, axis=0)


ROLL = 25           # rolling-median window on the eps SERIES (not positions):
                    # true eps is constant and model eps drifts over hundreds
                    # of frames, so this denoises without the perihelion bias
                    # of position smoothing


def eps_deviation(P_pl, P_star, gm, eps_true, smooth=False, roll=False):
    """Per-frame eps/eps_true - 1 for one planet (trimmed edges)."""
    if smooth:
        P_pl, P_star = _smooth(P_pl), _smooth(P_star)
    v = _fd_velocity(P_pl - P_star)
    r = np.linalg.norm(P_pl - P_star, axis=1)
    eps = 0.5 * (v ** 2).sum(axis=1) - gm / np.maximum(r, 1e-6)
    if roll:
        from scipy.ndimage import median_filter
        eps = median_filter(eps, size=ROLL, mode="nearest")
    return (eps / eps_true - 1.0)[TRIM:-TRIM]


def _scene_meta(td):
    objs = json.load(open(os.path.join(td, "objects.json")))
    sol = json.load(open(os.path.join(td, "solar.json")))
    return objs, sol


def _report(tag, devs):
    d = np.abs(np.concatenate(devs))
    print(f"{tag}: median |dev| {np.median(d)*100:.2f}% | p90 "
          f"{np.percentile(d, 90)*100:.2f}% | planets {len(devs)}",
          flush=True)
    return d


def stage_exact():
    out = {}
    for name, dirs in SETS.items():
        devs, devs_sm = [], []
        for simdir in dirs:
            n = sum(1 for x in os.listdir(simdir) if x.startswith("traj-"))
            for t in range(n):
                td = os.path.join(simdir, f"traj-{t}")
                objs, sol = _scene_meta(td)
                if sol["binary"]:
                    continue
                pos = np.load(os.path.join(td, "positions.npy"))
                pos = pos[CTX:CTX + STEPS]
                gm = sol["gm_tot"]
                for j, pl in enumerate(sol["planets"]):
                    eps_true = -gm / (2.0 * pl["a"])
                    args = (pos[:, 1 + j], pos[:, 0], gm, eps_true)
                    devs.append(eps_deviation(*args))
                    devs_sm.append(eps_deviation(*args, roll=True))
        d = _report(f"EXACT {name} (raw)", devs)
        ds = _report(f"EXACT {name} (rolled)", devs_sm)
        out[name] = {"raw_med": float(np.median(d)),
                     "raw_p90": float(np.percentile(d, 90)),
                     "rolled_med": float(np.median(ds)),
                     "rolled_p90": float(np.percentile(ds, 90))}
    json.dump(out, open("scratchpad/examples/solar_energy_exact.json", "w"))


def _rt_positions(simdir, n_scenes, ae):
    """GT frames -> VAE roundtrip -> tracked positions, per scene."""
    import torch

    from experiments.solar_metrics import track_bodies
    from src.frameio import load_frames
    got = []
    for t in range(n_scenes):
        td = os.path.join(simdir, f"traj-{t}")
        objs, sol = _scene_meta(td)
        if sol["binary"]:
            continue
        frames = load_frames(td)[CTX:CTX + STEPS]
        pos = np.load(os.path.join(td, "positions.npy"))
        with torch.no_grad():
            xg = (torch.from_numpy(frames).permute(0, 3, 1, 2).float()
                  .div_(255.0).cuda())
            rec = torch.cat([ae.decode(ae.encode(xg[i:i + 512])).clamp(0, 1)
                             for i in range(0, xg.shape[0], 512)])
        rec = (rec.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        tracked, presence = track_bodies(rec, objs, pos[CTX], refine=True)
        got.append((tracked, presence, sol))
    return got


def _dev_from_tracked(tracked, presence, sol, roll):
    gm = sol["gm_tot"]
    devs = []
    for j, pl in enumerate(sol["planets"]):
        if presence[:, 1 + j].min() < 0.5:
            continue
        eps_true = -gm / (2.0 * pl["a"])
        devs.append(eps_deviation(tracked[:, 1 + j], tracked[:, 0], gm,
                                  eps_true, roll=roll))
    return devs


def stage_roundtrip():
    from experiments.rollout import load_run
    ae, dit, cfg = load_run(RUN, "cuda")
    out = {}
    for name, dirs in SETS.items():
        per = max(N_RT // len(dirs), 1)
        scenes = []
        for simdir in dirs:
            scenes += _rt_positions(simdir, per, ae)
        for roll in (False, True):
            devs = []
            for tracked, presence, sol in scenes:
                devs += _dev_from_tracked(tracked, presence, sol, roll)
            tag = "rolled" if roll else "raw"
            d = _report(f"ROUNDTRIP {name} ({tag})", devs)
            curve = np.median(np.abs(np.stack(
                [x for x in devs if len(x) == STEPS - 2 * TRIM])), axis=0)
            out[f"{name}_{tag}"] = {
                "med": float(np.median(d)),
                "p90": float(np.percentile(d, 90)),
                "curve_med": curve.tolist(), "n_planets": len(devs)}
    json.dump(out,
              open("scratchpad/examples/solar_energy_roundtrip.json", "w"))


def stage_diag():
    """Where does the roundtrip eps error come from? Track roundtripped GT
    frames and compare against exact positions: per-planet jitter (radial /
    tangential std), systematic radial bias, and the eps error split into
    its kinetic and potential contributions."""
    from experiments.rollout import load_run
    ae, dit, cfg = load_run(RUN, "cuda")
    rows = []
    for name, dirs in SETS.items():
        scenes = []
        for simdir in dirs:
            got = _rt_positions(simdir, 60, ae)
            n_seen = 0
            for t in range(200):
                td = os.path.join(simdir, f"traj-{t}")
                if not os.path.isdir(td):
                    break
                objs, sol = _scene_meta(td)
                if sol["binary"]:
                    continue
                pos = np.load(os.path.join(td, "positions.npy"))
                scenes.append((got[n_seen], pos[CTX:CTX + STEPS], sol))
                n_seen += 1
                if n_seen == len(got):
                    break
        for (tracked, presence, sol), gt, sol2 in [
                (s[0], s[1], s[2]) for s in scenes]:
            gm = sol["gm_tot"]
            for j, pl in enumerate(sol["planets"]):
                if presence[:, 1 + j].min() < 0.5:
                    continue
                err = tracked[:, 1 + j] - gt[:, 1 + j]
                rel = gt[:, 1 + j] - gt[:, 0]
                r = np.linalg.norm(rel, axis=1)
                u_r = rel / r[:, None]
                u_t = np.stack([-u_r[:, 1], u_r[:, 0]], axis=1)
                e_r = (err * u_r).sum(axis=1)
                e_t = (err * u_t).sum(axis=1)
                v_true = _fd_velocity(gt[:, 1 + j] - gt[:, 0])
                v_trk = _fd_velocity(tracked[:, 1 + j] - tracked[:, 0])
                ke_err = (0.5 * (v_trk ** 2).sum(axis=1)
                          - 0.5 * (v_true ** 2).sum(axis=1))
                r_trk = np.linalg.norm(tracked[:, 1 + j] - tracked[:, 0],
                                       axis=1)
                pe_err = -gm / np.maximum(r_trk, 1e-6) + gm / r
                eps_true = -gm / (2.0 * pl["a"])
                sp = float(np.linalg.norm(v_true, axis=1).mean())
                rows.append((name, pl["cls"], float(pl["size"]), sp,
                             float(e_r.mean()), float(e_r.std()),
                             float(e_t.std()),
                             float(np.median(np.abs(ke_err / eps_true))),
                             float(np.median(np.abs(pe_err / eps_true)))))
    import collections
    by = collections.defaultdict(list)
    for name, cls, size, sp, rb, rs, ts, ke, pe in rows:
        by[(name, cls)].append((size, sp, rb, rs, ts, ke, pe))
    print(f"{'set':7s} {'class':9s} {'n':>4s} {'size':>5s} {'speed':>6s} "
          f"{'radBIAS':>8s} {'radSTD':>7s} {'tanSTD':>7s} "
          f"{'KEerr':>7s} {'PEerr':>7s}")
    for (name, cls), v in sorted(by.items()):
        a = np.array(v)
        print(f"{name:7s} {cls:9s} {len(v):4d} {a[:,0].mean():5.1f} "
              f"{a[:,1].mean():6.2f} {a[:,2].mean():+8.3f} "
              f"{a[:,3].mean():7.3f} {a[:,4].mean():7.3f} "
              f"{a[:,5].mean()*100:6.2f}% {a[:,6].mean()*100:6.2f}%",
              flush=True)


def stage_model():
    import time

    from experiments.deformation import _batched_rollout
    from experiments.rollout import load_run
    from experiments.solar_metrics import track_bodies
    from src.frameio import load_frames
    ae, dit, cfg = load_run(RUN, "cuda")
    out = {}
    for name, dirs in SETS.items():
        trajs = []
        for simdir in dirs:
            n = sum(1 for x in os.listdir(simdir) if x.startswith("traj-"))
            trajs += [os.path.join(simdir, f"traj-{t}") for t in range(n)]
        t0 = time.time()
        devs = []
        for gi in range(0, len(trajs), GROUP):
            ctxs, seeds, metas = [], [], []
            for td in trajs[gi:gi + GROUP]:
                objs, sol = _scene_meta(td)
                if sol["binary"]:
                    continue
                frames = load_frames(td)
                pos = np.load(os.path.join(td, "positions.npy"))
                ctxs.append(frames[:CTX])
                seeds.append(zlib.crc32(os.path.basename(td).encode())
                             & 0x7FFFFFFF)
                metas.append((objs, sol, pos[CTX]))
            if not ctxs:
                continue
            preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS, ns=2)
            for pred, (objs, sol, init) in zip(preds, metas):
                tracked, presence = track_bodies(pred, objs, init,
                                                 refine=True)
                devs += _dev_from_tracked(tracked, presence, sol, True)
            print(f"  {name} {min(gi + GROUP, len(trajs))}/{len(trajs)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        d = _report(f"MODEL {name} (rolled)", devs)
        full = [x for x in devs if len(x) == STEPS - 2 * TRIM]
        arr = np.stack(full)
        out[name] = {"med": float(np.median(d)),
                     "p90": float(np.percentile(d, 90)),
                     "n_planets": len(devs),
                     "curve_mean": arr.mean(axis=0).tolist(),
                     "curve_std": arr.std(axis=0).tolist(),
                     "curve_med_abs": np.median(np.abs(arr),
                                                axis=0).tolist(),
                     "per_planet_med": [float(np.median(x)) for x in full]}
    json.dump(out, open("scratchpad/examples/solar_energy_model.json", "w"))
    print("saved solar_energy_model.json")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "exact"
    if stage == "exact":
        stage_exact()
    elif stage == "roundtrip":
        stage_roundtrip()
    elif stage == "model":
        stage_model()
    elif stage == "diag":
        stage_diag()
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
