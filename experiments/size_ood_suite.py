"""Size-OOD battery for the balls-60k model (trained on r5-8): small r3-4 and
large r9-12 scene sets vs the in-distribution ballsv3 reference. Same
validated stack as the count suite (imports its worker); survival normalized
by per-scene ball counts (2-5 per scene here). Numbers only (per Viktor)."""
import json
import os
import zlib

import numpy as np

from experiments.deformation import _batched_rollout, detect_outline
from experiments.rollout import load_run
from experiments.count_ood_suite import _gt_count, _score
from experiments.energycurve_norot import energy_series, obj_mass_low
from src.frameio import load_frames

RUN = "runs/2026_07_30__02_46_32__balls_easy60k_norot_v3"
SETS = [("small r3-4 (OOD)", "scratchpad/examples/ood_sizeeval_small"),
        ("large r9-12 (OOD)", "scratchpad/examples/ood_sizeeval_large"),
        ("ref r5-8 (in dist)", "scratchpad/examples/energy_sim500_ballsv3")]
CTX = 5
STEPS = 300
N_SCENES = 500
GROUP = 128


def main():
    import time
    import torch
    from multiprocessing import get_context

    # size sets only: a renormalized (resized-toward-trained) ball must
    # still count as present; see shape_count._count_pair
    os.environ["CV3_WIDEN_BALL_BANDS"] = "1"
    ae, dit, cfg = load_run(RUN, "cuda")
    out = {}
    for label, simdir in SETS:
        trajs = [os.path.join(simdir, f"traj-{t}") for t in range(N_SCENES)]
        ext = simdir + "b"
        if os.path.isdir(ext):
            trajs += [os.path.join(ext, f"traj-{t}") for t in range(N_SCENES)]
        with get_context("spawn").Pool(24) as pool:
            g = pool.map(_gt_count, trajs)
        print(f"{label}: GT counting control deficit-frames "
              f"{sum(x[0] for x in g)} surplus-frames {sum(x[1] for x in g)}",
              flush=True)

        t0 = time.time()
        pool = get_context("spawn").Pool(20)
        pending, results, n_objs = None, [], []
        for gi in range(0, len(trajs), GROUP):
            group = trajs[gi:gi + GROUP]
            ctxs, seeds, metas, gts = [], [], [], []
            for td in group:
                frames_np = load_frames(td)
                objs = json.load(open(os.path.join(td, "objects.json")))
                ang = np.load(os.path.join(td, "angles.npy"))
                pos = np.load(os.path.join(td, "positions.npy"))
                vel = np.load(os.path.join(td, "velocities.npy"))
                thetas = ang[0, :, 0]
                ml = [obj_mass_low(o, thetas[i]) for i, o in enumerate(objs)]
                masses, lows = [m for m, _ in ml], [l for _, l in ml]
                E_sim = energy_series(pos, vel, masses, lows)
                metas.append((objs, thetas, pos[CTX - 1],
                              detect_outline(frames_np[0], objs, pos[0],
                                             thetas),
                              masses, lows, E_sim[CTX],
                              E_sim[CTX:CTX + STEPS]))
                n_objs.append(len(objs))
                ctxs.append(frames_np[:CTX])
                seeds.append(zlib.crc32(os.path.basename(td).encode()) & 0x7FFFFFFF)
                gts.append(frames_np[CTX:CTX + STEPS])
            preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS)
            flat = np.concatenate(gts)
            with torch.no_grad():
                xg = (torch.from_numpy(flat).permute(0, 3, 1, 2).float()
                      .div_(255.0).cuda())
                rec = torch.cat([ae.decode(ae.encode(xg[i:i + 1024])).clamp(0, 1)
                                 for i in range(0, xg.shape[0], 1024)])
            rec = (rec.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
            payloads, ofs = [], 0
            for pred, meta, gt in zip(preds, metas, gts):
                payloads.append((pred[:len(gt)], rec[ofs:ofs + len(gt)], *meta))
                ofs += len(gt)
            if pending is not None:
                results.extend(pending.get())
            pending = pool.map_async(_score, payloads)
            print(f"  {min(gi + GROUP, len(trajs))}/{len(trajs)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        results.extend(pending.get())
        pool.close()
        pool.join()

        deficit = np.stack([r[0] for r in results])
        nvec = np.array(n_objs, dtype=float)[:, None]
        surv = 1.0 - (deficit / nvec).mean(axis=0)
        dsum = np.stack([r[2] for r in results]).sum(axis=0)
        dsq = np.stack([r[3] for r in results]).sum(axis=0)
        dn = np.stack([r[4] for r in results]).sum(axis=0).astype(float)
        dmean = dsum / np.maximum(dn, 1)
        dstd = np.sqrt(np.maximum(dsq / np.maximum(dn, 1) - dmean ** 2, 0.0))
        e_mod = np.stack([r[5] for r in results])
        e_sim = np.stack([r[6] for r in results])
        out[label] = {"survival": surv.tolist(),
                      "deform_mean": dmean.tolist(),
                      "deform_std": dstd.tolist(),
                      "e_mod_mean": e_mod.mean(axis=0).tolist(),
                      "e_mod_std": e_mod.std(axis=0).tolist(),
                      "e_sim_mean": e_sim.mean(axis=0).tolist(),
                      "deform_n_curve": dn.tolist()}
        print(f"{label}: survival f80 {surv[79]*100:.1f}% f300 "
              f"{surv[-1]*100:.1f}% | deform f80 {dmean[79]:.3f} f300 "
              f"{dmean[-1]:.3f} (std {dstd[-1]:.3f}) | energy gap f300 "
              f"{(e_mod[:, -1] - e_sim[:, -1]).mean():+.3f} | "
              f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("scratchpad/examples/size_ood.json", "w"))
    print("saved size_ood.json")


if __name__ == "__main__":
    main()
