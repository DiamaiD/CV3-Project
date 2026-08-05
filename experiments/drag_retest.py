"""FINAL drag retest (Viktor-approved, all four counts): arc-fit both arms
under ONE extraction config -- model rollouts vs ceiling (VAE-roundtripped
true physics), per count over base+ext+hot+bank scene sets. Calibrated
model drag = model/ceiling x 0.02 with propagated SE. Plus the g-free
6-param consistency check on model arcs (also an independent gravity
readout). Saves drag_retest.json."""
import json
import os
import zlib

import numpy as np
import torch

from experiments.deformation import _batched_rollout
from experiments.rollout import load_run
from experiments.centroid_track import track_balls_centroid
from experiments.drag_arcfit import collect_c
from src.frameio import load_frames

RUN = "runs/2026_07_30__02_46_32__balls_easy60k_norot_v3"
COUNTS = [3, 5, 7, 9]
CTX = 5


def _n_traj(src):
    return sum(1 for d in os.listdir(src) if d.startswith("traj-"))


def _sets(cnt):
    out = []
    for src, steps in [(f"scratchpad/examples/ood_count_c{cnt}", 300),
                       (f"scratchpad/examples/ood_count_c{cnt}b", 300),
                       (f"scratchpad/examples/ood_hot_c{cnt}", 300),
                       (f"scratchpad/examples/dragbank_c{cnt}", 25)]:
        if os.path.isdir(src):
            out.append((src, _n_traj(src), steps))
    return out


def _med_se(cs):
    return float(np.median(cs)), float(1.253 * cs.std() / np.sqrt(len(cs)))


def main():
    import time
    ae, dit, cfg = load_run(RUN, "cuda")
    out = {}
    for cnt in COUNTS:
        t0 = time.time()
        model_tracks, ceil_tracks = [], []
        for src, n_set, steps in _sets(cnt):
            if not os.path.isdir(src):
                continue
            batch_ctx, batch_seed, batch_meta = [], [], []
            for t in range(n_set):
                td = os.path.join(src, f"traj-{t}")
                frames_np = load_frames(td)
                objs = json.load(open(os.path.join(td, "objects.json")))
                pos = np.load(os.path.join(td, "positions.npy"))
                batch_ctx.append(frames_np[:CTX])
                batch_seed.append(zlib.crc32(td.encode()) & 0x7FFFFFFF)
                batch_meta.append((objs, pos, frames_np))
                if len(batch_ctx) == 96 or t == n_set - 1:
                    preds = _batched_rollout(ae, dit, batch_ctx, batch_seed,
                                             steps)
                    for pred, (objs_, pos_, fr_) in zip(preds, batch_meta):
                        p, mc = track_balls_centroid(pred, objs_,
                                                     pos_[CTX - 1],
                                                     refine=True)
                        model_tracks.append((p, objs_, mc))
                        gt = fr_[CTX:CTX + steps]
                        with torch.no_grad():
                            xg = (torch.from_numpy(gt).permute(0, 3, 1, 2)
                                  .float().div_(255.0).cuda())
                            rec = torch.cat(
                                [ae.decode(ae.encode(xg[i:i + 512]))
                                 .clamp(0, 1)
                                 for i in range(0, xg.shape[0], 512)])
                        rec = (rec.permute(0, 2, 3, 1) * 255).byte().cpu() \
                            .numpy()
                        pc, mcc = track_balls_centroid(rec, objs_,
                                                       pos_[CTX - 1],
                                                       refine=True)
                        ceil_tracks.append((pc, objs_, mcc))
                    batch_ctx, batch_seed, batch_meta = [], [], []
        cs_m, n_m = collect_c(model_tracks)
        cs_c, n_c = collect_c(ceil_tracks)
        cs_g, gs_g, _ = collect_c(model_tracks, free_g=True)
        m_med, m_se = _med_se(cs_m)
        c_med, c_se = _med_se(cs_c)
        cal = m_med / c_med * 0.02
        cal_se = cal * np.sqrt((m_se / m_med) ** 2 + (c_se / c_med) ** 2)
        g6_med = float(np.median(gs_g))
        c6_med = float(np.median(cs_g))
        out[cnt] = {"model": [m_med, m_se, len(cs_m)],
                    "ceiling": [c_med, c_se, len(cs_c)],
                    "calibrated": [float(cal), float(cal_se)],
                    "freeg_c": c6_med, "freeg_g": g6_med,
                    "arcs_model": np.asarray(cs_m).tolist(),
                    "arcs_ceiling": np.asarray(cs_c).tolist(),
                    "arcs_freeg_c": np.asarray(cs_g).tolist(),
                    "arcs_freeg_g": np.asarray(gs_g).tolist()}
        print(f"c{cnt}: model {m_med:.5f}±{m_se:.5f} ({len(cs_m)} arcs) | "
              f"ceiling {c_med:.5f}±{c_se:.5f} ({len(cs_c)}) | CALIBRATED "
              f"{cal:.5f}±{cal_se:.5f} | g-free check: c {c6_med:.5f} "
              f"g {g6_med:+.4f} | {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("scratchpad/examples/drag_retest.json", "w"))
    print("saved drag_retest.json")


if __name__ == "__main__":
    main()
