"""Model restitution on the engineered bounce banks: roll the balls-60k
model on the bank scenes (full 7/9-ball OOD context, guaranteed clean bounce
geometry), track with the centroid tracker, extract the target material's
restitution. Saves bank_restitution.json."""
import json
import os
import zlib

import numpy as np

from environments.env_bouncing import MATERIALS
from experiments.deformation import _batched_rollout
from experiments.rollout import load_run
from experiments.centroid_track import track_balls_centroid
from experiments.constants_ood import _boot_ci, _collect
from src.frameio import load_frames

RUN = "runs/2026_07_30__02_46_32__balls_easy60k_norot_v3"
BANKS = [("Steel", 7), ("Sponge", 7), ("Steel", 9), ("Sponge", 9)]
CTX = 5
STEPS = 85
N = 250


def main():
    from multiprocessing import get_context
    ae, dit, cfg = load_run(RUN, "cuda")
    out = {}
    for mat, cnt in BANKS:
        bankdir = f"scratchpad/examples/bank_{mat}_c{cnt}"
        trajs = [os.path.join(bankdir, f"traj-{i}") for i in range(N)]
        ctxs, seeds, metas = [], [], []
        for td in trajs:
            frames_np = load_frames(td)
            objs = json.load(open(os.path.join(td, "objects.json")))
            pos = np.load(os.path.join(td, "positions.npy"))
            ctxs.append(frames_np[:CTX])
            seeds.append(zlib.crc32(td.encode()) & 0x7FFFFFFF)
            metas.append((objs, pos[CTX - 1], pos))
        preds = _batched_rollout(ae, dit, ctxs, seeds, STEPS)
        m_pay, g_pay = [], []
        for pred, (objs, init, pos) in zip(preds, metas):
            p, mc = track_balls_centroid(pred, objs, init, refine=True)
            m_pay.append((p, objs, mc))
            g_pay.append((pos[CTX:CTX + STEPS], objs, None))
        res = {}
        with get_context("spawn").Pool(12) as pool:
            for tag, pays in (("model", m_pay), ("control", g_pay)):
                got = pool.map(_collect, pays)
                vals = [r for g_ in got for m_, r in g_[1] if m_ == mat]
                lo, hi = _boot_ci(vals) if len(vals) >= 30 else (float("nan"),) * 2
                res[tag] = {"med": float(np.median(vals)), "n": len(vals),
                            "ci": [lo, hi]}
        out[f"{mat}_c{cnt}"] = res
        print(f"{mat} c{cnt}: model {res['model']['med']:.4f} "
              f"[{res['model']['ci'][0]:.3f},{res['model']['ci'][1]:.3f}] "
              f"(n={res['model']['n']}) | control {res['control']['med']:.4f} "
              f"(n={res['control']['n']}) | true "
              f"{MATERIALS[mat]['restitution']:.2f}", flush=True)
    json.dump(out, open("scratchpad/examples/bank_restitution.json", "w"))


if __name__ == "__main__":
    main()
