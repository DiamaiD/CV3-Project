"""GT-frame control for the counting metric: run the counter on REAL sim
frames, where the right answer is exactly zero missing and zero surplus.
Saves gt_count.json (curves + stats) for the shape_count chart."""
import json
import os

import numpy as np

from src.frameio import load_frames


def gt_count(td):
    from experiments.shape_count import _count_pair
    from experiments.deformation import detect_outline
    frames = load_frames(td)
    objs = json.load(open(os.path.join(td, "objects.json")))
    ang = np.load(os.path.join(td, "angles.npy"))
    pos = np.load(os.path.join(td, "positions.npy"))
    outline = detect_outline(frames[0], objs, pos[0], ang[0, :, 0])
    return _count_pair((frames[5:305], objs, ang[0, :, 0], outline))


def main():
    import sys
    from multiprocessing import get_context
    simdir = (sys.argv[1] if len(sys.argv) > 1
              else "scratchpad/examples/energy_sim500_v3")
    tag = "" if simdir.endswith("_v3") else "_v1n"
    trajs = [f"{simdir}/traj-{t}" for t in range(500)]
    with get_context("spawn").Pool(24) as pool:
        curves = pool.map(gt_count, trajs)
    m = np.stack([c[0] for c in curves])
    sp = np.stack([c[1] for c in curves])
    out = {"mean_missing": m.mean(axis=0).tolist(),
           "mean_surplus": sp.mean(axis=0).tolist(),
           "scenes_missing_sustained": float((m[:, -60:].min(axis=1) >= 1).mean()),
           "scenes_surplus_sustained": float((sp[:, -60:].min(axis=1) >= 1).mean()),
           "frames_deficit": int((m > 0).sum()), "frames_surplus": int((sp > 0).sum())}
    json.dump(out, open(f"scratchpad/examples/gt_count{tag}.json", "w"))
    bad_d = [i for i in range(500) if m[i].max() >= 1]
    bad_s = [i for i in range(500) if sp[i].max() >= 1]
    print(f"GT baseline: mean missing f80 {m[:, 79].mean():.4f} | "
          f"f300 {m[:, -1].mean():.4f} | sustained-loss scenes "
          f"{out['scenes_missing_sustained']*100:.1f}% | mean surplus f300 "
          f"{sp[:, -1].mean():.4f} | sustained-surplus scenes "
          f"{out['scenes_surplus_sustained']*100:.1f}%", flush=True)
    print(f"frames with deficit {out['frames_deficit']} | surplus "
          f"{out['frames_surplus']} (of 150000)")
    print(f"scenes with any deficit {bad_d} | any surplus {bad_s}")


if __name__ == "__main__":
    main()
