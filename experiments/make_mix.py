"""Build a mixed training dataset as a directory of symlinks.

Takes N source datasets, drops each source's held-out test tail (the last 10%
under the seed-42 split convention of experiments.rollout.test_split -- the same
tail vae_bench and surrogate evaluate on), and symlinks the remaining trajs into
one directory with unique names traj-0..traj-K. Training on the mix can then
never touch any source's eval trajectories, and every existing tool (FrameCache,
GUI, caches) works unchanged because the mix looks like a normal dataset.

  python -m experiments.make_mix --out data/mix3_full \
      --sources data/balls_20k,data/shapes_20k,data/tower_20k [--cap 6600]

--cap N keeps only the first N usable trajs per source (screening budgets).
"""
import argparse
import json
import os

from experiments.rollout import test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", required=True, help="comma-separated dataset dirs")
    ap.add_argument("--cap", type=int, default=0, help="max usable trajs per source (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sources = [s.strip().rstrip("/") for s in args.sources.split(",")]
    os.makedirs(args.out, exist_ok=False)   # refuse to overwrite an existing mix

    manifest = {"sources": {}, "cap": args.cap, "seed": args.seed}
    k = 0
    for src in sources:
        all_trajs, test = test_split(src, args.seed)
        n_tr, n_va = int(len(all_trajs) * 0.8), int(len(all_trajs) * 0.1)
        usable = all_trajs[: n_tr + n_va]          # train+val slices only
        if args.cap > 0:
            usable = usable[: args.cap]
        for td in usable:
            os.symlink(os.path.relpath(os.path.abspath(td), os.path.abspath(args.out)),
                       os.path.join(args.out, f"traj-{k}"))
            k += 1
        manifest["sources"][src] = {"total": len(all_trajs), "test_tail": len(test),
                                    "linked": len(usable)}
        print(f"{src}: {len(usable)} of {len(all_trajs)} trajs linked "
              f"({len(test)} test-tail excluded)")

    with open(os.path.join(args.out, "mix_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{args.out}: {k} trajs total")


if __name__ == "__main__":
    main()
