import os
import glob
import importlib
import numpy as np
from multiprocessing import get_context

# Viktor's box: 16 cores / 32 threads -- generation always uses all of them.
N_WORKERS = 32


def _worker(args):
    module, fn_name, kwargs, seed = args
    os.environ["TQDM_DISABLE"] = "1"  # 32 interleaved progress bars are noise
    np.random.seed(seed)
    fn = getattr(importlib.import_module(module), fn_name)
    fn(**kwargs)


def generate_parallel(module, fn_name, data_dir, n_trajectories, seed=None,
                      n_workers=N_WORKERS, progress_cb=None, **kwargs):
    """Run a trajectory generator across worker processes. The generators are
    chunk-safe via start_idx (every traj dir is independent); each worker gets
    its own RNG seed -- forked workers would otherwise inherit one shared numpy
    state and generate 32 identical datasets. Progress is tracked by counting
    finished trajectories (positions.npy is each trajectory's last artifact)."""
    n_workers = max(1, min(int(n_workers), int(n_trajectories)))
    counts = [n_trajectories // n_workers + (1 if i < n_trajectories % n_workers else 0)
              for i in range(n_workers)]
    seeds = np.random.SeedSequence(seed).generate_state(n_workers)
    tasks, start = [], 0
    for i, c in enumerate(counts):
        if c == 0:
            continue
        kw = dict(kwargs, data_dir=data_dir, n_trajectories=c, start_idx=start,
                  progress_cb=None)
        tasks.append((module, fn_name, kw, int(seeds[i]) % (2 ** 32)))
        start += c

    ctx = get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        result = pool.map_async(_worker, tasks)
        while not result.ready():
            result.wait(1.0)
            if progress_cb is not None:
                done = len(glob.glob(os.path.join(data_dir, "traj-*", "positions.npy")))
                progress_cb(done, n_trajectories)
        result.get()  # surface worker exceptions
    if progress_cb is not None:
        progress_cb(n_trajectories, n_trajectories)


def generate_bouncing_parallel(**kwargs):
    return generate_parallel("environments.env_bouncing", "generate_bouncing_data", **kwargs)


def generate_shapes_parallel(**kwargs):
    return generate_parallel("environments.env_shapes", "generate_shapes_data", **kwargs)


# Production backends (Chipmunk2D via pymunk) -- see environments/env_pymunk.py.
def generate_shapes_pymunk_parallel(**kwargs):
    return generate_parallel("environments.env_pymunk", "generate_shapes_pymunk", **kwargs)


def generate_tower_pymunk_parallel(**kwargs):
    return generate_parallel("environments.env_pymunk", "generate_tower_pymunk", **kwargs)


def generate_balls2_pymunk_parallel(**kwargs):
    return generate_parallel("environments.env_pymunk", "generate_balls2_pymunk", **kwargs)


def generate_solar_parallel(**kwargs):
    return generate_parallel("environments.env_solar", "generate_solar", **kwargs)
