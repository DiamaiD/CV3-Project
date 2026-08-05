import os
import glob
import numpy as np
import mmap as _mmap
import torch

# Read-only mmap of big caches must be MAP_SHARED: a MAP_PRIVATE mapping reserves
# copy-on-write commit for its full size and ENOMEMs when the file (66 GB full-mix
# cache) exceeds RAM+swap. We never write into loaded caches, so shared is safe.
try:
    torch.serialization.set_default_mmap_options(_mmap.MAP_SHARED)
except Exception:
    pass
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LatentPhysicsDataset(Dataset):
    def __init__(self, traj_dirs, context_len=5, horizon=1, transform=None):
        self.samples = []
        self.context_len = context_len
        self.horizon = horizon
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

        for traj in traj_dirs:
            frames = sorted(glob.glob(os.path.join(traj, "*.png")))
            if len(frames) < context_len + horizon:
                continue
            for i in range(len(frames) - context_len - horizon + 1):
                ctx = frames[i : i + context_len]
                target = frames[i + context_len : i + context_len + horizon]
                self.samples.append((ctx, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx_paths, target_paths = self.samples[idx]
        ctx_frames = torch.stack([self.transform(Image.open(p)) for p in ctx_paths])
        target_frames = torch.stack([self.transform(Image.open(p)) for p in target_paths])
        if self.horizon == 1:
            target_frames = target_frames[0]
        return ctx_frames, target_frames


class FrameCache:
    def __init__(self, traj_dirs, cache_device="cpu", disk_cache_path=None):
        traj_dirs = sorted(traj_dirs)
        dirs = {os.path.basename(t): t for t in traj_dirs}
        sig = {name: list(frame_meta(d)) for name, d in dirs.items()}

        frames = None
        ranges = None
        if disk_cache_path and os.path.exists(disk_cache_path):
            try:
                blob = torch.load(disk_cache_path, map_location="cpu", weights_only=True, mmap=True)
                if blob.get("sig") == sig:
                    frames = blob["frames"]
                    ranges = blob["ranges"]
                    print(f"[Cache] Memory-mapped {frames.shape[0]} frames from {disk_cache_path} "
                          f"(paged from disk on demand, not resident).")
                else:
                    print(f"[Cache] {disk_cache_path} is stale (dataset changed) -- rebuilding.")
            except Exception as e:
                print(f"[Cache] Failed to load {disk_cache_path} ({e}) -- rebuilding.")

        if frames is None:
            scratch = (disk_cache_path + ".decode.tmp") if disk_cache_path else None
            frames, ranges = self._decode(dirs, sig, scratch_path=scratch)
            if disk_cache_path:
                try:
                    torch.save({"frames": frames, "ranges": ranges, "sig": sig}, disk_cache_path)
                    print(f"[Cache] Saved frame cache to {disk_cache_path} "
                          f"({frames.shape[0]} frames, {frames.numel() / 1e9:.2f} GB).")
                    blob = torch.load(disk_cache_path, map_location="cpu", weights_only=True, mmap=True)
                    frames, ranges = blob["frames"], blob["ranges"]
                    print("[Cache] Re-opened as memory-map; decoded tensor released from RAM.")
                    if scratch and os.path.exists(scratch):
                        os.remove(scratch)
                except Exception as e:
                    # keep the decoded tensor (and its scratch backing, if any) --
                    # slower next start, never a crash
                    print(f"[Cache] Cache save/reload failed ({e}) -- using the decoded tensor directly.")

        self.sig = sig
        self.ranges = ranges
        self.frames = frames.to(cache_device)
        self.device = self.frames.device

    @staticmethod
    def _decode(dirs, sig, scratch_path=None):
        """Decode all frames into one uint8 tensor. When the tensor would crowd
        physical RAM (big mixes), it is allocated disk-backed via a scratch file
        and filled through the page cache instead -- same result, bounded RAM.
        Training speed is unaffected either way: the saved cache is reopened as
        a memory-map afterwards in both paths."""
        ranges, start = {}, 0
        names = sorted(dirs.keys())
        total = sum(sig[n][0] for n in names)
        first = load_frames(dirs[names[0]]) if names else None
        H, W = (first.shape[1:3]) if first is not None else (0, 0)
        nbytes = total * 3 * H * W
        phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        disk_backed = scratch_path is not None and nbytes > 0.5 * phys
        if disk_backed:
            print(f"[Cache] Decoding {total} frames into a disk-backed buffer "
                  f"({nbytes / 1e9:.1f} GB > half of {phys / 1e9:.0f} GB RAM)...")
            frames = torch.from_file(scratch_path, shared=True, size=nbytes,
                                     dtype=torch.uint8).view(total, 3, H, W)
        else:
            print(f"[Cache] Decoding {total} frames into memory (one-time)...")
            frames = torch.empty((total, 3, H, W), dtype=torch.uint8)
        for name in names:
            arr = load_frames(dirs[name])
            if arr.shape[0] == 0:
                continue
            frames[start:start + arr.shape[0]] = torch.from_numpy(arr).permute(0, 3, 1, 2)
            ranges[name] = (start, arr.shape[0])
            start += arr.shape[0]
        if start != frames.shape[0]:
            frames = frames[:start]
            if not disk_backed:
                frames = frames.contiguous()
        return frames, ranges

    def build_windows(self, traj_dirs, context_len, horizon):
        ctx_parts, tgt_parts = [], []
        for t in traj_dirs:
            name = os.path.basename(t)
            if name not in self.ranges:
                continue
            start, count = self.ranges[name]
            num = count - context_len - horizon + 1
            if num <= 0:
                continue
            base = start + torch.arange(num).unsqueeze(1)
            ctx_parts.append(base + torch.arange(context_len).unsqueeze(0))
            tgt_parts.append(base + context_len + torch.arange(horizon).unsqueeze(0))

        if not ctx_parts:
            empty = torch.empty((0, context_len), dtype=torch.long)
            return empty, torch.empty((0, horizon), dtype=torch.long)

        ctx_index = torch.cat(ctx_parts, dim=0).to(self.device)
        tgt_index = torch.cat(tgt_parts, dim=0).to(self.device)
        return ctx_index, tgt_index


class CachedLoader:
    def __init__(self, source, ctx_index, tgt_index, batch_size, device,
                 shuffle=False, horizon=1, sample_weights=None, aux=None):
        self.source = source
        self.aux = aux  # optional per-frame target tensor, indexed like source
        self.ctx_index = ctx_index
        self.tgt_index = tgt_index
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.horizon = horizon
        self.sample_weights = sample_weights
        self.is_uint8 = source.dtype == torch.uint8
        self.S = ctx_index.shape[0]

    def __len__(self):
        return (self.S + self.batch_size - 1) // self.batch_size

    def _to_float(self, x):
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        x = x.float()
        if self.is_uint8:
            x = x.div_(255.0)
        return x

    def __iter__(self):
        if self.shuffle and self.sample_weights is not None:
            order = torch.multinomial(self.sample_weights, self.S, replacement=True).to(self.ctx_index.device)
        else:
            order = (torch.randperm(self.S, device=self.ctx_index.device) if self.shuffle
                     else torch.arange(self.S, device=self.ctx_index.device))
        for i in range(0, self.S, self.batch_size):
            rows = order[i : i + self.batch_size]
            ctx = self._to_float(self.source[self.ctx_index[rows]])
            tgt = self._to_float(self.source[self.tgt_index[rows]])
            if self.horizon == 1:
                tgt = tgt[:, 0]
            if self.aux is not None:
                aux = self.aux[self.ctx_index[rows]].to(self.device, non_blocking=True).float()
                yield ctx, tgt, aux
            else:
                yield ctx, tgt


# ---------------------------------------------------------------------------
# Streaming frame layer: no monolithic decode-to-disk cache. Frames are decoded
# on demand by a persistent process pool (frame decode holds the GIL, so threads don't
# scale -- processes give ~36 us/frame, 8x a single core). Full-dataset passes
# stream in trajectory chunks with the next chunk prefetched while the current
# one trains, so decode hides behind GPU compute; RAM stays bounded (~2 chunks)
# regardless of dataset size, and no frame cache ever touches disk.
# ---------------------------------------------------------------------------
import atexit as _atexit
from multiprocessing import get_context as _get_context
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from src.frameio import load_frames, frame_meta


def _decode_traj(traj_dir):
    """Load one trajectory as (T, H, W, 3) uint8 RGB (packed .npz or legacy PNGs).
    Module-level so it pickles cheaply to spawn workers."""
    return load_frames(traj_dir)


_POOL = None
_POOL_N = None


def _get_pool(n_workers):
    """Lazily create one shared spawn pool, reused by every loader and closed at
    exit. Spawn (not fork) is required under CUDA; workers only import this module."""
    global _POOL, _POOL_N
    if _POOL is None or _POOL_N != n_workers:
        if _POOL is not None:
            _POOL.terminate()
        _POOL = _get_context("spawn").Pool(n_workers)
        _POOL_N = n_workers
        _atexit.register(lambda: _POOL is not None and _POOL.terminate())
    return _POOL


def _windows_for(ranges, context_len, horizon):
    """(ctx_index, tgt_index) CPU long tensors into a tensor laid out per `ranges`
    (name -> (start, count)). Same window math as FrameCache.build_windows."""
    ctx_parts, tgt_parts = [], []
    for name, (start, count) in ranges.items():
        num = count - context_len - horizon + 1
        if num <= 0:
            continue
        base = start + torch.arange(num).unsqueeze(1)
        ctx_parts.append(base + torch.arange(context_len).unsqueeze(0))
        tgt_parts.append(base + context_len + torch.arange(horizon).unsqueeze(0))
    if not ctx_parts:
        return (torch.empty((0, context_len), dtype=torch.long),
                torch.empty((0, horizon), dtype=torch.long))
    return torch.cat(ctx_parts, 0).long(), torch.cat(tgt_parts, 0).long()


class FrameStore:
    """Metadata-only view of a frame dataset: trajectory dirs, per-traj GLOBAL
    ranges (name -> (start, count)), a change signature and frame count. Holds no
    pixels and never decodes a frame at init (counts come from frame_meta). Full-
    dataset passes go through .stream()/.iter_frames(); small trajectory subsets
    decode into RAM via .subset()."""

    def __init__(self, traj_dirs):
        traj_dirs = sorted(traj_dirs)
        self.dirs = {os.path.basename(t): t for t in traj_dirs}
        self.ranges, self.sig, start = {}, {}, 0
        for name in sorted(self.dirs):
            n, mtime = frame_meta(self.dirs[name])
            if n == 0:
                continue
            self.ranges[name] = (start, n)
            self.sig[name] = [n, mtime]
            start += n
        self.n_frames = start
        first = next(iter(self.ranges), None)
        self.H, self.W = (load_frames(self.dirs[first]).shape[1:3] if first else (0, 0))
        print(f"[Store] {len(self.ranges)} trajectories, {self.n_frames} frames "
              f"(streamed on demand, no disk cache).")

    def build_windows(self, traj_dirs, context_len, horizon):
        names = [os.path.basename(t) for t in sorted(traj_dirs)]
        sub = {n: self.ranges[n] for n in names if n in self.ranges}
        return _windows_for(sub, context_len, horizon)

    def stream(self, traj_dirs, context_len, horizon, batch_size, device, shuffle=False,
               chunk_trajs=1000, n_workers=16, aux=None):
        return StreamingLoader(self, traj_dirs, context_len, horizon, batch_size, device,
                               shuffle, chunk_trajs, n_workers, aux)

    def iter_frames(self, batch_size=512, chunk_trajs=1000, n_workers=16):
        """Yield every frame as uint8 (B, 3, H, W) batches in GLOBAL order, so
        concatenated encoder outputs line up with the global ranges. Prefetched."""
        names = sorted(self.ranges.keys())
        chunks = [names[i:i + chunk_trajs] for i in range(0, len(names), chunk_trajs)]
        pool = _get_pool(n_workers)

        def decode(cn):
            arrays = pool.map(_decode_traj, [self.dirs[n] for n in cn])
            return torch.cat([torch.from_numpy(a).permute(0, 3, 1, 2) for a in arrays])

        with _ThreadPoolExecutor(max_workers=1) as pf:
            fut = pf.submit(decode, chunks[0]) if chunks else None
            for k in range(len(chunks)):
                big = fut.result()
                fut = pf.submit(decode, chunks[k + 1]) if k + 1 < len(chunks) else None
                for i in range(0, big.shape[0], batch_size):
                    yield big[i:i + batch_size]

    def subset(self, traj_dirs):
        return ResidentFrames(self, traj_dirs)


class StreamingLoader:
    """Re-iterable (ctx, tgt[, aux]) batch loader that streams trajectory chunks
    through the process pool, shuffling windows within each chunk, with the next
    chunk decoded in the background while the current one is consumed. Output
    shapes match CachedLoader: ctx (B, ctx, 3, H, W), tgt (B, 3, H, W) if
    horizon == 1 else (B, horizon, 3, H, W)."""

    def __init__(self, store, traj_dirs, context_len, horizon, batch_size, device,
                 shuffle=False, chunk_trajs=1000, n_workers=16, aux=None):
        self.store = store
        self.names = [os.path.basename(t) for t in sorted(traj_dirs) if os.path.basename(t) in store.ranges]
        self.context_len = context_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.chunk_trajs = chunk_trajs
        self.n_workers = n_workers
        self.aux = aux
        total = sum(max(0, store.ranges[n][1] - context_len - horizon + 1) for n in self.names)
        self.S = total
        self._len = (total + batch_size - 1) // batch_size

    def __len__(self):
        return self._len

    def _decode_chunk(self, chunk_names):
        pool = _get_pool(self.n_workers)
        arrays = pool.map(_decode_traj, [self.store.dirs[n] for n in chunk_names])
        parts, local_ranges, start, gparts = [], {}, 0, ([] if self.aux is not None else None)
        for name, arr in zip(chunk_names, arrays):
            cnt = arr.shape[0]
            parts.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
            local_ranges[name] = (start, cnt)
            if gparts is not None:
                gs = self.store.ranges[name][0]
                gparts.append(torch.arange(gs, gs + cnt))
            start += cnt
        frames = torch.cat(parts) if parts else torch.empty((0, 3, self.store.H, self.store.W),
                                                            dtype=torch.uint8)
        lctx, ltgt = _windows_for(local_ranges, self.context_len, self.horizon)
        cg = torch.cat(gparts) if gparts is not None else None
        return frames, lctx, ltgt, cg

    def _iter_chunk(self, chunk):
        frames, lctx, ltgt, cg = chunk
        S = lctx.shape[0]
        order = torch.randperm(S) if self.shuffle else torch.arange(S)
        for i in range(0, S, self.batch_size):
            rows = order[i:i + self.batch_size]
            ci, ti = lctx[rows], ltgt[rows]
            ctx = frames[ci].to(self.device, non_blocking=True).float().div_(255.0)
            tgt = frames[ti].to(self.device, non_blocking=True).float().div_(255.0)
            if self.horizon == 1:
                tgt = tgt[:, 0]
            if self.aux is not None:
                aux = self.aux[cg[ci]].to(self.device, non_blocking=True).float()
                yield ctx, tgt, aux
            else:
                yield ctx, tgt

    def __iter__(self):
        names = list(self.names)
        if self.shuffle:
            perm = torch.randperm(len(names)).tolist()
            names = [names[i] for i in perm]
        chunks = [names[i:i + self.chunk_trajs] for i in range(0, len(names), self.chunk_trajs)]
        with _ThreadPoolExecutor(max_workers=1) as pf:
            fut = pf.submit(self._decode_chunk, chunks[0]) if chunks else None
            for k in range(len(chunks)):
                chunk = fut.result()
                fut = pf.submit(self._decode_chunk, chunks[k + 1]) if k + 1 < len(chunks) else None
                yield from self._iter_chunk(chunk)


class ResidentFrames:
    """Decodes a (small) set of trajectories into RAM but presents GLOBAL frame
    indexing, so it drop-in replaces a full FrameCache for consumers that
    pair frames with the global latent cache (probe, collision eval, decoder
    retrain). frame_cache.frames[global_idx] and .build_windows() both work."""

    def __init__(self, store, traj_dirs, n_workers=16):
        self.ranges = store.ranges
        self.n_frames = store.n_frames
        names = [os.path.basename(t) for t in sorted(traj_dirs) if os.path.basename(t) in store.ranges]
        gmap = np.full(store.n_frames, -1, dtype=np.int64)
        parts, row = [], 0
        if names:
            pool = _get_pool(n_workers)
            arrays = pool.map(_decode_traj, [store.dirs[n] for n in names])
            for name, arr in zip(names, arrays):
                cnt = arr.shape[0]
                gs = store.ranges[name][0]
                gmap[gs:gs + cnt] = np.arange(row, row + cnt)
                parts.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
                row += cnt
        self._frames = (torch.cat(parts) if parts
                        else torch.empty((0, 3, store.H, store.W), dtype=torch.uint8))
        self._gmap = torch.from_numpy(gmap)

    @property
    def frames(self):
        return self

    @property
    def shape(self):
        return (self.n_frames,) + tuple(self._frames.shape[1:])

    def __getitem__(self, idx):
        return self._frames[self._gmap[idx]]

    def build_windows(self, traj_dirs, context_len, horizon):
        names = [os.path.basename(t) for t in sorted(traj_dirs)]
        sub = {n: self.ranges[n] for n in names if n in self.ranges}
        return _windows_for(sub, context_len, horizon)


def build_state_targets(frame_cache, data_dir, grid=8, frame_px=64):
    """Per-frame position targets for the VAE state-alignment loss: for each
    grid cell, [presence, dx, dy] of the ball center inside it (dx/dy = offset
    from the cell center in cell units, 0 where empty). Built from each traj's
    positions.npy; indexed identically to the global frame order."""
    cell = frame_px // grid
    N = getattr(frame_cache, "n_frames", None) or frame_cache.frames.shape[0]
    S = torch.zeros((N, 3, grid, grid), dtype=torch.float16)
    for name, (start, count) in frame_cache.ranges.items():
        pos = np.load(os.path.join(data_dir, name, "positions.npy"))
        T = min(count, pos.shape[0])
        for t in range(T):
            for x, y in pos[t]:
                col, row = x - 0.5, (frame_px - 0.5) - y
                j, i = int(col // cell), int(row // cell)
                if 0 <= i < grid and 0 <= j < grid:
                    S[start + t, 0, i, j] = 1.0
                    S[start + t, 1, i, j] = (col - (cell * j + cell / 2 - 0.5)) / cell
                    S[start + t, 2, i, j] = (row - (cell * i + cell / 2 - 0.5)) / cell
    print(f"[StateTargets] Built {N} frame targets ({S.numel() * 2 / 1e9:.2f} GB).")
    return S
