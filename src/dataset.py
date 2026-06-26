import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LatentPhysicsDataset(Dataset):
    def __init__(self, traj_dirs, context_len=5, horizon=1, transform=None):
        """
        traj_dirs:   List of paths to trajectory folders (e.g. ['data/traj-0', 'data/traj-1'])
        horizon:     Number of future frames to return as the target. horizon=1 gives the
                     classic single-step target (shape (C,H,W)); horizon>1 returns the next
                     `horizon` frames (shape (horizon,C,H,W)) for multi-step rollout training.
        """
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
    """Decode every PNG of a dataset once into a single contiguous uint8 tensor.

    Frames are stored in a canonical (sorted-by-trajectory-name) order so the on-disk cache is
    stable across runs regardless of how trajectories are later shuffled into train/val/test splits.
    `build_windows` then turns any subset of trajectories into absolute-frame index tensors, and
    `CachedLoader` gathers batches straight from memory (RAM or VRAM) with no per-epoch disk I/O.
    """

    def __init__(self, traj_dirs, cache_device="cpu", disk_cache_path=None):
        traj_dirs = sorted(traj_dirs)
        # name -> sorted list of frame paths. Signature is name -> (frame count, first-frame mtime).
        # The mtime catches an in-place REGENERATION (same names + counts but new pixels, e.g.
        # toggling anti-aliasing / sub-pixel rendering): without it a content change would silently
        # reuse the stale cache and quietly invalidate the whole run.
        listing = {os.path.basename(t): sorted(glob.glob(os.path.join(t, "*.png"))) for t in traj_dirs}
        sig = {name: [len(paths), os.path.getmtime(paths[0]) if paths else 0.0]
               for name, paths in listing.items()}

        frames = None
        ranges = None
        if disk_cache_path and os.path.exists(disk_cache_path):
            try:
                blob = torch.load(disk_cache_path, map_location="cpu", weights_only=True)
                if blob.get("sig") == sig:
                    frames = blob["frames"]
                    ranges = blob["ranges"]
                    print(f"[Cache] Loaded {frames.shape[0]} frames from {disk_cache_path}.")
                else:
                    print(f"[Cache] {disk_cache_path} is stale (dataset changed) -- rebuilding.")
            except Exception as e:
                print(f"[Cache] Failed to load {disk_cache_path} ({e}) -- rebuilding.")

        if frames is None:
            frames, ranges = self._decode(listing)
            if disk_cache_path:
                try:
                    torch.save({"frames": frames, "ranges": ranges, "sig": sig}, disk_cache_path)
                    print(f"[Cache] Saved frame cache to {disk_cache_path} "
                          f"({frames.shape[0]} frames, {frames.numel() / 1e9:.2f} GB).")
                except Exception as e:
                    print(f"[Cache] Could not save cache to {disk_cache_path}: {e}")

        self.ranges = ranges                 # name -> (start, count) into self.frames
        self.frames = frames.to(cache_device)  # (M, C, H, W) uint8
        self.device = self.frames.device

    @staticmethod
    def _decode(listing):
        """Decode all PNGs (canonical order) into one (M, C, H, W) uint8 tensor + name->range map."""
        chunks, ranges, start = [], {}, 0
        total = sum(len(p) for p in listing.values())
        print(f"[Cache] Decoding {total} frames into memory (one-time)...")
        for name in sorted(listing.keys()):
            paths = listing[name]
            if not paths:
                continue
            # np.array(Image.open(p).convert("RGB")) -> (H, W, 3) uint8; permute to (3, H, W).
            # Dividing by 255 later reproduces torchvision.ToTensor exactly.
            arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])  # (N,H,W,3)
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()                 # (N,3,H,W) uint8
            chunks.append(t)
            ranges[name] = (start, t.shape[0])
            start += t.shape[0]
        frames = torch.cat(chunks, dim=0)
        return frames, ranges

    def build_windows(self, traj_dirs, context_len, horizon):
        """Return (ctx_index, tgt_index) int64 tensors of absolute frame indices for these trajs.

        ctx_index: (S, context_len), tgt_index: (S, horizon). Mirrors the sliding-window logic of
        LatentPhysicsDataset but produces indices into the shared frame/latent cache instead of paths.
        """
        ctx_parts, tgt_parts = [], []
        for t in traj_dirs:
            name = os.path.basename(t)
            if name not in self.ranges:
                continue
            start, count = self.ranges[name]
            num = count - context_len - horizon + 1
            if num <= 0:
                continue
            base = start + torch.arange(num).unsqueeze(1)                     # (num, 1)
            ctx_parts.append(base + torch.arange(context_len).unsqueeze(0))   # (num, context_len)
            tgt_parts.append(base + context_len + torch.arange(horizon).unsqueeze(0))  # (num, horizon)

        if not ctx_parts:
            empty = torch.empty((0, context_len), dtype=torch.long)
            return empty, torch.empty((0, horizon), dtype=torch.long)

        ctx_index = torch.cat(ctx_parts, dim=0).to(self.device)
        tgt_index = torch.cat(tgt_parts, dim=0).to(self.device)
        return ctx_index, tgt_index


class CachedLoader:
    """Iterate (context, target) batches gathered from an in-memory cache tensor.

    source:    (M, *feat) cache tensor -- uint8 frames or float16 latents -- on `cache_device`.
    ctx_index: (S, context_len) int64 indices into `source` (same device as source).
    tgt_index: (S, horizon) int64.

    Yields ctx (B, context_len, *feat) and target (B, *feat) when horizon==1 else (B, horizon, *feat),
    as float32 on `device`. uint8 sources are scaled by 1/255 to reproduce ToTensor; other dtypes are
    just upcast to float32. Shuffling permutes the sample rows each epoch.
    """

    def __init__(self, source, ctx_index, tgt_index, batch_size, device,
                 shuffle=False, horizon=1):
        self.source = source
        self.ctx_index = ctx_index
        self.tgt_index = tgt_index
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.horizon = horizon
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
        order = (torch.randperm(self.S, device=self.ctx_index.device) if self.shuffle
                 else torch.arange(self.S, device=self.ctx_index.device))
        for i in range(0, self.S, self.batch_size):
            rows = order[i : i + self.batch_size]
            ctx = self._to_float(self.source[self.ctx_index[rows]])   # (B, T, *feat)
            tgt = self._to_float(self.source[self.tgt_index[rows]])   # (B, K, *feat)
            if self.horizon == 1:
                tgt = tgt[:, 0]
            yield ctx, tgt
