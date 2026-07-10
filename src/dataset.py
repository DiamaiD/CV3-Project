import os
import glob
import numpy as np
import torch
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

        self.ranges = ranges
        self.frames = frames.to(cache_device)
        self.device = self.frames.device

    @staticmethod
    def _decode(listing):
        chunks, ranges, start = [], {}, 0
        total = sum(len(p) for p in listing.values())
        print(f"[Cache] Decoding {total} frames into memory (one-time)...")
        for name in sorted(listing.keys()):
            paths = listing[name]
            if not paths:
                continue
            arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            chunks.append(t)
            ranges[name] = (start, t.shape[0])
            start += t.shape[0]
        frames = torch.cat(chunks, dim=0)
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
                 shuffle=False, horizon=1, sample_weights=None):
        self.source = source
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
            yield ctx, tgt
