import os
import glob
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