"""Trajectory frame I/O.

Frames are packed into ONE compressed .npz per trajectory (key "f", a
(T, H, W, 3) uint8 RGB array) instead of 100 separate PNGs. Because the whole
stack is compressed as a single stream, the frame-to-frame redundancy (a mostly
static white background) is exploited: ~3x smaller on disk than per-frame PNGs,
~4x faster to read, and ~100x fewer files. All measured on real data.

Backward compatible: load_frames reads the packed file when present, else falls
back to the legacy frame_XXX.png sequence, so old datasets keep working. RGB
channel order matches what PIL/cv2-BGR2RGB returned from the PNGs, so packed and
PNG datasets are byte-identical to every consumer.
"""
import os
import glob
import numpy as np

FRAMES_FILE = "frames.npz"


def save_frames(traj_dir, frames_rgb):
    """Store a trajectory's frames. frames_rgb: (T, H, W, 3) uint8, RGB order."""
    np.savez_compressed(os.path.join(traj_dir, FRAMES_FILE),
                        f=np.ascontiguousarray(frames_rgb, dtype=np.uint8))


def has_packed(traj_dir):
    return os.path.exists(os.path.join(traj_dir, FRAMES_FILE))


def load_frames(traj_dir):
    """(T, H, W, 3) uint8 RGB for a trajectory (packed .npz, else legacy PNGs)."""
    p = os.path.join(traj_dir, FRAMES_FILE)
    if os.path.exists(p):
        with np.load(p) as z:
            return z["f"]
    import cv2
    pngs = sorted(glob.glob(os.path.join(traj_dir, "frame_*.png")))
    return np.stack([cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB) for fp in pngs])


def frame_meta(traj_dir):
    """(count, mtime) without decoding frames -- for building windows/ranges/sig.
    The count comes from positions.npy (one row per frame, always written next to
    the frames by every generator), so a packed trajectory need not be decompressed
    just to be counted. Falls back to counting PNGs for legacy datasets."""
    packed = os.path.join(traj_dir, FRAMES_FILE)
    if os.path.exists(packed):
        pos = os.path.join(traj_dir, "positions.npy")
        if os.path.exists(pos):
            return int(np.load(pos, mmap_mode="r").shape[0]), os.path.getmtime(packed)
        with np.load(packed) as z:            # rare: no positions -> read the header
            return int(z["f"].shape[0]), os.path.getmtime(packed)
    pngs = sorted(glob.glob(os.path.join(traj_dir, "frame_*.png")))
    return len(pngs), (os.path.getmtime(pngs[0]) if pngs else 0.0)
