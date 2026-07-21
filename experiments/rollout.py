import os
import sys
import glob
import json
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import CNNVAE, DiffusionTransformer
from src.eval import flow_sample


def load_run(run_dir, device="cuda"):
    with open(os.path.join(run_dir, "run_config.json")) as f:
        cfg = json.load(f)
    ae = CNNVAE(latent_ch=cfg["latent_ch"], latent_grid=cfg["latent_grid"],
                base_ch=cfg.get("vae_base_ch", 64), block=cfg.get("vae_block", "res"),
                enc_res_blocks=cfg["vae_enc_res_blocks"],
                dec_res_blocks=cfg["vae_dec_res_blocks"]).to(device)
    ae.load_state_dict(torch.load(os.path.join(run_dir, "autoencoder.pth"),
                                  map_location=device, weights_only=True))
    ae.eval()
    dit = DiffusionTransformer(latent_ch=cfg["latent_ch"], context_len=cfg["context_len"],
                               grid=cfg["latent_grid"], chunk_len=cfg["chunk_len"],
                               d_model=cfg["dit_d_model"], n_layers=cfg["dit_n_layers"],
                               n_heads=cfg["dit_n_heads"], latent_scale=1.0).to(device)
    dit.load_state_dict(torch.load(os.path.join(run_dir, "dit.pth"),
                                   map_location=device, weights_only=True))
    dit.eval()
    return ae, dit, cfg


def test_split(data_dir, seed=42):
    random.seed(seed)
    all_trajs = glob.glob(os.path.join(data_dir, "traj-*"))
    random.shuffle(all_trajs)
    n_train, n_val = int(len(all_trajs) * 0.8), int(len(all_trajs) * 0.1)
    return all_trajs, all_trajs[n_train + n_val:]


def traj_frames_np(frame_cache, traj_dir):
    name = os.path.basename(traj_dir)
    start, count = frame_cache.ranges[name]
    return frame_cache.frames[start:start + count].permute(0, 2, 3, 1).contiguous().numpy()


@torch.no_grad()
def rollout(ae, dit, ctx, n_steps, num_steps=3, use_amp=True):
    B, T, C, H, W = ctx.shape
    z = ae.encode(ctx.reshape(-1, C, H, W)) / dit.latent_scale
    z_seq = z.view(B, T, *z.shape[1:])
    outs = []
    k = 0
    while k < n_steps:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            z_chunk = flow_sample(dit, z_seq, num_steps)
        for j in range(z_chunk.shape[1]):
            if k >= n_steps:
                break
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = ae.decode(z_chunk[:, j].float() * dit.latent_scale)
            outs.append(pred.float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu())
            k += 1
        z_seq = torch.cat([z_seq, z_chunk], dim=1)[:, -T:]
    return torch.stack(outs, dim=1)


@torch.no_grad()
def rollout_frames_np(ae, dit, real_frames_u8, ctx_len, n_steps, num_steps=3,
                      device="cuda", batch=128):
    out = []
    for i in range(0, len(real_frames_u8), batch):
        chunk = real_frames_u8[i:i + batch]
        ctx = torch.from_numpy(np.stack([f[:ctx_len] for f in chunk]))
        ctx = ctx.permute(0, 1, 4, 2, 3).float().div_(255.0).to(device)
        pred = rollout(ae, dit, ctx, n_steps, num_steps=num_steps)
        out.append(pred.permute(0, 1, 3, 4, 2).contiguous().numpy())
    return np.concatenate(out, axis=0)
