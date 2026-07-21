"""Standalone latent-predictability scoring on COMMON data.

The in-training LatentProbe scores each AE on its own dataset, so scores from
runs trained on different data were never strictly comparable. This script
scores any set of AE checkpoints on the SAME fixed dataset and windows:
identical frames, identical split convention (seed-42 shuffle, 80/10/10 --
matching src.main, so probe val windows are held out even for AEs trained on
this dataset), identical surrogate (fresh init, fixed seed, 30 ep / w192).

  python -m experiments.surrogate --runs runA,runB,... [--data data/bouncing_36k_v2]

Rank AEs by surrogate 1-step PSNR; trust separations >= ~0.3 dB (calibrated
against paired 10-ep DiT runs, 2026-07-16).
"""
import argparse
import json
import os

import torch

from experiments.rollout import test_split
from src.dataset import FrameCache
from src.models import CNNVAE
from src.train import score_latent_predictability


def load_ae(run_dir, device):
    with open(os.path.join(run_dir, "run_config.json")) as f:
        cfg = json.load(f)
    ae = CNNVAE(latent_ch=cfg["latent_ch"], latent_grid=cfg["latent_grid"],
                base_ch=cfg.get("vae_base_ch", 64), block=cfg.get("vae_block", "res"),
                mid_blocks=cfg.get("vae_mid_blocks", 1),
                enc_res_blocks=cfg["vae_enc_res_blocks"],
                dec_res_blocks=cfg["vae_dec_res_blocks"]).to(device)
    ae.load_state_dict(torch.load(os.path.join(run_dir, "autoencoder.pth"),
                                  map_location=device, weights_only=True))
    ae.eval()
    return ae, cfg


@torch.no_grad()
def encode_all(ae, frames, device, batch=512):
    z_all = None
    for i in range(0, frames.shape[0], batch):
        x = frames[i:i + batch].to(device).float().div_(255.0)
        z = ae.encode(x)
        if z_all is None:
            z_all = torch.empty((frames.shape[0], *z.shape[1:]), dtype=torch.float16)
        z_all[i:i + batch] = z.half().cpu()
    scale = z_all.float().std().item()
    return (z_all.float() / scale).half(), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run dirs (need run_config.json + autoencoder.pth)")
    ap.add_argument("--data", default="data/bouncing_36k_v2")
    ap.add_argument("--n-train-traj", type=int, default=500)
    ap.add_argument("--n-val-traj", type=int, default=100)
    ap.add_argument("--context-len", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_trajs, _ = test_split(args.data)  # seed-42 shuffled, same order as src.main
    n_tr, n_va = int(len(all_trajs) * 0.8), int(len(all_trajs) * 0.1)
    train_trajs = all_trajs[:n_tr][: args.n_train_traj]
    val_trajs = all_trajs[n_tr:n_tr + n_va][: args.n_val_traj]
    used = train_trajs + val_trajs
    print(f"Common-data surrogate | {args.data} | {len(train_trajs)} train / {len(val_trajs)} val trajs")
    cache = FrameCache(used)  # in-RAM, only the needed trajs

    results = {}
    for run in args.runs.split(","):
        run = run.strip().rstrip("/")
        ae, cfg = load_ae(run, device)
        z_all, scale = encode_all(ae, cache.frames, device)
        print(f"\n== {os.path.basename(run)} (enc{cfg['vae_enc_res_blocks']}/dec{cfg['vae_dec_res_blocks']}, "
              f"ch{cfg['latent_ch']}, trained on {cfg['data_dir']}) | latent scale {scale:.3f}")
        m = score_latent_predictability(ae, z_all, scale, cache, train_trajs, val_trajs,
                                        args.context_len, device, seed=args.seed,
                                        n_train_traj=len(train_trajs), n_val_traj=len(val_trajs))
        results[run] = {"config": f"enc{cfg['vae_enc_res_blocks']}/dec{cfg['vae_dec_res_blocks']}/ch{cfg['latent_ch']}",
                        "trained_on": cfg["data_dir"], "latent_scale": scale, **(m or {})}
        del ae, z_all
        torch.cuda.empty_cache()

    print(f"\n{'run':44s} | {'R²':>7s} | {'surrogate dB':>12s} | {'recon dB':>8s}")
    for run, r in sorted(results.items(), key=lambda kv: -kv[1].get("surrogate_psnr", 0)):
        print(f"{os.path.basename(run):44s} | {r.get('latent_r2', float('nan')):7.4f} | "
              f"{r.get('surrogate_psnr', float('nan')):12.2f} | {r.get('recon_psnr', float('nan')):8.2f}")

    os.makedirs("experiments/results", exist_ok=True)
    out = f"experiments/results/surrogate_{os.path.basename(args.data)}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
