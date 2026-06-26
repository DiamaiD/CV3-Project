import argparse
import torch
import random
import numpy as np
import glob
import os

from src.dataset import FrameCache, CachedLoader
from src.models import CNNVAE, DiffusionTransformer
from src.train import train_autoencoder, build_latent_cache, train_flow_matching
from src.eval import run_evaluation, save_rollout_video, save_vae_reconstructions, flow_sample
from src.utils import setup_run_folder


def run_training_pipeline(data_dir, env_name, context_len=5,
                          ae_batch_size=32, dyn_batch_size=64, ae_epochs=20, dyn_epochs=30,
                          ae_learning_rate=5e-4, ae_weight_decay=1e-2, ae_kl_weight=0.005,
                          ae_lpips_weight=0.0,
                          dyn_learning_rate=3e-4, dyn_weight_decay=1e-4,
                          eval_horizon=50, eval_max_batches=24, eval_best_of_n=1,
                          seed=None, ae_checkpoint="", cache_in_vram=False, latent_grid=8,
                          chunk_len=5, dit_d_model=256, dit_n_layers=6, dit_n_heads=8, inference_steps=10):
    """Two-stage latent world model:
      Phase 1 -- a continuous CNNVAE compresses 64x64 frames to 32x8x8 latents.
      Phase 2 -- a Diffusion Transformer learns the next-frame latent by Rectified Flow
                 (flow matching); inference integrates its velocity field with an Euler ODE solver.
    """
    run_dir, log_filepath = setup_run_folder(f"{env_name}_flow")

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"

    # Where the decoded frame / latent caches live. VRAM keeps everything resident on the GPU
    # (fastest, no per-batch host->device copies); otherwise cache in system RAM and move
    # each batch to the GPU on the fly.
    cache_device = "cuda" if (cache_in_vram and device == "cuda") else "cpu"

    if seed is not None and seed != "":
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")

    print(f"Using device: {device} | Latent Flow Matching | Context: {context_len} frames | Chunk: {chunk_len}")
    print(f"Cache: {cache_device.upper()} | Inference steps: {inference_steps} | Latent grid: {latent_grid}x{latent_grid}")
    print(f"VAE -> LR: {ae_learning_rate} | WD: {ae_weight_decay} | Batch: {ae_batch_size} | "
          f"Beta: {ae_kl_weight} | LPIPS: {ae_lpips_weight}")
    print(f"DiT -> LR: {dyn_learning_rate} | WD: {dyn_weight_decay} | Batch: {dyn_batch_size} | "
          f"d_model: {dit_d_model} | layers: {dit_n_layers} | heads: {dit_n_heads}")

    all_trajs = glob.glob(os.path.join(data_dir, "traj-*"))
    if not all_trajs:
        print(f"[Error] No trajectories found in {data_dir}. Did you run the generator?")
        return

    random.shuffle(all_trajs)
    n_train, n_val = int(len(all_trajs) * 0.8), int(len(all_trajs) * 0.1)
    train_trajs = all_trajs[:n_train]
    val_trajs = all_trajs[n_train:n_train + n_val]
    test_trajs = all_trajs[n_train + n_val:]

    # Decode every PNG once into an in-memory cache (RAM or VRAM). The disk cache lets repeat
    # runs skip PNG decoding entirely.
    frame_cache = FrameCache(all_trajs, cache_device=cache_device,
                             disk_cache_path=os.path.join(data_dir, "frames_cache.pt"))

    def pixel_loader(trajs, horizon, shuffle, bs):
        ctx_idx, tgt_idx = frame_cache.build_windows(trajs, context_len, horizon)
        return CachedLoader(frame_cache.frames, ctx_idx, tgt_idx, bs, device,
                            shuffle=shuffle, horizon=horizon)

    # ===== Phase 1: continuous VAE =====
    ae = CNNVAE(latent_ch=32, latent_grid=latent_grid).to(device)

    if ae_checkpoint and os.path.exists(ae_checkpoint):
        ae.load_state_dict(torch.load(ae_checkpoint, map_location=device))
        print(f"Loaded autoencoder from {ae_checkpoint} -- skipping Phase 1.")
    else:
        if ae_checkpoint:
            print(f"[Warn] AE checkpoint not found: {ae_checkpoint}. Training a new autoencoder.")
        ae = train_autoencoder(ae, pixel_loader(train_trajs, 1, True, ae_batch_size),
                               pixel_loader(val_trajs, 1, False, ae_batch_size),
                               epochs=ae_epochs, learning_rate=ae_learning_rate,
                               weight_decay=ae_weight_decay, kl_weight=ae_kl_weight,
                               lpips_weight=ae_lpips_weight, device=device)

    # Reconstruction baseline grid (truth | recon | abs error). Always written -- even when the
    # VAE is loaded from a checkpoint -- as a cheap fixed read on latent quality.
    save_vae_reconstructions(ae, pixel_loader(val_trajs, 1, False, ae_batch_size), device, run_dir)
    torch.save(ae.state_dict(), os.path.join(run_dir, "autoencoder.pth"))

    # Precompute the frozen-VAE latents (posterior means) once so the DiT phase never re-runs the
    # VAE. The KL already keeps them ~unit-Gaussian, so no extra normalization is needed.
    z_all, latent_scale = build_latent_cache(ae, frame_cache.frames, device, cache_device)
    latent_ch, grid = z_all.shape[1], z_all.shape[-1]   # (M, Cl, h, w) -> 32, 8

    def latent_loader(trajs, horizon, shuffle, bs):
        ctx_idx, tgt_idx = frame_cache.build_windows(trajs, context_len, horizon)
        return CachedLoader(z_all, ctx_idx, tgt_idx, bs, device, shuffle=shuffle, horizon=horizon)

    # ===== Phase 2: Flow Matching DiT (trained on K-frame chunks of next-frame latents) =====
    dit = DiffusionTransformer(latent_ch=latent_ch, context_len=context_len, grid=grid,
                               chunk_len=chunk_len, d_model=dit_d_model, n_layers=dit_n_layers,
                               n_heads=dit_n_heads, latent_scale=latent_scale).to(device)
    dit = train_flow_matching(dit, latent_loader(train_trajs, chunk_len, True, dyn_batch_size),
                              latent_loader(val_trajs, chunk_len, False, dyn_batch_size),
                              epochs=dyn_epochs, learning_rate=dyn_learning_rate,
                              weight_decay=dyn_weight_decay, device=device)
    torch.save(dit.state_dict(), os.path.join(run_dir, "dit.pth"))

    # ===== Phase 3: evaluation + rollout GIFs =====
    # Cap the rollout to a few hundred windows: each window costs eval_horizon x inference_steps
    # DiT forwards, so the full overlapping test set would be tens of minutes for no extra signal.
    run_evaluation(pixel_loader(test_trajs, eval_horizon, False, ae_batch_size), device, run_dir,
                   ae=ae, dit=dit, num_steps=inference_steps, max_batches=eval_max_batches,
                   best_of_n=eval_best_of_n)

    ae.eval(); dit.eval()

    def flow_predict_chunk(context):
        # context: (1, T, C, H, W) pixels -> (1, K, C, H, W) predicted next K frames.
        _, T, C, H, W = context.shape
        z = ae.encode(context.view(-1, C, H, W)) / dit.latent_scale   # into the DiT's normalized space
        z_seq = z.view(1, T, *z.shape[1:])
        z_chunk = flow_sample(dit, z_seq, inference_steps)          # (1, K, Cl, gh, gw), normalized
        K = z_chunk.shape[1]
        frames = ae.decode((z_chunk * dit.latent_scale).reshape(K, *z_chunk.shape[2:]))  # back to VAE scale
        return frames.unsqueeze(0)                                  # (1, K, C, H, W)

    for traj in test_trajs[:2]:
        save_rollout_video(flow_predict_chunk, traj, device, run_dir, context_len=context_len)

    print(f"Training Complete. All files saved in {run_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bouncing")
    parser.add_argument("--env_name", type=str, default="bouncing")
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--ae_batch_size", type=int, default=32, help="Batch size for the VAE phase (full-res frames)")
    parser.add_argument("--dyn_batch_size", type=int, default=64, help="Batch size for the DiT phase (tiny latents -> can be much larger)")
    parser.add_argument("--ae_epochs", type=int, default=20)
    parser.add_argument("--dyn_epochs", type=int, default=30)
    parser.add_argument("--ae_learning_rate", type=float, default=5e-4)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-2)
    parser.add_argument("--ae_kl_weight", type=float, default=0.005, help="VAE KL weight (beta). Lower if reconstructions blur / KL collapses; raise if the latent is barely regularized.")
    parser.add_argument("--ae_lpips_weight", type=float, default=0.0, help="Perceptual (LPIPS-VGG) loss weight on the VAE. 0 = off (pixel+KL only). ~1.0 makes latent L2 track perceptual quality, the key fix for the prediction-blur ceiling. Needs `pip install lpips`.")
    parser.add_argument("--latent_grid", type=int, default=8, help="VAE latent spatial size (8 -> 8x8, 16 -> 16x16). 16 makes motion more spatially local for the DiT at 4x token/cache cost. Requires retraining the VAE (8x8 checkpoints are incompatible).")
    parser.add_argument("--dyn_learning_rate", type=float, default=3e-4)
    parser.add_argument("--dyn_weight_decay", type=float, default=1e-4)
    parser.add_argument("--chunk_len", type=int, default=5, help="Chunk prediction: number of future frames (K) the DiT denoises jointly per call")
    parser.add_argument("--dit_d_model", type=int, default=256, help="DiT width (must be divisible by dit_n_heads)")
    parser.add_argument("--dit_n_layers", type=int, default=6, help="DiT depth")
    parser.add_argument("--dit_n_heads", type=int, default=8, help="DiT attention heads")
    parser.add_argument("--inference_steps", type=int, default=10, help="Euler ODE steps used to sample a frame")
    parser.add_argument("--eval_horizon", type=int, default=50, help="Rollout length used at eval time to report error growth vs horizon")
    parser.add_argument("--eval_max_batches", type=int, default=24, help="Cap on test batches rolled out at eval (each window costs eval_horizon x inference_steps DiT forwards)")
    parser.add_argument("--eval_best_of_n", type=int, default=1, help="Sample N independent rollouts per window and also report the best (per trajectory). Reveals if single-sample MSE punishes valid alternative futures. Multiplies eval cost by N.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs")
    parser.add_argument("--ae_checkpoint", type=str, default="", help="Path to a saved autoencoder.pth to reuse (skips Phase 1)")
    parser.add_argument("--cache_in_vram", action="store_true", help="Keep the decoded frame + latent caches in GPU VRAM instead of system RAM")
    args = parser.parse_args()

    run_training_pipeline(
        args.data_dir, args.env_name,
        context_len=args.context_len, ae_batch_size=args.ae_batch_size, dyn_batch_size=args.dyn_batch_size,
        ae_epochs=args.ae_epochs, dyn_epochs=args.dyn_epochs,
        ae_learning_rate=args.ae_learning_rate, ae_weight_decay=args.ae_weight_decay, ae_kl_weight=args.ae_kl_weight,
        ae_lpips_weight=args.ae_lpips_weight,
        dyn_learning_rate=args.dyn_learning_rate, dyn_weight_decay=args.dyn_weight_decay,
        eval_horizon=args.eval_horizon, eval_max_batches=args.eval_max_batches,
        eval_best_of_n=args.eval_best_of_n,
        seed=args.seed, ae_checkpoint=args.ae_checkpoint,
        cache_in_vram=args.cache_in_vram, latent_grid=args.latent_grid,
        chunk_len=args.chunk_len, dit_d_model=args.dit_d_model, dit_n_layers=args.dit_n_layers,
        dit_n_heads=args.dit_n_heads, inference_steps=args.inference_steps
    )
