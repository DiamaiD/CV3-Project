import argparse
import torch
import random
import numpy as np
import glob
import os

from src.dataset import FrameCache, CachedLoader
from src.models import CNNAutoencoder, LatentDynamicsConvGRU, PixelDynamicsCNN
from src.train import train_autoencoder, train_dynamics, train_pixel_model, build_latent_cache
from src.eval import run_evaluation, save_rollout_video
from src.utils import setup_run_folder

def run_training_pipeline(data_dir, env_name, model_type="Latent", context_len=5,
                          ae_batch_size=32, dyn_batch_size=32, ae_epochs=5, dyn_epochs=15,
                          ae_learning_rate=1e-3, ae_weight_decay=1e-4,
                          dyn_learning_rate=1e-3, dyn_weight_decay=1e-4, rollout_len=5,
                          eval_horizon=10, seed=None, ae_checkpoint="", cache_in_vram=False):

    run_dir, log_filepath = setup_run_folder(f"{env_name}_{model_type}")

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

    print(f"Using device: {device} | Model Type: {model_type} | Context: {context_len} frames")
    print(f"Cache: {cache_device.upper()} | Rollout: {rollout_len}")
    print(f"AE  -> LR: {ae_learning_rate} | WD: {ae_weight_decay} | Batch: {ae_batch_size}")
    print(f"Dyn -> LR: {dyn_learning_rate} | WD: {dyn_weight_decay} | Batch: {dyn_batch_size}")

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

    if model_type == "Latent":
        ae = CNNAutoencoder(latent_ch=32).to(device)
        dynamics = LatentDynamicsConvGRU(latent_ch=32, hidden_ch=64).to(device)

        if ae_checkpoint and os.path.exists(ae_checkpoint):
            ae.load_state_dict(torch.load(ae_checkpoint, map_location=device))
            print(f"Loaded autoencoder from {ae_checkpoint} -- skipping Phase 1.")
        else:
            if ae_checkpoint:
                print(f"[Warn] AE checkpoint not found: {ae_checkpoint}. Training a new autoencoder.")
            ae = train_autoencoder(ae, pixel_loader(train_trajs, 1, True, ae_batch_size), pixel_loader(val_trajs, 1, False, ae_batch_size),
                                   epochs=ae_epochs, learning_rate=ae_learning_rate, weight_decay=ae_weight_decay, device=device)

        # Precompute the frozen-AE latents once so the dynamics phase never re-runs the AE.
        z_all = build_latent_cache(ae, frame_cache.frames, device, cache_device)

        def latent_loader(trajs, horizon, shuffle, bs):
            ctx_idx, tgt_idx = frame_cache.build_windows(trajs, context_len, horizon)
            return CachedLoader(z_all, ctx_idx, tgt_idx, bs, device, shuffle=shuffle, horizon=horizon)

        train_dynamics(dynamics, latent_loader(train_trajs, rollout_len, True, dyn_batch_size),
                       latent_loader(val_trajs, rollout_len, False, dyn_batch_size),
                       epochs=dyn_epochs, learning_rate=dyn_learning_rate, weight_decay=dyn_weight_decay, device=device)

        # Eval runs the AE (encode/decode), so size it like the AE phase.
        run_evaluation(pixel_loader(test_trajs, eval_horizon, False, ae_batch_size), device, run_dir, "Latent", ae=ae, dynamics=dynamics)
        torch.save(ae.state_dict(), os.path.join(run_dir, "autoencoder.pth"))
        torch.save(dynamics.state_dict(), os.path.join(run_dir, "dynamics.pth"))

        ae.eval(); dynamics.eval()
        def latent_predict(context):
            _, T, C, H, W = context.shape
            z = ae.encode(context.view(-1, C, H, W))
            z_seq = z.view(1, T, *z.shape[1:])
            return ae.decode(dynamics(z_seq))
        for traj in test_trajs[:2]:
            save_rollout_video(latent_predict, traj, device, run_dir, context_len=context_len)

    elif model_type == "Pixel":
        pixel_model = PixelDynamicsCNN(context_len=context_len).to(device)

        # The pixel model IS the dynamics model here, so it uses the Dyn batch size.
        pixel_model = train_pixel_model(pixel_model, pixel_loader(train_trajs, rollout_len, True, dyn_batch_size),
                                        pixel_loader(val_trajs, rollout_len, False, dyn_batch_size),
                                        epochs=dyn_epochs, learning_rate=dyn_learning_rate, weight_decay=dyn_weight_decay, device=device)

        run_evaluation(pixel_loader(test_trajs, eval_horizon, False, dyn_batch_size), device, run_dir, "Pixel", pixel_model=pixel_model)
        torch.save(pixel_model.state_dict(), os.path.join(run_dir, "pixel_model.pth"))

        pixel_model.eval()
        for traj in test_trajs[:2]:
            save_rollout_video(lambda c: pixel_model(c), traj, device, run_dir, context_len=context_len)

    print(f"Training Complete. All files saved in {run_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bouncing")
    parser.add_argument("--env_name", type=str, default="bouncing")
    parser.add_argument("--model_type", type=str, default="Latent", choices=["Latent", "Pixel"])
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--ae_batch_size", type=int, default=32, help="Batch size for the autoencoder phase (full-res frames)")
    parser.add_argument("--dyn_batch_size", type=int, default=32, help="Batch size for the dynamics phase (tiny latents -> can be much larger)")
    parser.add_argument("--ae_epochs", type=int, default=5)
    parser.add_argument("--dyn_epochs", type=int, default=15)
    parser.add_argument("--ae_learning_rate", type=float, default=5e-4)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-3)
    parser.add_argument("--dyn_learning_rate", type=float, default=5e-4)
    parser.add_argument("--dyn_weight_decay", type=float, default=1e-3)
    parser.add_argument("--rollout_len", type=int, default=5, help="Multi-step rollout horizon for dynamics scheduled sampling")
    parser.add_argument("--eval_horizon", type=int, default=10, help="Rollout length used at eval time to report error growth vs horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs")
    parser.add_argument("--ae_checkpoint", type=str, default="", help="Path to a saved autoencoder.pth to reuse (skips Phase 1)")
    parser.add_argument("--cache_in_vram", action="store_true", help="Keep the decoded frame + latent caches in GPU VRAM instead of system RAM")
    args = parser.parse_args()

    run_training_pipeline(
        args.data_dir, args.env_name, args.model_type,
        context_len=args.context_len, ae_batch_size=args.ae_batch_size, dyn_batch_size=args.dyn_batch_size,
        ae_epochs=args.ae_epochs, dyn_epochs=args.dyn_epochs,
        ae_learning_rate=args.ae_learning_rate, ae_weight_decay=args.ae_weight_decay,
        dyn_learning_rate=args.dyn_learning_rate, dyn_weight_decay=args.dyn_weight_decay, rollout_len=args.rollout_len,
        eval_horizon=args.eval_horizon, seed=args.seed, ae_checkpoint=args.ae_checkpoint,
        cache_in_vram=args.cache_in_vram
    )
