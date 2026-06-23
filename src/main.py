import argparse
import torch
import random
import numpy as np
import glob
from torch.utils.data import DataLoader
import os

from src.dataset import LatentPhysicsDataset
from src.models import CNNAutoencoder, LatentDynamicsConvGRU, PixelDynamicsCNN
from src.train import train_autoencoder, train_dynamics, train_pixel_model
from src.eval import run_evaluation, save_rollout_video
from src.utils import setup_run_folder

def run_training_pipeline(data_dir, env_name, model_type="Latent", context_len=5,
                          batch_size=32, num_workers=8, ae_epochs=5, dyn_epochs=15,
                          learning_rate=1e-3, weight_decay=1e-4, rollout_len=5,
                          eval_horizon=10, seed=None, ae_checkpoint=""):

    run_dir, log_filepath = setup_run_folder(f"{env_name}_{model_type}")

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"

    if seed is not None and seed != "":
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")

    print(f"Using device: {device} | Model Type: {model_type} | Context: {context_len} frames")
    print(f"LR: {learning_rate} | WD: {weight_decay} | Workers: {num_workers} | Rollout: {rollout_len}")

    all_trajs = glob.glob(os.path.join(data_dir, "traj-*"))
    if not all_trajs:
        print(f"[Error] No trajectories found in {data_dir}. Did you run the generator?")
        return

    random.shuffle(all_trajs)
    n_train, n_val = int(len(all_trajs) * 0.8), int(len(all_trajs) * 0.1)
    
    val_test_workers = min(4, num_workers)
    train_loader = DataLoader(LatentPhysicsDataset(all_trajs[:n_train], context_len=context_len), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(LatentPhysicsDataset(all_trajs[n_train:n_train+n_val], context_len=context_len), batch_size=batch_size, shuffle=False, num_workers=val_test_workers, pin_memory=True)
    test_trajs = all_trajs[n_train+n_val:]
    test_loader = DataLoader(LatentPhysicsDataset(test_trajs, context_len=context_len, horizon=eval_horizon), batch_size=batch_size, shuffle=False, num_workers=val_test_workers, pin_memory=True)

    if model_type == "Latent":
        ae = CNNAutoencoder(latent_ch=32).to(device)
        dynamics = LatentDynamicsConvGRU(latent_ch=32, hidden_ch=64).to(device)

        if ae_checkpoint and os.path.exists(ae_checkpoint):
            ae.load_state_dict(torch.load(ae_checkpoint, map_location=device))
            print(f"Loaded autoencoder from {ae_checkpoint} -- skipping Phase 1.")
        else:
            if ae_checkpoint:
                print(f"[Warn] AE checkpoint not found: {ae_checkpoint}. Training a new autoencoder.")
            ae = train_autoencoder(ae, train_loader, val_loader, epochs=ae_epochs, learning_rate=learning_rate, weight_decay=weight_decay, device=device)

        dyn_train_loader = DataLoader(LatentPhysicsDataset(all_trajs[:n_train], context_len=context_len, horizon=rollout_len), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        dyn_val_loader = DataLoader(LatentPhysicsDataset(all_trajs[n_train:n_train+n_val], context_len=context_len, horizon=rollout_len), batch_size=batch_size, shuffle=False, num_workers=val_test_workers, pin_memory=True)
        train_dynamics(ae, dynamics, dyn_train_loader, dyn_val_loader, epochs=dyn_epochs, learning_rate=learning_rate, weight_decay=weight_decay, device=device)

        run_evaluation(test_loader, device, run_dir, "Latent", ae=ae, dynamics=dynamics)
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

        pix_train_loader = DataLoader(LatentPhysicsDataset(all_trajs[:n_train], context_len=context_len, horizon=rollout_len), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        pix_val_loader = DataLoader(LatentPhysicsDataset(all_trajs[n_train:n_train+n_val], context_len=context_len, horizon=rollout_len), batch_size=batch_size, shuffle=False, num_workers=val_test_workers, pin_memory=True)
        pixel_model = train_pixel_model(pixel_model, pix_train_loader, pix_val_loader, epochs=dyn_epochs, learning_rate=learning_rate, weight_decay=weight_decay, device=device)

        run_evaluation(test_loader, device, run_dir, "Pixel", pixel_model=pixel_model)
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ae_epochs", type=int, default=5)
    parser.add_argument("--dyn_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--rollout_len", type=int, default=5, help="Multi-step rollout horizon for dynamics scheduled sampling")
    parser.add_argument("--eval_horizon", type=int, default=10, help="Rollout length used at eval time to report error growth vs horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs")
    parser.add_argument("--ae_checkpoint", type=str, default="", help="Path to a saved autoencoder.pth to reuse (skips Phase 1)")
    args = parser.parse_args()

    run_training_pipeline(
        args.data_dir, args.env_name, args.model_type,
        context_len=args.context_len, batch_size=args.batch_size, num_workers=args.num_workers,
        ae_epochs=args.ae_epochs, dyn_epochs=args.dyn_epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, rollout_len=args.rollout_len,
        eval_horizon=args.eval_horizon, seed=args.seed, ae_checkpoint=args.ae_checkpoint
    )