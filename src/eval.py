import os
import glob
import json
import math
import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def _psnr(mse_val):
    """Peak signal-to-noise ratio for images in [0, 1] (so MAX_I = 1.0)."""
    if mse_val <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_val)


def run_evaluation(test_loader, device, run_dir, model_type, save_images=True, ae=None, dynamics=None, pixel_model=None):
    print(f"\n--- Phase 3: Final Test Set Evaluation ({model_type}) ---")

    vis_ctx, vis_target, vis_pred = None, None, None
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    # The model is fed its own predictions back in, so we can measure how error grows with horizon.
    # Persistence (copy the last real frame, held constant for every step) is the no-op baseline.
    metric_keys = ["model_pix_mse", "persist_pix_mse", "model_pix_mae", "model_lat_mse", "persist_lat_mse"]
    stats = None
    total = 0
    K = 0

    with torch.no_grad():
        for i, (ctx_frames, future_frames) in enumerate(test_loader):
            ctx_frames = ctx_frames.to(device)
            future_frames = future_frames.to(device)
            if future_frames.dim() == 4:
                future_frames = future_frames.unsqueeze(1)
            B, T, C, H, W = ctx_frames.shape
            K = future_frames.shape[1]
            last_frame = ctx_frames[:, -1]

            if stats is None:
                stats = {key: [0.0] * K for key in metric_keys}

            # Per-model rollout state.
            if model_type == "Latent":
                z = ae.encode(ctx_frames.view(-1, C, H, W))   # VAE posterior mean (mu)
                z_seq = z.view(B, T, *z.shape[1:])
                z_last = z_seq[:, -1].clone()   # encoded last real frame (persistence)
            else:
                context = ctx_frames

            for k in range(K):
                target = future_frames[:, k]

                if model_type == "Latent":
                    z_pred = dynamics(z_seq)
                    pred = ae.decode(z_pred)
                    z_target = ae.encode(target)
                    stats["model_lat_mse"][k] += mse(z_pred, z_target).item() * B
                    stats["persist_lat_mse"][k] += mse(z_last, z_target).item() * B
                    z_seq = torch.cat([z_seq[:, 1:], z_pred.unsqueeze(1)], dim=1)
                else:
                    pred = pixel_model(context)
                    context = torch.cat([context[:, 1:], pred.unsqueeze(1)], dim=1)

                # Pixel-space comparison works for both models.
                stats["model_pix_mse"][k] += mse(pred, target).item() * B
                stats["model_pix_mae"][k] += mae(pred, target).item() * B
                stats["persist_pix_mse"][k] += mse(last_frame, target).item() * B

                if i == 0 and k == 0 and save_images:
                    vis_ctx, vis_target, vis_pred = last_frame, target, pred

            total += B

    if not total:
        print("[Eval] No test samples (trajectories too short for context+horizon?). Skipping metrics.")
        return

    avg = {key: [s / total for s in stats[key]] for key in metric_keys}

    def _skill(model_v, persist_v):
        return (persist_v - model_v) / persist_v * 100 if persist_v > 0 else 0.0

    # Pixel-space table, one row per rollout step (both models).
    print("\nPixel-space metrics by rollout horizon (model = free-running, fed its own predictions):")
    print(f"{'Step':>4} | {'Model MSE':>10} | {'Persist MSE':>11} | {'Skill%':>7} | "
          f"{'Model MAE':>9} | {'Model PSNR':>10} | {'Persist PSNR':>12}")
    for k in range(K):
        m, p = avg["model_pix_mse"][k], avg["persist_pix_mse"][k]
        print(f"{k+1:>4} | {m:>10.5f} | {p:>11.5f} | {_skill(m, p):>6.1f}% | "
              f"{avg['model_pix_mae'][k]:>9.5f} | {_psnr(m):>10.2f} | {_psnr(p):>12.2f}")

    mean_m = sum(avg["model_pix_mse"]) / K
    mean_p = sum(avg["persist_pix_mse"]) / K
    print(f"  avg over {K} steps -> Model MSE {mean_m:.5f} | Persist MSE {mean_p:.5f} | Skill {_skill(mean_m, mean_p):+.1f}%")

    if model_type == "Latent":
        print("\nLatent-space metrics by rollout horizon (what the dynamics model optimizes):")
        print(f"{'Step':>4} | {'Model MSE':>10} | {'Persist MSE':>11} | {'Skill%':>7}")
        for k in range(K):
            m, p = avg["model_lat_mse"][k], avg["persist_lat_mse"][k]
            print(f"{k+1:>4} | {m:>10.5f} | {p:>11.5f} | {_skill(m, p):>6.1f}%")

    # Headline verdict uses single-step pixel MSE (step 1), the fairest cross-model number.
    m1, p1 = avg["model_pix_mse"][0], avg["persist_pix_mse"][0]
    verdict = "beats persistence" if m1 < p1 else "WORSE than persistence (no dynamics learned)"
    print(f"\n-> 1-step: Model is {_skill(m1, p1):+.1f}% vs persistence in pixel MSE  [{verdict}]")
    if K > 1:
        mK, pK = avg["model_pix_mse"][K-1], avg["persist_pix_mse"][K-1]
        print(f"-> {K}-step: Model is {_skill(mK, pK):+.1f}% vs persistence in pixel MSE")

    # Persist everything we just printed so runs can be compared without re-parsing stdout.
    results = {
        "model_type": model_type,
        "num_samples": total,
        "horizon": K,
        "per_step": {
            "model_pix_mse": avg["model_pix_mse"],
            "persist_pix_mse": avg["persist_pix_mse"],
            "model_pix_mae": avg["model_pix_mae"],
            "model_pix_psnr": [_psnr(v) for v in avg["model_pix_mse"]],
            "persist_pix_psnr": [_psnr(v) for v in avg["persist_pix_mse"]],
            "pix_skill_pct": [_skill(m, p) for m, p in zip(avg["model_pix_mse"], avg["persist_pix_mse"])],
        },
        "pix_mse_mean": {"model": mean_m, "persist": mean_p, "skill_pct": _skill(mean_m, mean_p)},
        "headline": {
            "step1_model_pix_mse": m1,
            "step1_persist_pix_mse": p1,
            "step1_skill_pct": _skill(m1, p1),
            "beats_persistence": m1 < p1,
        },
    }
    if K > 1:
        results["headline"]["stepK_skill_pct"] = _skill(mK, pK)
    if model_type == "Latent":
        results["per_step"]["model_lat_mse"] = avg["model_lat_mse"]
        results["per_step"]["persist_lat_mse"] = avg["persist_lat_mse"]
        results["per_step"]["lat_skill_pct"] = [
            _skill(m, p) for m, p in zip(avg["model_lat_mse"], avg["persist_lat_mse"])
        ]

    results_path = os.path.join(run_dir, f"eval_results_{model_type}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {model_type} evaluation results to: {results_path}")

    if save_images and vis_pred is not None:
        num_samples = min(8, vis_target.size(0))
        grid_imgs = []
        for j in range(num_samples):
            ctx_rgb = vis_ctx[j].cpu()
            target_rgb = vis_target[j].cpu()
            pred_rgb = vis_pred[j].cpu()

            error_rgb = torch.abs(target_rgb - pred_rgb) * 2.0
            error_rgb = torch.clamp(error_rgb, 0, 1)

            grid_imgs.extend([ctx_rgb, target_rgb, pred_rgb, error_rgb])

        img_grid = torchvision.utils.make_grid(grid_imgs, nrow=4, pad_value=0.5)

        img_np = img_grid.permute(1, 2, 0).cpu().numpy() * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Header lines: (text, y, font_scale, thickness, color_BGR).
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 10
        header_lines = [
            (f"Model: {model_type}", 25, 0.6, 2, (0, 0, 0)),
            ("Cols: [Frame t-1] | [Truth t] | [Pred t] | [Abs. Error]", 50, 0.5, 1, (0, 0, 0)),
            ("Abs. Error: Black = Perfect Match | Colors = Deviation in predicted physics/materials", 70, 0.45, 1, (0, 0, 200)),
        ]

        text_w = max(cv2.getTextSize(t, font, s, th)[0][0] for t, _, s, th, _ in header_lines)
        right_pad = max(0, (text_w + 2 * margin) - img_bgr.shape[1])
        header_height = 80
        img_padded = cv2.copyMakeBorder(img_bgr, header_height, 0, 0, right_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        for text, y, scale, thickness, color in header_lines:
            cv2.putText(img_padded, text, (margin, y), font, scale, color, thickness)

        save_path = os.path.join(run_dir, f"eval_predictions_{model_type}.png")
        cv2.imwrite(save_path, img_padded)

        print(f"Saved {model_type} evaluation image grid to: {save_path}")


@torch.no_grad()
def save_vae_reconstructions(ae, loader, device, run_dir, n_samples=8):
    """Save a [Truth | Reconstruction | Abs. Error] grid for the VAE.

    Encodes real frames to the posterior mean and decodes them -- the exact deterministic
    path the dynamics model relies on -- so the image is a direct read on how much detail the
    latent throws away. Written every run (even when the VAE is loaded from a checkpoint) as a
    fixed reconstruction baseline, and prints mean MSE / PSNR over the batch.
    """
    ae.eval()
    batch = next(iter(loader), None)
    if batch is None:
        print("[VAE] No frames available for reconstruction grid; skipping.")
        return
    ctx_frames, _ = batch
    frames = ctx_frames[:, -1].to(device)[:n_samples]      # one real frame per sample
    recon = ae.decode(ae.encode(frames)).clamp(0, 1)

    mse_val = nn.MSELoss()(recon, frames).item()
    psnr_val = _psnr(mse_val)
    print(f"[VAE] Reconstruction over {frames.size(0)} frames -> MSE {mse_val:.6f} | PSNR {psnr_val:.2f} dB")

    grid_imgs = []
    for j in range(frames.size(0)):
        truth, rec = frames[j].cpu(), recon[j].cpu()
        err = torch.clamp(torch.abs(truth - rec) * 2.0, 0, 1)
        grid_imgs.extend([truth, rec, err])

    img_grid = torchvision.utils.make_grid(grid_imgs, nrow=3, pad_value=0.5)
    img_np = np.clip(img_grid.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10
    header_lines = [
        (f"VAE reconstruction  |  MSE {mse_val:.6f}  PSNR {psnr_val:.2f} dB", 25, 0.6, 2, (0, 0, 0)),
        ("Cols: [Truth] | [Reconstruction] | [Abs. Error x2]", 50, 0.5, 1, (0, 0, 0)),
        ("Abs. Error: Black = Perfect Match | Colors = Detail lost by the latent", 70, 0.45, 1, (0, 0, 200)),
    ]
    text_w = max(cv2.getTextSize(t, font, s, th)[0][0] for t, _, s, th, _ in header_lines)
    right_pad = max(0, (text_w + 2 * margin) - img_bgr.shape[1])
    img_padded = cv2.copyMakeBorder(img_bgr, 80, 0, 0, right_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    for text, y, scale, thickness, color in header_lines:
        cv2.putText(img_padded, text, (margin, y), font, scale, color, thickness)

    save_path = os.path.join(run_dir, "vae_reconstructions.png")
    cv2.imwrite(save_path, img_padded)
    print(f"Saved VAE reconstruction grid to: {save_path}")


def _to_rgb(frame, out_wh):
    """(C,H,W) float tensor in [0,1] -> upscaled RGB uint8 image of size out_wh=(w,h)."""
    img = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(img, out_wh, interpolation=cv2.INTER_NEAREST)


def save_rollout_video(predict_fn, traj_dir, device, run_dir, context_len=5, n_steps=40, fps=10, scale=4):
    """Side-by-side rollout GIF comparing two prediction regimes:
      * Free-running: seed with `context_len` real frames, then feed the model's
        own predictions back in (errors compound over the horizon).
      * Teacher forcing: every step predicts a single frame from a sliding window
        of real frames, so each prediction is a clean 1-step-ahead forecast.
    `predict_fn(context)` takes a (1, T, C, H, W) tensor and returns (1, C, H, W).
    """
    to_tensor = transforms.ToTensor()
    frame_paths = sorted(glob.glob(os.path.join(traj_dir, "*.png")))
    n_steps = min(n_steps, len(frame_paths) - context_len)
    if n_steps <= 0:
        print(f"[Rollout] {traj_dir} too short ({len(frame_paths)} frames), skipping.")
        return

    frames = torch.stack([to_tensor(Image.open(p).convert("RGB"))
                          for p in frame_paths[:context_len + n_steps]]).to(device)
    _, H, W = frames.shape[1:]

    free_preds = []
    tf_preds = []
    with torch.no_grad():
        # Free-running: context absorbs the model's own predictions.
        context = frames[:context_len].unsqueeze(0)  # (1, T, C, H, W)
        for _ in range(n_steps):
            pred_frame = predict_fn(context)          # (1, C, H, W)
            free_preds.append(pred_frame.squeeze(0))
            context = torch.cat([context[:, 1:], pred_frame.unsqueeze(1)], dim=1)

        # Teacher forcing: each step's context is the real frames preceding the target.
        for k in range(n_steps):
            tf_context = frames[k:k + context_len].unsqueeze(0)
            tf_preds.append(predict_fn(tf_context).squeeze(0))

    free_preds = torch.stack(free_preds)              # (n_steps, C, H, W)
    tf_preds = torch.stack(tf_preds)
    gt = frames[context_len:context_len + n_steps]

    cell = (W * scale, H * scale)
    head_h = 22
    labels = ["Ground Truth", "Free-run Pred", "Free-run Err", "Teacher-forced Pred", "TF Err"]
    total_w = len(labels) * W * scale

    header = np.full((head_h, total_w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, label in enumerate(labels):
        cv2.putText(header, label, (k * W * scale + 8, 15), font, 0.4, (0, 0, 0), 1)

    gif_frames = []
    for k in range(n_steps):
        free_err = (gt[k] - free_preds[k]).abs().clamp(0, 1)
        tf_err = (gt[k] - tf_preds[k]).abs().clamp(0, 1)
        row = np.concatenate([
            _to_rgb(gt[k], cell),
            _to_rgb(free_preds[k], cell),
            _to_rgb(free_err, cell),
            _to_rgb(tf_preds[k], cell),
            _to_rgb(tf_err, cell),
        ], axis=1)
        gif_frames.append(Image.fromarray(np.concatenate([header, row], axis=0)))

    name = os.path.basename(traj_dir)
    gif_path = os.path.join(run_dir, f"rollout_{name}.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    print(f"Saved {n_steps}-step rollout GIF to: {gif_path}")