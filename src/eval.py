import os
import glob
import json
import math
import time
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


@torch.no_grad()
def flow_sample(dit, context_latents, num_steps):
    """Euler ODE solver for the rectified flow: integrate the DiT's learned velocity field from
    noise (t=0) to the predicted next-frame-chunk latents (t=1).

    context_latents: (B, T, Cl, h, w). Returns the chunk z_1: (B, K, Cl, h, w) where K =
    dit.chunk_len. Starts at z ~ N(0, I) and takes `num_steps` equal Euler steps
    z <- z + v(z, t) * dt,  dt = 1 / num_steps. One call produces all K frames at once. """
    dit.eval()
    B = context_latents.shape[0]
    device = context_latents.device
    z_t = torch.randn(B, dit.chunk_len, dit.latent_ch, dit.grid, dit.grid, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = dit(z_t, t, context_latents)
        z_t = z_t + v * dt
    return z_t


@torch.no_grad()
def _chunk_rollout(ae, dit, z_seq, future_frames, z_future, num_steps):
    """One free-running chunk rollout. Returns per-SAMPLE (not batch-averaged) per-step pixel and
    latent MSE -- (B, K) each -- plus the step-0 prediction (B, C, H, W). Per-sample errors let the
    caller do best-of-N selection across independent rollouts (each call re-draws the ODE noise). """
    B, K = future_frames.shape[0], future_frames.shape[1]
    T = z_seq.shape[1]
    pix = torch.empty(B, K, device=future_frames.device)
    lat = torch.empty(B, K, device=future_frames.device)
    step0 = None
    z_run = z_seq
    k = 0
    while k < K:
        z_chunk = flow_sample(dit, z_run, num_steps)         # (B, chunk, Cl, gh, gw)
        chunk = z_chunk.shape[1]
        for j in range(chunk):
            if k >= K:
                break
            z_pred = z_chunk[:, j]
            pred = ae.decode(z_pred * dit.latent_scale)          # normalized latent -> VAE scale
            pix[:, k] = ((pred - future_frames[:, k]) ** 2).flatten(1).mean(1)
            lat[:, k] = ((z_pred - z_future[:, k]) ** 2).flatten(1).mean(1)
            if k == 0:
                step0 = pred
            k += 1
        z_run = torch.cat([z_run, z_chunk], dim=1)[:, -T:]   # slide context by the chunk
    return pix, lat, step0


def run_evaluation(test_loader, device, run_dir, ae, dit, num_steps, save_images=True,
                   max_batches=None, best_of_n=1):
    """Free-running chunk-rollout evaluation for the VAE + Flow Matching DiT.

    At each horizon step the next-frame latent is sampled by integrating the DiT's velocity field
    (Euler ODE, `num_steps`) from noise, conditioned on the current latent context window; the
    prediction is decoded for the pixel metric and fed back as context (free-running, so error
    compounds with horizon). Persistence -- holding the last real frame/latent constant -- is the
    no-op baseline.

    `max_batches` caps how many test batches are rolled out (each window x rollout costs
    horizon x num_steps DiT forwards, so the full overlapping set is minutes to tens of minutes).

    `best_of_n` > 1 samples that many INDEPENDENT rollouts per window and additionally reports the
    best one (per trajectory, by mean pixel MSE). Single-sample MSE-to-truth punishes the flow
    model for committing to a valid-but-different future (e.g. which way a ball bounces); best-of-N
    asks whether at least one sample tracks the truth -- i.e. whether the model captured the
    dynamics distribution or genuinely failed. It multiplies eval cost by N. """
    model_type = "FlowMatch"
    print(f"\n--- Phase 3: Final Test Set Evaluation ({model_type}) ---")

    vis_ctx = vis_target = vis_pred = None
    bestN = best_of_n > 1

    metric_keys = ["model_pix_mse", "persist_pix_mse", "model_lat_mse", "persist_lat_mse"]
    if bestN:
        metric_keys += ["bestN_pix_mse", "bestN_lat_mse"]
    stats = None
    total = 0
    K = 0

    nb_eval = len(test_loader) if max_batches is None else min(len(test_loader), max_batches)
    print(f"Rolling out {nb_eval} test batches (capped from {len(test_loader)}), best_of_n={best_of_n}; "
          f"each window x rollout = horizon x {num_steps} Euler steps of DiT forwards.")

    t_start = time.time()
    with torch.no_grad():
        for i, (ctx_frames, future_frames) in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                break
            ctx_frames = ctx_frames.to(device)
            future_frames = future_frames.to(device)
            if future_frames.dim() == 4:
                future_frames = future_frames.unsqueeze(1)
            B, T, C, H, W = ctx_frames.shape
            K = future_frames.shape[1]
            last_frame = ctx_frames[:, -1]

            if stats is None:
                stats = {key: [0.0] * K for key in metric_keys}

            # Encode into the DiT's normalized latent space (/ scale); all latent metrics below and
            # the rollout context therefore live in that same space, consistent with training.
            z = ae.encode(ctx_frames.view(-1, C, H, W)) / dit.latent_scale   # VAE posterior mean (mu)
            z_seq = z.view(B, T, *z.shape[1:])
            z_last = z_seq[:, -1]                                   # last real frame/latent (persistence)
            z_future = ae.encode(future_frames.reshape(B * K, C, H, W)).view(B, K, *z.shape[1:]) / dit.latent_scale

            # Persistence baseline (hold the last real frame/latent constant), per sample per step.
            for k in range(K):
                stats["persist_pix_mse"][k] += ((last_frame - future_frames[:, k]) ** 2).flatten(1).mean(1).sum().item()
                stats["persist_lat_mse"][k] += ((z_last - z_future[:, k]) ** 2).flatten(1).mean(1).sum().item()

            # N independent rollouts (each re-draws the ODE noise inside flow_sample).
            pix_runs, lat_runs, step0_first = [], [], None
            for n in range(best_of_n):
                pix, lat, step0 = _chunk_rollout(ae, dit, z_seq, future_frames, z_future, num_steps)
                pix_runs.append(pix)
                lat_runs.append(lat)
                if n == 0:
                    step0_first = step0
            pix_stack = torch.stack(pix_runs)                      # (N, B, K)
            lat_stack = torch.stack(lat_runs)

            # Single-sample metrics use the first rollout.
            for k in range(K):
                stats["model_pix_mse"][k] += pix_stack[0, :, k].sum().item()
                stats["model_lat_mse"][k] += lat_stack[0, :, k].sum().item()

            # Best-of-N: per trajectory, keep the rollout with the lowest mean pixel MSE.
            if bestN:
                best = pix_stack.mean(dim=2).argmin(dim=0)         # (B,) winning rollout per sample
                ar = torch.arange(B, device=device)
                best_pix, best_lat = pix_stack[best, ar], lat_stack[best, ar]   # (B, K)
                for k in range(K):
                    stats["bestN_pix_mse"][k] += best_pix[:, k].sum().item()
                    stats["bestN_lat_mse"][k] += best_lat[:, k].sum().item()

            if i == 0 and save_images:
                vis_ctx, vis_target, vis_pred = last_frame, future_frames[:, 0], step0_first

            total += B
            if (i + 1) % 5 == 0 or (i + 1) == nb_eval:
                print(f"  [Eval] {i+1}/{nb_eval} batches | {time.time()-t_start:.0f}s elapsed", flush=True)

    elapsed = time.time() - t_start

    if not total:
        print("[Eval] No test samples (trajectories too short for context+horizon?). Skipping metrics.")
        return

    # Inference cost: one ODE solve (`num_steps` DiT forwards) produces a whole K=chunk_len chunk,
    # so per-frame cost is ~num_steps/K forwards; best_of_n multiplies the total.
    chunk_len = dit.chunk_len
    n_frames = total * K * best_of_n
    chunks_per_window = (K + chunk_len - 1) // chunk_len
    fwd_passes = total * best_of_n * chunks_per_window * num_steps
    ms_per_frame = 1000.0 * elapsed / max(1, n_frames)
    print(f"\n[Cost] inference_steps={num_steps}, chunk_len={chunk_len}, best_of_n={best_of_n} -> {num_steps} "
          f"DiT forwards per {chunk_len}-frame chunk (~{num_steps/chunk_len:.1f} forwards/frame).")
    print(f"[Cost] Sampled {n_frames} frames = {fwd_passes} forward passes in {elapsed:.1f}s "
          f"({ms_per_frame:.1f} ms/frame). Time scales ~linearly with inference_steps x best_of_n.")

    avg = {key: [s / total for s in stats[key]] for key in metric_keys}

    def _skill(model_v, persist_v):
        return (persist_v - model_v) / persist_v * 100 if persist_v > 0 else 0.0

    # Pixel-space table, one row per rollout step.
    print("\nPixel-space metrics by rollout horizon (model = free-running, fed its own predictions):")
    hdr = f"{'Step':>4} | {'Model MSE':>10} | {'Persist MSE':>11} | {'Skill%':>7} | {'Model PSNR':>10}"
    if bestN:
        hdr += f" | {'BestN MSE':>10} | {'BestN PSNR':>10} | {'BestN Sk%':>9}"
    print(hdr)
    for k in range(K):
        m, p = avg["model_pix_mse"][k], avg["persist_pix_mse"][k]
        row = f"{k+1:>4} | {m:>10.5f} | {p:>11.5f} | {_skill(m, p):>6.1f}% | {_psnr(m):>10.2f}"
        if bestN:
            bm = avg["bestN_pix_mse"][k]
            row += f" | {bm:>10.5f} | {_psnr(bm):>10.2f} | {_skill(bm, p):>8.1f}%"
        print(row)

    mean_m = sum(avg["model_pix_mse"]) / K
    mean_p = sum(avg["persist_pix_mse"]) / K
    line = f"  avg over {K} steps -> Model MSE {mean_m:.5f} | Persist MSE {mean_p:.5f} | Skill {_skill(mean_m, mean_p):+.1f}%"
    if bestN:
        mean_b = sum(avg["bestN_pix_mse"]) / K
        line += f" | BestN MSE {mean_b:.5f} | BestN Skill {_skill(mean_b, mean_p):+.1f}%"
    print(line)

    print("\nLatent-space metrics by rollout horizon (what the flow model optimizes):")
    print(f"{'Step':>4} | {'Model MSE':>10} | {'Persist MSE':>11} | {'Skill%':>7}")
    for k in range(K):
        m, p = avg["model_lat_mse"][k], avg["persist_lat_mse"][k]
        print(f"{k+1:>4} | {m:>10.5f} | {p:>11.5f} | {_skill(m, p):>6.1f}%")

    # Headline verdict uses single-step pixel MSE (step 1).
    m1, p1 = avg["model_pix_mse"][0], avg["persist_pix_mse"][0]
    verdict = "beats persistence" if m1 < p1 else "WORSE than persistence (no dynamics learned)"
    print(f"\n-> 1-step: Model is {_skill(m1, p1):+.1f}% vs persistence in pixel MSE  [{verdict}]")
    mK, pK = avg["model_pix_mse"][K-1], avg["persist_pix_mse"][K-1]
    if K > 1:
        print(f"-> {K}-step: Model is {_skill(mK, pK):+.1f}% vs persistence in pixel MSE")
    if bestN:
        mean_b = sum(avg["bestN_pix_mse"]) / K
        print(f"-> best-of-{best_of_n}: 1-step {_psnr(avg['bestN_pix_mse'][0]):.2f} dB vs single {_psnr(m1):.2f} dB | "
              f"avg skill {_skill(mean_b, mean_p):+.1f}% vs single {_skill(mean_m, mean_p):+.1f}%  "
              f"(big gap => metric was punishing valid alternative futures)")

    # Persist everything we just printed so runs can be compared without re-parsing stdout.
    results = {
        "model_type": model_type,
        "num_samples": total,
        "horizon": K,
        "inference_steps": num_steps,
        "best_of_n": best_of_n,
        "cost": {
            "frames_sampled": n_frames,
            "dit_forward_passes": fwd_passes,
            "eval_seconds": elapsed,
            "ms_per_frame": ms_per_frame,
        },
        "per_step": {
            "model_pix_mse": avg["model_pix_mse"],
            "persist_pix_mse": avg["persist_pix_mse"],
            "model_pix_psnr": [_psnr(v) for v in avg["model_pix_mse"]],
            "persist_pix_psnr": [_psnr(v) for v in avg["persist_pix_mse"]],
            "pix_skill_pct": [_skill(m, p) for m, p in zip(avg["model_pix_mse"], avg["persist_pix_mse"])],
            "model_lat_mse": avg["model_lat_mse"],
            "persist_lat_mse": avg["persist_lat_mse"],
            "lat_skill_pct": [_skill(m, p) for m, p in zip(avg["model_lat_mse"], avg["persist_lat_mse"])],
        },
        "pix_mse_mean": {"model": mean_m, "persist": mean_p, "skill_pct": _skill(mean_m, mean_p)},
        "headline": {
            "step1_model_pix_mse": m1,
            "step1_model_pix_psnr": _psnr(m1),
            "step1_skill_pct": _skill(m1, p1),
            "beats_persistence": m1 < p1,
        },
    }
    if K > 1:
        results["headline"]["stepK_skill_pct"] = _skill(mK, pK)
    if bestN:
        mean_b = sum(avg["bestN_pix_mse"]) / K
        results["per_step"]["bestN_pix_mse"] = avg["bestN_pix_mse"]
        results["per_step"]["bestN_pix_psnr"] = [_psnr(v) for v in avg["bestN_pix_mse"]]
        results["per_step"]["bestN_lat_mse"] = avg["bestN_lat_mse"]
        results["pix_mse_mean"]["bestN"] = mean_b
        results["headline"]["bestN_step1_pix_psnr"] = _psnr(avg["bestN_pix_mse"][0])
        results["headline"]["bestN_mean_skill_pct"] = _skill(mean_b, mean_p)

    results_path = os.path.join(run_dir, f"eval_results_{model_type}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {model_type} evaluation results to: {results_path}")

    if save_images and vis_pred is not None:
        num_samples = min(8, vis_target.size(0))
        # 1-step PSNR over the shown samples (single-sample prediction), echoed in the header.
        vis_mse = ((vis_pred[:num_samples] - vis_target[:num_samples]) ** 2).mean().item()
        vis_psnr = _psnr(vis_mse)
        grid_imgs = []
        for j in range(num_samples):
            ctx_rgb = vis_ctx[j].cpu()
            target_rgb = vis_target[j].cpu()
            pred_rgb = vis_pred[j].cpu()

            error_rgb = torch.clamp(torch.abs(target_rgb - pred_rgb) * 2.0, 0, 1)
            grid_imgs.extend([ctx_rgb, target_rgb, pred_rgb, error_rgb])

        img_grid = torchvision.utils.make_grid(grid_imgs, nrow=4, pad_value=0.5)

        img_np = img_grid.permute(1, 2, 0).cpu().numpy() * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Header lines: (text, y, font_scale, thickness, color_BGR).
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 10
        header_lines = [
            (f"Model: {model_type}  |  1-step PSNR {vis_psnr:.2f} dB", 25, 0.6, 2, (0, 0, 0)),
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


def _save_reconstruction_grid(frames, recon, run_dir, filename, title):
    """Write a [Truth | Reconstruction | Abs. Error x2] grid PNG with a header line `title`."""
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
        (title, 25, 0.6, 2, (0, 0, 0)),
        ("Cols: [Truth] | [Reconstruction] | [Abs. Error x2]", 50, 0.5, 1, (0, 0, 0)),
        ("Abs. Error: Black = Perfect Match | Colors = Detail lost by the latent", 70, 0.45, 1, (0, 0, 200)),
    ]
    text_w = max(cv2.getTextSize(t, font, s, th)[0][0] for t, _, s, th, _ in header_lines)
    right_pad = max(0, (text_w + 2 * margin) - img_bgr.shape[1])
    img_padded = cv2.copyMakeBorder(img_bgr, 80, 0, 0, right_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    for text, y, scale, thickness, color in header_lines:
        cv2.putText(img_padded, text, (margin, y), font, scale, color, thickness)

    save_path = os.path.join(run_dir, filename)
    cv2.imwrite(save_path, img_padded)
    print(f"Saved reconstruction grid to: {save_path}")


def _recon_batch_frames(loader, device, n_samples, tag):
    """Pull one batch from `loader` and return the first `n_samples` real frames, or None."""
    batch = next(iter(loader), None)
    if batch is None:
        print(f"[{tag}] No frames available for reconstruction grid; skipping.")
        return None
    ctx_frames, _ = batch
    return ctx_frames[:, -1].to(device)[:n_samples]      # one real frame per sample


@torch.no_grad()
def save_vae_reconstructions(ae, loader, device, run_dir, n_samples=8):
    """[Truth | Reconstruction | Abs. Error] grid for the VAE.

    Encodes real frames to the posterior mean and decodes them -- the exact deterministic
    path the dynamics model relies on -- so the image is a direct read on how much detail the
    latent throws away. Written every run (even when the VAE is loaded from a checkpoint).
    """
    ae.eval()
    frames = _recon_batch_frames(loader, device, n_samples, "VAE")
    if frames is None:
        return
    recon = ae.decode(ae.encode(frames)).clamp(0, 1)
    mse_val = nn.MSELoss()(recon, frames).item()
    psnr_val = _psnr(mse_val)

    # Perceptual (LPIPS) read on the recon: lower = sharper / more on the natural-image manifold.
    # This is the distance the perceptual VAE objective actually targets, so it is the honest
    # quality number once pixel-MSE stops being meaningful. Skipped if `lpips` isn't installed.
    from src.train import build_lpips
    perceptual = build_lpips(device)
    lpips_str = ""
    if perceptual is not None:
        lpips_val = perceptual(recon, frames, normalize=True).mean().item()
        lpips_str = f"  LPIPS {lpips_val:.4f}"
    print(f"[VAE] Reconstruction over {frames.size(0)} frames -> "
          f"MSE {mse_val:.6f} | PSNR {psnr_val:.2f} dB{lpips_str}")
    _save_reconstruction_grid(frames, recon, run_dir,
                              "vae_reconstructions.png",
                              f"VAE reconstruction  |  MSE {mse_val:.6f}  PSNR {psnr_val:.2f} dB{lpips_str}")


def _to_rgb(frame, out_wh):
    """(C,H,W) float tensor in [0,1] -> upscaled RGB uint8 image of size out_wh=(w,h)."""
    img = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(img, out_wh, interpolation=cv2.INTER_NEAREST)


def save_rollout_video(predict_chunk_fn, traj_dir, device, run_dir, context_len=5, n_steps=40, fps=10, scale=4):
    """Side-by-side rollout GIF comparing two prediction regimes, stepping in chunks of K frames:
      * Free-running: seed with `context_len` real frames, generate a K-frame chunk, append it,
        slide the context forward by K using the model's OWN predictions (errors compound).
      * Teacher forcing: predict each K-frame chunk from a window of REAL frames and slide by K
        over real frames, so predictions never feed back (clean K-step-ahead forecast).
    `predict_chunk_fn(context)` takes a (1, T, C, H, W) tensor and returns (1, K, C, H, W).
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
        # Free-running: context absorbs the model's own predictions, K frames at a time.
        context = frames[:context_len].unsqueeze(0)  # (1, T, C, H, W)
        while len(free_preds) < n_steps:
            chunk = predict_chunk_fn(context)[0]      # (K, C, H, W)
            free_preds.extend(chunk.unbind(0))
            context = torch.cat([context, chunk.unsqueeze(0)], dim=1)[:, -context_len:]

        # Teacher forcing: predict each chunk from REAL context, slide by K over real frames.
        p = 0
        while p < n_steps:
            tf_context = frames[p:p + context_len].unsqueeze(0)
            chunk = predict_chunk_fn(tf_context)[0]   # (K, C, H, W) -> gt positions p .. p+K-1
            tf_preds.extend(chunk.unbind(0))
            p += chunk.shape[0]

    free_preds = torch.stack(free_preds[:n_steps])    # (n_steps, C, H, W)
    tf_preds = torch.stack(tf_preds[:n_steps])
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