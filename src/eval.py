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
from src.frameio import load_frames


def _psnr(mse_val):
    if mse_val <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_val)


@torch.no_grad()
def flow_sample(dit, context_latents, num_steps):
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


def _traj_event_labels(traj_dir, count, gravity=-0.5, kick_thresh=1.5, pair_dist=18.0, post_frames=3):
    pos = np.load(os.path.join(traj_dir, "positions.npy"))
    vel = np.load(os.path.join(traj_dir, "velocities.npy"))
    T = min(len(pos), count)
    labels = np.zeros(count, dtype=np.int8)
    if T < 2:
        return labels
    dv = vel[1:T] - vel[:T - 1]
    dv[:, :, 1] -= gravity
    mag = np.linalg.norm(dv, axis=2)
    bb = np.zeros(T - 1, dtype=bool)
    wall = np.zeros(T - 1, dtype=bool)
    for t in range(T - 1):
        hit = np.where(mag[t] > kick_thresh)[0]
        paired = set()
        for a in range(len(hit)):
            for b in range(a + 1, len(hit)):
                i, j = hit[a], hit[b]
                if np.linalg.norm(pos[t, i] - pos[t, j]) < pair_dist:
                    paired.update((i, j))
        if paired:
            bb[t] = True
        if any(i not in paired for i in hit):
            wall[t] = True
    for f in range(1, T):
        t = f - 1
        if bb[t]:
            labels[f] = 3
        elif bb[max(0, t - post_frames):t].any():
            labels[f] = 2
        elif wall[t]:
            labels[f] = 1
    return labels


COLL_EVENT_NAMES = ["free flight", "wall bounce", "post ball-ball (1-3f)", "ball-ball contact"]


def build_event_labels(frame_cache, trajs):
    n_frames = getattr(frame_cache, "n_frames", None) or frame_cache.frames.shape[0]
    labels_g = torch.zeros(n_frames, dtype=torch.int8)
    n_missing = 0
    for t in sorted(trajs):
        name = os.path.basename(t)
        if name not in frame_cache.ranges:
            continue
        start, count = frame_cache.ranges[name]
        if not os.path.exists(os.path.join(t, "positions.npy")):
            n_missing += 1
            continue
        labels_g[start:start + count] = torch.from_numpy(_traj_event_labels(t, count))
    return labels_g, n_missing


@torch.no_grad()
def collision_conditioned_eval(ae, dit, z_all, frame_cache, test_trajs, context_len,
                               num_steps, run_dir, device, batch_size=256, n_traj=500):
    # Score a fixed, bounded number of held-out trajectories -- NOT a fraction of
    # the dataset (that would decode a huge subset into RAM on a large set). At 500
    # trajs even the rarest class (ball-ball contact, ~2-5% of windows) keeps
    # hundreds-to-thousands of samples and its mean sits within ~0.02px of the
    # 1000-traj value (measured on shapes), while decode/RAM stay flat at any dataset
    # size. Fixed seed for reproducibility.
    test_trajs = sorted(test_trajs)
    if n_traj and len(test_trajs) > n_traj:
        g = torch.Generator().manual_seed(0)
        pick = torch.randperm(len(test_trajs), generator=g)[:n_traj].tolist()
        test_trajs = [test_trajs[i] for i in pick]
    if hasattr(frame_cache, "subset"):   # FrameStore -> decode only these trajs
        frame_cache = frame_cache.subset(test_trajs)
    ctx_idx, tgt_idx = frame_cache.build_windows(test_trajs, context_len, 1)
    if ctx_idx.shape[0] == 0:
        print("[CollEval] No 1-step windows; skipping collision-conditioned eval.")
        return
    labels_g, n_missing = build_event_labels(frame_cache, test_trajs)
    if n_missing:
        print(f"[CollEval] {n_missing} test trajs lack positions.npy (labeled free flight).")
    win_labels = labels_g[tgt_idx[:, 0].cpu()]
    print(f"[CollEval] Scoring {ctx_idx.shape[0]} windows from {len(test_trajs)} held-out trajectories.")
    ae.eval(); dit.eval()
    scale = dit.latent_scale
    frames = frame_cache.frames
    use_amp = str(device).startswith("cuda")
    se = torch.zeros(4, dtype=torch.float64)
    cnt = torch.zeros(4, dtype=torch.float64)
    for i in range(0, ctx_idx.shape[0], batch_size):
        ci, ti = ctx_idx[i:i + batch_size], tgt_idx[i:i + batch_size, 0]
        z_ctx = z_all[ci].float().to(device)
        tgt = frames[ti].to(device).float().div_(255.0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            z_pred = flow_sample(dit, z_ctx, num_steps)[:, 0].float()
            pred = ae.decode(z_pred * scale).float()
        err = ((pred - tgt) ** 2).flatten(1).mean(1).double().cpu()
        lab = win_labels[i:i + batch_size]
        for k in range(4):
            m = lab == k
            if m.any():
                se[k] += err[m].sum()
                cnt[k] += int(m.sum())

    print("\n[CollEval] 1-step prediction error by event at the target frame:")
    print(f"{'event':>24} | {'windows':>8} | {'MSE':>10} | {'PSNR':>8} | {'vs free':>8}")
    results = {}
    free_mse = (se[0] / cnt[0]).item() if cnt[0] > 0 else float("nan")
    for k in range(4):
        if cnt[k] == 0:
            continue
        mse = (se[k] / cnt[k]).item()
        ratio = mse / free_mse if free_mse > 0 else float("nan")
        print(f"{COLL_EVENT_NAMES[k]:>24} | {int(cnt[k]):>8} | {mse:>10.6f} | {_psnr(mse):>7.2f} | {ratio:>7.2f}x")
        results[COLL_EVENT_NAMES[k]] = {"windows": int(cnt[k]), "mse": mse,
                                        "psnr": _psnr(mse), "mse_vs_free": ratio}
    path = os.path.join(run_dir, "collision_eval.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[CollEval] Saved to {path}")


@torch.no_grad()
def _chunk_rollout(ae, dit, z_seq, future_frames, z_future, num_steps, use_amp=False):
    B, K = future_frames.shape[0], future_frames.shape[1]
    T = z_seq.shape[1]
    pix = torch.empty(B, K, device=future_frames.device)
    lat = torch.empty(B, K, device=future_frames.device)
    step0 = None
    z_run = z_seq
    k = 0
    while k < K:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            z_chunk = flow_sample(dit, z_run, num_steps)
        chunk = z_chunk.shape[1]
        for j in range(chunk):
            if k >= K:
                break
            z_pred = z_chunk[:, j].float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = ae.decode(z_pred * dit.latent_scale).float()
            pix[:, k] = ((pred - future_frames[:, k]) ** 2).flatten(1).mean(1)
            lat[:, k] = ((z_pred - z_future[:, k]) ** 2).flatten(1).mean(1)
            if k == 0:
                step0 = pred
            k += 1
        z_run = torch.cat([z_run, z_chunk], dim=1)[:, -T:]
    return pix, lat, step0


def run_evaluation(test_loader, device, run_dir, ae, dit, num_steps, save_images=True,
                   max_batches=None, best_of_n=1, n_pngs=1, compile_mode="off"):
    model_type = "FlowMatch"
    print(f"\n--- Phase 4: Final Test Set Evaluation ({model_type}) ---")

    use_amp = str(device).startswith("cuda")
    dit_run = dit
    if compile_mode == "on" and use_amp:
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            dit_run = torch.compile(dit)
            print("[Eval] torch.compile ON (Inductor); the first batch runs slow while graphs build.")
        except Exception as e:
            dit_run = dit
            print(f"[Eval] torch.compile unavailable ({e}) -- running eager.")

    vis_ctx_l, vis_tgt_l, vis_pred_l = [], [], []
    n_vis = n_pngs * 8
    n_vis_collected = 0
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

            # encode in slices: one shot on B*K frames (2560 at the default
            # 32x80 eval) is the pipeline's VRAM peak -- with VAE + DiT loaded
            # it grazed 23.5GB at the 8x8 latent and would OOM at 16x16
            def _enc(x, chunk=512):
                return torch.cat([ae.encode(x[i:i + chunk])
                                  for i in range(0, x.shape[0], chunk)])
            z = _enc(ctx_frames.view(-1, C, H, W)) / dit.latent_scale
            z_seq = z.view(B, T, *z.shape[1:])
            z_last = z_seq[:, -1]
            z_future = _enc(future_frames.reshape(B * K, C, H, W)).view(B, K, *z.shape[1:]) / dit.latent_scale

            for k in range(K):
                stats["persist_pix_mse"][k] += ((last_frame - future_frames[:, k]) ** 2).flatten(1).mean(1).sum().item()
                stats["persist_lat_mse"][k] += ((z_last - z_future[:, k]) ** 2).flatten(1).mean(1).sum().item()

            pix_runs, lat_runs, step0_first = [], [], None
            for n in range(best_of_n):
                pix, lat, step0 = _chunk_rollout(ae, dit_run, z_seq, future_frames, z_future, num_steps, use_amp=use_amp)
                pix_runs.append(pix)
                lat_runs.append(lat)
                if n == 0:
                    step0_first = step0
            pix_stack = torch.stack(pix_runs)
            lat_stack = torch.stack(lat_runs)

            for k in range(K):
                stats["model_pix_mse"][k] += pix_stack[0, :, k].sum().item()
                stats["model_lat_mse"][k] += lat_stack[0, :, k].sum().item()

            if bestN:
                best = pix_stack.mean(dim=2).argmin(dim=0)
                ar = torch.arange(B, device=device)
                best_pix, best_lat = pix_stack[best, ar], lat_stack[best, ar]
                for k in range(K):
                    stats["bestN_pix_mse"][k] += best_pix[:, k].sum().item()
                    stats["bestN_lat_mse"][k] += best_lat[:, k].sum().item()

            if save_images and n_vis_collected < n_vis:
                take = min(B, n_vis - n_vis_collected)
                vis_ctx_l.append(last_frame[:take])
                vis_tgt_l.append(future_frames[:take, 0])
                vis_pred_l.append(step0_first[:take])
                n_vis_collected += take

            total += B
            if (i + 1) % 5 == 0 or (i + 1) == nb_eval:
                print(f"  [Eval] {i+1}/{nb_eval} batches | {time.time()-t_start:.0f}s elapsed", flush=True)

    elapsed = time.time() - t_start

    if not total:
        print("[Eval] No test samples (trajectories too short for context+horizon?). Skipping metrics.")
        return

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

    if save_images and n_vis_collected > 0:
        vis_ctx = torch.cat(vis_ctx_l)
        vis_target = torch.cat(vis_tgt_l)
        vis_pred = torch.cat(vis_pred_l)
        num_samples = min(n_vis, vis_target.size(0))
        per_page = 8
        n_pages = (num_samples + per_page - 1) // per_page

        for page in range(n_pages):
            lo, hi = page * per_page, min((page + 1) * per_page, num_samples)
            vis_mse = ((vis_pred[lo:hi] - vis_target[lo:hi]) ** 2).mean().item()
            vis_psnr = _psnr(vis_mse)
            grid_imgs = []
            for j in range(lo, hi):
                ctx_rgb = vis_ctx[j].cpu()
                target_rgb = vis_target[j].cpu()
                pred_rgb = vis_pred[j].cpu()

                error_rgb = torch.clamp(torch.abs(target_rgb - pred_rgb) * 2.0, 0, 1)
                grid_imgs.extend([ctx_rgb, target_rgb, pred_rgb, error_rgb])

            img_grid = torchvision.utils.make_grid(grid_imgs, nrow=4, pad_value=0.5)

            img_np = img_grid.permute(1, 2, 0).cpu().numpy() * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            font = cv2.FONT_HERSHEY_SIMPLEX
            margin = 10
            title = f"Model: {model_type}  |  1-step PSNR {vis_psnr:.2f} dB"
            if n_pages > 1:
                title += f"  |  Page {page + 1}/{n_pages} (samples {lo + 1}-{hi})"
            header_lines = [
                (title, 25, 0.6, 2, (0, 0, 0)),
                ("Cols: [Frame t-1] | [Truth t] | [Pred t] | [Abs. Error]", 50, 0.5, 1, (0, 0, 0)),
                ("Abs. Error: Black = Perfect Match | Colors = Deviation in predicted physics/materials", 70, 0.45, 1, (0, 0, 200)),
            ]

            text_w = max(cv2.getTextSize(t, font, s, th)[0][0] for t, _, s, th, _ in header_lines)
            right_pad = max(0, (text_w + 2 * margin) - img_bgr.shape[1])
            header_height = 80
            img_padded = cv2.copyMakeBorder(img_bgr, header_height, 0, 0, right_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            for text, y, scale, thickness, color in header_lines:
                cv2.putText(img_padded, text, (margin, y), font, scale, color, thickness)

            suffix = f"_{page + 1}" if n_pages > 1 else ""
            save_path = os.path.join(run_dir, f"eval_predictions_{model_type}{suffix}.png")
            cv2.imwrite(save_path, img_padded)
            print(f"Saved {model_type} evaluation image grid to: {save_path}")


def _save_reconstruction_grid(frames, recon, run_dir, filename, title):
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
    batch = next(iter(loader), None)
    if batch is None:
        print(f"[{tag}] No frames available for reconstruction grid; skipping.")
        return None
    ctx_frames, _ = batch
    return ctx_frames[:, -1].to(device)[:n_samples]


@torch.no_grad()
def save_vae_reconstructions(ae, loader, device, run_dir, n_samples=8):
    ae.eval()
    frames = _recon_batch_frames(loader, device, n_samples, "VAE")
    if frames is None:
        return
    recon = ae.decode(ae.encode(frames)).clamp(0, 1)
    mse_val = nn.MSELoss()(recon, frames).item()
    psnr_val = _psnr(mse_val)

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
    img = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(img, out_wh, interpolation=cv2.INTER_NEAREST)


def save_rollout_video(predict_chunk_fn, traj_dir, device, run_dir, context_len=5, n_steps=40, fps=10, scale=4):
    frames_np = load_frames(traj_dir)          # (T, H, W, 3) uint8 RGB, packed or PNG
    n_steps = min(n_steps, frames_np.shape[0] - context_len)
    if n_steps <= 0:
        print(f"[Rollout] {traj_dir} too short ({frames_np.shape[0]} frames), skipping.")
        return

    frames = (torch.from_numpy(frames_np[:context_len + n_steps]).permute(0, 3, 1, 2)
              .float().div_(255.0).to(device))
    _, H, W = frames.shape[1:]

    free_preds = []
    tf_preds = []
    with torch.no_grad():
        context = frames[:context_len].unsqueeze(0)
        while len(free_preds) < n_steps:
            chunk = predict_chunk_fn(context)[0]
            free_preds.extend(chunk.unbind(0))
            context = torch.cat([context, chunk.unsqueeze(0)], dim=1)[:, -context_len:]

        p = 0
        while p < n_steps:
            tf_context = frames[p:p + context_len].unsqueeze(0)
            chunk = predict_chunk_fn(tf_context)[0]
            tf_preds.extend(chunk.unbind(0))
            p += chunk.shape[0]

    free_preds = torch.stack(free_preds[:n_steps])
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