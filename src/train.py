import os
import time
import math
import random
import hashlib
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def build_warmup_cosine(optimizer, total_steps, warmup_frac=0.05, min_factor=0.0, warmup_steps=None):
    if warmup_steps is None:
        warmup_steps = max(1, int(total_steps * warmup_frac))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    decay_steps = max(1, total_steps - warmup_steps)
    min_factor = min(max(min_factor, 0.0), 1.0)

    def lr_factor(step):
        if step < warmup_steps:
            return 0.01 + (1.0 - 0.01) * (step / warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_factor + (1.0 - min_factor) * cos

    return optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def _decay_factor(shape, progress):
    """Monotone decay from 1.0 (progress 0) to 0.0 (progress 1). Shapes differ only in
    WHERE they linger: cosine's rate is zero at both endpoints so it crawls near the floor;
    linear holds one constant rate the whole way. That matters at a two-iteration handoff --
    with a matched floor->peak (e.g. 5e-5 -> 5e-5) cosine dwells near that LR at the tail of
    iter 1 AND the head of iter 2, over-training the handoff band; linear does not."""
    p = min(max(progress, 0.0), 1.0)
    if shape == "linear":
        return 1.0 - p
    return 0.5 * (1.0 + math.cos(math.pi * p))


def build_restart_schedule(optimizer, steps_1, lr_1, min_lr_1, steps_2, lr_2, min_lr_2,
                           warmup_frac=0.05, shape="cosine", shape_2="cosine"):
    """Anneal lr_1 -> min_lr_1 over steps_1 (linear warmup first), then a single warm
    restart: jump to lr_2 and anneal to min_lr_2 over steps_2. Weights and optimizer state
    carry over; only the LR jumps. `shape` picks iteration 1's decay curve and `shape_2`
    iteration 2's (see _decay_factor) -- they are independent. Factors are relative to lr_1
    (the optimizer's base LR). steps_2 == 0 reduces exactly to a single decay schedule."""
    warmup_steps = max(0, min(int(round(steps_1 * warmup_frac)), steps_1 - 1))
    decay_1 = max(1, steps_1 - warmup_steps)
    m1 = min(max(min_lr_1 / lr_1 if lr_1 > 0 else 0.0, 0.0), 1.0)
    r2 = lr_2 / lr_1 if lr_1 > 0 else 0.0
    m2 = min(max(min_lr_2 / lr_2 if lr_2 > 0 else 0.0, 0.0), 1.0)

    def lr_factor(step):
        if step < warmup_steps:
            return 0.01 + (1.0 - 0.01) * (step / warmup_steps)
        if step < steps_1 or steps_2 <= 0:
            d = _decay_factor(shape, (step - warmup_steps) / decay_1)
            return m1 + (1.0 - m1) * d
        d = _decay_factor(shape_2, (step - steps_1) / steps_2)
        return r2 * (m2 + (1.0 - m2) * d)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


@torch.no_grad()
def build_latent_cache(ae, store, device, cache_device, batch_size=512,
                       disk_cache_path=None, data_sig=None):
    ae.eval()
    M = store.n_frames

    # Latents are only valid for one exact (VAE weights, dataset) pair, so the
    # disk cache is keyed on both. A single file per dataset: any other VAE
    # overwrites it rather than piling up per-VAE caches.
    sig = None
    if disk_cache_path:
        h = hashlib.sha1()
        for k, v in sorted(ae.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().float().numpy().tobytes())
        sig = {"vae": h.hexdigest(), "data": data_sig}
        if os.path.exists(disk_cache_path):
            try:
                blob = torch.load(disk_cache_path, map_location="cpu", weights_only=True)
                if blob.get("sig") == sig:
                    z_all = blob["z"].to(cache_device)
                    latent_scale = blob["latent_scale"]
                    print(f"[Cache] Loaded latent cache from {disk_cache_path} "
                          f"({tuple(z_all.shape)} float16, scale {latent_scale.item():.4f}) -- skipping encode.")
                    return z_all, latent_scale
                print(f"[Cache] {disk_cache_path} is for a different VAE or dataset -- re-encoding.")
            except Exception as e:
                print(f"[Cache] Failed to load {disk_cache_path} ({e}) -- re-encoding.")

    z_all = None
    print(f"[Cache] Streaming-encoding {M} frames into latent cache (one-time)...")
    pos = 0
    for chunk in store.iter_frames(batch_size=batch_size):
        x = chunk.to(device, non_blocking=True).float().div_(255.0)
        z = ae.encode(x).half()
        if z_all is None:
            z_all = torch.empty((M, *z.shape[1:]), dtype=torch.float16, device=cache_device)
        z_all[pos : pos + z.shape[0]] = z.to(cache_device)
        pos += z.shape[0]
    print(f"[Cache] Latent cache ready: {tuple(z_all.shape)} float16 (~{z_all.numel() * 2 / 1e9:.2f} GB).")

    s = ss = 0.0
    cnt = 0
    for i in range(0, M, batch_size):
        c = z_all[i:i + batch_size].float()
        s += c.sum().item(); ss += (c * c).sum().item(); cnt += c.numel()
    gstd = float(max((ss / cnt - (s / cnt) ** 2) ** 0.5, 1e-6))
    z_all.div_(gstd)
    latent_scale = torch.tensor(gstd)
    print(f"[Cache] Latent scale (std) = {gstd:.4f}; normalized cache to ~unit variance.")

    if disk_cache_path:
        try:
            torch.save({"z": z_all.cpu(), "latent_scale": latent_scale, "sig": sig}, disk_cache_path)
            print(f"[Cache] Saved latent cache to {disk_cache_path} "
                  f"(~{z_all.numel() * 2 / 1e9:.2f} GB, replaces any previous VAE's cache).")
        except Exception as e:
            print(f"[Cache] Could not save latent cache to {disk_cache_path}: {e}")

    return z_all, latent_scale


class _LatentSurrogate(nn.Module):
    def __init__(self, in_ch, out_ch, width=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _probe_metrics(ae, sur, z_all, frames, ctx_va, tgt_va, context_len, grid, scale, batch_size, device):
    sur.eval()
    tgt_mean = z_all[tgt_va[:, 0]].float().mean().item()
    lat_se = lat_var = pix_se = rec_se = 0.0
    lat_n = pix_n = 0
    for i in range(0, ctx_va.shape[0], batch_size):
        ci, ti = ctx_va[i:i + batch_size], tgt_va[i:i + batch_size, 0]
        B = ci.shape[0]
        ctx = z_all[ci].float().to(device).reshape(B, context_len * z_all.shape[1], grid, grid)
        tgt = z_all[ti].float().to(device)
        pred = sur(ctx)
        lat_se += ((pred - tgt) ** 2).sum().item()
        lat_var += ((tgt - tgt_mean) ** 2).sum().item()
        lat_n += tgt.numel()
        frame = frames[ti].float().div(255.0).to(device)
        pix_se += ((ae.decode(pred * scale).clamp(0, 1) - frame) ** 2).sum().item()
        rec_se += ((ae.decode(ae.encode(frame)).clamp(0, 1) - frame) ** 2).sum().item()
        pix_n += frame.numel()
    psnr = lambda se: (10.0 * math.log10(1.0 / (se / pix_n)) if se > 0 else float("inf"))
    return 1.0 - lat_se / lat_var, psnr(pix_se), psnr(rec_se)


def score_latent_predictability(ae, z_all, latent_scale, frame_cache, train_trajs, val_trajs,
                                context_len, device, n_train_traj=500, n_val_traj=100,
                                epochs=30, width=192, batch_size=256, lr=1e-3, seed=0):
    if hasattr(frame_cache, "subset"):   # FrameStore -> decode only the probe trajs
        frame_cache = frame_cache.subset(train_trajs[:n_train_traj] + val_trajs[:n_val_traj])
    ctx_tr, tgt_tr = frame_cache.build_windows(train_trajs[:n_train_traj], context_len, 1)
    ctx_va, tgt_va = frame_cache.build_windows(val_trajs[:n_val_traj], context_len, 1)
    if ctx_tr.shape[0] == 0 or ctx_va.shape[0] == 0:
        print("[LatentProbe] Not enough windows to score latent predictability; skipping.")
        return None

    latent_ch, grid = z_all.shape[1], z_all.shape[-1]
    scale = float(latent_scale)
    frames = frame_cache.frames
    print(f"[LatentProbe] Scoring latent predictability: surrogate {epochs}ep/w{width} on "
          f"{ctx_tr.shape[0]} train / {ctx_va.shape[0]} val 1-step windows...")

    torch.manual_seed(seed)
    sur = _LatentSurrogate(context_len * latent_ch, latent_ch, width).to(device)
    opt = optim.Adam(sur.parameters(), lr=lr)
    S = ctx_tr.shape[0]
    sur.train()
    for _ in range(epochs):
        order = torch.randperm(S)
        for i in range(0, S, batch_size):
            rows = order[i:i + batch_size]
            ci, ti = ctx_tr[rows], tgt_tr[rows, 0]
            B = ci.shape[0]
            ctx = z_all[ci].float().to(device).reshape(B, context_len * latent_ch, grid, grid)
            tgt = z_all[ti].float().to(device)
            loss = F.mse_loss(sur(ctx), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    r2, sur_psnr, rec_psnr = _probe_metrics(ae, sur, z_all, frames, ctx_va, tgt_va,
                                            context_len, grid, scale, batch_size, device)
    print(f"[LatentProbe] Predictability R^2 {r2:.4f} | Surrogate 1-step PSNR {sur_psnr:.2f} dB | "
          f"Recon PSNR {rec_psnr:.2f} dB")
    print("[LatentProbe] Optimise the VAE for HIGH R^2 (predictability), not recon PSNR -- higher "
          "R^2 tracks a better downstream DiT.")
    return {"latent_r2": r2, "surrogate_psnr": sur_psnr, "recon_psnr": rec_psnr}


def _vae_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]


def build_lpips(device, net="alex"):
    try:
        import lpips
    except ImportError:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*'pretrained' is deprecated.*")
        warnings.filterwarnings("ignore", message=r".*Arguments other than a weight enum.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_only=False.*")
        net = lpips.LPIPS(net=net, verbose=False).to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def train_autoencoder(ae, train_loader, val_loader, epochs=5, learning_rate=1e-3,
                      weight_decay=1e-4, kl_weight=1.0, lpips_weight=0.0, lpips_net="alex",
                      focal_weight=0.0, pred_weight=0.0, state_weight=0.0, grad_clip=10.0,
                      precision="bf16", compile_mode="off", device="cuda"):
    print(f"--- Phase 1: Training Autoencoder (VAE) ---")
    optimizer = optim.AdamW(ae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps)
    ae.to(device)

    perceptual = build_lpips(device, net=lpips_net) if lpips_weight > 0 else None
    if lpips_weight > 0 and perceptual is None:
        print("[Warn] lpips not installed (pip install lpips). Falling back to pixel + KL only.")
        lpips_weight = 0.0
    elif perceptual is not None:
        backbone = getattr(perceptual, "pnet_type", "?")
        print(f"[VAE] Perceptual loss ON: LPIPS-{backbone}, weight {lpips_weight}.")
    if focal_weight > 0:
        print(f"[VAE] Focal pixel weighting ON: {focal_weight} (error-proportional, scale-normalized).")
    pred_head = None
    if pred_weight > 0:
        print(f"[VAE] Predictability loss ON: weight {pred_weight} (3x3 linear next-latent "
              f"predictor on consecutive window frames, variance-normalized).")
    state_head = None
    if state_weight > 0:
        print(f"[VAE] State-alignment loss ON: weight {state_weight} (1x1 linear head: latent -> "
              f"per-cell ball presence + offset).")

    on_cuda = str(device).startswith("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    use_amp = amp_dtype is not None and on_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))
    if use_amp:
        print(f"[VAE] Mixed precision ON: {precision} autocast.")

    ae_fwd = ae
    if compile_mode == "on" and on_cuda:
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            ae_fwd = torch.compile(ae)
            print("[VAE] torch.compile ON (Inductor); the first epoch runs slow while graphs build.")
        except Exception as e:
            ae_fwd = ae
            print(f"[VAE] torch.compile unavailable ({e}) -- running eager.")

    for epoch in range(epochs):
        start_time = time.time()

        ae.train()
        tr_recon = torch.zeros((), device=device)
        tr_kl = torch.zeros((), device=device)
        tr_perc = torch.zeros((), device=device)
        tr_gnorm = torch.zeros((), device=device)
        tr_pred = torch.zeros((), device=device)
        tr_state = torch.zeros((), device=device)
        nan_skips = 0
        for batch in train_loader:
            ctx_frames = batch[0]
            aux_ctx = batch[2] if len(batch) > 2 else None
            B, T, C, H, W = ctx_frames.shape
            x = ctx_frames.view(-1, C, H, W).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype if use_amp else torch.bfloat16, enabled=use_amp):
                recon, mu, logvar = ae_fwd(x)
                perc = perceptual(recon, x, normalize=True).mean() if perceptual is not None else None
            recon_err = (recon.float() - x) ** 2
            if focal_weight > 0:
                w = recon_err.detach()
                w = 1.0 + focal_weight * (w / w.mean().clamp_min(1e-12))
                recon_loss = (w * recon_err).sum() / (x.shape[0] * (1.0 + focal_weight))
            else:
                recon_loss = recon_err.sum() / x.shape[0]
            kl = _vae_kl(mu.float(), logvar.float())
            loss = recon_loss + kl_weight * kl
            if perc is not None:
                loss = loss + lpips_weight * perc.float()
                tr_perc += perc.detach().float()
            if pred_weight > 0 and T > 1:
                zc = mu.float().view(B, T, *mu.shape[1:])
                if pred_head is None:
                    pred_head = torch.nn.Conv2d(zc.shape[2], zc.shape[2], 3, padding=1).to(device)
                    optimizer.add_param_group({"params": pred_head.parameters()})
                zprev = zc[:, :-1].reshape(-1, *zc.shape[2:])
                znext = zc[:, 1:].reshape(-1, *zc.shape[2:])
                var = znext.detach().var().clamp_min(1e-8)
                pred_loss = F.mse_loss(pred_head(zprev), znext) / var
                loss = loss + pred_weight * pred_loss
                tr_pred += pred_loss.detach()
            if state_weight > 0 and aux_ctx is not None:
                mu_f = mu.float()
                if state_head is None:
                    state_head = torch.nn.Conv2d(mu_f.shape[1], aux_ctx.shape[2], 1).to(device)
                    optimizer.add_param_group({"params": state_head.parameters()})
                tgt_state = aux_ctx.reshape(-1, *aux_ctx.shape[2:])
                state_loss = F.mse_loss(state_head(mu_f), tgt_state)
                loss = loss + state_weight * state_loss
                tr_state += state_loss.detach()
            if not torch.isfinite(loss):
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_recon += recon_loss.detach()
            tr_kl += kl.detach()
            tr_gnorm += gnorm.detach()

        ae.eval()
        val_mse = torch.zeros((), device=device)
        val_kl = torch.zeros((), device=device)
        val_perc = torch.zeros((), device=device)
        with torch.no_grad():
            for ctx_frames, target_frame in val_loader:
                B, T, C, H, W = ctx_frames.shape
                x = ctx_frames.view(-1, C, H, W).to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype if use_amp else torch.bfloat16, enabled=use_amp):
                    recon, mu, logvar = ae_fwd(x)
                    vperc = perceptual(recon, x, normalize=True).mean() if perceptual is not None else None
                val_mse += F.mse_loss(recon.float(), x)
                val_kl += _vae_kl(mu.float(), logvar.float())
                if vperc is not None:
                    val_perc += vperc.float()

        nb_tr, nb_val = len(train_loader), len(val_loader)
        val_mse_mean = val_mse.item() / nb_val
        val_psnr = 10.0 * math.log10(1.0 / val_mse_mean) if val_mse_mean > 0 else float("inf")
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        perc_str = ""
        if perceptual is not None:
            perc_str = f"Train LPIPS: {tr_perc.item()/nb_tr:.4f} | Val LPIPS: {val_perc.item()/nb_val:.4f} | "
        print(f"AE Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e} | "
              f"GradNorm: {tr_gnorm.item()/nb_tr:.3f} | "
              f"Train Recon(sum): {tr_recon.item()/nb_tr:.4f} | "
              f"Train KL: {tr_kl.item()/nb_tr:.2f} | {perc_str}Val MSE: {val_mse_mean:.8f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val KL: {val_kl.item()/nb_val:.2f}"
              + (f" | Pred: {tr_pred.item()/nb_tr:.4f}" if pred_weight > 0 else "")
              + (f" | State: {tr_state.item()/nb_tr:.4f}" if state_weight > 0 else "")
              + (f" | NaN-skipped: {nan_skips}" if nan_skips else ""))
        if nan_skips > 0.5 * nb_tr:
            print(f"[VAE] Divergence: {nan_skips}/{nb_tr} non-finite steps in epoch {epoch+1} "
                  f"-- weights are almost certainly NaN, aborting training early.")
            break
    return ae


def train_flow_matching(model, train_loader, val_loader, epochs=15, learning_rate=3e-4,
                        min_lr=1e-6, warmup_frac=0.05, lr_schedule="cosine",
                        lr_schedule_2="cosine", epochs_2=0,
                        learning_rate_2=2e-4,
                        min_lr_2=2e-6, weight_decay=1e-4, grad_clip=3.0, ema_decay=0.999,
                        context_noise=0.0, precision="bf16", compile_mode="off",
                        t_dist="logit_normal", loss_type="mse", huber_c=1.0, device="cuda"):
    print("--- Phase 2: Training Flow Matching DiT (chunk prediction) ---")
    model.to(device)
    try:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    except (RuntimeError, TypeError, ValueError):
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    epochs_2 = max(0, int(epochs_2))
    total_epochs = epochs + epochs_2
    warmup_frac = min(max(warmup_frac, 0.0), 1.0)
    steps_1 = epochs * steps_per_epoch
    warmup_steps = max(0, min(int(round(steps_1 * warmup_frac)), steps_1 - 1))
    scheduler = build_restart_schedule(optimizer, steps_1, learning_rate, min_lr,
                                       epochs_2 * steps_per_epoch, learning_rate_2, min_lr_2,
                                       warmup_frac=warmup_frac, shape=lr_schedule,
                                       shape_2=lr_schedule_2)
    print(f"LR warmup: off ({lr_schedule} decay starts at full LR)" if warmup_steps == 0
          else f"LR warmup: {warmup_frac:g} of the first iteration ({warmup_steps} steps, "
               f"{warmup_frac * epochs:.2g} epochs)")
    if epochs_2 > 0:
        print(f"LR schedule: warm restart after epoch {epochs} -- iter 1 ({lr_schedule}): {learning_rate:.1e} -> "
              f"{min_lr:.1e} over {epochs} ep, then iter 2 ({lr_schedule_2}): {learning_rate_2:.1e} -> {min_lr_2:.1e} "
              f"over {epochs_2} ep (weights, optimizer state and EMA carry over)")
    elif min_lr > 0:
        print(f"LR floor: {lr_schedule} anneals from {learning_rate:.1e} to {min_lr:.1e} over the full schedule "
              f"(reaches the floor at the final step)")

    batch_size = getattr(train_loader, "batch_size", 0)
    token_rows = batch_size * getattr(model, "seq_len", 0)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    use_amp = (amp_dtype is not None and str(device).startswith("cuda") and token_rows >= 8192
               and (precision == "fp16" or torch.cuda.is_bf16_supported()))
    if amp_dtype is not None and not use_amp:
        if 0 < token_rows < 8192:
            print(f"[FM] {precision} skipped: {batch_size} x {model.seq_len} = {token_rows} token-rows/step "
                  f"is launch-bound (measured slower). Needs >= 8192 (bigger batch or more tokens).")
        else:
            print(f"[FM] {precision} not supported on this device -- training in fp32.")
    elif use_amp:
        scaling = "dynamic loss scaling" if precision == "fp16" else "no loss scaling"
        print(f"[FM] Mixed precision ON: {precision} autocast (fp32 master weights, {scaling}).")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))

    _loss_desc = f"pseudo-Huber (c={huber_c:g})" if loss_type == "huber" else "MSE"
    print(f"[FM] Objective: {_loss_desc} loss | t ~ {t_dist}.")

    fwd = model
    if compile_mode == "on" and str(device).startswith("cuda"):
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            fwd = torch.compile(model, dynamic=False, mode="max-autotune")
            print("[FM] torch.compile ON (Inductor, max-autotune); the first epoch runs slow while "
                  "kernels autotune (cached across runs).")
        except Exception as e:
            fwd = model
            print(f"[FM] torch.compile unavailable ({e}) -- training eager.")

    use_ema = ema_decay is not None and ema_decay > 0.0
    ema = {n: p.detach().clone() for n, p in model.named_parameters()} if use_ema else None
    if use_ema:
        ema_list = list(ema.values())
        ema_params = [p for _, p in model.named_parameters()]
    if use_ema:
        print(f"[FM] Weight EMA on: decay {ema_decay} (warmed up); final eval + dit.pth use the EMA weights.")
    if context_noise > 0.0:
        print(f"[FM] Context-latent noise on: std {context_noise} (training only; rollout-robustness regularizer).")
    gstep = 0

    def _flow_batch(z_seq, z_future, add_ctx_noise=False):
        if z_future.dim() == 4:
            z_future = z_future.unsqueeze(1)
        z1 = z_future.to(device)
        z_seq = z_seq.to(device)
        if add_ctx_noise and context_noise > 0.0:
            z_seq = z_seq + context_noise * torch.randn_like(z_seq)
        z0 = torch.randn_like(z1)
        if t_dist == "logit_normal":
            t = torch.sigmoid(torch.randn(z1.shape[0], device=device))
        else:
            t = torch.rand(z1.shape[0], device=device)
        t_b = t.view(-1, 1, 1, 1, 1)
        z_t = t_b * z1 + (1.0 - t_b) * z0
        v_target = z1 - z0
        m = fwd if (not batch_size or z1.shape[0] == batch_size) else model
        with torch.autocast(device_type="cuda", dtype=amp_dtype if use_amp else torch.bfloat16,
                            enabled=use_amp):
            v_pred = m(z_t, t, z_seq)
        vp, vt = v_pred.float(), v_target.float()
        if loss_type == "huber":
            loss = (torch.sqrt((vp - vt) ** 2 + huber_c * huber_c) - huber_c).mean()
        else:
            loss = F.mse_loss(vp, vt)
        return loss

    for epoch in range(total_epochs):
        start_time = time.time()

        model.train()
        tr_loss = torch.zeros((), device=device)
        tr_gnorm = torch.zeros((), device=device)
        n_overflow = 0
        for z_seq, z_future in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _flow_batch(z_seq, z_future, add_ctx_noise=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if math.isfinite(gnorm.item()):
                scaler.step(optimizer)
                if use_ema:
                    d = min(ema_decay, (gstep + 1) / (gstep + 11))
                    with torch.no_grad():
                        torch._foreach_mul_(ema_list, d)
                        torch._foreach_add_(ema_list, ema_params, alpha=1.0 - d)
                    gstep += 1
                tr_loss += loss.detach()
                tr_gnorm += gnorm.detach()
            else:
                n_overflow += 1
            scaler.update()
            scheduler.step()

        model.eval()
        val_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for z_seq, z_future in val_loader:
                val_loss += _flow_batch(z_seq, z_future)

        nb_tr, nb_val = max(len(train_loader) - n_overflow, 1), len(val_loader)
        epoch_time = time.time() - start_time
        skipped = f" | Skipped: {n_overflow} overflow" if n_overflow else ""
        print(f"FM Epoch {epoch+1}/{total_epochs} | Time: {epoch_time:.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | GradNorm: {tr_gnorm.item()/nb_tr:.3f} | "
              f"Train Loss: {tr_loss.item()/nb_tr:.6f} | Val Loss: {val_loss.item()/nb_val:.6f}{skipped}")

    if use_ema:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(ema[n])
    return model


@torch.no_grad()
def retrain_decoder(ae, dit, z_all, frame_cache, train_trajs, val_trajs, context_len,
                    epochs=3, learning_rate=5e-4, weight_decay=0.0, lpips_weight=1.0,
                    lpips_net="alex", rollout_k=5, clean_frac=0.3, num_steps=10, batch_size=64,
                    grad_clip=10.0, n_train_traj=1000, n_val_traj=100, precision="bf16",
                    compile_mode="off", device="cuda"):
    from src.eval import flow_sample
    if hasattr(frame_cache, "subset"):   # FrameStore -> decode only the decoder-train trajs
        frame_cache = frame_cache.subset(train_trajs[:n_train_traj] + val_trajs[:n_val_traj])
    ctx_tr, tgt_tr = frame_cache.build_windows(train_trajs[:n_train_traj], context_len, rollout_k)
    ctx_va, tgt_va = frame_cache.build_windows(val_trajs[:n_val_traj], context_len, rollout_k)
    if ctx_tr.shape[0] == 0 or ctx_va.shape[0] == 0:
        print("[DecTrain] Not enough windows for decoder training; skipping Phase 3.")
        return ae

    print("--- Phase 3: Decoder Training (on DiT-predicted latents) ---")
    print(f"[DecTrain] {ctx_tr.shape[0]} train / {ctx_va.shape[0]} val windows | rollout depth 1..{rollout_k} | "
          f"clean_frac {clean_frac} | {num_steps} Euler steps | encoder + DiT frozen.")

    frames = frame_cache.frames
    scale = dit.latent_scale
    dit.eval()
    for p in dit.parameters():
        p.requires_grad_(False)
    dec_params = [p for m in (ae.dec_in, ae.dec_mid, ae.dec, ae.dec_norm, ae.conv_out)
                  for p in m.parameters()]
    optimizer = optim.AdamW(dec_params, lr=learning_rate, weight_decay=weight_decay)
    nb_tr = (ctx_tr.shape[0] + batch_size - 1) // batch_size
    scheduler = build_warmup_cosine(optimizer, epochs * nb_tr)

    perceptual = build_lpips(device, net=lpips_net) if lpips_weight > 0 else None
    if lpips_weight > 0 and perceptual is None:
        print("[Warn] lpips not installed (pip install lpips). Falling back to pixel loss only.")
        lpips_weight = 0.0
    elif perceptual is not None:
        backbone = getattr(perceptual, "pnet_type", "?")
        print(f"[DecTrain] Perceptual loss ON: LPIPS-{backbone}, weight {lpips_weight}.")

    on_cuda = str(device).startswith("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    use_amp = amp_dtype is not None and on_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))
    if use_amp:
        print(f"[DecTrain] Mixed precision ON: {precision} autocast.")
    dit_run, decode_run = dit, ae.decode
    if compile_mode == "on" and on_cuda:
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            dit_run = torch.compile(dit)
            decode_run = torch.compile(ae.decode)
            print("[DecTrain] torch.compile ON (Inductor) for DiT rollouts + decoder; "
                  "the first pass runs slow while graphs build.")
        except Exception as e:
            dit_run, decode_run = dit, ae.decode
            print(f"[DecTrain] torch.compile unavailable ({e}) -- running eager.")

    def _frames_at(idx):
        return frames[idx].to(device).float().div_(255.0)

    def _latents_at(idx):
        return z_all[idx].float().to(device)

    def _autocast():
        return torch.autocast(device_type="cuda", dtype=amp_dtype if use_amp else torch.bfloat16, enabled=use_amp)

    @torch.no_grad()
    def _rollout_cache(ctx_idx):
        N = ctx_idx.shape[0]
        out = torch.empty((N, rollout_k, z_all.shape[1], z_all.shape[-2], z_all.shape[-1]),
                          dtype=torch.float16, device=z_all.device)
        for i in range(0, N, batch_size):
            z_run = _latents_at(ctx_idx[i:i + batch_size])
            T = z_run.shape[1]
            produced = 0
            with _autocast():
                while produced < rollout_k:
                    z_chunk = flow_sample(dit_run, z_run, num_steps)
                    for f in range(z_chunk.shape[1]):
                        if produced >= rollout_k:
                            break
                        out[i:i + batch_size, produced] = z_chunk[:, f].to(device=z_all.device, dtype=torch.float16)
                        produced += 1
                    z_run = torch.cat([z_run, z_chunk], dim=1)[:, -T:]
        return out

    print(f"[DecTrain] Precomputing frozen-DiT rollouts (depth {rollout_k}) once for "
          f"{ctx_tr.shape[0]} train + {ctx_va.shape[0]} val windows...")
    t0 = time.time()
    roll_tr = _rollout_cache(ctx_tr)
    roll_va = _rollout_cache(ctx_va)
    print(f"[DecTrain] Rollout cache ready in {time.time() - t0:.1f}s "
          f"({(roll_tr.numel() + roll_va.numel()) * 2 / 1e9:.2f} GB); DiT not called again.")
    torch.set_grad_enabled(True)

    @torch.no_grad()
    def _val_metrics():
        ae.eval()
        se = {"pred1": 0.0, "predK": 0.0, "recon": 0.0}
        n = 0
        for i in range(0, ctx_va.shape[0], batch_size):
            tg = tgt_va[i:i + batch_size]
            f1 = _frames_at(tg[:, 0])
            with _autocast():
                d1 = ae.decode(roll_va[i:i + batch_size][:, 0].float().to(device) * scale).float()
                rc = ae.decode(_latents_at(tg[:, 0]) * scale).float()
            se["pred1"] += ((d1 - f1) ** 2).sum().item()
            se["recon"] += ((rc - f1) ** 2).sum().item()
            if rollout_k > 1:
                fK = _frames_at(tg[:, rollout_k - 1])
                with _autocast():
                    dK = ae.decode(roll_va[i:i + batch_size][:, rollout_k - 1].float().to(device) * scale).float()
                se["predK"] += ((dK - fK) ** 2).sum().item()
            n += f1.numel()
        psnr = lambda k: 10.0 * math.log10(1.0 / (se[k] / n)) if se[k] > 0 else float("inf")
        return psnr("pred1"), psnr("predK"), psnr("recon")

    for epoch in range(epochs):
        start_time = time.time()
        ae.train()
        tr_loss = torch.zeros((), device=device)
        tr_gnorm = torch.zeros((), device=device)
        order = torch.randperm(ctx_tr.shape[0])
        for i in range(0, ctx_tr.shape[0], batch_size):
            rows = order[i:i + batch_size]
            tg = tgt_tr[rows]
            j = random.randint(1, rollout_k)
            if random.random() < clean_frac:
                z_pred = _latents_at(tg[:, j - 1])
            else:
                z_pred = roll_tr[rows][:, j - 1].float().to(device)
            target = _frames_at(tg[:, j - 1])

            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad(), _autocast():
                recon = decode_run(z_pred * scale)
                perc = perceptual(recon, target, normalize=True).mean() if perceptual is not None else None
                loss = F.mse_loss(recon.float(), target, reduction="sum") / target.shape[0]
                if perc is not None:
                    loss = loss + lpips_weight * perc.float()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(dec_params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss += loss.detach()
            tr_gnorm += gnorm.detach()

        p1, pK, pr = _val_metrics()
        print(f"DecTrain Epoch {epoch+1}/{epochs} | Time: {time.time()-start_time:.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | GradNorm: {tr_gnorm.item()/nb_tr:.3f} | "
              f"Train Loss(sum): {tr_loss.item()/nb_tr:.4f} | Val decode(DiT@1): {p1:.2f} dB | "
              f"decode(DiT@{rollout_k}): {pK:.2f} dB | recon: {pr:.2f} dB")

    ae.eval()
    return ae
