import time
import math
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def build_warmup_cosine(optimizer, total_steps, warmup_frac=0.05):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_factor(step):
        if step < warmup_steps:
            return 0.01 + (1.0 - 0.01) * (step / warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


@torch.no_grad()
def build_latent_cache(ae, frames, device, cache_device, batch_size=512):
    """Encode every cached frame through the frozen VAE once into a float16 latent cache.

    frames: (M, C, H, W) uint8 on `cache_device`. Returns (M, Cl, h, w) float16 on
    `cache_device`. Uses the deterministic posterior mean (ae.encode -> mu) as 'the latent',
    so the dynamics phase trains directly on these and the VAE never runs inside the per-epoch
    loop. The VAE's KL term already keeps mu ~unit-Gaussian per channel, so no separate
    normalization is applied here. float16 storage roughly halves memory vs float32 (~2 GB vs
    ~4 GB for 500k 32x8x8 latents) at negligible accuracy cost for an MSE objective.
    """
    ae.eval()
    M = frames.shape[0]
    z_all = None
    print(f"[Cache] Encoding {M} frames into latent cache (one-time)...")
    for i in range(0, M, batch_size):
        chunk = frames[i : i + batch_size]
        if chunk.device != torch.device(device):
            chunk = chunk.to(device, non_blocking=True)
        x = chunk.float().div_(255.0)
        z = ae.encode(x).half()
        if z_all is None:
            z_all = torch.empty((M, *z.shape[1:]), dtype=torch.float16, device=cache_device)
        z_all[i : i + batch_size] = z.to(cache_device)
    print(f"[Cache] Latent cache ready: {tuple(z_all.shape)} float16 (~{z_all.numel() * 2 / 1e9:.2f} GB).")

    # LDM-style scale factor: divide the cache by its std so the flow model always trains on
    # ~unit-variance latents, regardless of the VAE's KL weight. Without this a low-KL VAE produces
    # large-magnitude latents that break the z_0 ~ N(0, I) flow-matching prior -- the noise and the
    # data end up at different radii and the rectified-flow ODE is poorly conditioned. The scalar is
    # stored on the DiT and re-applied at inference (encode -> /scale, decode -> *scale), decoupling
    # the VAE's latent scale from the dynamics model.
    s = ss = 0.0
    cnt = 0
    for i in range(0, M, batch_size):
        c = z_all[i:i + batch_size].float()
        s += c.sum().item(); ss += (c * c).sum().item(); cnt += c.numel()
    gstd = float(max((ss / cnt - (s / cnt) ** 2) ** 0.5, 1e-6))
    z_all.div_(gstd)
    latent_scale = torch.tensor(gstd)
    print(f"[Cache] Latent scale (std) = {gstd:.4f}; normalized cache to ~unit variance.")
    return z_all, latent_scale


class _LatentSurrogate(nn.Module):
    """Tiny deterministic next-latent predictor used ONLY to score latent predictability.

    Three 3x3 conv layers over the channel-stacked context (T*Cl -> Cl); at an 8x8 grid this
    already sees the whole field and can form temporal differences (velocity). Kept small and fixed
    so it is a cheap, fair probe -- it is NOT part of the world model. """
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
    tgt_mean = z_all[tgt_va[:, 0]].float().mean().item()   # global mean for the R^2 denominator
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
    """Cheap, DiT-free proxy for how PREDICTABLE this VAE's latent is -- validated offline to rank
    VAEs the same way the full DiT's 1-step PSNR does (for gaps > ~1 dB; sub-1-dB is DiT noise).

    Trains a small deterministic surrogate to predict the next latent from the `context_len` context
    latents (1-step) on a capped trajectory subset, on the SAME normalized cache the DiT trains on,
    then reports:
      * Predictability R^2   -- fraction of next-latent variance the surrogate explains (PRIMARY).
      * Surrogate 1-step PSNR -- decode(prediction)*scale vs the true next frame (secondary; folds in
        decoder quality + latent scale, so it is less discriminative than R^2).
      * Recon PSNR           -- on the same val frames, for the reconstruction axis / 2D view.

    Optimise the VAE for HIGH R^2, NOT recon PSNR: a sharper VAE can reconstruct better yet be LESS
    predictable, which hurts the downstream DiT. Fixed seed/arch/budget => comparable across runs. """
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
    """KL(N(mu, sigma^2) || N(0, 1)) summed over latent dims, averaged over the batch."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]


def build_lpips(device, net="alex"):
    """Frozen LPIPS perceptual metric, or None if the `lpips` package is unavailable.

    LPIPS scores two images by the distance between their activations in a pretrained,
    perceptually-calibrated backbone -- so it penalises the BLUR that pixel MSE rewards (blur is
    the L2-optimal hedge under uncertainty). Used purely as a differentiable loss: the backbone and
    the learned linear weights are frozen, so this module is never trained and is not part of any
    optimizer. First call downloads the backbone weights.

    `net` is the backbone: "alex" (AlexNet, the LPIPS paper's default -- shallow, several times
    cheaper to fwd+bwd per step) or "vgg" (deeper, slightly smoother gradients but markedly slower."""
    try:
        import lpips
    except ImportError:
        return None
    # lpips builds the torchvision backbone via the legacy `pretrained=` API, which fires two
    # torchvision deprecation UserWarnings on first construction. It also loads its own pretrained
    # weights via torch.load(weights_only=False), firing a FutureWarning we can't fix from our side
    # (it's inside the lpips package). Silence just those three at the construction site.
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
                      grad_clip=10.0, device="cuda"):
    """Train the convolutional VAE: reconstruction + lpips_weight * LPIPS + beta * KL.

    Pixel reconstruction is summed over pixels (per image) so it sits on the same scale as the
    summed-over-dims KL, which makes `kl_weight` (beta) an O(1) knob. beta applies at FULL
    strength from step 1 -- no warmup. (A linear warmup used to guard against posterior collapse,
    but at the tiny betas used here (<= 0.01) collapse never materialises, and the warmup silently
    zeroed beta on short runs. Collapse would show as Val KL cratering toward 0 in the logs; ours
    rides at 13k-23k.) Tuning by the logged numbers: if Val PSNR drops much and KL collapses
    toward 0, lower kl_weight; if KL stays very large (latent barely regularized), raise it.

    `lpips_weight` > 0 adds a perceptual (LPIPS-VGG) term. This is the key fix for a latent whose
    L2 distance does NOT track perceptual quality: it reshapes the latent geometry so that the
    downstream flow model (which only ever minimises latent MSE) is implicitly optimising
    perceptual quality, and it stops the decoder from rewarding blur. The pixel-MSE + KL balance
    (and hence the unit-scale latent the dynamics phase assumes) is left intact, so this is a
    drop-in: the perceptual term rides on top. LPIPS is taken in [0,1] via normalize=True.
    Watch `Train LPIPS` falling alongside Val PSNR; expect Val PSNR itself to dip slightly vs a
    pure-MSE VAE -- sharp is not MSE-optimal, and that is the point.
    """
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

    for epoch in range(epochs):
        start_time = time.time()

        ae.train()
        tr_recon = torch.zeros((), device=device)
        tr_kl = torch.zeros((), device=device)
        tr_perc = torch.zeros((), device=device)
        tr_gnorm = torch.zeros((), device=device)
        for ctx_frames, target_frame in train_loader:
            B, T, C, H, W = ctx_frames.shape
            x = ctx_frames.view(-1, C, H, W).to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = ae(x)
            recon_loss = F.mse_loss(recon, x, reduction="sum") / x.shape[0]
            kl = _vae_kl(mu, logvar)
            loss = recon_loss + kl_weight * kl
            if perceptual is not None:
                perc = perceptual(recon, x, normalize=True).mean()   # inputs in [0,1]
                loss = loss + lpips_weight * perc
                tr_perc += perc.detach()
            loss.backward()
            # Clip the global grad norm before the step: absorbs the loss spikes that a deeper LPIPS
            # backbone (e.g. VGG) can trigger, so one bad step never poisons AdamW's moments. The
            # sum-reduced recon makes these norms large, so the cap is loose by default -- the logged
            # GradNorm (pre-clip) is the readout: if it sits well below grad_clip the cap is a pure
            # safety net; if it rides at the cap, the cap is biting and should be raised.
            gnorm = torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            tr_recon += recon_loss.detach()
            tr_kl += kl.detach()
            tr_gnorm += gnorm.detach()

        ae.eval()
        val_mse = torch.zeros((), device=device)   # mean MSE -> comparable PSNR
        val_kl = torch.zeros((), device=device)
        val_perc = torch.zeros((), device=device)
        with torch.no_grad():
            for ctx_frames, target_frame in val_loader:
                B, T, C, H, W = ctx_frames.shape
                x = ctx_frames.view(-1, C, H, W).to(device)
                recon, mu, logvar = ae(x)
                val_mse += F.mse_loss(recon, x)   # mean over all elements
                val_kl += _vae_kl(mu, logvar)
                if perceptual is not None:
                    val_perc += perceptual(recon, x, normalize=True).mean()

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
              f"Val PSNR: {val_psnr:.2f} dB | Val KL: {val_kl.item()/nb_val:.2f}")
    return ae


def train_flow_matching(model, train_loader, val_loader, epochs=15, learning_rate=3e-4,
                        weight_decay=1e-4, grad_clip=3.0, ema_decay=0.999, context_noise=0.0,
                        precision="bf16", spike_factor=4.0, compile_mode="off",
                        t_dist="logit_normal", loss_type="mse", huber_c=1.0, device="cuda"):
    """Train the Diffusion Transformer with Rectified Flow (flow matching), CHUNK prediction.

    The loaders yield (z_seq, z_1): the context-frame latents and the ground-truth chunk of the
    next K = chunk_len frame latents (horizon == K), gathered from the frozen-VAE latent cache.
    z_1 has shape (B, K, Cl, h, w) and the whole chunk is denoised at a single noise level:

        z_0 ~ N(0, I)                       (noise, same (B,K,Cl,h,w) shape as z_1)
        t   ~ U(0, 1)                       (one continuous time per sample, shared across the chunk)
        z_t  = t * z_1 + (1 - t) * z_0      (linear interpolation -- the rectified-flow path)
        v*   = z_1 - z_0                    (constant target velocity along that straight path)
        loss = MSE(model(z_t, t, z_seq), v*)

    Predicting the K-frame chunk jointly (rather than one autoregressive step) is what curbs
    exposure bias on long rollouts. At inference an Euler ODE solver integrates the learned
    velocity from noise (t=0) to the data (t=1), producing K frames per call; see eval.flow_sample.

    `context_noise` > 0 adds Gaussian noise (this std, in the ~unit-variance normalized latent
    space) to the CONTEXT latents during training only -- never the target or the val pass. This
    attacks exposure bias from the other side: at autoregressive rollout the DiT is fed its own
    imperfect predictions as context, a distribution it never sees when trained on clean cached
    latents. Perturbing the context teaches it to stay robust to that drift, which typically
    trades a hair of 1-step accuracy for steadier long-horizon rollouts. Start ~0.02-0.1.
    """
    print("--- Phase 2: Training Flow Matching DiT (chunk prediction) ---")
    model.to(device)
    # Fused AdamW runs the whole optimizer step in one kernel -- a free speedup on CUDA. It
    # requires the params to already live on the GPU, hence model.to(device) above. Falls back
    # cleanly on CPU / older torch.
    try:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    except (RuntimeError, TypeError, ValueError):
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps)

    # Mixed-precision autocast: matmuls/attention run in the reduced dtype (half the memory
    # traffic, tensor-core throughput) while master weights, grads, optimizer state and the loss
    # stay fp32. Eval/checkpoints are untouched -- only training-time compute precision changes.
    #   "bf16"  fp32's exponent range but only a ~3-significant-digit mantissa. No loss scaling
    #           needed -- but nothing skips a poisoned step either: measured DIVERGENT on the
    #           temporal DiT at every LR tried (fp32 identical-seed runs stable).
    #   "fp16"  8x finer mantissa, narrow exponent range, so gradients ride a dynamic loss scale;
    #           the scaler SKIPS any step whose grads hit inf/nan (built-in spike guard).
    #   "fp32"  full precision, no autocast -- the proven-stable reference.
    #
    # Only worth it when the matmuls are fat enough to amortize autocast's cast/dispatch overhead
    # (LayerNorm/softmax stay fp32). The right size measure is token-rows per step = batch x
    # seq_len, NOT batch alone: the squash layout at batch 64 is 64 x 64 = 4096 rows and bf16 made
    # epochs MEASURABLY SLOWER there (48s -> 54s, launch-bound), while the temporal layout at the
    # same batch is 64 x 384 = 24576 rows of real tensor-core work. Threshold 8192 reproduces the
    # old batch>=128 rule for squash and lets big-token layouts qualify at small batch.
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
    # Disabled scaler (bf16/fp32) turns every scaler call below into a pass-through of the plain
    # backward/clip/step sequence, so all three precisions share one training loop.
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))

    # Gradient-spike guard: reject a step whose PRE-CLIP grad norm dwarfs the running EMA of recent
    # ACCEPTED norms (by spike_factor x). The identical-seed fp32 runs never spike, so a spike here
    # is a low-precision artifact -- bf16 mantissa noise, or compiled-fp16 op reordering that keeps
    # the spike FINITE so the loss-scaler's inf/nan skip misses it. Norm-clipping caps a step's
    # magnitude but NOT its direction, so a finite spike still poisons AdamW's moments and derails
    # the run; dropping the whole step is the fix. Only accepted steps feed the EMA, so the baseline
    # can't be dragged up by the very spikes it must catch. spike_factor <= 0 disables the guard
    # (nonfinite grads are ALWAYS skipped regardless -- that alone rescues bf16 from inf/nan steps).
    use_spike_guard = bool(spike_factor and spike_factor > 0.0)
    spike_warmup = 50          # accepted steps to seed the baseline before the guard may skip
    # Deadlock escape: the EMA baseline only updates on ACCEPTED steps, so if the model ever enters
    # a sustained high-gradient regime, every step trips the guard, the baseline freezes, and the
    # model can never move again (observed: a whole epoch of skips, GradNorm/Loss -> 0). After this
    # many CONSECUTIVE skips, force one step through -- it is still grad-clipped, so bounded -- and
    # let it re-seed the EMA toward the new regime so the guard adapts instead of freezing.
    max_consec_skip = 20
    consec_skip = 0
    # Absolute floor: never treat a step as a spike while its norm is this small, even if it beats
    # the relative test. Late in training the accepted norm decays to ~0.02, so 4x the EMA becomes a
    # tiny threshold that benign step-noise trips constantly -- one run skipped ~20% of its late
    # steps and diverged onto a worse path. Real spikes are >>1 (seen: 2..24000), so this floor sits
    # well below them: it only silences the useless late-training over-triggering, tightening
    # run-to-run reproducibility. Clipping still bounds anything that slips through.
    spike_floor = 0.5
    gnorm_ema = None
    n_accept = 0
    if use_spike_guard:
        print(f"[FM] Grad-spike guard on: drop steps whose pre-clip norm > {spike_factor:g}x the "
              f"running mean (active after {spike_warmup} steps).")
    _loss_desc = f"pseudo-Huber (c={huber_c:g})" if loss_type == "huber" else "MSE"
    print(f"[FM] Objective: {_loss_desc} loss | t ~ {t_dist}.")

    # Weight EMA: keep a slowly-averaged shadow of the parameters and deploy THAT (final eval + saved
    # checkpoint). Flow-matching gradients are very noisy (a fresh random t and fresh noise each step),
    # so the live weights oscillate around the loss basin; averaging the trajectory lands on a flatter,
    # better-generalizing point -- standard for diffusion/flow models, usually a few tenths of a dB for
    # free. The decay is warmed up via min(decay, (s+1)/(s+11)) so the average is not polluted by the
    # random init. ema_decay <= 0 disables it. Only parameters are averaged; the lone buffer
    # (latent_scale) is constant, so it is left untouched.
    # torch.compile: optional graph capture / codegen for the DiT forward+backward. Only the local
    # `fwd` wrapper is compiled -- optimizer, EMA, checkpoint and the returned model keep the raw
    # module (same parameter tensors), so state_dicts never grow the _orig_mod. prefix and dit.pth
    # stays loadable everywhere. Modes:
    #   "cudagraphs"       CUDA-graph capture WITHOUT codegen: no Triton/MSVC needed, replays the
    #                      whole fwd/bwd as one graph. Directly attacks the kernel-LAUNCH overhead
    #                      that dominates at batch 64 (the reason bf16 alone made epochs slower).
    #   "default"          Inductor codegen: fuses pointwise chains (adaLN modulate, SiLU, residual
    #                      adds) into fewer kernels. On Windows needs MSVC + a triton-windows wheel
    #                      matching the torch version.
    #   "reduce-overhead"  Inductor + CUDA graphs: both of the above; the most it can give here.
    # First steps compile/capture (minutes on Windows) and each new batch shape (e.g. the last
    # partial batch, the val pass) triggers one recompile -- both amortize to nothing over a run.
    fwd = model
    if compile_mode != "off" and str(device).startswith("cuda"):
        try:
            # NOT `import torch._dynamo` -- that would rebind `torch` as a function-local for the
            # whole body and break every earlier torch.* line with UnboundLocalError.
            from torch import _dynamo
            _dynamo.config.suppress_errors = True   # backend failure -> eager fallback, not a crash
            # dynamic=False: never promote the batch dim to a symbolic shape. CUDA graphs replay
            # buffers of a FIXED size, so a "dynamic" recompile silently reuses the 64-wide graph
            # for a smaller batch and crashes in the input copy; off-size batches bypass the
            # wrapper in _flow_batch instead.
            if compile_mode == "cudagraphs":
                fwd = torch.compile(model, backend="cudagraphs", dynamic=False)
            elif compile_mode == "reduce-overhead":
                fwd = torch.compile(model, mode="reduce-overhead", dynamic=False)
            else:
                fwd = torch.compile(model, dynamic=False)
            print(f"[FM] torch.compile ON ({compile_mode}); the first epoch runs slow while graphs build.")
        except Exception as e:
            fwd = model
            print(f"[FM] torch.compile unavailable ({e}) -- training eager.")

    use_ema = ema_decay is not None and ema_decay > 0.0
    ema = {n: p.detach().clone() for n, p in model.named_parameters()} if use_ema else None
    if use_ema:
        print(f"[FM] Weight EMA on: decay {ema_decay} (warmed up); final eval + dit.pth use the EMA weights.")
    if context_noise > 0.0:
        print(f"[FM] Context-latent noise on: std {context_noise} (training only; rollout-robustness regularizer).")
    gstep = 0

    def _flow_batch(z_seq, z_future, add_ctx_noise=False):
        # z_seq (B,T,Cl,h,w); z_future (B,K,Cl,h,w) (chunk). horizon==1 -> add the chunk axis.
        if z_future.dim() == 4:
            z_future = z_future.unsqueeze(1)
        z1 = z_future.to(device)                           # (B, K, Cl, h, w)
        z_seq = z_seq.to(device)
        if add_ctx_noise and context_noise > 0.0:
            # Perturb the CONTEXT only (the regime faced at autoregressive rollout); target stays clean.
            z_seq = z_seq + context_noise * torch.randn_like(z_seq)
        z0 = torch.randn_like(z1)
        # Timestep sampling. "uniform" is plain U(0,1). "logit_normal" (Stable Diffusion 3, Esser
        # et al. 2024) draws t = sigmoid(N(0,1)), concentrating samples on the MIDDLE of the
        # trajectory where velocity prediction is hardest and spending fewer on the near-noise /
        # near-clean endpoints -- a consistent quality win for rectified-flow transformers at zero
        # extra cost. It does not change the optimum, only where training effort is spent.
        if t_dist == "logit_normal":
            t = torch.sigmoid(torch.randn(z1.shape[0], device=device))
        else:
            t = torch.rand(z1.shape[0], device=device)     # (B,) in [0,1], shared over the chunk
        t_b = t.view(-1, 1, 1, 1, 1)
        z_t = t_b * z1 + (1.0 - t_b) * z0
        v_target = z1 - z0
        # Only full-size batches take the compiled path; the last partial train/val batch of an
        # epoch runs eager. A captured CUDA graph can only replay the batch size it was recorded
        # at, and one eager straggler per epoch is cheaper than capturing a graph for its shape.
        m = fwd if (not batch_size or z1.shape[0] == batch_size) else model
        with torch.autocast(device_type="cuda", dtype=amp_dtype if use_amp else torch.bfloat16,
                            enabled=use_amp):
            v_pred = m(z_t, t, z_seq)
        # Loss in fp32 for a stable reduction. "mse" is the standard flow-matching L2 objective
        # (regression to the conditional mean velocity). "huber" is pseudo-Huber
        # sqrt(err^2 + c^2) - c: L2 near zero, L1 in the tail, so its per-element gradient
        # SATURATES at 1 for large errors -- an outlier sample can no longer produce an unbounded
        # gradient, which curbs the precision-induced spikes at the source (complements the guard).
        vp, vt = v_pred.float(), v_target.float()
        if loss_type == "huber":
            loss = (torch.sqrt((vp - vt) ** 2 + huber_c * huber_c) - huber_c).mean()
        else:
            loss = F.mse_loss(vp, vt)
        return loss

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        tr_loss = torch.zeros((), device=device)
        tr_gnorm = torch.zeros((), device=device)
        n_spike = 0
        n_overflow = 0
        for z_seq, z_future in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _flow_batch(z_seq, z_future, add_ctx_noise=True)
            scaler.scale(loss).backward()
            # Unscale BEFORE clipping so the clip threshold means what it says AND the guard reads a
            # real (not loss-scaled) gradient norm.
            scaler.unscale_(optimizer)
            # clip_grad_norm_ returns the PRE-clip total norm -- the signal the guard reads. The
            # clip stays as a magnitude backstop for any finite spike inside the warmup window.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            gval = gnorm.item()
            finite = math.isfinite(gval)
            # A spike = past warmup, finite, and far above the healthy baseline. Nonfinite grads are
            # always dropped (this is what makes even bf16 non-divergent). A dropped step calls
            # neither scaler.step nor optimizer.step, so AdamW's moments never see the bad gradient.
            spike = (use_spike_guard and finite and gnorm_ema is not None
                     and n_accept >= spike_warmup and gval > spike_factor * gnorm_ema
                     and gval > spike_floor)
            # Deadlock escape: after too many skips in a row, force a (still-clipped) finite step
            # through so the model can never freeze permanently in a high-gradient regime.
            if spike and consec_skip >= max_consec_skip:
                spike = False
            take_step = finite and not spike
            if take_step:
                scaler.step(optimizer)          # fp16 re-checks inf here; plain optimizer.step otherwise
                gnorm_ema = gval if gnorm_ema is None else 0.98 * gnorm_ema + 0.02 * gval
                n_accept += 1
                consec_skip = 0
                if use_ema:
                    d = min(ema_decay, (gstep + 1) / (gstep + 11))   # warm up the decay from ~0
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            ema[n].mul_(d).add_(p.detach(), alpha=1.0 - d)
                    gstep += 1
                tr_loss += loss.detach()
                tr_gnorm += gnorm.detach()
            elif not finite:
                n_overflow += 1
                consec_skip += 1
            else:
                n_spike += 1
                consec_skip += 1
            scaler.update()      # keeps the fp16 loss scale calibrated; no-op when the scaler is off
            scheduler.step()     # schedule tracks iterations; the rare skip doesn't shift it materially

        model.eval()
        val_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for z_seq, z_future in val_loader:
                val_loss += _flow_batch(z_seq, z_future)

        nb_tr, nb_val = max(len(train_loader) - n_spike - n_overflow, 1), len(val_loader)
        epoch_time = time.time() - start_time
        parts = ([f"{n_spike} spike"] if n_spike else []) + ([f"{n_overflow} overflow"] if n_overflow else [])
        skipped = f" | Skipped: {', '.join(parts)}" if parts else ""
        print(f"FM Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | GradNorm: {tr_gnorm.item()/nb_tr:.3f} | "
              f"Train Loss: {tr_loss.item()/nb_tr:.6f} | Val Loss: {val_loss.item()/nb_val:.6f}{skipped}")

    # Swap the averaged weights into the model so eval + the saved checkpoint use them.
    if use_ema:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(ema[n])
    return model


@torch.no_grad()
def _dit_predict_latents(dit, z_ctx, n_frames, num_steps):
    """Free-running DiT rollout in the normalized latent space: predict `n_frames` ahead of the
    context window and return the latent at that offset, (B, Cl, g, g). Mirrors the feedback loop
    of eval._chunk_rollout (each chunk is sampled from noise and fed back as context)."""
    from src.eval import flow_sample
    T = z_ctx.shape[1]
    z_run = z_ctx
    produced = 0
    while produced < n_frames:
        z_chunk = flow_sample(dit, z_run, num_steps)         # (B, chunk_len, Cl, g, g)
        take = min(z_chunk.shape[1], n_frames - produced)
        last = z_chunk[:, take - 1]
        produced += take
        z_run = torch.cat([z_run, z_chunk], dim=1)[:, -T:]
    return last


def retrain_decoder(ae, dit, z_all, frame_cache, train_trajs, val_trajs, context_len,
                    epochs=3, learning_rate=5e-4, weight_decay=0.0, lpips_weight=1.0,
                    lpips_net="alex", rollout_k=5, clean_frac=0.3, num_steps=10, batch_size=64,
                    grad_clip=10.0, n_train_traj=1000, n_val_traj=100, device="cuda"):
    """Phase 3: train the decoder of `ae` on the DiT's own predicted latents.

    The encoder and the DiT stay frozen, so the latent space, the latent cache and dit.pth all
    remain valid -- only the rendering changes (the Stable-Diffusion swap-the-decoder trick). The
    caller decides what `ae` is: the pipeline passes a FRESH full-capacity decoder (encoder weights
    copied from the Phase-1 VAE, decoder randomly initialized) so a weak Phase-1 decoder -- used
    only to force the encoder into an explicit, predictable latent -- is replaced from scratch by a
    rich renderer. A rollout-time decoder also gets DiT samples, which carry prediction error and
    drift off-manifold with horizon; training on those makes it a robust renderer / mild
    error-corrector (re-sharpening smeared balls) and lets the pixel-feedback GIF rollout
    self-correct via decode->re-encode.

    Each batch decodes either a CLEAN cached latent (prob `clean_frac`; anchors reconstruction
    quality) or a DiT latent from a free-running rollout of random depth j in [1, rollout_k]
    (robustness to realistic drift), and is scored against the TRUE frame at that offset with the
    Phase-1 loss (sum-MSE + lpips_weight * LPIPS). `num_steps` should match the deployment
    inference_steps so the decoder sees the latents it will actually render. It cannot fix wrong
    dynamics -- a latent without a ball stays ball-less -- so expect sharper rollouts, not better
    physics. Val metrics: decode(DiT@1) / decode(DiT@K) PSNR (the training target) and recon PSNR
    (clean-latent decode); the Before line is the from-scratch baseline."""
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
    # Only the decoder half of the VAE is optimised; the encoder is never even called here
    # (context/clean latents come from the cache), so it cannot drift.
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

    def _frames_at(idx):
        return frames[idx].to(device).float().div_(255.0)

    def _latents_at(idx):
        return z_all[idx].float().to(device)

    @torch.no_grad()
    def _val_metrics():
        ae.eval()
        se = {"pred1": 0.0, "predK": 0.0, "recon": 0.0}
        n = 0
        for i in range(0, ctx_va.shape[0], batch_size):
            ci, tg = ctx_va[i:i + batch_size], tgt_va[i:i + batch_size]
            z_ctx = _latents_at(ci)
            f1 = _frames_at(tg[:, 0])
            se["pred1"] += ((ae.decode(_dit_predict_latents(dit, z_ctx, 1, num_steps) * scale) - f1) ** 2).sum().item()
            se["recon"] += ((ae.decode(_latents_at(tg[:, 0]) * scale) - f1) ** 2).sum().item()
            if rollout_k > 1:
                fK = _frames_at(tg[:, rollout_k - 1])
                se["predK"] += ((ae.decode(_dit_predict_latents(dit, z_ctx, rollout_k, num_steps) * scale) - fK) ** 2).sum().item()
            n += f1.numel()
        psnr = lambda k: 10.0 * math.log10(1.0 / (se[k] / n)) if se[k] > 0 else float("inf")
        return psnr("pred1"), psnr("predK"), psnr("recon")

    p1, pK, pr = _val_metrics()
    print(f"[DecTrain] Before: decode(DiT@1) {p1:.2f} dB | decode(DiT@{rollout_k}) {pK:.2f} dB | "
          f"recon {pr:.2f} dB")

    for epoch in range(epochs):
        start_time = time.time()
        ae.train()
        tr_loss = torch.zeros((), device=device)
        tr_gnorm = torch.zeros((), device=device)
        order = torch.randperm(ctx_tr.shape[0])
        for i in range(0, ctx_tr.shape[0], batch_size):
            rows = order[i:i + batch_size]
            ci, tg = ctx_tr[rows], tgt_tr[rows]
            j = random.randint(1, rollout_k)
            if random.random() < clean_frac:
                z_pred = _latents_at(tg[:, j - 1])                                # clean cached latent
            else:
                z_pred = _dit_predict_latents(dit, _latents_at(ci), j, num_steps)  # drifted DiT latent
            target = _frames_at(tg[:, j - 1])

            optimizer.zero_grad(set_to_none=True)
            recon = ae.decode(z_pred * scale)
            loss = F.mse_loss(recon, target, reduction="sum") / target.shape[0]
            if perceptual is not None:
                loss = loss + lpips_weight * perceptual(recon, target, normalize=True).mean()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(dec_params, max_norm=grad_clip)
            optimizer.step()
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
