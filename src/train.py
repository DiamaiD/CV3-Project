import time
import math
import warnings
import torch
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
    # torchvision deprecation UserWarnings on first construction. Nothing actionable on our
    # side, so silence just those two at the construction site.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*'pretrained' is deprecated.*")
        warnings.filterwarnings("ignore", message=r".*Arguments other than a weight enum.*")
        net = lpips.LPIPS(net=net, verbose=False).to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def train_autoencoder(ae, train_loader, val_loader, epochs=5, learning_rate=1e-3,
                      weight_decay=1e-4, kl_weight=1.0, kl_anneal_frac=0.3, lpips_weight=0.0,
                      grad_clip=10.0, device="cuda"):
    """Train the convolutional VAE: reconstruction + lpips_weight * LPIPS + beta * KL.

    Pixel reconstruction is summed over pixels (per image) so it sits on the same scale as the
    summed-over-dims KL, which makes `kl_weight` (beta) an O(1) knob. beta is linearly warmed
    up from 0 over the first `kl_anneal_frac` of training so the decoder can establish sharp
    reconstructions before the KL pressure kicks in -- the main guard against posterior
    collapse. Tuning by the logged numbers: if Val PSNR drops much and KL collapses toward 0,
    lower kl_weight; if KL stays very large (latent barely regularized), raise it.

    `lpips_weight` > 0 adds a perceptual (LPIPS-VGG) term. This is the key fix for a latent whose
    L2 distance does NOT track perceptual quality: it reshapes the latent geometry so that the
    downstream flow model (which only ever minimises latent MSE) is implicitly optimising
    perceptual quality, and it stops the decoder from rewarding blur. The pixel-MSE + KL balance
    (and hence the unit-scale latent the dynamics phase assumes) is left intact, so this is a
    drop-in: the perceptual term rides on top. LPIPS is taken in [0,1] via normalize=True.
    Watch `Train LPIPS` falling alongside Val PSNR; expect Val PSNR itself to dip slightly vs a
    pure-MSE VAE -- sharp is not MSE-optimal, and that is the point.
    """
    print("--- Phase 1: Training Autoencoder (VAE) ---")
    optimizer = optim.AdamW(ae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps)
    ae.to(device)

    perceptual = build_lpips(device) if lpips_weight > 0 else None
    if lpips_weight > 0 and perceptual is None:
        print("[Warn] lpips not installed (pip install lpips). Falling back to pixel + KL only.")
        lpips_weight = 0.0
    elif perceptual is not None:
        backbone = getattr(perceptual, "pnet_type", "?")
        print(f"[VAE] Perceptual loss ON: LPIPS-{backbone}, weight {lpips_weight}.")

    warmup_epochs = max(1, int(epochs * kl_anneal_frac))

    for epoch in range(epochs):
        start_time = time.time()
        beta = kl_weight * min(1.0, epoch / warmup_epochs)

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
            loss = recon_loss + beta * kl
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
              f"Beta: {beta:.5f} | Train Recon(sum): {tr_recon.item()/nb_tr:.4f} | "
              f"Train KL: {tr_kl.item()/nb_tr:.2f} | {perc_str}Val MSE: {val_mse_mean:.8f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val KL: {val_kl.item()/nb_val:.2f}")
    return ae


def train_flow_matching(model, train_loader, val_loader, epochs=15, learning_rate=3e-4,
                        weight_decay=1e-4, grad_clip=3.0, ema_decay=0.999, context_noise=0.0,
                        device="cuda"):
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps)
    model.to(device)

    # Weight EMA: keep a slowly-averaged shadow of the parameters and deploy THAT (final eval + saved
    # checkpoint). Flow-matching gradients are very noisy (a fresh random t and fresh noise each step),
    # so the live weights oscillate around the loss basin; averaging the trajectory lands on a flatter,
    # better-generalizing point -- standard for diffusion/flow models, usually a few tenths of a dB for
    # free. The decay is warmed up via min(decay, (s+1)/(s+11)) so the average is not polluted by the
    # random init. ema_decay <= 0 disables it. Only parameters are averaged; the lone buffer
    # (latent_scale) is constant, so it is left untouched.
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
        t = torch.rand(z1.shape[0], device=device)         # (B,) in [0,1], shared over the chunk
        t_b = t.view(-1, 1, 1, 1, 1)
        z_t = t_b * z1 + (1.0 - t_b) * z0
        v_target = z1 - z0
        v_pred = model(z_t, t, z_seq)
        return F.mse_loss(v_pred, v_target)

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        tr_loss = torch.zeros((), device=device)
        tr_gnorm = torch.zeros((), device=device)
        for z_seq, z_future in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _flow_batch(z_seq, z_future, add_ctx_noise=True)
            loss.backward()
            # Clip the global grad norm before the step: absorbs the occasional loss spike (the cause
            # of the 1024-width NaN divergence) so it never poisons AdamW's moments. clip_grad_norm_
            # returns the PRE-clip total norm, which we log -- if it sits well below grad_clip the cap
            # is just a safety net; if it rides at the cap, the cap is biting and should be raised.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            if use_ema:
                d = min(ema_decay, (gstep + 1) / (gstep + 11))   # warm up the decay from ~0
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        ema[n].mul_(d).add_(p.detach(), alpha=1.0 - d)
            gstep += 1
            tr_loss += loss.detach()
            tr_gnorm += gnorm.detach()

        model.eval()
        val_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for z_seq, z_future in val_loader:
                val_loss += _flow_batch(z_seq, z_future)

        nb_tr, nb_val = len(train_loader), len(val_loader)
        epoch_time = time.time() - start_time
        print(f"FM Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | GradNorm: {tr_gnorm.item()/nb_tr:.3f} | "
              f"Train Loss: {tr_loss.item()/nb_tr:.5f} | Val Loss: {val_loss.item()/nb_val:.5f}")

    # Swap the averaged weights into the model so eval + the saved checkpoint use them.
    if use_ema:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(ema[n])
    return model
