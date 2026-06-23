import time
import math
import torch
import torch.nn as nn
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
    """Encode every cached frame through the frozen AE once into a float16 latent cache.

    frames: (M, C, H, W) uint8 on `cache_device`. Returns (M, Cl, h, w) float16 on `cache_device`.
    The dynamics phase then trains directly on these latents, so the AE never runs inside the
    per-epoch training loop. float16 storage roughly halves the memory vs float32 (~2 GB vs ~4 GB
    for 500k 32x8x8 latents) at negligible accuracy cost for an MSE objective.
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
    return z_all


def train_autoencoder(ae, train_loader, val_loader, epochs=5, learning_rate=1e-3, weight_decay=1e-4, device="cuda"):
    print("--- Phase 1: Training Autoencoder ---")
    optimizer = optim.AdamW(ae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps)
    criterion = nn.MSELoss()
    ae.to(device)

    for epoch in range(epochs):
        start_time = time.time()

        ae.train()
        train_loss = torch.zeros((), device=device)
        for ctx_frames, target_frame in train_loader:
            B, T, C, H, W = ctx_frames.shape
            x = ctx_frames.view(-1, C, H, W).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(ae(x), x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.detach()

        ae.eval()
        val_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for ctx_frames, target_frame in val_loader:
                B, T, C, H, W = ctx_frames.shape
                x = ctx_frames.view(-1, C, H, W).to(device)
                val_loss += criterion(ae(x), x)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        print(f"AE Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e} | Train Loss: {train_loss.item()/len(train_loader):.8f} | Val Loss: {val_loss.item()/len(val_loader):.8f}")
    return ae


@torch.no_grad()
def _latent_rollout_loss(dynamics, loader, criterion, device, max_batches=None):
    """Free-running (eps=0) multi-step rollout loss on cached latents (no AE forward)."""
    dynamics.eval()
    total = torch.zeros((), device=device)
    nb = 0
    for bi, (z_seq, z_future) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        if z_future.dim() == 4:  # horizon==1 -> restore the K dim
            z_future = z_future.unsqueeze(1)
        K = z_future.shape[1]
        step_loss = torch.zeros((), device=device)
        for k in range(K):
            z_pred = dynamics(z_seq)
            step_loss += criterion(z_pred, z_future[:, k])
            z_seq = torch.cat([z_seq[:, 1:], z_pred.unsqueeze(1)], dim=1)
        total += step_loss / K
        nb += 1
    return (total / max(1, nb)).item()


def _pixel_rollout_loss(model, loader, criterion, device, max_batches=None):
    """Free-running (eps=0) multi-step rollout loss in pixel space (Pixel model)."""
    model.eval()
    total = torch.zeros((), device=device)
    nb = 0
    with torch.no_grad():
        for bi, (ctx_frames, future_frames) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            ctx_frames = ctx_frames.to(device)
            future_frames = future_frames.to(device)
            if future_frames.dim() == 4:
                future_frames = future_frames.unsqueeze(1)
            K = future_frames.shape[1]
            context = ctx_frames
            step_loss = torch.zeros((), device=device)
            for k in range(K):
                pred = model(context)
                step_loss += criterion(pred, future_frames[:, k])
                context = torch.cat([context[:, 1:], pred.unsqueeze(1)], dim=1)
            total += step_loss / K
            nb += 1
    return (total / max(1, nb)).item()


def train_dynamics(dynamics, train_loader, val_loader, epochs=15, learning_rate=1e-3, weight_decay=1e-4, device="cuda"):
    """Train the latent dynamics model on precomputed latents.

    `train_loader`/`val_loader` yield (z_seq, z_future) latent batches gathered from the cache
    (see CachedLoader over the latent tensor), so the frozen AE is never re-run here.
    """
    print("--- Phase 2: Training Latent Dynamics Model---")
    optimizer = optim.AdamW(dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps)
    criterion = nn.MSELoss()
    dynamics.to(device)

    for epoch in range(epochs):
        start_time = time.time()

        eps = 1.0 - epoch / max(1, epochs - 1)

        dynamics.train()
        train_loss = torch.zeros((), device=device)
        for z_seq, z_future in train_loader:
            # z_seq (B, T, Cl, h, w), z_future (B, K, Cl, h, w) -- float32 on device.
            if z_future.dim() == 4:  # horizon==1 -> restore the K dim
                z_future = z_future.unsqueeze(1)
            B, K = z_seq.shape[0], z_future.shape[1]

            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for k in range(K):
                z_pred = dynamics(z_seq)                    # (B, Cl, h, w)
                z_true = z_future[:, k]
                loss = loss + criterion(z_pred, z_true)
                teacher = torch.rand(B, 1, 1, 1, device=device) < eps
                z_next = torch.where(teacher, z_true, z_pred.detach())
                z_seq = torch.cat([z_seq[:, 1:], z_next.unsqueeze(1)], dim=1)
            loss = loss / K
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.detach()

        train_rollout = _latent_rollout_loss(dynamics, train_loader, criterion, device, max_batches=30)
        val_rollout = _latent_rollout_loss(dynamics, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        print(f"Dyn Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e} | Eps: {eps:.2f} | Train(ss): {train_loss.item()/len(train_loader):.5f} | Train(roll): {train_rollout:.5f} | Val(roll): {val_rollout:.5f}")


def train_pixel_model(model, train_loader, val_loader, epochs=15, learning_rate=1e-3, weight_decay=1e-4, device="cuda"):
    print("--- Training Pixel Dynamics Model ---")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps)
    criterion = nn.MSELoss()
    model.to(device)

    for epoch in range(epochs):
        start_time = time.time()

        eps = 1.0 - epoch / max(1, epochs - 1)

        model.train()
        train_loss = torch.zeros((), device=device)
        for ctx_frames, future_frames in train_loader:
            ctx_frames = ctx_frames.to(device)
            future_frames = future_frames.to(device)
            if future_frames.dim() == 4:
                future_frames = future_frames.unsqueeze(1)
            B, K = ctx_frames.shape[0], future_frames.shape[1]

            optimizer.zero_grad(set_to_none=True)
            context = ctx_frames
            loss = 0.0
            for k in range(K):
                pred = model(context)                    # (B, C, H, W)
                target = future_frames[:, k]
                loss = loss + criterion(pred, target)
                teacher = torch.rand(B, 1, 1, 1, device=device) < eps
                nxt = torch.where(teacher, target, pred.detach())
                context = torch.cat([context[:, 1:], nxt.unsqueeze(1)], dim=1)
            loss = loss / K
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.detach()

        train_rollout = _pixel_rollout_loss(model, train_loader, criterion, device, max_batches=30)
        val_rollout = _pixel_rollout_loss(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        print(f"Pixel Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e} | Eps: {eps:.2f} | Train(ss): {train_loss.item()/len(train_loader):.5f} | Train(roll): {train_rollout:.5f} | Val(roll): {val_rollout:.5f}")
    return model
