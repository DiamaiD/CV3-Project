"""What does the 32x8x8 latent space actually encode?

Three probes, all against ground-truth states:

  A: capacity -- AE round-trip (encode+decode) PSNR restricted to ball regions,
     split by the ball's situation (free / wall contact / ball-ball contact).
     If contact regions reconstruct far above the model's contact-prediction
     PSNR, the representation was never the contact bottleneck.
  B: content -- per-cell linear probes: from one cell's 32 channels, ridge-
     decode the nearest ball's offset, distance, radius and material, reported
     by distance-to-ball bins (tests the 'each cell knows the closest ball'
     hypothesis directly).
  C: stress -- decode a ball's exact position from the 3x3 cell neighborhood
     around it, comparing contact vs free configurations.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.extractor import extract_frame, snap_radius, MAT_NAMES
from experiments.rollout import load_run, test_split

CELL = 8
GRID = 8
BINS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]


def cell_centers():
    cx = np.array([[8 * j + 4.0 for j in range(GRID)] for _ in range(GRID)])
    cy = np.array([[60.0 - 8 * i for _ in range(GRID)] for i in range(GRID)])
    return cx, cy


def ridge_fit(X, Y, lam=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    ymu = Y.mean(0)
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * len(X) / X.shape[1] * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T @ (Y - ymu))
    return mu, sd, ymu, W


def ridge_apply(model, X):
    mu, sd, ymu, W = model
    return ((X - mu) / sd) @ W + ymu


def load_traj_meta(td, pos0):
    img = None
    import cv2
    img = cv2.cvtColor(cv2.imread(os.path.join(td, "frame_000.png")), cv2.COLOR_BGR2RGB)
    dets = extract_frame(img)
    if len(dets) != pos0.shape[0]:
        return None
    radii, mats, used = np.zeros(len(dets)), [None] * len(dets), set()
    for d in dets:
        order = np.argsort(np.linalg.norm(pos0 - [d["x"], d["y"]], axis=1))
        j = next(k for k in order if k not in used)
        if np.linalg.norm(pos0[j] - [d["x"], d["y"]]) > 2.0:
            return None
        used.add(j)
        radii[j], mats[j] = snap_radius(d["radius"]), d["mat"]
    return radii, mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/2026_07_14__13_56_10__bouncing_36k_mix_flow")
    ap.add_argument("--data", default="data/bouncing_20k_b25")
    ap.add_argument("--n-traj", type=int, default=250)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import cv2
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, dit, cfg = load_run(args.run, device)
    print(f"Latent analysis | AE from {args.run} | data {args.data}")

    import glob, random
    random.seed(args.seed)
    trajs = glob.glob(os.path.join(args.data, "traj-*"))
    random.shuffle(trajs)
    n_tr, n_va = int(len(trajs) * 0.8), int(len(trajs) * 0.1)
    test = trajs[n_tr + n_va:][:args.n_traj]

    frames_all, entries = [], []   # entries: (frame_idx_in_batch, pos, radii, mats)
    for td in test:
        pos = np.load(os.path.join(td, "positions.npy"))
        meta = load_traj_meta(td, pos[0])
        if meta is None:
            continue
        radii, mats = meta
        for t in range(0, pos.shape[0], args.stride):
            img = cv2.cvtColor(cv2.imread(os.path.join(td, f"frame_{t:03d}.png")), cv2.COLOR_BGR2RGB)
            entries.append((len(frames_all), pos[t], radii, mats))
            frames_all.append(img)
    frames_np = np.stack(frames_all)
    print(f"[data] {len(test)} trajs -> {len(frames_np)} frames")

    # encode + decode in batches
    mus, recons = [], []
    with torch.no_grad():
        for s in range(0, len(frames_np), 256):
            x = torch.from_numpy(frames_np[s:s + 256]).to(device).permute(0, 3, 1, 2).float() / 255.0
            mu = ae.encode(x)
            rec = ae.decode(mu).clamp(0, 1)
            mus.append(mu.cpu().float().numpy())
            recons.append((rec.cpu().numpy() * 255.0))
    mus = np.concatenate(mus)
    recons = np.concatenate(recons).transpose(0, 2, 3, 1)
    print(f"[encode] latents {mus.shape}")

    # ---------- A: recon PSNR by ball situation, ball-region masked ----------
    sit_mse = {"free": [], "wall": [], "bb contact": []}
    for fi, pos, radii, mats in entries:
        N = len(radii)
        for i in range(N):
            x, y, r = pos[i, 0], pos[i, 1], radii[i]
            near_wall = (x - r < 1.0) or (x + r > 63.0) or (y - r < 1.0) or (y + r > 63.0)
            in_contact = any(np.linalg.norm(pos[i] - pos[j]) < radii[i] + radii[j] + 1.0
                             for j in range(N) if j != i)
            col, row = x - 0.5, 63.5 - y
            c0, c1 = int(max(col - r - 2, 0)), int(min(col + r + 3, 64))
            r0, r1 = int(max(row - r - 2, 0)), int(min(row + r + 3, 64))
            if c1 <= c0 or r1 <= r0:
                continue
            gt = frames_np[fi][r0:r1, c0:c1].astype(np.float64)
            rc = recons[fi][r0:r1, c0:c1].astype(np.float64)
            mse = ((gt - rc) ** 2).mean()
            key = "bb contact" if in_contact else ("wall" if near_wall else "free")
            sit_mse[key].append(mse)
    recon_by_class = {}
    print("\n[A] AE round-trip, ball-region PSNR by situation:")
    for k, v in sit_mse.items():
        pooled = 10 * np.log10(255.0 ** 2 / np.mean(v))
        recon_by_class[k] = {"psnr_pooled": float(pooled), "n_regions": len(v)}
        print(f"  {k:12s}: {pooled:6.2f} dB  (n={len(v):,})")

    # ---------- B: per-cell probes for the nearest ball ----------
    cx, cy = cell_centers()
    X_rows, T_dx, T_dy, T_dist, T_rad, T_mat, T_traj = [], [], [], [], [], [], []
    for ei, (fi, pos, radii, mats) in enumerate(entries):
        feat = mus[fi].reshape(32, -1).T          # (64, 32)
        d = np.sqrt((cx.ravel()[:, None] - pos[None, :, 0]) ** 2
                    + (cy.ravel()[:, None] - pos[None, :, 1]) ** 2)
        nb = d.argmin(1)
        X_rows.append(feat)
        T_dx.append(pos[nb, 0] - cx.ravel())
        T_dy.append(pos[nb, 1] - cy.ravel())
        T_dist.append(d.min(1))
        T_rad.append(radii[nb])
        T_mat.append(np.array([MAT_NAMES.index(mats[b]) for b in nb]))
        T_traj.append(np.full(64, ei))
    X = np.concatenate(X_rows)
    T_dx, T_dy = np.concatenate(T_dx), np.concatenate(T_dy)
    T_dist, T_rad = np.concatenate(T_dist), np.concatenate(T_rad)
    T_mat, T_traj = np.concatenate(T_mat), np.concatenate(T_traj)

    split = T_traj % 5 != 0    # 80/20 by entry
    Y_reg = np.stack([T_dx, T_dy, T_dist, T_rad], 1)
    Y_mat = np.eye(4)[T_mat]

    probe_bins = {}
    print("\n[B] per-bin linear probes from ONE cell's 32 channels -> nearest ball (held-out trajs):")
    print("  dist bin   |  offset RMSE (0-model) |  dist RMSE  |  radius RMSE (0-model) |  mat acc | n")
    for lo, hi in BINS:
        tr = split & (T_dist >= lo) & (T_dist < hi)
        te = (~split) & (T_dist >= lo) & (T_dist < hi)
        if tr.sum() < 500 or te.sum() < 50:
            continue
        m_reg = ridge_fit(X[tr], Y_reg[tr])
        m_mat = ridge_fit(X[tr], Y_mat[tr])
        P_reg = ridge_apply(m_reg, X[te])
        P_mat = ridge_apply(m_mat, X[te]).argmax(1)
        g = Y_reg[te]
        off = float(np.sqrt(((P_reg[:, :2] - g[:, :2]) ** 2).sum(1).mean()))
        off0 = float(np.sqrt(((g[:, :2] - Y_reg[tr][:, :2].mean(0)) ** 2).sum(1).mean()))
        dr = float(np.sqrt(((P_reg[:, 2] - g[:, 2]) ** 2).mean()))
        rr = float(np.sqrt(((P_reg[:, 3] - g[:, 3]) ** 2).mean()))
        rr0 = float(np.sqrt(((g[:, 3] - Y_reg[tr][:, 3].mean()) ** 2).mean()))
        acc = float((P_mat == T_mat[te]).mean())
        tag = f"{lo:.0f}-{hi:.0f}px" if hi < 1e8 else f">{lo:.0f}px"
        probe_bins[tag] = {"offset_rmse": off, "offset_rmse_baseline": off0, "dist_rmse": dr,
                           "radius_rmse": rr, "radius_rmse_baseline": rr0,
                           "material_acc": acc, "n": int(te.sum())}
        print(f"  {tag:9s}  |  {off:6.2f} ({off0:5.2f})     |  {dr:5.2f} px   |  {rr:5.2f} ({rr0:4.2f})      |  {acc:5.1%}  | {int(te.sum()):,}")

    # ---------- C: position decoding, 3x3 neighborhood, contact vs free ----------
    Xc, Yc, is_contact, traj_id = [], [], [], []
    for ei, (fi, pos, radii, mats) in enumerate(entries):
        lat = mus[fi]
        N = len(radii)
        for i in range(N):
            x, y = pos[i]
            j = int(np.clip((x - 0.5) // 8, 0, 7))
            irow = int(np.clip((63.5 - y) // 8, 0, 7))
            if not (1 <= j <= 6 and 1 <= irow <= 6):
                continue
            feat = lat[:, irow - 1:irow + 2, j - 1:j + 2].ravel()
            Xc.append(feat)
            Yc.append((x - (8 * j + 4.0), y - (60.0 - 8 * irow)))
            is_contact.append(any(np.linalg.norm(pos[i] - pos[k]) < radii[i] + radii[k] + 1.0
                                  for k in range(N) if k != i))
            traj_id.append(ei)
    Xc, Yc = np.array(Xc), np.array(Yc)
    is_contact, traj_id = np.array(is_contact), np.array(traj_id)
    sp = traj_id % 5 != 0
    m_pos = ridge_fit(Xc[sp], Yc[sp])
    P = ridge_apply(m_pos, Xc[~sp])
    err = np.sqrt(((P - Yc[~sp]) ** 2).sum(1))
    ct = is_contact[~sp]
    pos_probe = {"free_med_px": float(np.median(err[~ct])), "free_p95_px": float(np.percentile(err[~ct], 95)),
                 "contact_med_px": float(np.median(err[ct])), "contact_p95_px": float(np.percentile(err[ct], 95)),
                 "n_free": int((~ct).sum()), "n_contact": int(ct.sum())}
    print(f"\n[C] ball position from 3x3 cell neighborhood (linear, held-out):")
    print(f"  free:    median {pos_probe['free_med_px']:.3f} px | p95 {pos_probe['free_p95_px']:.3f} (n={pos_probe['n_free']:,})")
    print(f"  contact: median {pos_probe['contact_med_px']:.3f} px | p95 {pos_probe['contact_p95_px']:.3f} (n={pos_probe['n_contact']:,})")

    out_dir = os.path.join(args.run, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "latent_analysis.json")
    with open(out, "w") as f:
        json.dump({"run": args.run, "data": args.data, "n_frames": len(frames_np),
                   "recon_by_situation": recon_by_class, "cell_probe_by_distance": probe_bins,
                   "position_probe_3x3": pos_probe}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
