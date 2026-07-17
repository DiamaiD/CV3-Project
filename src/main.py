import os

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))

import argparse
import json
import logging
import torch
import random
import numpy as np
import glob

logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from src.dataset import FrameCache, CachedLoader
from src.models import CNNVAE, DiffusionTransformer
from src.train import (train_autoencoder, build_latent_cache, train_flow_matching,
                       score_latent_predictability, retrain_decoder)
from src.eval import (run_evaluation, save_rollout_video, save_vae_reconstructions, flow_sample,
                      collision_conditioned_eval, build_event_labels)
from src.utils import setup_run_folder, teardown_run_logging


def run_training_pipeline(*args, **kwargs):
    try:
        return _run_pipeline(*args, **kwargs)
    finally:
        teardown_run_logging()


def _parse_event_weights(s):
    s = str(s).strip()
    if not s or s in ("0", "off"):
        return None
    try:
        w = [float(x) for x in s.split(",")]
    except ValueError:
        w = []
    if len(w) != 4 or any(v < 0 for v in w) or sum(w) <= 0:
        print(f"[Warn] Event weights '{s}' invalid (want 4 non-negative numbers: free,wall,post,bb) "
              f"-- uniform sampling.")
        return None
    return w


def _run_pipeline(data_dir, env_name, context_len=5,
                          ae_batch_size=32, dyn_batch_size=64, ae_epochs=20, dyn_epochs=30,
                          ae_learning_rate=5e-4, ae_weight_decay=1e-2, ae_kl_weight=0.005,
                          ae_lpips_weight=0.0, ae_lpips_net="alex", ae_focal_weight=0.0,
                          ae_probe="on", ae_grad_clip=10.0, ae_precision="bf16",
                          dyn_learning_rate=3e-4, dit_min_lr=1e-6, dit_warmup_frac=0.05,
                          dyn_epochs_2=0, dyn_learning_rate_2=2e-4, dit_min_lr_2=2e-6,
                          dyn_weight_decay=1e-4, dit_grad_clip=3.0,
                          dit_ema_decay=0.999, dit_context_noise=0.0, dit_precision="bf16",
                          compile_mode="off", dit_event_weights="off",
                          dit_t_dist="logit_normal", dit_loss="mse", dit_huber_c=1.0,
                          dec_epochs=0, dec_learning_rate=1e-4, dec_lpips_weight=1.0, dec_lpips_net="alex",
                          dec_rollout_k=5, dec_clean_frac=0.3, dec_grad_clip=10.0, dec_res_blocks=1,
                          dec_n_train_traj=1000, dec_checkpoint="", dec_precision="bf16",
                          eval_horizon=50, eval_max_batches=24, eval_best_of_n=1,
                          eval_n_pngs=1, eval_n_gifs=2, eval_gif_len=40,
                          seed=None, ae_checkpoint="", dit_checkpoint="", latent_grid=8, latent_ch=32,
                          vae_enc_res_blocks=1, vae_dec_res_blocks=1,
                          chunk_len=5, dit_d_model=256, dit_n_layers=6, dit_n_heads=8, inference_steps=10):
    run_config = dict(locals())
    run_dir, log_filepath = setup_run_folder(f"{env_name}_flow")
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    cache_device = "cpu"

    if seed is not None and seed != "":
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")

    print(f"Using device: {device} | Latent Flow Matching | Context: {context_len} frames | Chunk: {chunk_len}")
    print(f"Cache: {cache_device.upper()} | Inference steps: {inference_steps} | Latent: {latent_ch}x{latent_grid}x{latent_grid} | "
          f"ResBlocks enc/dec: {vae_enc_res_blocks}/{vae_dec_res_blocks} (Phase 3 dec: {dec_res_blocks})")
    print(f"VAE -> LR: {ae_learning_rate} | WD: {ae_weight_decay} | Batch: {ae_batch_size} | "
          f"Beta: {ae_kl_weight} | LPIPS: {ae_lpips_weight}")
    print(f"DiT -> LR: {dyn_learning_rate} | WD: {dyn_weight_decay} | Batch: {dyn_batch_size} | "
          f"d_model: {dit_d_model} | layers: {dit_n_layers} | heads: {dit_n_heads} | "
          f"context layout: temporal tokens")

    all_trajs = glob.glob(os.path.join(data_dir, "traj-*"))
    if not all_trajs:
        print(f"[Error] No trajectories found in {data_dir}. Did you run the generator?")
        return

    random.shuffle(all_trajs)
    n_train, n_val = int(len(all_trajs) * 0.8), int(len(all_trajs) * 0.1)
    train_trajs = all_trajs[:n_train]
    val_trajs = all_trajs[n_train:n_train + n_val]
    test_trajs = all_trajs[n_train + n_val:]

    frame_cache = FrameCache(all_trajs, cache_device=cache_device,
                             disk_cache_path=os.path.join(data_dir, "frames_cache.pt"))

    def pixel_loader(trajs, horizon, shuffle, bs):
        ctx_idx, tgt_idx = frame_cache.build_windows(trajs, context_len, horizon)
        return CachedLoader(frame_cache.frames, ctx_idx, tgt_idx, bs, device,
                            shuffle=shuffle, horizon=horizon)

    ae = CNNVAE(latent_ch=latent_ch, latent_grid=latent_grid,
                enc_res_blocks=vae_enc_res_blocks, dec_res_blocks=vae_dec_res_blocks).to(device)

    vae_trained = not (ae_checkpoint and os.path.exists(ae_checkpoint))
    if not vae_trained:
        ae.load_state_dict(torch.load(ae_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded autoencoder from {ae_checkpoint} -- skipping Phase 1.")
    else:
        if ae_checkpoint:
            print(f"[Warn] AE checkpoint not found: {ae_checkpoint}. Training a new autoencoder.")
        ae = train_autoencoder(ae, pixel_loader(train_trajs, 1, True, ae_batch_size),
                               pixel_loader(val_trajs, 1, False, ae_batch_size),
                               epochs=ae_epochs, learning_rate=ae_learning_rate,
                               weight_decay=ae_weight_decay, kl_weight=ae_kl_weight,
                               lpips_weight=ae_lpips_weight, lpips_net=ae_lpips_net,
                               focal_weight=ae_focal_weight,
                               grad_clip=ae_grad_clip, precision=ae_precision, compile_mode=compile_mode, device=device)

    save_vae_reconstructions(ae, pixel_loader(val_trajs, 1, False, ae_batch_size), device, run_dir)
    torch.save(ae.state_dict(), os.path.join(run_dir, "autoencoder.pth"))

    z_all, latent_scale = build_latent_cache(ae, frame_cache.frames, device, cache_device)
    latent_ch, grid = z_all.shape[1], z_all.shape[-1]

    if vae_trained and ae_probe == "on":
        score_latent_predictability(ae, z_all, latent_scale, frame_cache, train_trajs, val_trajs,
                                    context_len, device)

    def latent_loader(trajs, horizon, shuffle, bs):
        ctx_idx, tgt_idx = frame_cache.build_windows(trajs, context_len, horizon)
        return CachedLoader(z_all, ctx_idx, tgt_idx, bs, device, shuffle=shuffle, horizon=horizon)

    dit_loaded = bool(dit_checkpoint) and os.path.exists(dit_checkpoint)
    if dit_checkpoint and not dit_loaded:
        print(f"[Warn] DiT checkpoint not found: {dit_checkpoint}. Training a new DiT.")
    if dyn_epochs <= 0 and not dit_loaded:
        print("DiT epochs = 0 and no DiT checkpoint -> stopping after the VAE stage "
              "(no DiT training, decoder training or evaluation).")
        print(f"Training Complete. All files saved in {run_dir}.")
        return

    dit = DiffusionTransformer(latent_ch=latent_ch, context_len=context_len, grid=grid,
                               chunk_len=chunk_len, d_model=dit_d_model, n_layers=dit_n_layers,
                               n_heads=dit_n_heads, latent_scale=latent_scale,
                               ).to(device)
    if dit_loaded:
        dit.load_state_dict(torch.load(dit_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded DiT from {dit_checkpoint} -- skipping Phase 2.")
    else:
        fm_train = latent_loader(train_trajs, chunk_len, True, dyn_batch_size)
        ev = _parse_event_weights(dit_event_weights)
        if ev is not None:
            labels_g, n_missing = build_event_labels(frame_cache, train_trajs)
            if n_missing == len(train_trajs):
                print("[FM] Event weights set but no positions.npy in the dataset -- uniform sampling.")
            else:
                sev = labels_g[fm_train.tgt_index.cpu()].max(dim=1).values.long()
                fm_train.sample_weights = torch.tensor(ev)[sev].to(fm_train.ctx_index.device)
                nat = torch.bincount(sev, minlength=4).double()
                mix = nat * torch.tensor(ev, dtype=torch.float64)
                nat, mix = 100 * nat / nat.sum(), 100 * mix / mix.sum()
                wtxt = "/".join(f"{v:g}" for v in ev)
                print(f"[FM] Event-weighted sampling ON (free/wall/post/bb = {wtxt}): "
                      f"batch mix {mix[0]:.0f}/{mix[1]:.0f}/{mix[2]:.0f}/{mix[3]:.0f}% "
                      f"vs natural {nat[0]:.0f}/{nat[1]:.0f}/{nat[2]:.0f}/{nat[3]:.0f}%.")
        dit = train_flow_matching(dit, fm_train,
                                  latent_loader(val_trajs, chunk_len, False, dyn_batch_size),
                                  epochs=dyn_epochs, learning_rate=dyn_learning_rate,
                                  min_lr=dit_min_lr, warmup_frac=dit_warmup_frac,
                                  epochs_2=dyn_epochs_2, learning_rate_2=dyn_learning_rate_2,
                                  min_lr_2=dit_min_lr_2,
                                  weight_decay=dyn_weight_decay, grad_clip=dit_grad_clip,
                                  ema_decay=dit_ema_decay, context_noise=dit_context_noise,
                                  precision=dit_precision,
                                  compile_mode=compile_mode, t_dist=dit_t_dist,
                                  loss_type=dit_loss, huber_c=dit_huber_c, device=device)
    torch.save(dit.state_dict(), os.path.join(run_dir, "dit.pth"))

    dec_loaded = bool(dec_checkpoint) and os.path.exists(dec_checkpoint)
    if dec_checkpoint and not dec_loaded:
        print(f"[Warn] Decoder checkpoint not found: {dec_checkpoint}. Training Phase 3 as configured.")
    if dec_loaded:
        ae_full = CNNVAE(latent_ch=latent_ch, latent_grid=latent_grid,
                         enc_res_blocks=vae_enc_res_blocks, dec_res_blocks=dec_res_blocks).to(device)
        ae_full.load_state_dict(torch.load(dec_checkpoint, map_location=device, weights_only=True))
        ae = ae_full.eval()
        print(f"Loaded final VAE (retrained decoder) from {dec_checkpoint} -- skipping Phase 3.")
    elif dec_epochs > 0:
        ae_full = CNNVAE(latent_ch=latent_ch, latent_grid=latent_grid,
                         enc_res_blocks=vae_enc_res_blocks, dec_res_blocks=dec_res_blocks).to(device)
        enc_state = {k: v for k, v in ae.state_dict().items()
                     if k.startswith(("conv_in", "enc", "to_mu", "to_logvar"))}
        ae_full.load_state_dict(enc_state, strict=False)
        ae = retrain_decoder(ae_full, dit, z_all, frame_cache, train_trajs, val_trajs, context_len,
                             epochs=dec_epochs, learning_rate=dec_learning_rate,
                             lpips_weight=dec_lpips_weight, lpips_net=dec_lpips_net,
                             rollout_k=dec_rollout_k,
                             clean_frac=dec_clean_frac, num_steps=inference_steps,
                             batch_size=ae_batch_size, grad_clip=dec_grad_clip,
                             n_train_traj=dec_n_train_traj, precision=dec_precision, compile_mode=compile_mode, device=device)
    if dec_loaded or dec_epochs > 0:
        torch.save(ae.state_dict(), os.path.join(run_dir, "autoencoder_final.pth"))

    run_evaluation(pixel_loader(test_trajs, eval_horizon, False, ae_batch_size), device, run_dir,
                   ae=ae, dit=dit, num_steps=inference_steps, max_batches=eval_max_batches,
                   best_of_n=eval_best_of_n, n_pngs=eval_n_pngs, compile_mode=compile_mode)

    ae.eval(); dit.eval()

    collision_conditioned_eval(ae, dit, z_all, frame_cache, test_trajs, context_len,
                               num_steps=inference_steps, run_dir=run_dir, device=device)

    def flow_predict_chunk(context):
        _, T, C, H, W = context.shape
        z = ae.encode(context.view(-1, C, H, W)) / dit.latent_scale
        z_seq = z.view(1, T, *z.shape[1:])
        z_chunk = flow_sample(dit, z_seq, inference_steps)
        K = z_chunk.shape[1]
        frames = ae.decode((z_chunk * dit.latent_scale).reshape(K, *z_chunk.shape[2:]))
        return frames.unsqueeze(0)

    for traj in test_trajs[:eval_n_gifs]:
        save_rollout_video(flow_predict_chunk, traj, device, run_dir, context_len=context_len,
                           n_steps=eval_gif_len)

    print(f"Training Complete. All files saved in {run_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bouncing")
    parser.add_argument("--env_name", type=str, default="bouncing")
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--ae_batch_size", type=int, default=32, help="Batch size for the VAE phase (full-res frames)")
    parser.add_argument("--dyn_batch_size", type=int, default=64, help="Batch size for the DiT phase (tiny latents -> can be much larger)")
    parser.add_argument("--ae_epochs", type=int, default=20)
    parser.add_argument("--dyn_epochs", type=int, default=30)
    parser.add_argument("--ae_learning_rate", type=float, default=5e-4)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-2)
    parser.add_argument("--ae_kl_weight", type=float, default=0.005, help="VAE KL weight (beta). Lower if reconstructions blur / KL collapses; raise if the latent is barely regularized.")
    parser.add_argument("--ae_lpips_weight", type=float, default=0.0, help="Perceptual (LPIPS) loss weight on the VAE. 0 = off (pixel+KL only). ~1.0 makes latent L2 track perceptual quality, the key fix for the prediction-blur ceiling. Needs `pip install lpips`.")
    parser.add_argument("--ae_lpips_net", type=str, default="alex", choices=["alex", "vgg"], help="LPIPS backbone for Phase 1. alex is cheaper and historically gave the more PREDICTABLE latent (better DiT); vgg pushes recon sharper but traded predictability away in every run so far.")
    parser.add_argument("--ae_probe", choices=["on", "off"], default="on", help="In-run LatentProbe after VAE training. Turn off for AEs trained on non-temporal data (e.g. synthesized frame sets), where 1-step windows are meaningless -- score those with experiments/surrogate.py on real data instead.")
    parser.add_argument("--ae_focal_weight", type=float, default=0.0, help="Error-focused pixel weighting on the VAE recon loss (0 = off). Each pixel's squared error is upweighted by 1 + w*(err/mean_err), detached and scale-normalized by (1+w) so the recon/LPIPS balance stays fixed. Concentrates capacity on hard pixels (ball-ball contact regions carry ~8x the squared error of free flight).")
    parser.add_argument("--ae_grad_clip", type=float, default=10.0, help="Max global grad norm for the VAE (clipped each step). Safety net against the loss spikes a deeper LPIPS backbone (e.g. VGG) can trigger. The sum-reduced recon makes norms large, so this is loose; the logged GradNorm (pre-clip) shows the steady-state -- tighten toward ~2-3x it once observed.")
    parser.add_argument("--latent_grid", type=int, default=8, help="VAE latent spatial size (8 -> 8x8, 16 -> 16x16). 16 makes motion more spatially local for the DiT at 4x token/cache cost. Requires retraining the VAE (8x8 checkpoints are incompatible).")
    parser.add_argument("--latent_ch", type=int, default=32, help="VAE latent channels per grid cell. More channels raise the reconstruction ceiling and give the DiT a richer per-cell code, at linearly more latent-cache size (does NOT change DiT token count -- that's chunk_len x grid^2). Requires retraining the VAE (checkpoints with a different channel count are incompatible); the DiT adapts automatically from the cache shape.")
    parser.add_argument("--dyn_learning_rate", type=float, default=3e-4)
    parser.add_argument("--dit_min_lr", type=float, default=1e-6, help="Phase 2 LR floor: the cosine anneal is rescaled to end exactly at this LR instead of zero (0 = decay to zero). Measured on 20k runs: below ~1e-6 val loss stops improving and the train/val gap keeps widening -- annealing into that region only overfits.")
    parser.add_argument("--dit_warmup_frac", type=float, default=0.05, help="Phase 2 LR warmup length as a fraction of the full schedule (0.05 = 5%%, the long-standing default). 0 = no warmup (cosine decay starts at full LR). Warmup steps come out of the same total budget, so shorter warmup means a longer decay.")
    parser.add_argument("--dyn_epochs_2", type=int, default=0, help="Warm-restart second iteration, in epochs. 0 = off (single cosine schedule, the default). >0: after --dyn_epochs finish annealing --dyn_learning_rate down to --dit_min_lr, the LR jumps to --dyn_learning_rate_2 and cosine-anneals to --dit_min_lr_2 over this many epochs. Weights, optimizer state and EMA carry over; warmup applies to the first iteration only. All other hyperparameters are shared by both iterations.")
    parser.add_argument("--dyn_learning_rate_2", type=float, default=2e-4, help="Peak LR of the warm-restart iteration (only used when --dyn_epochs_2 > 0).")
    parser.add_argument("--dit_min_lr_2", type=float, default=2e-6, help="LR floor of the warm-restart iteration (only used when --dyn_epochs_2 > 0).")
    parser.add_argument("--dyn_weight_decay", type=float, default=1e-4)
    parser.add_argument("--dit_grad_clip", type=float, default=3.0, help="Max global grad norm for the DiT (clipped each step). Safety net against loss-spike NaN divergence; the logged GradNorm shows whether it's biting. Set ~2-3x above the steady-state norm (which rode 1.2-1.4 late in training, so 1.0 was clipping healthy steps).")
    parser.add_argument("--dit_ema_decay", type=float, default=0.999, help="Weight-EMA decay for the DiT (0 = off). Eval + saved checkpoint use the averaged weights. Standard diffusion/flow trick, usually worth a few tenths of a dB.")
    parser.add_argument("--dit_precision", choices=["bf16", "fp16", "fp32"], default="bf16", help="DiT training compute precision. bf16/fp16 halve memory traffic (~2x epoch speed at >=8192 token-rows/step) with fp32 master weights; fp16 adds dynamic loss scaling whose inf/nan step-skip doubles as a spike guard (bf16 measured divergent on the temporal layout). fp32 = no autocast, the proven-stable reference. Weights/eval/checkpoints are fp32 in every mode.")
    parser.add_argument("--dit_event_weights", type=str, default="off", help="Event-weighted window sampling for DiT training: 4 comma-separated weights (free,wall,post,bb) applied to target-frame event classes, e.g. 0.5,2,2,4 to oversample collisions. Requires positions.npy/velocities.npy in the dataset. off = uniform.")
    parser.add_argument("--dit_t_dist", choices=["logit_normal", "uniform"], default="logit_normal", help="Flow-matching timestep sampling. logit_normal (SD3) draws t=sigmoid(N(0,1)), concentrating training on the hard middle of the trajectory; uniform is plain U(0,1). Same optimum, different emphasis -- logit_normal is a consistent quality win at no extra cost.")
    parser.add_argument("--dit_loss", choices=["mse", "huber"], default="mse", help="DiT velocity loss. mse = standard flow-matching L2 (regression to the conditional mean velocity). huber = pseudo-Huber sqrt(err^2+c^2)-c, whose per-element gradient saturates for large errors, so an outlier sample can't produce an unbounded gradient -- curbs precision-induced spikes at the source.")
    parser.add_argument("--dit_huber_c", type=float, default=1.0, help="Pseudo-Huber transition constant c (only used when --dit_loss huber). Errors >> c behave like L1, << c like L2. Velocity targets have per-element std ~1.4 in the normalized latent space, so c~1.0 balances robustness against fidelity; smaller c = more robust but further from L2's optimum.")
    parser.add_argument("--dit_fp32", action="store_true", help="Legacy alias for --dit_precision fp32 (takes precedence when set).")
    parser.add_argument("--compile", choices=["off", "on"], default="off", help="torch.compile (Inductor) for all training phases. Falls back to eager if compilation fails; checkpoints are unaffected either way.")
    parser.add_argument("--ae_precision", choices=["bf16", "fp16", "fp32"], default="bf16", help="Phase 1 (VAE) compute precision (autocast; fp16 adds gradient scaling).")
    parser.add_argument("--dec_precision", choices=["bf16", "fp16", "fp32"], default="bf16", help="Phase 3 (decoder) compute precision (autocast; fp16 adds gradient scaling).")
    parser.add_argument("--dit_context_noise", type=float, default=0.0, help="Std of Gaussian noise added to the DiT's CONTEXT latents during training (0 = off; in normalized-latent units). Rollout-robustness regularizer against exposure bias; trades a little 1-step accuracy for steadier long horizons. Try ~0.02-0.1.")
    parser.add_argument("--vae_enc_res_blocks", type=int, default=1, help="Residual blocks per level in the VAE ENCODER. 0 = weak encoder (plain conv+downsample, no bottleneck/attention) -- it cannot write an entangled latent at all; 0/0 with the decoder approximates the pre-residual VAE. A reused ae_checkpoint must match this setting.")
    parser.add_argument("--vae_dec_res_blocks", type=int, default=1, help="Residual blocks per level in the PHASE-1 decoder. 0 = weak decoder (plain conv+upsample, no bottleneck/attention): forces the encoder to write an explicit, predictable latent; pair with dec_epochs>0 so Phase 3 trains a full decoder for rendering. A reused ae_checkpoint must match this setting.")
    parser.add_argument("--dec_epochs", type=int, default=0, help="Phase 3: epochs to train a FRESH full-capacity decoder from scratch on clean + DiT-predicted latents (0 = off). Encoder + DiT stay frozen so the latent space and dit.pth remain valid; only the rendering changes. Runs after DiT training, before the final eval.")
    parser.add_argument("--dec_learning_rate", type=float, default=5e-4, help="Phase 3 decoder LR (the decoder trains from scratch, so the full VAE-scale LR is appropriate).")
    parser.add_argument("--dec_lpips_weight", type=float, default=1.0, help="Phase 3 perceptual (LPIPS) weight, same convention as ae_lpips_weight.")
    parser.add_argument("--dec_lpips_net", type=str, default="alex", choices=["alex", "vgg"], help="LPIPS backbone for Phase 3. Here the latent is FROZEN, so vgg's sharper gradients cannot hurt predictability -- it only shapes the renderer; worth trying vgg for crisper rollouts.")
    parser.add_argument("--dec_rollout_k", type=int, default=5, help="Phase 3 max rollout depth K: each batch decodes a DiT latent from a free-running rollout of random depth 1..K, so the decoder sees realistically drifted latents, not just 1-step error.")
    parser.add_argument("--dec_clean_frac", type=float, default=0.3, help="Phase 3 fraction of batches that decode CLEAN cached encoder latents instead of DiT predictions -- anchors reconstruction quality so the decoder does not overfit to rendering model error.")
    parser.add_argument("--dec_grad_clip", type=float, default=10.0, help="Max global grad norm for the Phase 3 decoder training (clipped each step).")
    parser.add_argument("--dec_res_blocks", type=int, default=1, help="Residual blocks per level in the PHASE-3 decoder that is trained from scratch (1 = full residual decoder with bottleneck+attention, 0 = weak plain decoder). Lets you match or exceed the Phase-1 decoder capacity for the final renderer.")
    parser.add_argument("--dec_n_train_traj", type=int, default=1000, help="Phase 3 data cap: trajectories used to build the decoder-training windows (~90+ windows each at rollout_k=5; the full train split is ~4000). Epoch time scales ~linearly; raise it if the decoder underfits or for the final run.")
    parser.add_argument("--dit_checkpoint", type=str, default="", help="Path to a saved dit.pth to reuse (skips Phase 2, like --ae_checkpoint skips Phase 1). Must match the DiT architecture flags and belong to the same run as the reused VAE -- a DiT only understands the latent space it was trained on.")
    parser.add_argument("--dec_checkpoint", type=str, default="", help="Path to a saved autoencoder_final.pth to reuse as the eval renderer (skips Phase 3 training). For DiT sweeps on a fixed VAE: keeps pixel metrics comparable without paying for decoder training every run. Must come from a run with the same latent space and match the enc/dec res-block flags.")
    parser.add_argument("--chunk_len", type=int, default=5, help="Chunk prediction: number of future frames (K) the DiT denoises jointly per call")
    parser.add_argument("--dit_d_model", type=int, default=256, help="DiT width (must be divisible by dit_n_heads)")
    parser.add_argument("--dit_n_layers", type=int, default=6, help="DiT depth")
    parser.add_argument("--dit_n_heads", type=int, default=8, help="DiT attention heads")
    parser.add_argument("--inference_steps", type=int, default=10, help="Euler ODE steps used to sample a frame")
    parser.add_argument("--eval_horizon", type=int, default=50, help="Rollout length used at eval time to report error growth vs horizon")
    parser.add_argument("--eval_max_batches", type=int, default=24, help="Cap on test batches rolled out at eval (each window costs eval_horizon x inference_steps DiT forwards)")
    parser.add_argument("--eval_best_of_n", type=int, default=1, help="Sample N independent rollouts per window and also report the best (per trajectory). Reveals if single-sample MSE punishes valid alternative futures. Multiplies eval cost by N.")
    parser.add_argument("--eval_n_pngs", type=int, default=1, help="Number of eval prediction grid PNGs (8 scenarios each, collected across test batches).")
    parser.add_argument("--eval_n_gifs", type=int, default=2, help="Number of test trajectories rendered as rollout GIFs.")
    parser.add_argument("--eval_gif_len", type=int, default=40, help="Rollout steps per GIF; clamped to trajectory length minus context_len.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs")
    parser.add_argument("--ae_checkpoint", type=str, default="", help="Path to a saved autoencoder.pth to reuse (skips Phase 1)")
    args = parser.parse_args()

    run_training_pipeline(
        args.data_dir, args.env_name,
        context_len=args.context_len, ae_batch_size=args.ae_batch_size, dyn_batch_size=args.dyn_batch_size,
        ae_epochs=args.ae_epochs, dyn_epochs=args.dyn_epochs,
        ae_learning_rate=args.ae_learning_rate, ae_weight_decay=args.ae_weight_decay, ae_kl_weight=args.ae_kl_weight,
        ae_lpips_weight=args.ae_lpips_weight, ae_lpips_net=args.ae_lpips_net,
        ae_focal_weight=args.ae_focal_weight, ae_probe=args.ae_probe, ae_grad_clip=args.ae_grad_clip,
        dyn_learning_rate=args.dyn_learning_rate, dit_min_lr=args.dit_min_lr,
        dit_warmup_frac=args.dit_warmup_frac,
        dyn_epochs_2=args.dyn_epochs_2, dyn_learning_rate_2=args.dyn_learning_rate_2,
        dit_min_lr_2=args.dit_min_lr_2,
        dyn_weight_decay=args.dyn_weight_decay,
        dit_grad_clip=args.dit_grad_clip, dit_ema_decay=args.dit_ema_decay,
        dit_precision=("fp32" if args.dit_fp32 else args.dit_precision),
        dit_event_weights=args.dit_event_weights,
        dit_t_dist=args.dit_t_dist, dit_loss=args.dit_loss, dit_huber_c=args.dit_huber_c,
        compile_mode=args.compile,
        ae_precision=args.ae_precision, dec_precision=args.dec_precision,
        dit_context_noise=args.dit_context_noise,
        dec_epochs=args.dec_epochs, dec_learning_rate=args.dec_learning_rate,
        dec_lpips_weight=args.dec_lpips_weight, dec_lpips_net=args.dec_lpips_net,
        dec_rollout_k=args.dec_rollout_k,
        dec_clean_frac=args.dec_clean_frac, dec_grad_clip=args.dec_grad_clip,
        dec_res_blocks=args.dec_res_blocks, dec_n_train_traj=args.dec_n_train_traj,
        dec_checkpoint=args.dec_checkpoint,
        eval_horizon=args.eval_horizon, eval_max_batches=args.eval_max_batches,
        eval_best_of_n=args.eval_best_of_n,
        eval_n_pngs=args.eval_n_pngs, eval_n_gifs=args.eval_n_gifs, eval_gif_len=args.eval_gif_len,
        seed=args.seed, ae_checkpoint=args.ae_checkpoint, dit_checkpoint=args.dit_checkpoint,
        latent_grid=args.latent_grid, latent_ch=args.latent_ch,
        vae_enc_res_blocks=args.vae_enc_res_blocks, vae_dec_res_blocks=args.vae_dec_res_blocks,
        chunk_len=args.chunk_len, dit_d_model=args.dit_d_model, dit_n_layers=args.dit_n_layers,
        dit_n_heads=args.dit_n_heads, inference_steps=args.inference_steps
    )
