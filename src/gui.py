import customtkinter as ctk
import threading
import os
import json
import shutil
from tkinter import filedialog
from src.main import run_training_pipeline

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


CONFIG_FILE = "configs/model_config.json"

class TrainingGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Physics Video Model Trainer")
        self.geometry("1200x850")
        
        self.huge_font = ctk.CTkFont(family="DejaVu Sans Mono", size=18)
        self.bold_font = ctk.CTkFont(family="DejaVu Sans Mono", size=18, weight="bold")

        os.makedirs("configs", exist_ok=True)

        self.general_frame = ctk.CTkFrame(self)
        self.general_frame.pack(pady=(10, 0), padx=20, fill="x")

        self.env_label = ctk.CTkLabel(self.general_frame, text="Environment:", font=self.bold_font)
        self.env_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.env_menu = ctk.CTkOptionMenu(self.general_frame, values=self._list_environments(), font=self.huge_font, width=150)
        self.env_menu.grid(row=0, column=1, padx=10, pady=10)

        self.ctx_label = ctk.CTkLabel(self.general_frame, text="Ctx Frames:", font=self.bold_font)
        self.ctx_label.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.ctx_entry = ctk.CTkEntry(self.general_frame, width=80, font=self.huge_font)
        self.ctx_entry.grid(row=0, column=3, padx=10, pady=10)

        self.seed_label = ctk.CTkLabel(self.general_frame, text="Seed:", font=self.bold_font)
        self.seed_label.grid(row=0, column=4, padx=10, pady=10, sticky="e")
        self.seed_entry = ctk.CTkEntry(self.general_frame, width=150, font=self.huge_font, placeholder_text="(blank = random)")
        self.seed_entry.grid(row=0, column=5, padx=10, pady=10, sticky="w")

        self.compile_label = ctk.CTkLabel(self.general_frame, text="Compile:", font=self.bold_font)
        self.compile_label.grid(row=0, column=6, padx=10, pady=10, sticky="e")
        self.compile_menu = ctk.CTkOptionMenu(self.general_frame, values=["off", "on"], font=self.huge_font, width=100)
        self.compile_menu.grid(row=0, column=7, padx=10, pady=10, sticky="w")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(pady=5, padx=20, fill="x")
        self.tabview.add("VAE")
        self.tabview.add("Dynamics")
        self.tabview.add("Evaluation")

        vae_tab = self.tabview.tab("VAE")
        dyn_tab = self.tabview.tab("Dynamics")
        eval_tab = self.tabview.tab("Evaluation")

        self.ae_frame = ctk.CTkFrame(vae_tab)
        self.ae_frame.pack(pady=10, padx=10, fill="x")

        self.ae_section_label = ctk.CTkLabel(self.ae_frame, text="Autoencoder (VAE)", font=self.bold_font)
        self.ae_section_label.grid(row=0, column=0, columnspan=8, padx=10, pady=(10, 0), sticky="w")

        self.ae_lr_label = ctk.CTkLabel(self.ae_frame, text="Learn Rate:", font=self.bold_font)
        self.ae_lr_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.ae_lr_entry = ctk.CTkEntry(self.ae_frame, width=150, font=self.huge_font)
        self.ae_lr_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.ae_wd_label = ctk.CTkLabel(self.ae_frame, text="Weight Decay:", font=self.bold_font)
        self.ae_wd_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.ae_wd_entry = ctk.CTkEntry(self.ae_frame, width=150, font=self.huge_font)
        self.ae_wd_entry.grid(row=1, column=3, padx=10, pady=10, sticky="w")

        self.ae_batch_label = ctk.CTkLabel(self.ae_frame, text="Batch:", font=self.bold_font)
        self.ae_batch_label.grid(row=1, column=4, padx=10, pady=10, sticky="e")
        self.ae_batch_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.ae_batch_entry.grid(row=1, column=5, padx=10, pady=10, sticky="w")

        self.ae_epochs_label = ctk.CTkLabel(self.ae_frame, text="Epochs:", font=self.bold_font)
        self.ae_epochs_label.grid(row=1, column=6, padx=10, pady=10, sticky="e")
        self.ae_epochs_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.ae_epochs_entry.grid(row=1, column=7, padx=10, pady=10, sticky="w")

        self.ae_kl_label = ctk.CTkLabel(self.ae_frame, text="KL Weight (beta):", font=self.bold_font)
        self.ae_kl_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.ae_kl_entry = ctk.CTkEntry(self.ae_frame, width=150, font=self.huge_font)
        self.ae_kl_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.latent_grid_label = ctk.CTkLabel(self.ae_frame, text="Latent Grid:", font=self.bold_font)
        self.latent_grid_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.latent_grid_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.latent_grid_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        self.lpips_label = ctk.CTkLabel(self.ae_frame, text="LPIPS Weight:", font=self.bold_font)
        self.lpips_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.lpips_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.lpips_entry.grid(row=2, column=5, padx=10, pady=10, sticky="w")

        self.latent_ch_label = ctk.CTkLabel(self.ae_frame, text="Latent Ch:", font=self.bold_font)
        self.latent_ch_label.grid(row=2, column=6, padx=10, pady=10, sticky="e")
        self.latent_ch_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.latent_ch_entry.grid(row=2, column=7, padx=10, pady=10, sticky="w")

        self.ae_clip_label = ctk.CTkLabel(self.ae_frame, text="Grad Clip:", font=self.bold_font)
        self.ae_clip_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.ae_clip_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.ae_clip_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        self.dec_res_label = ctk.CTkLabel(self.ae_frame, text="Dec ResBlocks:", font=self.bold_font)
        self.dec_res_label.grid(row=3, column=2, padx=10, pady=10, sticky="e")
        self.dec_res_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.dec_res_entry.grid(row=3, column=3, padx=10, pady=10, sticky="w")

        self.enc_res_label = ctk.CTkLabel(self.ae_frame, text="Enc ResBlocks:", font=self.bold_font)
        self.enc_res_label.grid(row=3, column=4, padx=10, pady=10, sticky="e")
        self.enc_res_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.enc_res_entry.grid(row=3, column=5, padx=10, pady=10, sticky="w")

        self.ae_lpips_net_label = ctk.CTkLabel(self.ae_frame, text="LPIPS Net:", font=self.bold_font)
        self.ae_lpips_net_label.grid(row=3, column=6, padx=10, pady=10, sticky="e")
        self.ae_lpips_net_menu = ctk.CTkOptionMenu(self.ae_frame, values=["alex", "vgg"], font=self.huge_font, width=100)
        self.ae_lpips_net_menu.grid(row=3, column=7, padx=10, pady=10, sticky="w")

        self.ae_base_ch_label = ctk.CTkLabel(self.ae_frame, text="Base Ch:", font=self.bold_font)
        self.ae_base_ch_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.ae_base_ch_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.ae_base_ch_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        self.ae_block_label = ctk.CTkLabel(self.ae_frame, text="Block:", font=self.bold_font)
        self.ae_block_label.grid(row=4, column=2, padx=10, pady=10, sticky="e")
        self.ae_block_menu = ctk.CTkOptionMenu(self.ae_frame, values=["res", "convnext", "mobile"], font=self.huge_font, width=120)
        self.ae_block_menu.grid(row=4, column=3, padx=10, pady=10, sticky="w")

        self.ae_label = ctk.CTkLabel(self.ae_frame, text="Reuse AE:", font=self.bold_font)
        self.ae_label.grid(row=5, column=0, padx=10, pady=10, sticky="e")
        self.ae_entry = ctk.CTkEntry(self.ae_frame, width=400, font=self.huge_font, placeholder_text="path to autoencoder.pth (blank = train new)")
        self.ae_entry.grid(row=5, column=1, columnspan=6, padx=10, pady=10, sticky="ew")
        self.ae_browse_button = ctk.CTkButton(self.ae_frame, text="Browse", width=80, font=self.bold_font, command=self._browse_ae)
        self.ae_browse_button.grid(row=5, column=7, padx=10, pady=10)

        self.dyn_frame = ctk.CTkFrame(dyn_tab)
        self.dyn_frame.pack(pady=10, padx=10, fill="x")

        self.dyn_section_label = ctk.CTkLabel(self.dyn_frame, text="Flow Matching (DiT)", font=self.bold_font)
        self.dyn_section_label.grid(row=0, column=0, columnspan=8, padx=10, pady=(10, 0), sticky="w")

        # --- Iteration 1 (first leg):  Epochs | Learn Rate | Min LR | LR Curve ---
        self.dyn_epochs_label = ctk.CTkLabel(self.dyn_frame, text="Epochs:", font=self.bold_font)
        self.dyn_epochs_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.dyn_epochs_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dyn_epochs_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.dyn_lr_label = ctk.CTkLabel(self.dyn_frame, text="Learn Rate:", font=self.bold_font)
        self.dyn_lr_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.dyn_lr_entry = ctk.CTkEntry(self.dyn_frame, width=150, font=self.huge_font)
        self.dyn_lr_entry.grid(row=1, column=3, padx=10, pady=10, sticky="w")

        self.dit_min_lr_label = ctk.CTkLabel(self.dyn_frame, text="Min LR:", font=self.bold_font)
        self.dit_min_lr_label.grid(row=1, column=4, padx=10, pady=10, sticky="e")
        self.dit_min_lr_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_min_lr_entry.grid(row=1, column=5, padx=10, pady=10, sticky="w")

        self.lr_sched_label = ctk.CTkLabel(self.dyn_frame, text="LR Curve:", font=self.bold_font)
        self.lr_sched_label.grid(row=1, column=6, padx=10, pady=10, sticky="e")
        self.lr_sched_menu = ctk.CTkOptionMenu(self.dyn_frame, values=["cosine", "linear"], font=self.huge_font, width=110)
        self.lr_sched_menu.grid(row=1, column=7, padx=10, pady=10, sticky="w")

        # --- Iteration 2 (warm restart; Epochs 2 = 0 disables it): same columns as iter 1 ---
        self.dyn_epochs2_label = ctk.CTkLabel(self.dyn_frame, text="Epochs 2:", font=self.bold_font)
        self.dyn_epochs2_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.dyn_epochs2_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dyn_epochs2_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.dyn_lr2_label = ctk.CTkLabel(self.dyn_frame, text="LR 2:", font=self.bold_font)
        self.dyn_lr2_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.dyn_lr2_entry = ctk.CTkEntry(self.dyn_frame, width=150, font=self.huge_font)
        self.dyn_lr2_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        self.dit_min_lr2_label = ctk.CTkLabel(self.dyn_frame, text="Min LR 2:", font=self.bold_font)
        self.dit_min_lr2_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.dit_min_lr2_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_min_lr2_entry.grid(row=2, column=5, padx=10, pady=10, sticky="w")

        self.lr_sched_2_label = ctk.CTkLabel(self.dyn_frame, text="LR Curve 2:", font=self.bold_font)
        self.lr_sched_2_label.grid(row=2, column=6, padx=10, pady=10, sticky="e")
        self.lr_sched_2_menu = ctk.CTkOptionMenu(self.dyn_frame, values=["cosine", "linear"], font=self.huge_font, width=110)
        self.lr_sched_2_menu.grid(row=2, column=7, padx=10, pady=10, sticky="w")

        # --- Shared schedule / optimizer knobs ---
        self.dit_warmup_label = ctk.CTkLabel(self.dyn_frame, text="Warmup Frac:", font=self.bold_font)
        self.dit_warmup_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.dit_warmup_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_warmup_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        self.dyn_batch_label = ctk.CTkLabel(self.dyn_frame, text="Batch:", font=self.bold_font)
        self.dyn_batch_label.grid(row=3, column=2, padx=10, pady=10, sticky="e")
        self.dyn_batch_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dyn_batch_entry.grid(row=3, column=3, padx=10, pady=10, sticky="w")

        self.dyn_wd_label = ctk.CTkLabel(self.dyn_frame, text="Weight Decay:", font=self.bold_font)
        self.dyn_wd_label.grid(row=3, column=4, padx=10, pady=10, sticky="e")
        self.dyn_wd_entry = ctk.CTkEntry(self.dyn_frame, width=150, font=self.huge_font)
        self.dyn_wd_entry.grid(row=3, column=5, padx=10, pady=10, sticky="w")

        self.dit_clip_label = ctk.CTkLabel(self.dyn_frame, text="Grad Clip:", font=self.bold_font)
        self.dit_clip_label.grid(row=3, column=6, padx=10, pady=10, sticky="e")
        self.dit_clip_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_clip_entry.grid(row=3, column=7, padx=10, pady=10, sticky="w")

        # --- Architecture ---
        self.dit_dmodel_label = ctk.CTkLabel(self.dyn_frame, text="DiT Width:", font=self.bold_font)
        self.dit_dmodel_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.dit_dmodel_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_dmodel_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        self.dit_layers_label = ctk.CTkLabel(self.dyn_frame, text="DiT Layers:", font=self.bold_font)
        self.dit_layers_label.grid(row=4, column=2, padx=10, pady=10, sticky="e")
        self.dit_layers_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_layers_entry.grid(row=4, column=3, padx=10, pady=10, sticky="w")

        self.dit_heads_label = ctk.CTkLabel(self.dyn_frame, text="DiT Heads:", font=self.bold_font)
        self.dit_heads_label.grid(row=4, column=4, padx=10, pady=10, sticky="e")
        self.dit_heads_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_heads_entry.grid(row=4, column=5, padx=10, pady=10, sticky="w")

        self.chunk_len_label = ctk.CTkLabel(self.dyn_frame, text="Chunk Len:", font=self.bold_font)
        self.chunk_len_label.grid(row=4, column=6, padx=10, pady=10, sticky="e")
        self.chunk_len_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.chunk_len_entry.grid(row=4, column=7, padx=10, pady=10, sticky="w")

        # --- Objective / regularization ---
        self.ctx_noise_label = ctk.CTkLabel(self.dyn_frame, text="Ctx Noise:", font=self.bold_font)
        self.ctx_noise_label.grid(row=5, column=0, padx=10, pady=10, sticky="e")
        self.ctx_noise_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.ctx_noise_entry.grid(row=5, column=1, padx=10, pady=10, sticky="w")

        self.loss_label = ctk.CTkLabel(self.dyn_frame, text="Loss:", font=self.bold_font)
        self.loss_label.grid(row=5, column=2, padx=10, pady=10, sticky="e")
        self.loss_menu = ctk.CTkOptionMenu(self.dyn_frame, values=["mse", "huber"], font=self.huge_font, width=100)
        self.loss_menu.grid(row=5, column=3, padx=10, pady=10, sticky="w")

        self.huberc_label = ctk.CTkLabel(self.dyn_frame, text="Huber C:", font=self.bold_font)
        self.huberc_label.grid(row=5, column=4, padx=10, pady=10, sticky="e")
        self.huberc_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.huberc_entry.grid(row=5, column=5, padx=10, pady=10, sticky="w")

        self.ema_decay_label = ctk.CTkLabel(self.dyn_frame, text="EMA Decay:", font=self.bold_font)
        self.ema_decay_label.grid(row=5, column=6, padx=10, pady=10, sticky="e")
        self.ema_decay_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.ema_decay_entry.grid(row=5, column=7, padx=10, pady=10, sticky="w")

        # --- Continuation / reuse ---
        self.dit_continue_label = ctk.CTkLabel(self.dyn_frame, text="Continue:", font=self.bold_font)
        self.dit_continue_label.grid(row=6, column=0, padx=10, pady=(10, 0), sticky="e")
        self.dit_continue_switch = ctk.CTkSwitch(self.dyn_frame, text="", onvalue="on", offvalue="off", width=48)
        self.dit_continue_switch.grid(row=6, column=1, padx=10, pady=(10, 0), sticky="w")

        self.dit_label = ctk.CTkLabel(self.dyn_frame, text="Reuse DiT:", font=self.bold_font)
        self.dit_label.grid(row=7, column=0, padx=10, pady=10, sticky="e")
        self.dit_entry = ctk.CTkEntry(self.dyn_frame, width=400, font=self.huge_font, placeholder_text="path to dit.pth (blank = train new)")
        self.dit_entry.grid(row=7, column=1, columnspan=6, padx=10, pady=10, sticky="ew")
        self.dit_browse_button = ctk.CTkButton(self.dyn_frame, text="Browse", width=80, font=self.bold_font, command=self._browse_dit)
        self.dit_browse_button.grid(row=7, column=7, padx=10, pady=10)

        self.eval_frame = ctk.CTkFrame(eval_tab)
        self.eval_frame.pack(pady=10, padx=10, fill="x")

        self.eval_section_label = ctk.CTkLabel(self.eval_frame, text="Evaluation / Inference", font=self.bold_font)
        self.eval_section_label.grid(row=0, column=0, columnspan=8, padx=10, pady=(10, 0), sticky="w")

        self.infsteps_label = ctk.CTkLabel(self.eval_frame, text="Infer Steps:", font=self.bold_font)
        self.infsteps_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.infsteps_entry = ctk.CTkEntry(self.eval_frame, width=80, font=self.huge_font)
        self.infsteps_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.eval_horizon_label = ctk.CTkLabel(self.eval_frame, text="Eval Horizon:", font=self.bold_font)
        self.eval_horizon_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.eval_horizon_entry = ctk.CTkEntry(self.eval_frame, width=80, font=self.huge_font)
        self.eval_horizon_entry.grid(row=1, column=3, padx=10, pady=10, sticky="w")

        self.eval_batches_label = ctk.CTkLabel(self.eval_frame, text="Eval Batches:", font=self.bold_font)
        self.eval_batches_label.grid(row=1, column=4, padx=10, pady=10, sticky="e")
        self.eval_batches_entry = ctk.CTkEntry(self.eval_frame, width=80, font=self.huge_font)
        self.eval_batches_entry.grid(row=1, column=5, padx=10, pady=10, sticky="w")

        self.best_of_n_label = ctk.CTkLabel(self.eval_frame, text="Best-of-N:", font=self.bold_font)
        self.best_of_n_label.grid(row=1, column=6, padx=10, pady=10, sticky="e")
        self.best_of_n_entry = ctk.CTkEntry(self.eval_frame, width=80, font=self.huge_font)
        self.best_of_n_entry.grid(row=1, column=7, padx=10, pady=10, sticky="w")

        self.eval_png_label = ctk.CTkLabel(self.eval_frame, text="Eval PNGs:", font=self.bold_font)
        self.eval_png_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.eval_png_entry = ctk.CTkEntry(self.eval_frame, width=80, font=self.huge_font)
        self.eval_png_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.eval_gifs_label = ctk.CTkLabel(self.eval_frame, text="Rollout GIFs:", font=self.bold_font)
        self.eval_gifs_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.eval_gifs_entry = ctk.CTkEntry(self.eval_frame, width=80, font=self.huge_font)
        self.eval_gifs_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        self.eval_gif_len_label = ctk.CTkLabel(self.eval_frame, text="GIF Length:", font=self.bold_font)
        self.eval_gif_len_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.eval_gif_len_entry = ctk.CTkEntry(self.eval_frame, width=80, font=self.huge_font)
        self.eval_gif_len_entry.grid(row=2, column=5, padx=10, pady=10, sticky="w")

        self.coll_eval_label = ctk.CTkLabel(self.eval_frame, text="CollEval:", font=self.bold_font)
        self.coll_eval_label.grid(row=2, column=6, padx=10, pady=10, sticky="e")
        self.coll_eval_menu = ctk.CTkOptionMenu(self.eval_frame, values=["on", "off"], font=self.huge_font, width=100)
        self.coll_eval_menu.grid(row=2, column=7, padx=10, pady=10, sticky="w")

        self.actions_frame = ctk.CTkFrame(self)
        self.actions_frame.pack(pady=(0, 5), padx=20, fill="x")
        self.actions_frame.grid_columnconfigure(0, weight=1)
        self.actions_frame.grid_columnconfigure(3, weight=1)

        self.save_button = ctk.CTkButton(self.actions_frame, text="Save Config", font=self.bold_font, command=self.save_settings)
        self.save_button.grid(row=0, column=1, padx=20, pady=10)

        self.start_button = ctk.CTkButton(self.actions_frame, text="START TRAINING", font=self.bold_font, fg_color="green", hover_color="darkgreen", command=self.start_training_thread)
        self.start_button.grid(row=0, column=2, padx=20, pady=10)

        self.log_textbox = ctk.CTkTextbox(self, width=1100, height=380, font=self.huge_font)
        self.log_textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.current_log_file = None
        self.is_training = False
        self.last_read_pos = 0

        self.load_settings()

    def _list_environments(self):
        data_root = "data"
        envs = []
        if os.path.isdir(data_root):
            envs = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))
        return envs or ["bouncing"]

    def _set_env(self, env):
        vals = self._list_environments()
        if env not in vals:
            vals = vals + [env]
        self.env_menu.configure(values=vals)
        self.env_menu.set(env)

    def _browse_ae(self):
        path = filedialog.askopenfilename(title="Select autoencoder.pth",
                                          filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            self.ae_entry.delete(0, "end")
            self.ae_entry.insert(0, path)

    def _browse_dit(self):
        path = filedialog.askopenfilename(title="Select dit.pth",
                                          filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            self.dit_entry.delete(0, "end")
            self.dit_entry.insert(0, path)

    @staticmethod
    def _set_entry(entry, value):
        entry.delete(0, "end")
        if value:
            entry.insert(0, value)
        else:
            entry._activate_placeholder()

    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    c = json.load(f)
                self._set_env(c.get("env_name", "bouncing"))
                self.ctx_entry.delete(0, "end"); self.ctx_entry.insert(0, str(c.get("context_len", 5)))
                self.ae_lr_entry.delete(0, "end"); self.ae_lr_entry.insert(0, str(c.get("ae_learning_rate", 0.0005)))
                self.dyn_lr_entry.delete(0, "end"); self.dyn_lr_entry.insert(0, str(c.get("dyn_learning_rate", 0.0005)))
                self.dit_min_lr_entry.delete(0, "end"); self.dit_min_lr_entry.insert(0, str(c.get("dit_min_lr", 1e-06)))
                self.dit_warmup_entry.delete(0, "end"); self.dit_warmup_entry.insert(0, str(c.get("dit_warmup_frac", 0.05)))
                self.dyn_epochs2_entry.delete(0, "end"); self.dyn_epochs2_entry.insert(0, str(c.get("dyn_epochs_2", 0)))
                self.dyn_lr2_entry.delete(0, "end"); self.dyn_lr2_entry.insert(0, str(c.get("dyn_learning_rate_2", 0.0002)))
                self.dit_min_lr2_entry.delete(0, "end"); self.dit_min_lr2_entry.insert(0, str(c.get("dit_min_lr_2", 2e-06)))
                self.ae_wd_entry.delete(0, "end"); self.ae_wd_entry.insert(0, str(c.get("ae_weight_decay", 0.001)))
                self.dyn_wd_entry.delete(0, "end"); self.dyn_wd_entry.insert(0, str(c.get("dyn_weight_decay", 0.001)))
                self.ae_batch_entry.delete(0, "end"); self.ae_batch_entry.insert(0, str(c.get("ae_batch_size", 32)))
                self.dyn_batch_entry.delete(0, "end"); self.dyn_batch_entry.insert(0, str(c.get("dyn_batch_size", 32)))
                self.ae_epochs_entry.delete(0, "end"); self.ae_epochs_entry.insert(0, str(c.get("ae_epochs", 10)))
                self.ae_kl_entry.delete(0, "end"); self.ae_kl_entry.insert(0, str(c.get("ae_kl_weight", 0.005)))
                self.lpips_entry.delete(0, "end"); self.lpips_entry.insert(0, str(c.get("ae_lpips_weight", 0.0)))
                self.ae_lpips_net_menu.set(c.get("ae_lpips_net", "alex"))
                self.latent_grid_entry.delete(0, "end"); self.latent_grid_entry.insert(0, str(c.get("latent_grid", 8)))
                self.latent_ch_entry.delete(0, "end"); self.latent_ch_entry.insert(0, str(c.get("latent_ch", 32)))
                self.dit_dmodel_entry.delete(0, "end"); self.dit_dmodel_entry.insert(0, str(c.get("dit_d_model", 256)))
                self.dit_layers_entry.delete(0, "end"); self.dit_layers_entry.insert(0, str(c.get("dit_n_layers", 6)))
                self.dit_heads_entry.delete(0, "end"); self.dit_heads_entry.insert(0, str(c.get("dit_n_heads", 8)))
                self.infsteps_entry.delete(0, "end"); self.infsteps_entry.insert(0, str(c.get("inference_steps", 10)))
                self.dyn_epochs_entry.delete(0, "end"); self.dyn_epochs_entry.insert(0, str(c.get("dyn_epochs", 30)))
                self.eval_horizon_entry.delete(0, "end"); self.eval_horizon_entry.insert(0, str(c.get("eval_horizon", 50)))
                self.eval_batches_entry.delete(0, "end"); self.eval_batches_entry.insert(0, str(c.get("eval_max_batches", 24)))
                self.chunk_len_entry.delete(0, "end"); self.chunk_len_entry.insert(0, str(c.get("chunk_len", 5)))
                self.best_of_n_entry.delete(0, "end"); self.best_of_n_entry.insert(0, str(c.get("eval_best_of_n", 1)))
                self.eval_png_entry.delete(0, "end"); self.eval_png_entry.insert(0, str(c.get("eval_n_pngs", 1)))
                self.eval_gifs_entry.delete(0, "end"); self.eval_gifs_entry.insert(0, str(c.get("eval_n_gifs", 2)))
                self.eval_gif_len_entry.delete(0, "end"); self.eval_gif_len_entry.insert(0, str(c.get("eval_gif_len", 40)))
                self.ema_decay_entry.delete(0, "end"); self.ema_decay_entry.insert(0, str(c.get("dit_ema_decay", 0.999)))
                self.ctx_noise_entry.delete(0, "end"); self.ctx_noise_entry.insert(0, str(c.get("dit_context_noise", 0.0)))
                _cm = str(c.get("compile", c.get("dit_compile", "off")))
                self.compile_menu.set("off" if _cm in ("off", "") else "on")
                self.coll_eval_menu.set(c.get("coll_eval", "on"))
                self.loss_menu.set(c.get("dit_loss", "mse"))
                self.huberc_entry.delete(0, "end"); self.huberc_entry.insert(0, str(c.get("dit_huber_c", 1.0)))
                self.ae_clip_entry.delete(0, "end"); self.ae_clip_entry.insert(0, str(c.get("ae_grad_clip", 10.0)))
                self.dec_res_entry.delete(0, "end"); self.dec_res_entry.insert(0, str(c.get("vae_dec_res_blocks", 1)))
                self.enc_res_entry.delete(0, "end"); self.enc_res_entry.insert(0, str(c.get("vae_enc_res_blocks", 1)))
                self.ae_base_ch_entry.delete(0, "end"); self.ae_base_ch_entry.insert(0, str(c.get("vae_base_ch", 64)))
                self.ae_block_menu.set(c.get("vae_block", "res"))
                self.dit_clip_entry.delete(0, "end"); self.dit_clip_entry.insert(0, str(c.get("dit_grad_clip", 3.0)))
                self.seed_entry.delete(0, "end"); self.seed_entry.insert(0, str(c.get("seed", "42")))
                self._set_entry(self.ae_entry, str(c.get("ae_checkpoint", "")))
                self._set_entry(self.dit_entry, str(c.get("dit_checkpoint", "")))
                self.lr_sched_menu.set(c.get("dit_lr_schedule", "cosine"))
                self.lr_sched_2_menu.set(c.get("dit_lr_schedule_2", "cosine"))
                if str(c.get("dit_continue", "off")).lower() in ("on", "true", "1"):
                    self.dit_continue_switch.select()
                else:
                    self.dit_continue_switch.deselect()
                self.log_textbox.insert("end", f"[System] Settings loaded from {CONFIG_FILE}\n")
            except Exception as e:
                self.log_textbox.insert("end", f"[Error] Load config failed: {e}\n")
        else:
            self._set_env("bouncing")
            self.ctx_entry.insert(0, "5")
            self.ae_lr_entry.insert(0, "0.0005")
            self.dyn_lr_entry.insert(0, "0.0003")
            self.ae_wd_entry.insert(0, "0.01")
            self.dyn_wd_entry.insert(0, "0.0001")
            self.ae_batch_entry.insert(0, "32")
            self.dyn_batch_entry.insert(0, "64")
            self.ae_epochs_entry.insert(0, "20")
            self.ae_kl_entry.insert(0, "0.005")
            self.lpips_entry.insert(0, "1.0")
            self.ae_lpips_net_menu.set("alex")
            self.latent_grid_entry.insert(0, "8")
            self.latent_ch_entry.insert(0, "32")
            self.dit_dmodel_entry.insert(0, "256")
            self.dit_layers_entry.insert(0, "6")
            self.dit_heads_entry.insert(0, "8")
            self.infsteps_entry.insert(0, "10")
            self.dyn_epochs_entry.insert(0, "30")
            self.eval_horizon_entry.insert(0, "50")
            self.eval_batches_entry.insert(0, "24")
            self.chunk_len_entry.insert(0, "5")
            self.best_of_n_entry.insert(0, "1")
            self.eval_png_entry.insert(0, "1")
            self.eval_gifs_entry.insert(0, "2")
            self.eval_gif_len_entry.insert(0, "40")
            self.ema_decay_entry.insert(0, "0.999")
            self.ctx_noise_entry.insert(0, "0.0")
            self.compile_menu.set("off")
            self.loss_menu.set("mse")
            self.lr_sched_menu.set("cosine")
            self.lr_sched_2_menu.set("cosine")
            self.dit_continue_switch.deselect()
            self.huberc_entry.insert(0, "1.0")
            self.ae_clip_entry.insert(0, "10.0")
            self.dit_clip_entry.insert(0, "3.0")
            self.dec_res_entry.insert(0, "1")
            self.enc_res_entry.insert(0, "1")
            self.ae_base_ch_entry.insert(0, "64")
            self.ae_block_menu.set("res")

    def save_settings(self):
        try:
            config = {
                "env_name": self.env_menu.get(),
                "context_len": int(self.ctx_entry.get()),
                "ae_learning_rate": float(self.ae_lr_entry.get()),
                "dyn_learning_rate": float(self.dyn_lr_entry.get()),
                "dit_min_lr": float(self.dit_min_lr_entry.get()),
                "dit_warmup_frac": float(self.dit_warmup_entry.get()),
                "dyn_epochs_2": int(self.dyn_epochs2_entry.get()),
                "dyn_learning_rate_2": float(self.dyn_lr2_entry.get()),
                "dit_min_lr_2": float(self.dit_min_lr2_entry.get()),
                "ae_weight_decay": float(self.ae_wd_entry.get()),
                "dyn_weight_decay": float(self.dyn_wd_entry.get()),
                "ae_batch_size": int(self.ae_batch_entry.get()),
                "dyn_batch_size": int(self.dyn_batch_entry.get()),
                "ae_epochs": int(self.ae_epochs_entry.get()),
                "ae_kl_weight": float(self.ae_kl_entry.get()),
                "ae_lpips_weight": float(self.lpips_entry.get()),
                "ae_lpips_net": self.ae_lpips_net_menu.get(),
                "latent_grid": int(self.latent_grid_entry.get()),
                "latent_ch": int(self.latent_ch_entry.get()),
                "dit_d_model": int(self.dit_dmodel_entry.get()),
                "dit_n_layers": int(self.dit_layers_entry.get()),
                "dit_n_heads": int(self.dit_heads_entry.get()),
                "inference_steps": int(self.infsteps_entry.get()),
                "dyn_epochs": int(self.dyn_epochs_entry.get()),
                "eval_horizon": int(self.eval_horizon_entry.get()),
                "eval_max_batches": int(self.eval_batches_entry.get()),
                "chunk_len": int(self.chunk_len_entry.get()),
                "eval_best_of_n": int(self.best_of_n_entry.get()),
                "eval_n_pngs": int(self.eval_png_entry.get()),
                "eval_n_gifs": int(self.eval_gifs_entry.get()),
                "eval_gif_len": int(self.eval_gif_len_entry.get()),
                "coll_eval": self.coll_eval_menu.get(),
                "dit_ema_decay": float(self.ema_decay_entry.get()),
                "dit_context_noise": float(self.ctx_noise_entry.get()),
                "compile": self.compile_menu.get(),
                "dit_loss": self.loss_menu.get(),
                "dit_huber_c": float(self.huberc_entry.get()),
                "ae_grad_clip": float(self.ae_clip_entry.get()),
                "dit_grad_clip": float(self.dit_clip_entry.get()),
                "vae_dec_res_blocks": int(self.dec_res_entry.get()),
                "vae_enc_res_blocks": int(self.enc_res_entry.get()),
                "vae_base_ch": int(self.ae_base_ch_entry.get()),
                "vae_block": self.ae_block_menu.get(),
                "seed": self.seed_entry.get().strip(),
                "ae_checkpoint": self.ae_entry.get().strip(),
                "dit_checkpoint": self.dit_entry.get().strip(),
                "dit_lr_schedule": self.lr_sched_menu.get(),
                "dit_lr_schedule_2": self.lr_sched_2_menu.get(),
                "dit_continue": self.dit_continue_switch.get(),
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            self.log_textbox.insert("end", f"[System] Settings saved to {CONFIG_FILE}!\n")
            self.log_textbox.see("end")
            return config
        except ValueError:
            self.log_textbox.insert("end", "[Error] All numerical fields must contain valid numbers!\n")
            return None

    def start_training_thread(self):
        if self.is_training: return
        config = self.save_settings()
        if not config: return
            
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.insert("end", "[System] Starting Latent Flow Matching run...\n")
        self.is_training = True
        self.start_button.configure(state="disabled", text="Training...")

        thread = threading.Thread(target=self._run_training, args=(config,), daemon=True)
        thread.start()
        self.after(1000, self.poll_log_file) 

    def _run_training(self, c):
        run_training_pipeline(
            data_dir=f"data/{c['env_name']}", env_name=c['env_name'],
            context_len=c['context_len'], ae_batch_size=c['ae_batch_size'], dyn_batch_size=c['dyn_batch_size'],
            ae_epochs=c['ae_epochs'], dyn_epochs=c['dyn_epochs'],
            ae_learning_rate=c['ae_learning_rate'], ae_weight_decay=c['ae_weight_decay'],
            ae_kl_weight=c.get('ae_kl_weight', 0.005), ae_lpips_weight=c.get('ae_lpips_weight', 0.0),
            ae_lpips_net=c.get('ae_lpips_net', 'alex'),
            dyn_learning_rate=c['dyn_learning_rate'], dit_min_lr=c.get('dit_min_lr', 1e-6),
            dit_warmup_frac=c.get('dit_warmup_frac', 0.05),
            dit_lr_schedule=c.get('dit_lr_schedule', "cosine"),
            dit_lr_schedule_2=c.get('dit_lr_schedule_2', "cosine"),
            dyn_epochs_2=c.get('dyn_epochs_2', 0), dyn_learning_rate_2=c.get('dyn_learning_rate_2', 2e-4),
            dit_min_lr_2=c.get('dit_min_lr_2', 2e-6),
            dyn_weight_decay=c['dyn_weight_decay'],
            eval_horizon=c.get('eval_horizon', 50), eval_max_batches=c.get('eval_max_batches', 24),
            seed=(c.get('seed') or None), ae_checkpoint=c.get('ae_checkpoint', ""),
            dit_checkpoint=c.get('dit_checkpoint', ""),
            dit_continue=(str(c.get('dit_continue', "off")).lower() in ("on", "true", "1")),
            latent_grid=c.get('latent_grid', 8), latent_ch=c.get('latent_ch', 32),
            vae_enc_res_blocks=c.get('vae_enc_res_blocks', 1),
            vae_base_ch=c.get('vae_base_ch', 64),
            vae_block=c.get('vae_block', 'res'),
            vae_dec_res_blocks=c.get('vae_dec_res_blocks', 1),
            chunk_len=c.get('chunk_len', 5), eval_best_of_n=c.get('eval_best_of_n', 1),
            eval_n_pngs=c.get('eval_n_pngs', 1), eval_n_gifs=c.get('eval_n_gifs', 2),
            eval_gif_len=c.get('eval_gif_len', 40),
            coll_eval=c.get('coll_eval', "on"),
            dit_ema_decay=c.get('dit_ema_decay', 0.999),
            dit_context_noise=c.get('dit_context_noise', 0.0),
            compile_mode=c.get('compile', "off"),
            dit_loss=c.get('dit_loss', "mse"), dit_huber_c=c.get('dit_huber_c', 1.0),
            ae_grad_clip=c.get('ae_grad_clip', 10.0), dit_grad_clip=c.get('dit_grad_clip', 3.0),
            dit_d_model=c.get('dit_d_model', 256), dit_n_layers=c.get('dit_n_layers', 6),
            dit_n_heads=c.get('dit_n_heads', 8), inference_steps=c.get('inference_steps', 10)
        )
        self.is_training = False
        self.after(0, lambda: self.start_button.configure(state="normal", text="START TRAINING"))

    def poll_log_file(self):
        if self.current_log_file is None:
            runs_dir = os.path.join(os.getcwd(), "runs")
            if os.path.exists(runs_dir):
                folders = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)]
                if folders:
                    newest_folder = max(folders, key=os.path.getmtime)
                    possible_log = os.path.join(newest_folder, "log.txt")
                    if os.path.exists(possible_log):
                        self.current_log_file = possible_log
                        self.last_read_pos = 0

        if self.current_log_file and os.path.exists(self.current_log_file):
            with open(self.current_log_file, 'r', encoding='utf-8') as f:
                f.seek(self.last_read_pos)
                new_text = f.read()
                self.last_read_pos = f.tell()
                if new_text:
                    self.log_textbox.insert("end", new_text)
                    self.log_textbox.see("end")

        if self.is_training:
            self.after(500, self.poll_log_file)
        else:
            self.current_log_file = None

if __name__ == "__main__":
    app = TrainingGUI()
    app.mainloop()