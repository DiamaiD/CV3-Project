import customtkinter as ctk
import threading
import os
import json
import shutil
from tkinter import filedialog
from src.main import run_training_pipeline
from environments.env_bouncing import generate_bouncing_data

CONFIG_FILE = "configs/model_config.json"

class TrainingGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Physics Video Model Trainer")
        self.geometry("1200x850")
        
        self.huge_font = ctk.CTkFont(family="Consolas", size=18)
        self.bold_font = ctk.CTkFont(family="Consolas", size=18, weight="bold")

        os.makedirs("configs", exist_ok=True)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(pady=10, padx=20, fill="x")
        self.tabview.add("Training")
        self.tabview.add("Data")

        train_tab = self.tabview.tab("Training")

        # ===== General =====
        self.general_frame = ctk.CTkFrame(train_tab)
        self.general_frame.pack(pady=(10, 5), padx=10, fill="x")

        self.env_label = ctk.CTkLabel(self.general_frame, text="Environment:", font=self.bold_font)
        self.env_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.env_menu = ctk.CTkOptionMenu(self.general_frame, values=self._list_environments(), font=self.huge_font, width=150)
        self.env_menu.grid(row=0, column=1, padx=10, pady=10)

        self.ctx_label = ctk.CTkLabel(self.general_frame, text="Ctx Frames:", font=self.bold_font)
        self.ctx_label.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.ctx_entry = ctk.CTkEntry(self.general_frame, width=80, font=self.huge_font)
        self.ctx_entry.grid(row=0, column=3, padx=10, pady=10)

        self.seed_label = ctk.CTkLabel(self.general_frame, text="Seed:", font=self.bold_font)
        self.seed_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.seed_entry = ctk.CTkEntry(self.general_frame, width=150, font=self.huge_font, placeholder_text="(blank = random)")
        self.seed_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # Checked = keep the decoded frame + latent caches resident in GPU VRAM (fastest, no
        # per-batch host->device copies). Unchecked = cache in system RAM and stream batches to the GPU.
        self.vram_var = ctk.BooleanVar(value=False)
        self.vram_check = ctk.CTkCheckBox(self.general_frame, text="Load dataset into VRAM",
                                          font=self.bold_font, variable=self.vram_var)
        self.vram_check.grid(row=1, column=2, columnspan=3, padx=10, pady=10, sticky="w")

        # ===== Autoencoder (Phase 1: continuous VAE) =====
        self.ae_frame = ctk.CTkFrame(train_tab)
        self.ae_frame.pack(pady=5, padx=10, fill="x")

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

        # VAE latent spatial size: 8 -> 8x8, 16 -> 16x16 (motion more spatially local for the DiT,
        # at 4x token / latent-cache cost). Changing this requires retraining the VAE.
        self.latent_grid_label = ctk.CTkLabel(self.ae_frame, text="Latent Grid:", font=self.bold_font)
        self.latent_grid_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.latent_grid_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.latent_grid_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        # Perceptual (LPIPS-VGG) loss weight. >0 makes latent L2 track perceptual quality (needs
        # `pip install lpips`); 0 = pixel + KL only. Key fix for the prediction-blur ceiling.
        self.lpips_label = ctk.CTkLabel(self.ae_frame, text="LPIPS Weight:", font=self.bold_font)
        self.lpips_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.lpips_entry = ctk.CTkEntry(self.ae_frame, width=80, font=self.huge_font)
        self.lpips_entry.grid(row=2, column=5, padx=10, pady=10, sticky="w")

        self.ae_label = ctk.CTkLabel(self.ae_frame, text="Reuse AE:", font=self.bold_font)
        self.ae_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.ae_entry = ctk.CTkEntry(self.ae_frame, width=400, font=self.huge_font, placeholder_text="path to autoencoder.pth (blank = train new)")
        self.ae_entry.grid(row=3, column=1, columnspan=6, padx=10, pady=10, sticky="ew")
        self.ae_browse_button = ctk.CTkButton(self.ae_frame, text="Browse", width=80, font=self.bold_font, command=self._browse_ae)
        self.ae_browse_button.grid(row=3, column=7, padx=10, pady=10)

        # ===== Flow Matching (DiT) (Phase 2) =====
        self.dyn_frame = ctk.CTkFrame(train_tab)
        self.dyn_frame.pack(pady=5, padx=10, fill="x")

        self.dyn_section_label = ctk.CTkLabel(self.dyn_frame, text="Flow Matching (DiT)", font=self.bold_font)
        self.dyn_section_label.grid(row=0, column=0, columnspan=8, padx=10, pady=(10, 0), sticky="w")

        self.dyn_lr_label = ctk.CTkLabel(self.dyn_frame, text="Learn Rate:", font=self.bold_font)
        self.dyn_lr_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.dyn_lr_entry = ctk.CTkEntry(self.dyn_frame, width=150, font=self.huge_font)
        self.dyn_lr_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.dyn_wd_label = ctk.CTkLabel(self.dyn_frame, text="Weight Decay:", font=self.bold_font)
        self.dyn_wd_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.dyn_wd_entry = ctk.CTkEntry(self.dyn_frame, width=150, font=self.huge_font)
        self.dyn_wd_entry.grid(row=1, column=3, padx=10, pady=10, sticky="w")

        self.dyn_batch_label = ctk.CTkLabel(self.dyn_frame, text="Batch:", font=self.bold_font)
        self.dyn_batch_label.grid(row=1, column=4, padx=10, pady=10, sticky="e")
        self.dyn_batch_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dyn_batch_entry.grid(row=1, column=5, padx=10, pady=10, sticky="w")

        self.dyn_epochs_label = ctk.CTkLabel(self.dyn_frame, text="Epochs:", font=self.bold_font)
        self.dyn_epochs_label.grid(row=1, column=6, padx=10, pady=10, sticky="e")
        self.dyn_epochs_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dyn_epochs_entry.grid(row=1, column=7, padx=10, pady=10, sticky="w")

        self.dit_dmodel_label = ctk.CTkLabel(self.dyn_frame, text="DiT Width:", font=self.bold_font)
        self.dit_dmodel_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.dit_dmodel_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_dmodel_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.dit_layers_label = ctk.CTkLabel(self.dyn_frame, text="DiT Layers:", font=self.bold_font)
        self.dit_layers_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.dit_layers_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_layers_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        self.dit_heads_label = ctk.CTkLabel(self.dyn_frame, text="DiT Heads:", font=self.bold_font)
        self.dit_heads_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.dit_heads_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.dit_heads_entry.grid(row=2, column=5, padx=10, pady=10, sticky="w")

        # Euler ODE steps used to sample one frame from the flow.
        self.infsteps_label = ctk.CTkLabel(self.dyn_frame, text="Infer Steps:", font=self.bold_font)
        self.infsteps_label.grid(row=2, column=6, padx=10, pady=10, sticky="e")
        self.infsteps_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.infsteps_entry.grid(row=2, column=7, padx=10, pady=10, sticky="w")

        # Max rollout length at eval time; the report tests every horizon from 1 to this.
        self.eval_horizon_label = ctk.CTkLabel(self.dyn_frame, text="Eval Horizon:", font=self.bold_font)
        self.eval_horizon_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.eval_horizon_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.eval_horizon_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # Cap on test batches rolled out at eval (each window = horizon x infer-steps DiT forwards).
        self.eval_batches_label = ctk.CTkLabel(self.dyn_frame, text="Eval Batches:", font=self.bold_font)
        self.eval_batches_label.grid(row=3, column=2, padx=10, pady=10, sticky="e")
        self.eval_batches_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.eval_batches_entry.grid(row=3, column=3, padx=10, pady=10, sticky="w")

        # Chunk prediction: number of future frames (K) the DiT denoises jointly per call.
        self.chunk_len_label = ctk.CTkLabel(self.dyn_frame, text="Chunk Len:", font=self.bold_font)
        self.chunk_len_label.grid(row=3, column=4, padx=10, pady=10, sticky="e")
        self.chunk_len_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.chunk_len_entry.grid(row=3, column=5, padx=10, pady=10, sticky="w")

        # Best-of-N eval: sample N rollouts per window, also report the best (multiplies eval cost).
        self.best_of_n_label = ctk.CTkLabel(self.dyn_frame, text="Best-of-N:", font=self.bold_font)
        self.best_of_n_label.grid(row=3, column=6, padx=10, pady=10, sticky="e")
        self.best_of_n_entry = ctk.CTkEntry(self.dyn_frame, width=80, font=self.huge_font)
        self.best_of_n_entry.grid(row=3, column=7, padx=10, pady=10, sticky="w")

        # ===== Actions =====
        self.actions_frame = ctk.CTkFrame(train_tab)
        self.actions_frame.pack(pady=(5, 10), padx=10, fill="x")
        # Spacer columns on the outside keep the two buttons grouped together in the centre.
        self.actions_frame.grid_columnconfigure(0, weight=1)
        self.actions_frame.grid_columnconfigure(3, weight=1)

        self.save_button = ctk.CTkButton(self.actions_frame, text="Save Config", font=self.bold_font, command=self.save_settings)
        self.save_button.grid(row=0, column=1, padx=20, pady=10)

        self.start_button = ctk.CTkButton(self.actions_frame, text="START TRAINING", font=self.bold_font, fg_color="green", hover_color="darkgreen", command=self.start_training_thread)
        self.start_button.grid(row=0, column=2, padx=20, pady=10)

        # --- Data tab ---
        self.datagen_frame = ctk.CTkFrame(self.tabview.tab("Data"))
        self.datagen_frame.pack(pady=10, padx=10, fill="x")

        self.datagen_title = ctk.CTkLabel(self.datagen_frame, text="Generate a dataset (saved to data/<name>; regenerating replaces it)", font=self.bold_font)
        self.datagen_title.grid(row=0, column=0, columnspan=6, padx=10, pady=(10, 0), sticky="w")

        self.dataname_label = ctk.CTkLabel(self.datagen_frame, text="Name:", font=self.bold_font)
        self.dataname_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.dataname_entry = ctk.CTkEntry(self.datagen_frame, width=150, font=self.huge_font)
        self.dataname_entry.grid(row=1, column=1, columnspan=3, padx=10, pady=10, sticky="w")

        self.res_label = ctk.CTkLabel(self.datagen_frame, text="Resolution:", font=self.bold_font)
        self.res_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.res_entry = ctk.CTkEntry(self.datagen_frame, width=80, font=self.huge_font)
        self.res_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.balls_min_label = ctk.CTkLabel(self.datagen_frame, text="Balls Min:", font=self.bold_font)
        self.balls_min_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.balls_min_entry = ctk.CTkEntry(self.datagen_frame, width=80, font=self.huge_font)
        self.balls_min_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        self.speed_min_label = ctk.CTkLabel(self.datagen_frame, text="Speed Min:", font=self.bold_font)
        self.speed_min_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.speed_min_entry = ctk.CTkEntry(self.datagen_frame, width=80, font=self.huge_font)
        self.speed_min_entry.grid(row=2, column=5, padx=10, pady=10, sticky="w")

        self.traj_label = ctk.CTkLabel(self.datagen_frame, text="Trajectories:", font=self.bold_font)
        self.traj_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.traj_entry = ctk.CTkEntry(self.datagen_frame, width=80, font=self.huge_font)
        self.traj_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        self.balls_max_label = ctk.CTkLabel(self.datagen_frame, text="Balls Max:", font=self.bold_font)
        self.balls_max_label.grid(row=3, column=2, padx=10, pady=10, sticky="e")
        self.balls_max_entry = ctk.CTkEntry(self.datagen_frame, width=80, font=self.huge_font)
        self.balls_max_entry.grid(row=3, column=3, padx=10, pady=10, sticky="w")

        self.speed_max_label = ctk.CTkLabel(self.datagen_frame, text="Speed Max:", font=self.bold_font)
        self.speed_max_label.grid(row=3, column=4, padx=10, pady=10, sticky="e")
        self.speed_max_entry = ctk.CTkEntry(self.datagen_frame, width=80, font=self.huge_font)
        self.speed_max_entry.grid(row=3, column=5, padx=10, pady=10, sticky="w")

        self.generate_button = ctk.CTkButton(self.datagen_frame, text="Generate Data", font=self.bold_font, command=self.start_generation)
        self.generate_button.grid(row=4, column=0, columnspan=3, pady=10)
        self.datagen_status = ctk.CTkLabel(self.datagen_frame, text="idle", font=self.huge_font)
        self.datagen_status.grid(row=4, column=3, columnspan=3, pady=10)

        # --- Log Console (shared by both tabs) ---
        self.log_textbox = ctk.CTkTextbox(self, width=1100, height=380, font=self.huge_font)
        self.log_textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.current_log_file = None
        self.is_training = False
        self.is_generating = False
        self.last_read_pos = 0

        self.load_settings()

    def _list_environments(self):
        """Available environments = dataset folders under data/ (env maps to data/<env>)."""
        data_root = "data"
        envs = []
        if os.path.isdir(data_root):
            envs = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))
        return envs or ["bouncing"]

    def _set_env(self, env):
        """Select an environment, adding it to the dropdown if it isn't already listed."""
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

    def start_generation(self):
        if self.is_training or self.is_generating:
            return
        try:
            name = self.dataname_entry.get().strip()
            res = int(self.res_entry.get())
            n_traj = int(self.traj_entry.get())
            bmin, bmax = int(self.balls_min_entry.get()), int(self.balls_max_entry.get())
            smin, smax = float(self.speed_min_entry.get()), float(self.speed_max_entry.get())
        except ValueError:
            self.log_textbox.insert("end", "[Error] Data-generation fields must be valid numbers!\n")
            return
        # Guard against deleting unintended paths (name must be a plain folder name).
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            self.log_textbox.insert("end", "[Error] Dataset name must be a non-empty plain folder name.\n")
            return
        if res % 8 != 0:
            self.log_textbox.insert("end", f"[Error] Resolution must be a multiple of 8 (got {res}).\n")
            return
        if bmin < 1 or bmax < bmin:
            self.log_textbox.insert("end", "[Error] Require 1 <= Balls Min <= Balls Max.\n")
            return

        data_dir = os.path.join("data", name)
        self.is_generating = True
        self.generate_button.configure(state="disabled", text="Generating...")
        self.log_textbox.insert("end", f"[System] Generating {n_traj} trajectories ({res}x{res}) into {data_dir} ...\n")
        self.log_textbox.see("end")
        threading.Thread(target=self._run_generation,
                         args=(data_dir, n_traj, res, bmin, bmax, smin, smax), daemon=True).start()

    def _run_generation(self, data_dir, n_traj, res, bmin, bmax, smin, smax):
        def cb(done, total):
            self.after(0, lambda d=done, t=total: self.datagen_status.configure(text=f"{d}/{t}"))
        try:
            # Remove the old dataset so no stale trajectories remain.
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
                self.after(0, lambda: self.log_textbox.insert("end", f"[System] Removed existing dataset at {data_dir}.\n"))
            generate_bouncing_data(data_dir=data_dir, n_trajectories=n_traj, width=res, height=res,
                                   n_balls_min=bmin, n_balls_max=bmax, speed_min=smin, speed_max=smax,
                                   progress_cb=cb)
            self.after(0, lambda: self._on_generation_done(data_dir, n_traj))
        except Exception as e:
            self.after(0, lambda err=e: self.log_textbox.insert("end", f"[Error] Generation failed: {err}\n"))
            self.after(0, lambda: self.generate_button.configure(state="normal", text="Generate Data"))
            self.is_generating = False

    def _on_generation_done(self, data_dir, n_traj):
        self.is_generating = False
        self.generate_button.configure(state="normal", text="Generate Data")
        self.datagen_status.configure(text="Done")
        self.log_textbox.insert("end", f"[System] Generated {n_traj} trajectories in {data_dir}.\n")
        self.log_textbox.see("end")
        self._set_env(os.path.basename(data_dir))

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
                self.ae_wd_entry.delete(0, "end"); self.ae_wd_entry.insert(0, str(c.get("ae_weight_decay", 0.001)))
                self.dyn_wd_entry.delete(0, "end"); self.dyn_wd_entry.insert(0, str(c.get("dyn_weight_decay", 0.001)))
                self.ae_batch_entry.delete(0, "end"); self.ae_batch_entry.insert(0, str(c.get("ae_batch_size", 32)))
                self.dyn_batch_entry.delete(0, "end"); self.dyn_batch_entry.insert(0, str(c.get("dyn_batch_size", 32)))
                self.ae_epochs_entry.delete(0, "end"); self.ae_epochs_entry.insert(0, str(c.get("ae_epochs", 10)))
                self.ae_kl_entry.delete(0, "end"); self.ae_kl_entry.insert(0, str(c.get("ae_kl_weight", 0.005)))
                self.lpips_entry.delete(0, "end"); self.lpips_entry.insert(0, str(c.get("ae_lpips_weight", 0.0)))
                self.latent_grid_entry.delete(0, "end"); self.latent_grid_entry.insert(0, str(c.get("latent_grid", 8)))
                self.dit_dmodel_entry.delete(0, "end"); self.dit_dmodel_entry.insert(0, str(c.get("dit_d_model", 256)))
                self.dit_layers_entry.delete(0, "end"); self.dit_layers_entry.insert(0, str(c.get("dit_n_layers", 6)))
                self.dit_heads_entry.delete(0, "end"); self.dit_heads_entry.insert(0, str(c.get("dit_n_heads", 8)))
                self.infsteps_entry.delete(0, "end"); self.infsteps_entry.insert(0, str(c.get("inference_steps", 10)))
                self.dyn_epochs_entry.delete(0, "end"); self.dyn_epochs_entry.insert(0, str(c.get("dyn_epochs", 30)))
                self.eval_horizon_entry.delete(0, "end"); self.eval_horizon_entry.insert(0, str(c.get("eval_horizon", 50)))
                self.eval_batches_entry.delete(0, "end"); self.eval_batches_entry.insert(0, str(c.get("eval_max_batches", 24)))
                self.chunk_len_entry.delete(0, "end"); self.chunk_len_entry.insert(0, str(c.get("chunk_len", 5)))
                self.best_of_n_entry.delete(0, "end"); self.best_of_n_entry.insert(0, str(c.get("eval_best_of_n", 1)))
                self.seed_entry.delete(0, "end"); self.seed_entry.insert(0, str(c.get("seed", "42")))
                self._set_entry(self.ae_entry, str(c.get("ae_checkpoint", "")))
                self.vram_var.set(bool(c.get("cache_in_vram", False)))
                self.dataname_entry.delete(0, "end"); self.dataname_entry.insert(0, str(c.get("datagen_name", c.get("env_name", "bouncing"))))
                self.res_entry.delete(0, "end"); self.res_entry.insert(0, str(c.get("resolution", 64)))
                self.traj_entry.delete(0, "end"); self.traj_entry.insert(0, str(c.get("n_trajectories", 5000)))
                self.balls_min_entry.delete(0, "end"); self.balls_min_entry.insert(0, str(c.get("n_balls_min", 1)))
                self.balls_max_entry.delete(0, "end"); self.balls_max_entry.insert(0, str(c.get("n_balls_max", 5)))
                self.speed_min_entry.delete(0, "end"); self.speed_min_entry.insert(0, str(c.get("speed_min", 3.0)))
                self.speed_max_entry.delete(0, "end"); self.speed_max_entry.insert(0, str(c.get("speed_max", 8.0)))
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
            self.latent_grid_entry.insert(0, "8")
            self.dit_dmodel_entry.insert(0, "256")
            self.dit_layers_entry.insert(0, "6")
            self.dit_heads_entry.insert(0, "8")
            self.infsteps_entry.insert(0, "10")
            self.dyn_epochs_entry.insert(0, "30")
            self.eval_horizon_entry.insert(0, "50")
            self.eval_batches_entry.insert(0, "24")
            self.chunk_len_entry.insert(0, "5")
            self.best_of_n_entry.insert(0, "1")
            self.dataname_entry.insert(0, "bouncing")
            self.res_entry.insert(0, "64")
            self.traj_entry.insert(0, "5000")
            self.balls_min_entry.insert(0, "1")
            self.balls_max_entry.insert(0, "5")
            self.speed_min_entry.insert(0, "3.0")
            self.speed_max_entry.insert(0, "8.0")

    def save_settings(self):
        try:
            config = {
                "env_name": self.env_menu.get(),
                "context_len": int(self.ctx_entry.get()),
                "ae_learning_rate": float(self.ae_lr_entry.get()),
                "dyn_learning_rate": float(self.dyn_lr_entry.get()),
                "ae_weight_decay": float(self.ae_wd_entry.get()),
                "dyn_weight_decay": float(self.dyn_wd_entry.get()),
                "ae_batch_size": int(self.ae_batch_entry.get()),
                "dyn_batch_size": int(self.dyn_batch_entry.get()),
                "ae_epochs": int(self.ae_epochs_entry.get()),
                "ae_kl_weight": float(self.ae_kl_entry.get()),
                "ae_lpips_weight": float(self.lpips_entry.get()),
                "latent_grid": int(self.latent_grid_entry.get()),
                "dit_d_model": int(self.dit_dmodel_entry.get()),
                "dit_n_layers": int(self.dit_layers_entry.get()),
                "dit_n_heads": int(self.dit_heads_entry.get()),
                "inference_steps": int(self.infsteps_entry.get()),
                "dyn_epochs": int(self.dyn_epochs_entry.get()),
                "eval_horizon": int(self.eval_horizon_entry.get()),
                "eval_max_batches": int(self.eval_batches_entry.get()),
                "chunk_len": int(self.chunk_len_entry.get()),
                "eval_best_of_n": int(self.best_of_n_entry.get()),
                "seed": self.seed_entry.get().strip(),
                "ae_checkpoint": self.ae_entry.get().strip(),
                "datagen_name": self.dataname_entry.get().strip(),
                "resolution": int(self.res_entry.get()),
                "n_trajectories": int(self.traj_entry.get()),
                "n_balls_min": int(self.balls_min_entry.get()),
                "n_balls_max": int(self.balls_max_entry.get()),
                "speed_min": float(self.speed_min_entry.get()),
                "speed_max": float(self.speed_max_entry.get()),
                "cache_in_vram": bool(self.vram_var.get())
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
        if self.is_training or self.is_generating: return
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
            dyn_learning_rate=c['dyn_learning_rate'], dyn_weight_decay=c['dyn_weight_decay'],
            eval_horizon=c.get('eval_horizon', 50), eval_max_batches=c.get('eval_max_batches', 24),
            seed=(c.get('seed') or None), ae_checkpoint=c.get('ae_checkpoint', ""),
            cache_in_vram=c.get('cache_in_vram', False), latent_grid=c.get('latent_grid', 8),
            chunk_len=c.get('chunk_len', 5), eval_best_of_n=c.get('eval_best_of_n', 1),
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