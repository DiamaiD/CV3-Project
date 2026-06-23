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

        self.settings_frame = ctk.CTkFrame(self.tabview.tab("Training"))
        self.settings_frame.pack(pady=10, padx=10, fill="x")

        # --- ROW 0: Core Architecture ---
        self.env_label = ctk.CTkLabel(self.settings_frame, text="Environment:", font=self.bold_font)
        self.env_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.env_menu = ctk.CTkOptionMenu(self.settings_frame, values=self._list_environments(), font=self.huge_font, width=150)
        self.env_menu.grid(row=0, column=1, padx=10, pady=10)

        self.model_label = ctk.CTkLabel(self.settings_frame, text="Architecture:", font=self.bold_font)
        self.model_label.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.model_menu = ctk.CTkOptionMenu(self.settings_frame, values=["Latent", "Pixel"], font=self.huge_font, command=self._on_model_change)
        self.model_menu.grid(row=0, column=3, padx=10, pady=10)

        self.ctx_label = ctk.CTkLabel(self.settings_frame, text="Ctx Frames:", font=self.bold_font)
        self.ctx_label.grid(row=0, column=4, padx=10, pady=10, sticky="e")
        self.ctx_entry = ctk.CTkEntry(self.settings_frame, width=80, font=self.huge_font)
        self.ctx_entry.grid(row=0, column=5, padx=10, pady=10)

        # --- ROW 1: Optimization Params ---
        self.lr_label = ctk.CTkLabel(self.settings_frame, text="Learn Rate:", font=self.bold_font)
        self.lr_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.lr_entry = ctk.CTkEntry(self.settings_frame, width=150, font=self.huge_font)
        self.lr_entry.grid(row=1, column=1, padx=10, pady=10)

        self.wd_label = ctk.CTkLabel(self.settings_frame, text="Weight Decay:", font=self.bold_font)
        self.wd_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.wd_entry = ctk.CTkEntry(self.settings_frame, width=150, font=self.huge_font)
        self.wd_entry.grid(row=1, column=3, padx=10, pady=10)

        self.batch_label = ctk.CTkLabel(self.settings_frame, text="Batch Size:", font=self.bold_font)
        self.batch_label.grid(row=1, column=4, padx=10, pady=10, sticky="e")
        self.batch_entry = ctk.CTkEntry(self.settings_frame, width=80, font=self.huge_font)
        self.batch_entry.grid(row=1, column=5, padx=10, pady=10)

        # --- ROW 2: Workers & Epochs ---
        self.workers_label = ctk.CTkLabel(self.settings_frame, text="Num Workers:", font=self.bold_font)
        self.workers_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.workers_entry = ctk.CTkEntry(self.settings_frame, width=80, font=self.huge_font)
        self.workers_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.ae_epochs_label = ctk.CTkLabel(self.settings_frame, text="AE Epochs:", font=self.bold_font)
        self.ae_epochs_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.ae_epochs_entry = ctk.CTkEntry(self.settings_frame, width=80, font=self.huge_font)
        self.ae_epochs_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        self.dyn_epochs_label = ctk.CTkLabel(self.settings_frame, text="Dyn Epochs:", font=self.bold_font)
        self.dyn_epochs_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.dyn_epochs_entry = ctk.CTkEntry(self.settings_frame, width=80, font=self.huge_font)
        self.dyn_epochs_entry.grid(row=2, column=5, padx=10, pady=10)

        # --- ROW 3: Rollout (scheduled sampling horizon) + Seed ---
        self.rollout_label = ctk.CTkLabel(self.settings_frame, text="Rollout Len:", font=self.bold_font)
        self.rollout_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.rollout_entry = ctk.CTkEntry(self.settings_frame, width=80, font=self.huge_font)
        self.rollout_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        self.seed_label = ctk.CTkLabel(self.settings_frame, text="Seed:", font=self.bold_font)
        self.seed_label.grid(row=3, column=2, padx=10, pady=10, sticky="e")
        self.seed_entry = ctk.CTkEntry(self.settings_frame, width=150, font=self.huge_font, placeholder_text="(blank = random)")
        self.seed_entry.grid(row=3, column=3, padx=10, pady=10, sticky="w")

        # Max rollout length at eval time; the report tests every horizon from 1 to this.
        self.eval_horizon_label = ctk.CTkLabel(self.settings_frame, text="Eval Horizon:", font=self.bold_font)
        self.eval_horizon_label.grid(row=3, column=4, padx=10, pady=10, sticky="e")
        self.eval_horizon_entry = ctk.CTkEntry(self.settings_frame, width=80, font=self.huge_font)
        self.eval_horizon_entry.grid(row=3, column=5, padx=10, pady=10, sticky="w")

        # --- ROW 4: Reuse trained autoencoder (Latent only; blank = train fresh) ---
        self.ae_label = ctk.CTkLabel(self.settings_frame, text="Reuse AE:", font=self.bold_font)
        self.ae_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.ae_entry = ctk.CTkEntry(self.settings_frame, font=self.huge_font, placeholder_text="path to autoencoder.pth (blank = train new)")
        self.ae_entry.grid(row=4, column=1, columnspan=4, padx=10, pady=10, sticky="ew")
        self.ae_browse_button = ctk.CTkButton(self.settings_frame, text="Browse", width=80, font=self.bold_font, command=self._browse_ae)
        self.ae_browse_button.grid(row=4, column=5, padx=10, pady=10)

        # --- ROW 5: Actions ---
        self.save_button = ctk.CTkButton(self.settings_frame, text="Save Config", font=self.bold_font, command=self.save_settings)
        self.save_button.grid(row=5, column=0, columnspan=3, pady=10)

        self.start_button = ctk.CTkButton(self.settings_frame, text="START TRAINING", font=self.bold_font, fg_color="green", hover_color="darkgreen", command=self.start_training_thread)
        self.start_button.grid(row=5, column=3, columnspan=3, pady=10)

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

    def _on_model_change(self, choice):
        if choice == "Pixel":
            self.ae_epochs_entry.configure(state="disabled", fg_color="gray20", text_color="gray50")
            self.ae_epochs_label.configure(text_color="gray50")
        else:
            self.ae_epochs_entry.configure(state="normal", fg_color=["#F9F9FA", "#343638"], text_color=["black", "white"])
            self.ae_epochs_label.configure(text_color=["black", "white"])

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
                self.model_menu.set(c.get("model_type", "Latent"))
                self.ctx_entry.delete(0, "end"); self.ctx_entry.insert(0, str(c.get("context_len", 5)))
                self.lr_entry.delete(0, "end"); self.lr_entry.insert(0, str(c.get("learning_rate", 0.0005)))
                self.wd_entry.delete(0, "end"); self.wd_entry.insert(0, str(c.get("weight_decay", 0.001)))
                self.batch_entry.delete(0, "end"); self.batch_entry.insert(0, str(c.get("batch_size", 32)))
                self.workers_entry.delete(0, "end"); self.workers_entry.insert(0, str(c.get("num_workers", 8)))
                self.ae_epochs_entry.delete(0, "end"); self.ae_epochs_entry.insert(0, str(c.get("ae_epochs", 10)))
                self.dyn_epochs_entry.delete(0, "end"); self.dyn_epochs_entry.insert(0, str(c.get("dyn_epochs", 15)))
                self.rollout_entry.delete(0, "end"); self.rollout_entry.insert(0, str(c.get("rollout_len", 5)))
                self.eval_horizon_entry.delete(0, "end"); self.eval_horizon_entry.insert(0, str(c.get("eval_horizon", 10)))
                self.seed_entry.delete(0, "end"); self.seed_entry.insert(0, str(c.get("seed", "42")))
                self._set_entry(self.ae_entry, str(c.get("ae_checkpoint", "")))
                self.dataname_entry.delete(0, "end"); self.dataname_entry.insert(0, str(c.get("datagen_name", c.get("env_name", "bouncing"))))
                self.res_entry.delete(0, "end"); self.res_entry.insert(0, str(c.get("resolution", 64)))
                self.traj_entry.delete(0, "end"); self.traj_entry.insert(0, str(c.get("n_trajectories", 5000)))
                self.balls_min_entry.delete(0, "end"); self.balls_min_entry.insert(0, str(c.get("n_balls_min", 1)))
                self.balls_max_entry.delete(0, "end"); self.balls_max_entry.insert(0, str(c.get("n_balls_max", 5)))
                self.speed_min_entry.delete(0, "end"); self.speed_min_entry.insert(0, str(c.get("speed_min", 3.0)))
                self.speed_max_entry.delete(0, "end"); self.speed_max_entry.insert(0, str(c.get("speed_max", 8.0)))
                self._on_model_change(self.model_menu.get())
                self.log_textbox.insert("end", f"[System] Settings loaded from {CONFIG_FILE}\n")
            except Exception as e:
                self.log_textbox.insert("end", f"[Error] Load config failed: {e}\n")
        else:
            self._set_env("bouncing")
            self.model_menu.set("Latent")
            self.ctx_entry.insert(0, "5")
            self.lr_entry.insert(0, "0.0005")
            self.wd_entry.insert(0, "0.001")
            self.batch_entry.insert(0, "32")
            self.workers_entry.insert(0, "8")
            self.ae_epochs_entry.insert(0, "5")
            self.dyn_epochs_entry.insert(0, "15")
            self.rollout_entry.insert(0, "5")
            self.eval_horizon_entry.insert(0, "10")
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
                "model_type": self.model_menu.get(),
                "context_len": int(self.ctx_entry.get()),
                "learning_rate": float(self.lr_entry.get()),
                "weight_decay": float(self.wd_entry.get()),
                "batch_size": int(self.batch_entry.get()),
                "num_workers": int(self.workers_entry.get()),
                "ae_epochs": int(self.ae_epochs_entry.get()),
                "dyn_epochs": int(self.dyn_epochs_entry.get()),
                "rollout_len": int(self.rollout_entry.get()),
                "eval_horizon": int(self.eval_horizon_entry.get()),
                "seed": self.seed_entry.get().strip(),
                "ae_checkpoint": self.ae_entry.get().strip(),
                "datagen_name": self.dataname_entry.get().strip(),
                "resolution": int(self.res_entry.get()),
                "n_trajectories": int(self.traj_entry.get()),
                "n_balls_min": int(self.balls_min_entry.get()),
                "n_balls_max": int(self.balls_max_entry.get()),
                "speed_min": float(self.speed_min_entry.get()),
                "speed_max": float(self.speed_max_entry.get())
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
        self.log_textbox.insert("end", f"[System] Starting {config['model_type']} run...\n")
        self.is_training = True
        self.start_button.configure(state="disabled", text="Training...")

        thread = threading.Thread(target=self._run_training, args=(config,), daemon=True)
        thread.start()
        self.after(1000, self.poll_log_file) 

    def _run_training(self, c):
        run_training_pipeline(
            data_dir=f"data/{c['env_name']}", env_name=c['env_name'], model_type=c['model_type'],
            context_len=c['context_len'], batch_size=c['batch_size'], num_workers=c['num_workers'],
            ae_epochs=c['ae_epochs'], dyn_epochs=c['dyn_epochs'],
            learning_rate=c['learning_rate'], weight_decay=c['weight_decay'],
            rollout_len=c.get('rollout_len', 5), eval_horizon=c.get('eval_horizon', 10),
            seed=(c.get('seed') or None), ae_checkpoint=c.get('ae_checkpoint', "")
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