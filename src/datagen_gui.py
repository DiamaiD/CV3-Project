import customtkinter as ctk
import threading
import os
import json
import shutil

from environments.env_shapes import DEFAULT_KIND_WEIGHTS
from environments.env_tower import PROJECTILE_KINDS
from environments.parallel import generate_parallel, N_WORKERS

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "configs/datagen_config.json"

ENV_MODULES = {
    "Bouncing": ("environments.env_bouncing", "generate_bouncing_data"),
    "Balls v2": ("environments.env_pymunk", "generate_balls2_pymunk"),
    "Shapes": ("environments.env_pymunk", "generate_shapes_pymunk"),
    "Tower": ("environments.env_pymunk", "generate_tower_pymunk"),
}


class DatagenGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CV3 Dataset Generator")
        self.geometry("1100x760")
        self.huge_font = ctk.CTkFont(family="DejaVu Sans Mono", size=18)
        self.bold_font = ctk.CTkFont(family="DejaVu Sans Mono", size=18, weight="bold")
        os.makedirs("configs", exist_ok=True)

        self.common = ctk.CTkFrame(self)
        self.common.pack(pady=(10, 0), padx=20, fill="x")
        self._entry(self.common, "Name:", "name_entry", 0, 0, width=180)
        self._entry(self.common, "Trajectories:", "traj_entry", 0, 2)
        self._entry(self.common, "Frames:", "frames_entry", 0, 4)
        self._entry(self.common, "Resolution:", "res_entry", 0, 6)
        self.ss_label = ctk.CTkLabel(self.common, text="Supersample:", font=self.bold_font)
        self.ss_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.ss_menu = ctk.CTkOptionMenu(self.common, values=["1", "2", "4"], font=self.huge_font, width=80)
        self.ss_menu.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self._entry(self.common, "Workers:", "workers_entry", 1, 2)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(pady=5, padx=20, fill="x")
        for env in ENV_MODULES:
            self.tabview.add(env)

        b = ctk.CTkFrame(self.tabview.tab("Bouncing"))
        b.pack(pady=10, padx=10, fill="x")
        self._entry(b, "Balls Min:", "b_min_entry", 0, 0)
        self._entry(b, "Balls Max:", "b_max_entry", 0, 2)
        self._entry(b, "Speed Min:", "b_smin_entry", 0, 4)
        self._entry(b, "Speed Max:", "b_smax_entry", 0, 6)
        self._entry(b, "Radius Min:", "b_rmin_entry", 1, 0)
        self._entry(b, "Radius Max:", "b_rmax_entry", 1, 2)

        v = ctk.CTkFrame(self.tabview.tab("Balls v2"))
        v.pack(pady=10, padx=10, fill="x")
        self.v_info = ctk.CTkLabel(v, text="Balls on the Chipmunk2D engine: rotation, ball-ball friction, sleeping.",
                                   font=self.bold_font)
        self.v_info.grid(row=0, column=0, columnspan=8, padx=10, pady=(10, 0), sticky="w")
        self._entry(v, "Balls Min:", "v_min_entry", 1, 0)
        self._entry(v, "Balls Max:", "v_max_entry", 1, 2)
        self._entry(v, "Speed Min:", "v_smin_entry", 1, 4)
        self._entry(v, "Speed Max:", "v_smax_entry", 1, 6)
        self._entry(v, "Radius Min:", "v_rmin_entry", 2, 0)
        self._entry(v, "Radius Max:", "v_rmax_entry", 2, 2)
        self._entry(v, "Spin Max:", "v_spin_entry", 2, 4)
        self.v_marker_label = ctk.CTkLabel(v, text="Markers:", font=self.bold_font)
        self.v_marker_label.grid(row=2, column=6, padx=10, pady=10, sticky="e")
        self.v_marker_menu = ctk.CTkOptionMenu(v, values=["on", "off"], font=self.huge_font, width=90)
        self.v_marker_menu.grid(row=2, column=7, padx=10, pady=10, sticky="w")

        s = ctk.CTkFrame(self.tabview.tab("Shapes"))
        s.pack(pady=10, padx=10, fill="x")
        self._entry(s, "Objects Min:", "s_min_entry", 0, 0)
        self._entry(s, "Objects Max:", "s_max_entry", 0, 2)
        self._entry(s, "Speed Min:", "s_smin_entry", 0, 4)
        self._entry(s, "Speed Max:", "s_smax_entry", 0, 6)
        self._entry(s, "Spin Max:", "s_spin_entry", 1, 0)
        self._entry(s, "Vertex Jitter:", "s_jitter_entry", 1, 2)
        self._entry(s, "Size Min:", "s_zmin_entry", 1, 4)
        self._entry(s, "Size Max:", "s_zmax_entry", 1, 6)
        self.s_marker_label = ctk.CTkLabel(s, text="Markers:", font=self.bold_font)
        self.s_marker_label.grid(row=2, column=4, padx=10, pady=10, sticky="e")
        self.s_marker_menu = ctk.CTkOptionMenu(s, values=["on", "off"], font=self.huge_font, width=90)
        self.s_marker_menu.grid(row=2, column=5, padx=10, pady=10, sticky="w")
        self.s_kind_label = ctk.CTkLabel(s, text="Kind weights:", font=self.bold_font)
        self.s_kind_label.grid(row=2, column=0, padx=10, pady=(14, 4), sticky="w", columnspan=4)
        self.s_kind_entries = {}
        for i, kind in enumerate(DEFAULT_KIND_WEIGHTS):
            row, col = 3 + i // 4, (i % 4) * 2
            self._entry(s, f"{kind}:", f"s_kind_{kind}", row, col)
            self.s_kind_entries[kind] = getattr(self, f"s_kind_{kind}")

        t = ctk.CTkFrame(self.tabview.tab("Tower"))
        t.pack(pady=10, padx=10, fill="x")
        self._entry(t, "Storeys Min:", "t_min_entry", 0, 0)
        self._entry(t, "Storeys Max:", "t_max_entry", 0, 2)
        self._entry(t, "Arrive Min:", "t_amin_entry", 0, 4)
        self._entry(t, "Arrive Max:", "t_amax_entry", 0, 6)
        self._entry(t, "Vertex Jitter:", "t_jitter_entry", 1, 4)
        self.t_marker_label = ctk.CTkLabel(t, text="Markers:", font=self.bold_font)
        self.t_marker_label.grid(row=1, column=6, padx=10, pady=10, sticky="e")
        self.t_marker_menu = ctk.CTkOptionMenu(t, values=["on", "off"], font=self.huge_font, width=90)
        self.t_marker_menu.grid(row=1, column=7, padx=10, pady=10, sticky="w")
        self.t_kind_label = ctk.CTkLabel(t, text="Projectile weights:", font=self.bold_font)
        self.t_kind_label.grid(row=1, column=0, padx=10, pady=(14, 4), sticky="w", columnspan=4)
        self.t_kind_entries = {}
        for i, kind in enumerate(PROJECTILE_KINDS):
            self._entry(t, f"{kind}:", f"t_kind_{kind}", 2, i * 2)
            self.t_kind_entries[kind] = getattr(self, f"t_kind_{kind}")

        actions = ctk.CTkFrame(self)
        actions.pack(pady=(0, 5), padx=20, fill="x")
        self.generate_button = ctk.CTkButton(actions, text="GENERATE", font=self.bold_font,
                                             command=self.start_generation, height=40)
        self.generate_button.pack(side="left", padx=10, pady=10)
        self.save_button = ctk.CTkButton(actions, text="Save Config", font=self.bold_font,
                                         command=self.save_config_clicked, height=40, width=120)
        self.save_button.pack(side="left", padx=10, pady=10)
        self.status_label = ctk.CTkLabel(actions, text="idle", font=self.huge_font)
        self.status_label.pack(side="left", padx=20)

        self.log = ctk.CTkTextbox(self, width=1050, height=280, font=self.huge_font)
        self.log.pack(pady=10, padx=20, fill="both", expand=True)

        self.is_generating = False
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _entry(self, parent, label, attr, row, col, width=90):
        lab = ctk.CTkLabel(parent, text=label, font=self.bold_font)
        lab.grid(row=row, column=col, padx=10, pady=10, sticky="e")
        ent = ctk.CTkEntry(parent, width=width, font=self.huge_font)
        ent.grid(row=row, column=col + 1, padx=10, pady=10, sticky="w")
        setattr(self, attr, ent)

    def _set(self, attr, value):
        e = getattr(self, attr)
        e.delete(0, "end")
        e.insert(0, str(value))

    def load_settings(self):
        c = {}
        if os.path.exists(CONFIG_FILE):
            try:
                c = json.load(open(CONFIG_FILE))
            except Exception:
                c = {}
        self._set("name_entry", c.get("name", "shapes_36k"))
        self._set("traj_entry", c.get("n_trajectories", 36000))
        self._set("frames_entry", c.get("max_frames", 100))
        self._set("res_entry", c.get("resolution", 64))
        self.ss_menu.set(str(c.get("supersample", 4)))
        try:
            self.tabview.set(c.get("env", "Shapes"))
        except Exception:
            pass
        self._set("workers_entry", c.get("n_workers", N_WORKERS))
        self._set("b_min_entry", c.get("b_min", 1)); self._set("b_max_entry", c.get("b_max", 5))
        self._set("b_smin_entry", c.get("b_smin", 3.0)); self._set("b_smax_entry", c.get("b_smax", 8.0))
        self._set("b_rmin_entry", c.get("b_rmin", 5)); self._set("b_rmax_entry", c.get("b_rmax", 8))
        self._set("v_min_entry", c.get("v_min", 1)); self._set("v_max_entry", c.get("v_max", 6))
        self._set("v_smin_entry", c.get("v_smin", 3.0)); self._set("v_smax_entry", c.get("v_smax", 8.0))
        self._set("v_rmin_entry", c.get("v_rmin", 5)); self._set("v_rmax_entry", c.get("v_rmax", 8))
        self._set("v_spin_entry", c.get("v_spin", 0.25))
        self.v_marker_menu.set(c.get("v_markers", "on"))
        self.s_marker_menu.set(c.get("s_markers", "on"))
        self.t_marker_menu.set(c.get("t_markers", "on"))
        self._set("s_min_entry", c.get("s_min", 1)); self._set("s_max_entry", c.get("s_max", 5))
        self._set("s_smin_entry", c.get("s_smin", 3.0)); self._set("s_smax_entry", c.get("s_smax", 8.0))
        self._set("s_spin_entry", c.get("s_spin", 0.25))
        self._set("s_jitter_entry", c.get("s_jitter", 0.4))
        self._set("s_zmin_entry", c.get("s_zmin", 5)); self._set("s_zmax_entry", c.get("s_zmax", 8))
        for kind, default in DEFAULT_KIND_WEIGHTS.items():
            self._set(f"s_kind_{kind}", c.get(f"s_kind_{kind}", default))
        self._set("t_min_entry", c.get("t_min", 1)); self._set("t_max_entry", c.get("t_max", 3))
        self._set("t_amin_entry", c.get("t_amin", 6.0)); self._set("t_amax_entry", c.get("t_amax", 8.0))
        self._set("t_jitter_entry", c.get("t_jitter", 0.4))
        for kind, default in PROJECTILE_KINDS.items():
            self._set(f"t_kind_{kind}", c.get(f"t_kind_{kind}", default))

    def save_settings(self):
        c = {"env": self.tabview.get(),
             "name": self.name_entry.get().strip(),
             "n_trajectories": int(self.traj_entry.get()),
             "max_frames": int(self.frames_entry.get()),
             "resolution": int(self.res_entry.get()),
             "supersample": int(self.ss_menu.get()),
             "n_workers": int(self.workers_entry.get()),
             "b_min": int(self.b_min_entry.get()), "b_max": int(self.b_max_entry.get()),
             "b_smin": float(self.b_smin_entry.get()), "b_smax": float(self.b_smax_entry.get()),
             "b_rmin": int(self.b_rmin_entry.get()), "b_rmax": int(self.b_rmax_entry.get()),
             "v_min": int(self.v_min_entry.get()), "v_max": int(self.v_max_entry.get()),
             "v_smin": float(self.v_smin_entry.get()), "v_smax": float(self.v_smax_entry.get()),
             "v_rmin": int(self.v_rmin_entry.get()), "v_rmax": int(self.v_rmax_entry.get()),
             "v_spin": float(self.v_spin_entry.get()),
             "v_markers": self.v_marker_menu.get(),
             "s_markers": self.s_marker_menu.get(),
             "t_markers": self.t_marker_menu.get(),
             "s_min": int(self.s_min_entry.get()), "s_max": int(self.s_max_entry.get()),
             "s_smin": float(self.s_smin_entry.get()), "s_smax": float(self.s_smax_entry.get()),
             "s_spin": float(self.s_spin_entry.get()),
             "s_jitter": float(self.s_jitter_entry.get()),
             "s_zmin": int(self.s_zmin_entry.get()), "s_zmax": int(self.s_zmax_entry.get()),
             "t_min": int(self.t_min_entry.get()), "t_max": int(self.t_max_entry.get()),
             "t_amin": float(self.t_amin_entry.get()), "t_amax": float(self.t_amax_entry.get()),
             "t_jitter": float(self.t_jitter_entry.get())}
        for kind, e in self.s_kind_entries.items():
            c[f"s_kind_{kind}"] = float(e.get())
        for kind, e in self.t_kind_entries.items():
            c[f"t_kind_{kind}"] = float(e.get())
        with open(CONFIG_FILE, "w") as f:
            json.dump(c, f, indent=4)

    def save_config_clicked(self):
        try:
            self.save_settings()
            self.log.insert("end", f"Config saved to {CONFIG_FILE}\n")
        except ValueError as e:
            self.log.insert("end", f"[Error] Invalid fields, config not saved: {e}\n")

    def _on_close(self):
        try:
            self.save_settings()
        except Exception:
            pass
        self.destroy()

    def start_generation(self):
        if self.is_generating:
            return
        env = self.tabview.get()
        try:
            name = self.name_entry.get().strip()
            n_traj = int(self.traj_entry.get())
            frames = int(self.frames_entry.get())
            res = int(self.res_entry.get())
            ss = int(self.ss_menu.get())
            kwargs = {"max_frames": frames, "width": res, "height": res, "supersample": ss}
            n_workers = int(self.workers_entry.get())
            if env == "Bouncing":
                kwargs.update(n_balls_min=int(self.b_min_entry.get()),
                              n_balls_max=int(self.b_max_entry.get()),
                              speed_min=float(self.b_smin_entry.get()),
                              speed_max=float(self.b_smax_entry.get()),
                              radius_min=int(self.b_rmin_entry.get()),
                              radius_max=int(self.b_rmax_entry.get()))
            elif env == "Balls v2":
                kwargs.update(n_balls_min=int(self.v_min_entry.get()),
                              n_balls_max=int(self.v_max_entry.get()),
                              speed_min=float(self.v_smin_entry.get()),
                              speed_max=float(self.v_smax_entry.get()),
                              radius_min=int(self.v_rmin_entry.get()),
                              radius_max=int(self.v_rmax_entry.get()),
                              spin_max=float(self.v_spin_entry.get()),
                              markers=self.v_marker_menu.get())
            elif env == "Shapes":
                weights = {k: float(e.get()) for k, e in self.s_kind_entries.items()}
                weights = {k: v for k, v in weights.items() if v > 0}
                if not weights:
                    raise ValueError("all shape kind weights are zero")
                kwargs.update(n_objects_min=int(self.s_min_entry.get()),
                              n_objects_max=int(self.s_max_entry.get()),
                              speed_min=float(self.s_smin_entry.get()),
                              speed_max=float(self.s_smax_entry.get()),
                              spin_max=float(self.s_spin_entry.get()),
                              vertex_jitter=float(self.s_jitter_entry.get()),
                              size_min=int(self.s_zmin_entry.get()),
                              size_max=int(self.s_zmax_entry.get()),
                              kind_weights=weights,
                              markers=self.s_marker_menu.get())
            else:
                weights = {k: float(e.get()) for k, e in self.t_kind_entries.items()}
                weights = {k: v for k, v in weights.items() if v > 0}
                if not weights:
                    raise ValueError("all projectile weights are zero")
                kwargs.update(n_storeys_min=int(self.t_min_entry.get()),
                              n_storeys_max=int(self.t_max_entry.get()),
                              arrive_min=float(self.t_amin_entry.get()),
                              arrive_max=float(self.t_amax_entry.get()),
                              vertex_jitter=float(self.t_jitter_entry.get()),
                              projectile_kinds=weights,
                              markers=self.t_marker_menu.get())
        except ValueError as e:
            self.log.insert("end", f"[Error] Invalid fields: {e}\n")
            return
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            self.log.insert("end", "[Error] Dataset name must be a non-empty plain folder name.\n")
            return
        if res % 8 != 0:
            self.log.insert("end", f"[Error] Resolution must be a multiple of 8 (got {res}).\n")
            return

        self.save_settings()
        data_dir = os.path.join("data", name)
        self.is_generating = True
        self.generate_button.configure(state="disabled", text="Generating...")
        self.log.insert("end", f"[System] {env}: {n_traj} trajectories ({res}x{res}, ss{ss}) -> {data_dir}\n")
        self.log.see("end")
        module, fn = ENV_MODULES[env]
        threading.Thread(target=self._run, args=(module, fn, data_dir, n_traj, max(1, n_workers), kwargs),
                         daemon=True).start()

    def _run(self, module, fn, data_dir, n_traj, n_workers, kwargs):
        def cb(done, total):
            self.after(0, lambda d=done, t=total: self.status_label.configure(text=f"{d}/{t}"))
        try:
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
                self.after(0, lambda: self.log.insert("end", f"[System] Removed existing dataset at {data_dir}.\n"))
            generate_parallel(module, fn, data_dir=data_dir, n_trajectories=n_traj,
                              n_workers=n_workers, progress_cb=cb, **kwargs)
            self.after(0, lambda: self.log.insert("end", "[System] Generation complete.\n"))
        except Exception as e:
            self.after(0, lambda err=e: self.log.insert("end", f"[Error] Generation failed: {err}\n"))
        finally:
            self.is_generating = False
            self.after(0, lambda: self.generate_button.configure(state="normal", text="GENERATE"))
            self.after(0, lambda: self.log.see("end"))


if __name__ == "__main__":
    DatagenGUI().mainloop()
