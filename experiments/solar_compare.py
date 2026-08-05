"""Comparison graphs for the solar models: mean PLANET position error
vs rollout frame (suns excluded by construction of orbit_errors) and mean
|radius drift| vs frame. 300-frame rollouts, 500 scenes, each model at its
protocol ns (chunk-1 models ns3, chunk-5 models ns2). Reads the
cmp_*.json files."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("no noise, 23ep (ns3)", "cmp_nonoise23", "#64748B"),
          ("noise 0.1, 11ep (ns3)", "cmp_noise11", "#2563EB"),
          ("chunk5 + noise, 11ep (ns2)", "cmp_chunk5", "#E8710A"),
          ("chunk5 + noise + finetune (ns2)", "cmp_ft_ns2", "#059669")]


def _plot(key, ylabel, title, fname, note):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    x = np.arange(1, 301)
    for label, tag, color in MODELS:
        d = json.load(open(f"scratchpad/examples/{tag}.json"))
        y = np.array(d[key])
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.annotate(label.split(" (")[0], (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=8.5,
                    va="center", fontweight="bold")
    ax.set_xlabel("rollout frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(1, 385)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    from scratchpad.figsave import save, train_end
    train_end(ax, 25, y=0.55)
    fig.tight_layout()
    save(fig, fname)


def main():
    _plot("pos_curve", "mean planet position error (px)",
          "Planet position error over 300-frame rollouts",
          "solar_cmp_pos",
          "500 scenes/model; planets only (suns excluded); error vs exact "
          "Kepler ground truth; each model at its protocol inference steps")
    _plot("rad_curve", "mean radius drift (px)",
          "Orbit radius drift over 300-frame rollouts",
          "solar_cmp_radius",
          "500 scenes/model; |tracked radius - true radius| per planet, "
          "averaged; radius stability = the orbit keeps its rail")


if __name__ == "__main__":
    main()
