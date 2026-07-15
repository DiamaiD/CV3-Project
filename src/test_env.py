from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for candidate in (REPO_ROOT, CURRENT_DIR):
	candidate_str = str(candidate)
	if candidate_str not in sys.path:
		sys.path.insert(0, candidate_str)

from environments.env_bouncing import generate_bouncing_data
from environments.env_bouncing_rigid import generate_bouncing_data as generate_bouncing_rigid_data
from environments.env_domino import generate_domino_data
from environments.env_inclined_plane import generate_slope_data


ENVIRONMENT_SPECS = {
	"env_bouncing": {
		"generator": generate_bouncing_data,
		"data_dir": "bouncing",
	},
	"env_bouncing_rigid": {
		"generator": generate_bouncing_rigid_data,
		"data_dir": "rigid_bouncing",
	},
	"env_domino": {
		"generator": generate_domino_data,
		"data_dir": "domino",
	},
	"env_inclined_plane": {
		"generator": generate_slope_data,
		"data_dir": "inclined_plane",
	},
}

DEFAULT_PREVIEW_FPS = 20


def _load_traj_frames(traj_dir: Path) -> list[Image.Image]:
	frame_paths = sorted(traj_dir.glob("frame_*.png"))
	frames: list[Image.Image] = []
	for frame_path in frame_paths:
		with Image.open(frame_path) as frame:
			frames.append(frame.convert("RGB"))
	return frames


def _save_gif(frames: list[Image.Image], gif_path: Path, fps: int) -> None:
	if not frames:
		raise RuntimeError(f"No frames were generated for {gif_path.name}")

	gif_path.parent.mkdir(parents=True, exist_ok=True)
	duration_ms = max(1, round(1000 / fps))
	first_frame, remaining_frames = frames[0], frames[1:]
	first_frame.save(
		gif_path,
		save_all=True,
		append_images=remaining_frames,
		duration=duration_ms,
		loop=0,
		optimize=False,
		disposal=2,
	)


def _resolve_env_name(env_name: str) -> tuple[str, dict]:
	key = env_name.strip().lower()
	if key not in ENVIRONMENT_SPECS:
		available = ", ".join(sorted(ENVIRONMENT_SPECS))
		raise SystemExit(f"Unknown environment '{env_name}'. Available options: {available}")
	return key, ENVIRONMENT_SPECS[key]


def generate_env_gifs(env_name: str, n_gifs: int, seconds: int = 5,
					  output_dir: str | Path = "outputs/test_env") -> list[Path]:
	resolved_name, spec = _resolve_env_name(env_name)
	fps = DEFAULT_PREVIEW_FPS
	max_frames = int(seconds * fps)
	output_root = Path(output_dir) / resolved_name
	output_root.mkdir(parents=True, exist_ok=True)

	generated_paths: list[Path] = []
	temp_dir = Path(tempfile.mkdtemp(prefix=f"{resolved_name}_preview_"))
	try:
		spec["generator"](data_dir=str(temp_dir / spec["data_dir"]), n_trajectories=n_gifs, max_frames=max_frames)

		for traj_idx in range(n_gifs):
			traj_dir = temp_dir / spec["data_dir"] / f"traj-{traj_idx}"
			frames = _load_traj_frames(traj_dir)
			gif_path = output_root / f"traj-{traj_idx}.gif"
			_save_gif(frames, gif_path, fps)
			generated_paths.append(gif_path)
	finally:
		shutil.rmtree(temp_dir, ignore_errors=True)

	return generated_paths


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate short GIF previews for the physics environments.")
	parser.add_argument("-e", "--env", default="env_bouncing",
						help="Environment to preview: env_bouncing, env_bouncing_rigid, env_domino, env_inclined_plane")
	parser.add_argument("-n", "--n_gifs", type=int, default=5,
						help="Number of GIF rollouts to generate")
	parser.add_argument("--seconds", type=int, default=5,
						help="Length of each GIF in seconds")
	parser.add_argument("--output_dir", type=str, default="outputs/test_env",
						help="Directory where GIFs will be written")
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)

	generated_paths = generate_env_gifs(
		env_name=args.env,
		n_gifs=args.n_gifs,
		seconds=args.seconds,
		output_dir=args.output_dir,
	)

	for gif_path in generated_paths:
		print(gif_path)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
