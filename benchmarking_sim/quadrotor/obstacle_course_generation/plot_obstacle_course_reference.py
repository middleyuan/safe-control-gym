"""Plot an obstacle-course reference `.npy` file in 3D.

This script is paired with `generate_obstacle_course_reference.py` and can
overlay the obstacle segments from the copied YAML config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


CONFIG_STEM_PREFIX = "final_trajectory_prototype_"


def find_default_config_path(base_dir: Path) -> Path:
    trajectory_configs_dir = base_dir / "trajectory_configs"
    config_candidates = sorted(trajectory_configs_dir.glob("*.yaml"), key=lambda path: (path.stat().st_mtime, path.name))
    if config_candidates:
        return config_candidates[-1]

    fallback_path = base_dir / "configs" / "quadrotor_3D_attitude_tracking.yaml"
    return fallback_path


def config_suffix(config_path: Path) -> str:
    stem = config_path.stem
    if stem.startswith(CONFIG_STEM_PREFIX):
        return stem[len(CONFIG_STEM_PREFIX):]
    return stem


def default_npy_path(config_path: Path) -> Path:
    return config_path.with_name(f"custom_snap_ref_traj_{config_suffix(config_path)}.npy")


def load_reference(npy_path: Path) -> dict:
    data = np.load(npy_path, allow_pickle=True)
    if data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {npy_path}, got {type(data)!r}")
    if "POS_REF" not in data:
        raise KeyError(f"{npy_path} does not contain POS_REF")
    return data


def load_strings(config_path: Path) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    task_info = config.get("task_config", {}).get("task_info", {}) if isinstance(config, dict) else {}
    strings = task_info.get("strings", [])
    return [segment for segment in strings if isinstance(segment, dict) and "start" in segment and "end" in segment]


def plot_reference(pos_ref: np.ndarray, strings: list[dict] | None, output_path: Path | None, show: bool) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    xs = pos_ref[:, 0]
    ys = pos_ref[:, 1]
    zs = pos_ref[:, 2]

    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(pos_ref)))
    ax.plot(xs, ys, zs, color="0.35", linewidth=2, label="Trajectory")
    ax.scatter(xs, ys, zs, c=colors, s=8)

    if strings:
        first = True
        for obstacle in strings:
            start = np.asarray(obstacle["start"], dtype=float)
            end = np.asarray(obstacle["end"], dtype=float)
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="crimson", linewidth=2.5, label="Obstacle" if first else "")
            ax.scatter([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="crimson", s=18)
            first = False

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("Obstacle Course Reference Trajectory")
    ax.grid(True)
    ax.set_zlim(bottom=0)
    ax.set_box_aspect([1, 1, 0.7])
    ax.legend()

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_config = find_default_config_path(base_dir)
    parser = argparse.ArgumentParser(description="Plot the obstacle-course reference .npy file")
    parser.add_argument(
        "npy_path",
        nargs="?",
        default=str(default_npy_path(default_config)),
        help="Path to the reference .npy file",
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Optional YAML config used to overlay obstacle strings",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG path. Defaults to <npy stem>_3d.png next to the input file.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npy_path = Path(args.npy_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else npy_path.with_name(f"{npy_path.stem}_3d.png")

    ref_data = load_reference(npy_path)
    pos_ref = np.asarray(ref_data["POS_REF"], dtype=float)
    strings = load_strings(Path(args.config).expanduser().resolve()) if args.config else None

    plot_reference(pos_ref, strings, output_path, args.show)
    print(f"Saved 3D plot to {output_path}")


if __name__ == "__main__":
    main()