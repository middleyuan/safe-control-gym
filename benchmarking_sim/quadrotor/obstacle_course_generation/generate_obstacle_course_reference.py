"""Generate an obstacle-course reference `.npy` from a YAML config.

This is the obstacle-course reference generation pipeline:
- it reads the obstacle-course YAML,
- uses the existing safe_control_gym TrajectoryPlanner,
- computes reference derivatives numerically,
- and writes the same style of dict saved by the larger environment path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from safe_control_gym.envs.gym_pybullet_drones.trajectory_utils import (
    PiecewisePolynomialTrajectory,
    PolynomialSize,
    TrajectoryPlanner,
    Waypoint,
    _parse_references,
    _solve_closed_form,
    _solve_constrained,
    compute_trajectory_derivatives,
)


DEFAULT_WARMSTART_STEPS = 60
CONFIG_STEM_PREFIX = "final_trajectory_prototype_"


def format_time_folder(value: float) -> str:
    rounded = round(value, 6)
    if float(rounded).is_integer():
        return str(int(rounded))
    return str(rounded).replace(".", "_")


def find_latest_data_config_path(base_dir: Path) -> Path | None:
    data_dir = base_dir / "data"
    if not data_dir.exists():
        return None

    config_candidates = sorted(
        data_dir.glob("*/config/*.yaml"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if config_candidates:
        return config_candidates[-1]
    return None


def infer_time_folder_from_config_path(config_path: Path, base_dir: Path) -> str:
    data_dir = base_dir / "data"
    try:
        rel = config_path.resolve().relative_to(data_dir.resolve())
        parts = rel.parts
        if len(parts) >= 3 and parts[1] == "config":
            return parts[0]
    except ValueError:
        pass

    config = load_config(config_path)
    task_config = config.get("task_config", {})
    episode_len_sec = task_config.get("episode_len_sec")
    if episode_len_sec is not None:
        return format_time_folder(float(episode_len_sec))
    return config_suffix(config_path)


def find_default_config_path(base_dir: Path) -> Path:
    latest_data_config = find_latest_data_config_path(base_dir)
    if latest_data_config is not None:
        return latest_data_config

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


def default_output_path(config_path: Path) -> Path:
    base_dir = Path(__file__).resolve().parent
    time_folder = infer_time_folder_from_config_path(config_path, base_dir)
    return base_dir / "data" / time_folder / "trajectory" / f"custom_snap_ref_traj_{time_folder}.npy"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML dict in {config_path}, got {type(config)!r}")
    return config


def validate_waypoints(waypoints: list[dict]) -> None:
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required to generate a trajectory")

    times = [float(wp["time"]) for wp in waypoints]
    if any(curr <= prev for prev, curr in zip(times, times[1:])):
        raise ValueError("Waypoint times must be strictly increasing")

    for idx, wp in enumerate(waypoints):
        position = wp.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"Waypoint {idx} must contain a 3D position list")


def enforce_strictly_increasing_waypoint_times(waypoints: list[dict], min_delta: float = 1e-6) -> bool:
    adjusted = False
    prev_time: float | None = None
    for wp in waypoints:
        curr_time = float(wp["time"])
        if prev_time is not None and curr_time <= prev_time:
            curr_time = prev_time + min_delta
            wp["time"] = curr_time
            adjusted = True
        prev_time = curr_time
    return adjusted


def init_custom_waypoints(waypoints: list[dict]) -> list[Waypoint]:
    return [Waypoint(time=float(wp["time"]), position=wp["position"]) for wp in waypoints]


def resolve_warmstart_steps(planner_steps: int | None) -> int:
    if planner_steps is not None:
        return max(2, planner_steps)
    return DEFAULT_WARMSTART_STEPS


def generate_minimum_snap_trajectory(
    references: list[Waypoint],
    degree: int = 6,
    idx_minimized_orders: int | tuple[int, ...] = 5,
    num_continuous_orders: int = 3,
    algorithm: str = "closed-form",
    optimize_options: dict | None = None,
    acc_limits: tuple[float, float] = (-np.inf, np.inf),
) -> PiecewisePolynomialTrajectory:
    if algorithm == "closed-form":
        print("Using closed-form solution for minimum snap, acceleration limits will be ignored")

    if degree < 2:
        raise ValueError("Polynomial degree too low")

    derivative_weights = np.zeros(degree)
    idx_minimized_orders = np.asarray(idx_minimized_orders, dtype=np.int32).ravel()
    if (idx_minimized_orders < 2).any():
        raise ValueError("Minimizing 0th- or 1st-order derivatives does not make sense")
    if (idx_minimized_orders > degree).any():
        raise ValueError("Cannot minimize derivatives whose order is higher than the polynomial degree")
    derivative_weights[idx_minimized_orders] = 1

    if num_continuous_orders < 3:
        raise ValueError("Trajectory continuity below acceleration does not make sense here")
    if num_continuous_orders > degree:
        raise ValueError("Continuous derivative order cannot exceed polynomial degree")

    t_ref, refs = _parse_references(references, num_continuous_orders)
    if (t_ref < 0.0).any():
        raise ValueError("Waypoint timestamp is negative")

    durations = np.diff(t_ref).astype("float64")
    if (durations <= 1e-8).any():
        raise ValueError("The time duration for transiting between waypoints is too small")

    poly_dim = PolynomialSize(
        n_poly=refs.shape[0] - 1,
        n_cfs=degree + 1,
        dim=refs.shape[2],
    )

    if algorithm == "constrained":
        coeffs = _solve_constrained(
            refs,
            durations,
            poly_dim,
            derivative_weights,
            num_continuous_orders,
            optimize_options,
            acc_limits,
        )
    elif algorithm == "closed-form":
        coeffs = _solve_closed_form(
            refs,
            durations,
            poly_dim,
            derivative_weights,
            num_continuous_orders,
            optimize_options,
        )
    else:
        raise ValueError("Unrecognized algorithm")

    return PiecewisePolynomialTrajectory(t_ref, durations, coeffs)


def build_reference(config: dict, planner_steps: int | None = None) -> dict[str, np.ndarray]:
    task_config = config.get("task_config", {})
    task_info = task_config.get("task_info", {})

    waypoints = task_info.get("waypoints", [])
    strings = task_info.get("strings", [])
    if not waypoints:
        raise ValueError("No waypoints found in task_config.task_info.waypoints")

    if enforce_strictly_increasing_waypoint_times(waypoints):
        print("Adjusted non-increasing waypoint timestamps to be strictly increasing.")

    validate_waypoints(waypoints)

    ctrl_freq = float(task_config.get("ctrl_freq", 60))
    sample_time = 1.0 / ctrl_freq
    episode_len_sec = float(task_config.get("episode_len_sec", waypoints[-1]["time"]))
    times = np.arange(0.0, episode_len_sec + sample_time, sample_time)

    if strings:
        warmstart_steps = resolve_warmstart_steps(planner_steps)
        planner = TrajectoryPlanner(waypoint_list=waypoints, string_list=strings, N=warmstart_steps)
        minimum_snap_waypoints = planner.waypoints.copy()
    else:
        minimum_snap_waypoints = init_custom_waypoints(waypoints)

    polys = generate_minimum_snap_trajectory(
        minimum_snap_waypoints,
        degree=6,
        idx_minimized_orders=5,
        num_continuous_orders=3,
        algorithm="closed-form",
    )
    pvaj = compute_trajectory_derivatives(polys, times, 4)
    pos_ref = pvaj[0, :, :]
    vel_ref = pvaj[1, :, :]
    acc_ref = pvaj[2, :, :]
    jrk_ref = pvaj[3, :, :]
    spd_ref = np.linalg.norm(vel_ref, axis=1)

    return {
        "POS_REF": pos_ref,
        "VEL_REF": vel_ref,
        "ACC_REF": acc_ref,
        "JRK_REF": jrk_ref,
        "SPD_REF": spd_ref,
    }


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate an obstacle-course reference .npy")
    parser.add_argument(
        "--config",
        default=str(find_default_config_path(base_dir)),
        help="Path to the obstacle-course YAML config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npy path. Defaults to custom_snap_ref_traj_<config suffix>.npy next to the config.",
    )
    parser.add_argument(
        "--planner-steps",
        type=int,
        default=None,
        help="Override the warm-start obstacle-avoidance planner horizon N. Default matches the original method: N=60.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(config_path)

    config = load_config(config_path)
    ref_data = build_reference(config, planner_steps=args.planner_steps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, ref_data, allow_pickle=True)
    print(f"Saved reference trajectory to {output_path}")


if __name__ == "__main__":
    main()
