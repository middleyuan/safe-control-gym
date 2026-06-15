# Obstacle Course Generation

This folder is the minimal bundle for generating the obstacle-course reference trajectory.

It intentionally avoids copying the full `quadrotor.py` and `benchmark_env.py` modules.
The reference path is generated from the copied YAML config in two stages:

1. A coarse obstacle-avoidance warm-start is generated with the existing CasADi-based `TrajectoryPlanner`.
2. A minimum-snap polynomial trajectory is fit through those warm-start waypoints and sampled at the control rate.

The final `.npy` stores `POS_REF` / `VEL_REF` / `ACC_REF` / `JRK_REF` / `SPD_REF`.

By default, the warm-start planner uses `N=60`, which matches the original env-side method. You can override that with `--planner-steps`.

## Included

- `generate_obstacle_course_reference.py`: builds the `.npy` reference file from the YAML.
- `plot_obstacle_course_reference.py`: renders the reference trajectory and optional obstacle overlay.
- `trajectory_configs/*.yaml`: generated obstacle-course configs used by default.
- `configs/quadrotor_3D_attitude_tracking.yaml`: fallback config if no generated trajectory config exists.
- `generator_scripts/obstacle_course_generator.py`: copied obstacle-course generator.
- `generator_scripts/obstacle_course_generator_v6_0.py`: copied obstacle-course generator variant.
- `generator_scripts/obstacle_course_generator_v6_1.py`: copied obstacle-course generator variant.
- `generator_scripts/obstacle_course_generator_v6_2.py`: copied obstacle-course generator variant.

## Usage

From the repo root:

```bash
python3 benchmarking_sim/quadrotor/obstacle_course_generation/generate_obstacle_course_reference.py
```

By default this reads the newest YAML from `trajectory_configs/` and writes `custom_snap_ref_traj_<config suffix>.npy` next to that YAML unless you pass `--output`.

To choose a custom warm-start horizon:

```bash
python3 benchmarking_sim/quadrotor/obstacle_course_generation/generate_obstacle_course_reference.py --planner-steps 180
```

To plot the generated reference:

```bash
python3 benchmarking_sim/quadrotor/obstacle_course_generation/plot_obstacle_course_reference.py
```

## Notes

- The original source files under `examples/pid` and `safe_control_gym` are unchanged.
- The generator still depends on the main `safe_control_gym` trajectory utilities and the CasADi solver stack.
- The warm-start obstacle planner horizon can change, but the underlying optimization methods remain the same: obstacle-aware warm-start first, minimum-snap smoothing second.
- The copied generator scripts are snapshots for reference and local editing inside this bundle.
- If you want the historical generator variants too, keep using the broader bundle under
  `benchmarking_sim/quadrotor/obstacle_course_generation/`.
