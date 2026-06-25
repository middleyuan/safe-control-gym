# Obstacle Course Generation

This folder contains the obstacle-course generation, plotting, and PID test workflow for the quadrotor reference trajectories.

The reference path is generated in two stages:

1. A coarse obstacle-avoidance warm-start is generated with the CasADi-based `TrajectoryPlanner`.
2. A minimum-snap polynomial trajectory is fit through those warm-start waypoints and sampled at the control rate.

The generated `.npy` stores `POS_REF`, `VEL_REF`, `ACC_REF`, `JRK_REF`, and `SPD_REF`.

By default, the warm-start planner uses `N=60`, which matches the original env-side method. You can override that with `--planner-steps`.

## Commands

Run these from [benchmarking_sim/quadrotor/obstacle_course_generation](benchmarking_sim/quadrotor/obstacle_course_generation).

### Generate the reference `.npy`

```bash
python3 generate_obstacle_course_reference.py --config data/13/config/config.yaml
python3 generate_obstacle_course_reference.py --config data/15_5/config/config.yaml
python3 generate_obstacle_course_reference.py --config data/20_5/config/config.yaml
```

These commands regenerate the obstacle-course reference files for the 13, 15.5, and 20.5 second trajectories.

### Plot the reference trajectories

```bash
export MPLBACKEND=Agg

python3 plot_obstacle_course_reference.py data/13/trajectory/custom_snap_ref_traj_13.npy --config data/13/config/config.yaml
python3 plot_obstacle_course_reference.py data/15_5/trajectory/custom_snap_ref_traj_15_5.npy --config data/15_5/config/config.yaml
python3 plot_obstacle_course_reference.py data/20_5/trajectory/custom_snap_ref_traj_20_5.npy --config data/20_5/config/config.yaml
```

Each plot command generates:
- a 3D trajectory plot
- 2D `xy`, `xz`, and `yz` projection plots with a color gradient showing progress along the trajectory

### Run PID test scripts

```bash
./pid_tests/run_pid_test_13.sh
./pid_tests/run_pid_test_15_5.sh
./pid_tests/run_pid_test_20_5.sh
```

These run the PID controller against the corresponding reference `.npy` files and save the resulting figures in the per-time `pid_tests` folders.

## Folder Structure

The generated artifacts are organized by trajectory duration:

- `data/13/`
- `data/15_5/`
- `data/20_5/`

Each time folder contains:

- `config/`
	- `config.yaml`: canonical config used by the generator and plotting scripts
	- `final_trajectory_prototype_*.yaml`: versioned YAML snapshot from the generator script
- `trajectory/`
	- `custom_snap_ref_traj_*.npy`: generated reference trajectory used by the PID controller
- `plots/`
	- `*_3d.png`: 3D trajectory plot
	- `*_xy.png`: 2D XY projection
	- `*_xz.png`: 2D XZ projection
	- `*_yz.png`: 2D YZ projection
- `warmstart_data/`
	- generator warm-start YAML and figures from the obstacle-course builder
- `test_results/` or `pid_tests/<time>/plotted_results/`
	- PID run outputs and saved figures, depending on which runner you use

For PID test runs, the current layout is:

- `pid_tests/13/bash_script/`
- `pid_tests/13/plotted_results/`
- `pid_tests/15_5/bash_script/`
- `pid_tests/15_5/plotted_results/`
- `pid_tests/20_5/bash_script/`
- `pid_tests/20_5/plotted_results/`

The `bash_script` folder keeps the run script copy and log for each run, and `plotted_results` contains the generated figures from `pid_experiment.py`.

## Included Files

- `generate_obstacle_course_reference.py`: builds the `.npy` reference file from the YAML config.
- `plot_obstacle_course_reference.py`: renders the reference trajectory and optional obstacle overlay.
- `configs/quadrotor_3D_attitude_tracking.yaml`: fallback config if no generated trajectory config exists.
- `generator_scripts/obstacle_course_generator.py`: obstacle-course generator.
- `generator_scripts/obstacle_course_generator_v6_0.py`: copied obstacle-course generator variant.
- `generator_scripts/obstacle_course_generator_v6_1.py`: copied obstacle-course generator variant.
- `generator_scripts/obstacle_course_generator_v6_2.py`: copied obstacle-course generator variant.
- `pid_tests/run_pid_test_13.sh`: PID runner for the 13-second trajectory.
- `pid_tests/run_pid_test_15_5.sh`: PID runner for the 15.5-second trajectory.
- `pid_tests/run_pid_test_20_5.sh`: PID runner for the 20.5-second trajectory.

## Notes

- The plotting script now saves both 3D and 2D projection figures.
- The PID scripts are thin wrappers around `examples/pid/pid_experiment.py`.
- The default plot output directory is the `plots/` folder within the matching `data/<time>/` directory.
