#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_DIR="$REPO_ROOT/benchmarking_sim/quadrotor/obstacle_course_generation/data/13"
NPY_PATH="$DATA_DIR/trajectory/custom_snap_ref_traj_13.npy"
EP_LEN="13.0"

STAMP="$(date +%Y%m%d_%H%M%S)"
PID_TEST_DIR="$REPO_ROOT/benchmarking_sim/quadrotor/obstacle_course_generation/pid_tests/13"
BASH_SCRIPT_DIR="$PID_TEST_DIR/bash_script"
PLOTTED_RESULTS_DIR="$PID_TEST_DIR/plotted_results"
RESULT_DIR="$PLOTTED_RESULTS_DIR/pid_$STAMP"
mkdir -p "$BASH_SCRIPT_DIR" "$RESULT_DIR"

cp "$0" "$BASH_SCRIPT_DIR/run_pid_test_13_$STAMP.sh"

cd "$REPO_ROOT/examples/pid"

# PID Experiment for 13s obstacle reference.
python3 ./pid_experiment.py \
    --task quadrotor \
    --algo pid \
    --overrides \
        ./config_overrides/pid.yaml \
        ./config_overrides/quadrotor_3D_attitude/other/custom_snap_tracking.yaml \
    --kv_overrides \
        task_config.task_info.custom_snap_ref_traj="$NPY_PATH" \
        task_config.episode_len_sec="$EP_LEN" \
        plot=True \
        output_dir="$RESULT_DIR" \
    2>&1 | tee "$BASH_SCRIPT_DIR/run_pid_test_13_$STAMP.log"

echo "Saved PID test outputs in: $RESULT_DIR"
