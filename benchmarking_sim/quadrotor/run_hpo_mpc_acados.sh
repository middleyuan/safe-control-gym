#!/bin/bash
# filepath: /home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/run_hpo_mpc_acados.sh

# HPO for MPC Acados: tune q_mpc and r_mpc to minimize xyz RMSE and input state noise

CONFIG_BASE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/config_overrides/mpc_acados_quadrotor_3D_attitude_tracking_100.yaml"
EXPERIMENT_SCRIPT="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/mb_experiment.py"
RESULTS_DIR="./mpc_acados/hpo_results"
N_TRIALS=30

mkdir -p "$RESULTS_DIR"

for trial in $(seq 1 $N_TRIALS); do
    # Randomly sample q_mpc and r_mpc values (example ranges, adjust as needed)
    q1=$(python -c "import random; print(round(random.uniform(10, 60), 2))")
    q2=$(python -c "import random; print(round(random.uniform(1, 10), 2))")
    q3=$(python -c "import random; print(round(random.uniform(10, 60), 2))")
    q4=$(python -c "import random; print(round(random.uniform(1, 10), 2))")
    q5=$(python -c "import random; print(round(random.uniform(50, 150), 2))")
    q6=$(python -c "import random; print(round(random.uniform(1, 10), 2))")
    q7=$(python -c "import random; print(round(random.uniform(10, 30), 2))")
    q8=$(python -c "import random; print(round(random.uniform(10, 30), 2))")
    q9=$(python -c "import random; print(round(random.uniform(10, 30), 2))")
    q10=0.001
    q11=0.001
    q12=0.001
    q13=$(python -c "import random; print(round(random.uniform(1, 10), 2))")

    r1=$(python -c "import random; print(round(random.uniform(5, 30), 2))")
    r2=$(python -c "import random; print(round(random.uniform(10, 30), 2))")
    r3=$(python -c "import random; print(round(random.uniform(10, 30), 2))")
    r4=$(python -c "import random; print(round(random.uniform(10, 30), 2))")

    TRIAL_CONFIG="$RESULTS_DIR/mpc_acados_trial_${trial}.yaml"
    cp "$CONFIG_BASE" "$TRIAL_CONFIG"

    # Use Python to update q_mpc and r_mpc
    python - <<END
import yaml
with open("$TRIAL_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg['algo_config']['q_mpc'] = [$q1, $q2, $q3, $q4, $q5, $q6, $q7, $q8, $q9, $q10, $q11, $q12, $q13]
cfg['algo_config']['r_mpc'] = [$r1, $r2, $r3, $r4]
with open("$TRIAL_CONFIG", "w") as f:
    yaml.dump(cfg, f)
END

    cp "$TRIAL_CONFIG" "./config_overrides/"

    # Pass positional arguments as expected by mb_experiment.py
    python $EXPERIMENT_SCRIPT mpc_acados $trial "" "" # You can add more positional args if needed

    # Optionally, pass the seed and output_dir as environment variables or handle them in Python
done
echo "HPO finished. See $RESULTS_DIR/hpo_summary.txt for results."