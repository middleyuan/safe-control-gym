import os
import subprocess
import yaml
import numpy as np
import glob
import shutil
from datetime import datetime
import optuna

CONFIG_BASE = "/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/config_overrides/mpc_acados_quadrotor_3D_attitude_tracking_100.yaml"
EXPERIMENT_SCRIPT = "/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/mb_experiment.py"
RESULTS_DIR = "./mpc_acados/hpo_results"

N_TRIALS = 100
EXPERIMENT_CONFIG_PATH = "/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/config_overrides/mpc_acados_quadrotor_3D_attitude_tracking_100.yaml"

# Create a timestamped folder for this HPO run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RESULTS_DIR, f"hpo_run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Prepare log_rmse.txt file in the run_dir
log_rmse_path = os.path.join(run_dir, "log_rmse.txt")
with open(log_rmse_path, "w") as log_file:
    log_file.write("trial,rmse\n")

trial_results = []

def objective(trial):
    # Sample q_mpc and r_mpc using Optuna
    q_mpc = [
        trial.suggest_float("q_mpc_0", 10, 60),
        trial.suggest_float("q_mpc_1", 1, 5),
        trial.suggest_float("q_mpc_2", 10, 60),
        trial.suggest_float("q_mpc_3", 1, 5),
        trial.suggest_float("q_mpc_4", 10, 150),
        trial.suggest_float("q_mpc_5", 1, 5),
        trial.suggest_float("q_mpc_6", 5, 30),
        trial.suggest_float("q_mpc_7", 5, 30),
        trial.suggest_float("q_mpc_8", 5, 30),
        trial.suggest_float("q_mpc_9", 0.0001, 0.1),
        trial.suggest_float("q_mpc_10", 0.0001, 0.1),
        trial.suggest_float("q_mpc_11", 0.0001, 0.1),
        trial.suggest_float("q_mpc_12", 1, 10)
    ]
    r_mpc = [
        trial.suggest_float("r_mpc_0", 3, 30),
        trial.suggest_float("r_mpc_1", 3, 30),
        trial.suggest_float("r_mpc_2", 3, 30),
        trial.suggest_float("r_mpc_3", 3, 30)
    ]

    trial_number = trial.number + 1
    output_dir = f"{run_dir}/trial_history/trial_{trial_number}"
    os.makedirs(output_dir, exist_ok=True)
    trial_config_path = f"{run_dir}/trial_history/trial_{trial_number}/mpc_acados_trial_{trial_number}.yaml"
    with open(CONFIG_BASE) as f:
        cfg = yaml.safe_load(f)
    cfg['algo_config']['q_mpc'] = q_mpc
    cfg['algo_config']['r_mpc'] = r_mpc
    with open(trial_config_path, "w") as f:
        yaml.dump(cfg, f)

    shutil.copy(trial_config_path, EXPERIMENT_CONFIG_PATH)

    cmd = [
        "python", EXPERIMENT_SCRIPT,
        "mpc_acados", f"mpc_acados_trial_{trial_number}.yaml", str(trial_number), "", ""
    ]
    print(f"Running trial {trial_number}: q_mpc={q_mpc}, r_mpc={r_mpc}")
    subprocess.run(cmd, check=True)

    results_candidates = glob.glob("./mpc_acados/results/temp/seed*/")
    if results_candidates:
        latest_results = max(results_candidates, key=os.path.getmtime)
        for fname in os.listdir(latest_results):
            src = os.path.join(latest_results, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                shutil.copy(src, dst)
        print(f"Copied top-level files from {latest_results} to {output_dir}")
    else:
        print(f"No results folder found for trial {trial_number}")

    metrics_file = os.path.join(output_dir, "metrics.txt")
    rmse = None
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            for line in f:
                if line.startswith("rmse:"):
                    try:
                        rmse = float(line.split(":")[1].strip())
                    except Exception:
                        rmse = None
                    break

    trial_results.append({
        "trial": trial_number,
        "q_mpc": q_mpc,
        "r_mpc": r_mpc,
        "rmse": rmse
    })

    with open(log_rmse_path, "a") as log_file:
        log_file.write(f"{trial_number},{rmse if rmse is not None else 'NaN'}\n")

    # Optuna minimizes the objective, so return rmse (or a large value if not found)
    return rmse if rmse is not None else 1e6

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

# Summarize results
trial_results = [r for r in trial_results if r["rmse"] is not None]
trial_results.sort(key=lambda x: x["rmse"])
print("\n=== HPO Results (sorted by RMSE) ===")
for r in trial_results:
    print(f"Trial {r['trial']}: RMSE={r['rmse']:.4f}, q_mpc={r['q_mpc']}, r_mpc={r['r_mpc']}")

if trial_results:
    best = trial_results[0]
    print(f"\nBest trial: {best['trial']} with RMSE={best['rmse']:.4f}")
    print(f"Best q_mpc: {best['q_mpc']}")
    print(f"Best r_mpc: {best['r_mpc']}")
    best_trial_src = os.path.join(run_dir, "trial_history", f"trial_{best['trial']}")
    best_trial_dst = os.path.join(run_dir, "best_trial")
    if os.path.exists(best_trial_dst):
        shutil.rmtree(best_trial_dst)
    shutil.copytree(best_trial_src, best_trial_dst)
else:
    print("No valid RMSE results found.")