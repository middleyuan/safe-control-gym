import os
import subprocess
import yaml
import numpy as np
import glob
import shutil
from datetime import datetime  # <-- Add this import

CONFIG_BASE = "/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/config_overrides/mpc_acados_quadrotor_3D_attitude_tracking_100.yaml"
EXPERIMENT_SCRIPT = "/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/mb_experiment.py"
RESULTS_DIR = "./mpc_acados/hpo_results"

N_TRIALS = 5
EXPERIMENT_CONFIG_PATH = "/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/config_overrides/mpc_acados_quadrotor_3D_attitude_tracking_100.yaml"

# --- NEW: Create a timestamped folder for this HPO run ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RESULTS_DIR, f"hpo_run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# --- NEW: Prepare log_rmse.txt file in the run_dir ---
log_rmse_path = os.path.join(run_dir, "log_rmse.txt")
with open(log_rmse_path, "w") as log_file:
    log_file.write("trial,rmse\n")

trial_results = []

for trial in range(1, N_TRIALS + 1):
    # Sample q_mpc and r_mpc
    q_mpc = [
        round(np.random.uniform(10, 60), 2),
        round(np.random.uniform(1, 10), 2),
        round(np.random.uniform(10, 60), 2),
        round(np.random.uniform(1, 10), 2),
        round(np.random.uniform(50, 150), 2),
        round(np.random.uniform(1, 10), 2),
        round(np.random.uniform(10, 30), 2),
        round(np.random.uniform(10, 30), 2),
        round(np.random.uniform(10, 30), 2),
        0.001, 0.001, 0.001,
        round(np.random.uniform(1, 10), 2)
    ]
    r_mpc = [
        round(np.random.uniform(5, 30), 2),
        round(np.random.uniform(10, 30), 2),
        round(np.random.uniform(10, 30), 2),
        round(np.random.uniform(10, 30), 2)
    ]
    output_dir = f"{run_dir}/trial_history/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    # Prepare trial config
    trial_config_path = f"{run_dir}/trial_history/trial_{trial}/mpc_acados_trial_{trial}.yaml"
    with open(CONFIG_BASE) as f:
        cfg = yaml.safe_load(f)
    cfg['algo_config']['q_mpc'] = q_mpc
    cfg['algo_config']['r_mpc'] = r_mpc
    with open(trial_config_path, "w") as f:
        yaml.dump(cfg, f)

    # Copy to config_overrides for compatibility
    shutil.copy(trial_config_path, EXPERIMENT_CONFIG_PATH)


    cmd = [
        "python", EXPERIMENT_SCRIPT,
        "mpc_acados", f"mpc_acados_trial_{trial}.yaml", str(trial), "", ""
    ]
    print(f"Running trial {trial}: q_mpc={q_mpc}, r_mpc={r_mpc}")
    subprocess.run(cmd, check=True)

    # Find the latest results folder after the run
    results_candidates = glob.glob("./mpc_acados/results/temp/seed*/")
    if results_candidates:
        latest_results = max(results_candidates, key=os.path.getmtime)
        # Copy only files directly under the latest_results folder (ignore subdirectories)
        for fname in os.listdir(latest_results):
            src = os.path.join(latest_results, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src):  # only copy files (not directories)
                shutil.copy(src, dst)
        print(f"Copied top-level files from {latest_results} to {output_dir}")
    else:
        print(f"No results folder found for trial {trial}")

    # Parse RMSE from metrics.txt
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
        "trial": trial,
        "q_mpc": q_mpc,
        "r_mpc": r_mpc,
        "rmse": rmse
    })

    # --- NEW: Update log_rmse.txt after each trial ---
    with open(log_rmse_path, "a") as log_file:
        log_file.write(f"{trial},{rmse if rmse is not None else 'NaN'}\n")

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