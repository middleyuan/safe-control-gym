"""
Plot RL convergence analysis for different methods.

This script creates convergence plots showing training progress and sample efficiency
for RL methods, optionally including domain randomization variants.

Usage:
    python plot_convergence.py [method] [--include-dr]
    
Arguments:
    method: 'ppo', 'sac', 'dppo', 'ppo_mpc', or 'all' (default: 'all')
    --include-dr: Include domain randomization variants (optional)
    
Examples:
    python plot_convergence.py all                    # Plot all RL methods
    python plot_convergence.py ppo                    # Plot only PPO
    python plot_convergence.py all --include-dr       # Include domain randomization
    python plot_convergence.py sac --include-dr       # SAC with DR comparison

Note: This script looks for training data in ./data/nominal/ directory structure.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors, STEPS_PER_SECOND, plotting_data_dir

# script dir
script_dir = Path(__file__).parent.resolve()
data_dir = plotting_data_dir(script_dir)
default_named_input_data_root = data_dir / 'Final_june'

def normalize_input_data_root(input_data_root):
    """Resolve a curated plotting input root relative to plotting/data."""
    root = Path(input_data_root).expanduser()
    if not root.is_absolute():
        root = data_dir / root
    return root.resolve()

def default_input_data_root():
    """Return the same default curated input root used by the data processor."""
    env_root = os.environ.get('SCG_PLOTTING_INPUT_ROOT')
    if env_root:
        return normalize_input_data_root(env_root)
    if default_named_input_data_root.exists():
        return default_named_input_data_root.resolve()
    return data_dir.resolve()

input_data_root = default_input_data_root()

def load_from_log_file(path):
    """Return x, y sequence data from the stat csv."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        # Labels
        xk, yk = [k.strip() for k in lines[0].strip().split(',')]
        # Values
        x, y = [], []
        for line in lines[1:]:
            data = line.strip().split(',')
            x.append(float(data[0].strip()))
            y.append(float(data[1].strip()))
        x = np.array(x)
        y = np.array(y)
        return xk, x, yk, y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, np.array([]), None, np.array([])

def load_rl_training_data(method_type='all', input_root=None):
    """Load RL training data from log files."""
    
    if input_root is None:
        input_root = input_data_root

    exp_name = "nominal"
    
    # Define data paths for each method
    data_paths = {
        "PPO": input_root / exp_name / "quadrotor_2D_attitude_ppo_data",
        "SAC": input_root / exp_name / "quadrotor_2D_attitude_sac_data",
        "DPPO": input_root / exp_name / "quadrotor_2D_attitude_dppo_data",
        "PPO-MPC": input_root / exp_name / "quadrotor_2D_attitude_ppo_mpc_data",
    }
    
    # Filter methods based on input
    if method_type != 'all':
        method_upper = method_type.upper()
        if method_upper == 'PPO_MPC':
            method_upper = 'PPO-MPC'
        
        if method_upper in data_paths:
            data_paths = {method_upper: data_paths[method_upper]}
        else:
            print(f"Warning: Method '{method_type}' not found. Available: {list(data_paths.keys())}")
            return {}
    
    # Load training data
    n_seeds = 10
    seeds = list(range(n_seeds))
    perf_data = defaultdict(lambda: defaultdict())
    
    print("Loading RL training data...")
    print(f"  Input root: {input_root}")
    for method in data_paths.keys():
        print(f"  Loading {method}...")
        method_paths = data_paths[method]
        if not isinstance(method_paths, list):
            method_paths = [method_paths]
        data_path = next((path for path in method_paths if path.exists()), method_paths[0])
        if not data_path.exists():
            print(f"    Warning: Data directory not found: {data_path}")
            if len(method_paths) > 1:
                print("    Checked alternatives:")
                for path in method_paths:
                    print(f"      {path}")
            continue
            
        for seed_dir in data_path.glob('seed*'):
            if seed_dir.is_dir():
                try:
                    # Extract seed number from 'seedX_...' format
                    seed_part = seed_dir.name.split('_')[0]  # Get 'seedX' part
                    seed = int(seed_part[4:])  # Extract number from 'seedX'
                    if seed in seeds:
                        base_path = seed_dir
                        
                        # Load training logs
                        logs_dir = base_path / "logs" / "stat_eval"
                        if logs_dir.exists():
                            xk, x, lk, l = load_from_log_file(logs_dir / "ep_length.log")
                            xk, x, yk, y = load_from_log_file(logs_dir / "ep_return.log")
                            xk, x, zk, z = load_from_log_file(logs_dir / "ep_return_std.log")
                            xk, x, yk, m = load_from_log_file(logs_dir / "rmse.log")
                            xk, x, yk, n = load_from_log_file(logs_dir / "rmse_std.log")
                            
                            if len(x) > 0:  # Only store if we have valid data
                                perf_data[method][seed] = {
                                    "data": x, 
                                    "ep_return": y, 
                                    "ep_return_std": z, 
                                    "rmse": m, 
                                    "rmse_std": n, 
                                    "ep_length": l
                                }
                                print(f"    Loaded seed {seed}: {len(x)} data points")
                except Exception as e:
                    print(f"    Error loading seed {seed_dir.name}: {e}")
    
    return perf_data

def load_gp_mpc_data():
    """Load GP-MPC convergence data."""
    gp_file = data_dir / 'gpmpc_acados_TP_hpo_convergence_results.npy'
    
    if gp_file.exists():
        try:
            gp_mpc_data = np.load(gp_file, allow_pickle=True).item()
            # Convert training steps to time (seconds)
            gp_mpc_data['train_steps'] = gp_mpc_data['train_steps'] / STEPS_PER_SECOND
            print(f"  Loaded GP-MPC data with {len(gp_mpc_data['train_steps'])} points")
            return gp_mpc_data
        except Exception as e:
            print(f"  Error loading GP-MPC data: {e}")
    else:
        print(f"  GP-MPC data not found: {gp_file}")
    
    return None

def load_domain_randomization_convergence(method_type='all'):
    """Load domain randomization training data if available."""
    # This would need to be implemented based on where DR training data is stored
    # For now, return empty dict
    print("Domain randomization convergence data not yet implemented")
    return {}

def create_convergence_plot(perf_data, gp_mpc_data=None, dr_data=None, include_dr=False):
    """Create convergence plot showing RMSE vs training time."""
    
    if not perf_data and not gp_mpc_data:
        print("No data to plot")
        return
    
    # Setup plot
    fig = plt.figure(figsize=(8, 2))
    
    # Plot parameters
    mean_fn = np.percentile
    perc = 80  # Use 80th percentile as the mean
    n_seeds = 10
    
    # Add reference line
    plt.axhline(y=0.1, linestyle='-.', color='k', alpha=0.7, label='RMSE=0.1m')
    
    # Plot GP-MPC if available
    if gp_mpc_data is not None:
        rmse_mean = mean_fn(gp_mpc_data['rmse'], perc, axis=0)
        rmse_std = gp_mpc_data['rmse'].std(axis=0)
        
        plt.plot(gp_mpc_data['train_steps'], rmse_mean, 
                color=plot_colors.get("GP-MPC", 'purple'), 
                linewidth=2, label='GP-MPC')
        plt.fill_between(gp_mpc_data['train_steps'], 
                        np.clip(rmse_mean - rmse_std, 0, 10), 
                        np.clip(rmse_mean + rmse_std, 0, 10), 
                        color=plot_colors.get("GP-MPC", 'purple'), alpha=0.25)
    
    # Plot RL methods
    legends = {
        "PPO": "PPO",
        "SAC": "SAC", 
        "DPPO": "DPPO",
        "PPO-MPC": "PPO-MPC"
    }
    
    for method in perf_data.keys():
        if not perf_data[method]:
            continue
            
        # Collect data from all seeds
        seeds_with_data = list(perf_data[method].keys())
        if not seeds_with_data:
            continue
            
        # Get data length from first seed
        first_seed = seeds_with_data[0]
        data_length = len(perf_data[method][first_seed]["data"])
        
        # Initialize array for all seeds
        temp = np.zeros((n_seeds, 6, data_length))
        valid_seeds = 0
        
        for seed in range(n_seeds):
            if seed in perf_data[method]:
                seed_data = perf_data[method][seed]
                if len(seed_data["data"]) == data_length:
                    temp[valid_seeds, 0, :] = seed_data["data"]
                    temp[valid_seeds, 1, :] = seed_data["ep_return"]
                    temp[valid_seeds, 2, :] = seed_data["ep_return_std"]
                    temp[valid_seeds, 3, :] = seed_data["rmse"]
                    temp[valid_seeds, 4, :] = seed_data["rmse_std"]
                    temp[valid_seeds, 5, :] = seed_data["ep_length"]
                    valid_seeds += 1
        
        if valid_seeds == 0:
            continue
            
        # Use only valid seeds
        temp = temp[:valid_seeds, :, :]
        
        # Calculate mean and std
        rmse_mean = mean_fn(temp[:, 3, :], perc, axis=0)
        rmse_lower = mean_fn(temp[:, 3, :] - temp[:, 4, :], perc, axis=0)
        rmse_upper = mean_fn(temp[:, 3, :] + temp[:, 4, :], perc, axis=0)
        
        # Convert training steps to time (seconds)
        h_axis = (temp[0, 0, :] + 1) / STEPS_PER_SECOND
        
        # Plot method
        plt.plot(h_axis, rmse_mean, 
                color=plot_colors.get(method, 'gray'), 
                linewidth=2, label=legends.get(method, method))
        plt.fill_between(h_axis, 
                        np.clip(rmse_lower, 0, 10),  
                        np.clip(rmse_upper, 0, 10), 
                        color=plot_colors.get(method, 'gray'), alpha=0.25)
        
        print(f"{method}: Final RMSE = {rmse_mean[-1]:.6f} ± {temp[:, 3, -1].std():.6f}")
    
    # Plot domain randomization data if available
    if include_dr and dr_data:
        # This would plot DR variants with dashed lines
        pass
    
    # Formatting
    plt.legend(ncol=2, loc='upper right')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training Time [s]", fontsize=12)
    plt.ylabel("RMSE [m]", fontsize=12)
    plt.title("Training Convergence Analysis", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable limits
    plt.ylim(0.005, 10)
    
    # Save plot
    convergence_dir = script_dir / 'convergence'
    convergence_dir.mkdir(exist_ok=True)
    
    dr_suffix = "_with_dr" if include_dr and dr_data else ""
    plot_name = f"convergence{dr_suffix}"
    
    plt.savefig(convergence_dir / f"{plot_name}.pdf", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(convergence_dir / f"{plot_name}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"Convergence plot saved as {convergence_dir / plot_name}")
    
    plt.show()
    return fig

def print_sample_efficiency_analysis(perf_data, gp_mpc_data=None):
    """Print sample efficiency analysis."""
    print("\n=== SAMPLE EFFICIENCY ANALYSIS ===")
    
    threshold_factor = 1.2
    
    # Analyze GP-MPC if data is available
    if gp_mpc_data is not None:
        method = "GP-MPC"
        final_rmse = gp_mpc_data['rmse'][:, -1]  # Final RMSE for all seeds
        rmse_ref = final_rmse.mean()
        threshold = threshold_factor * rmse_ref
        
        # Find when GP-MPC reaches threshold performance
        rmse_array = gp_mpc_data['rmse'].mean(axis=0)  # Average across seeds
        train_steps = gp_mpc_data['train_steps']
        mask = rmse_array < threshold
        if np.any(mask):
            time_to_threshold = train_steps[mask][0]  # Already in seconds
            steps_to_threshold = time_to_threshold * STEPS_PER_SECOND  # Convert back to steps for consistent reporting
            print(f"{method}: Reaches {threshold:.6f} RMSE at step {steps_to_threshold:.0f} ({time_to_threshold:.1f} s)")
        else:
            print(f"{method}: Did not reach threshold {threshold:.6f} RMSE")
        
        print(f"{method}: Final RMSE = {rmse_ref:.6f} ± {final_rmse.std():.6f}")
    
    for method in perf_data.keys():
        if not perf_data[method]:
            continue
            
        seeds_with_data = list(perf_data[method].keys())
        if not seeds_with_data:
            continue
            
        # Get final performance across seeds
        final_rmse = []
        for seed in seeds_with_data:
            seed_data = perf_data[method][seed]
            if len(seed_data["rmse"]) > 0:
                final_rmse.append(seed_data["rmse"][-1])
        
        if final_rmse:
            final_rmse = np.array(final_rmse)
            rmse_ref = final_rmse.mean()
            threshold = threshold_factor * rmse_ref
            
            # Find when method reaches threshold performance
            for seed in seeds_with_data:
                seed_data = perf_data[method][seed]
                rmse_array = np.array(seed_data["rmse"])
                mask = rmse_array < threshold
                if np.any(mask):
                    steps_to_threshold = seed_data["data"][mask][0]
                    time_to_threshold = steps_to_threshold / STEPS_PER_SECOND  # Convert to seconds
                    print(f"{method}: Reaches {threshold:.6f} RMSE at step {steps_to_threshold:.0f} ({time_to_threshold:.1f} s)")
                    break
            else:
                print(f"{method}: Did not reach threshold {threshold:.6f} RMSE")
            
            print(f"{method}: Final RMSE = {rmse_ref:.6f} ± {final_rmse.std():.6f}")

def main():
    """Main function to create convergence plots."""
    
    # Parse command line arguments
    method_type = 'all'
    include_dr = False
    global input_data_root
    
    positional_args = [
        arg for arg in sys.argv[1:]
        if not arg.startswith('--')
    ]

    if positional_args:
        method_type = positional_args[0].lower()
        if method_type not in ['ppo', 'sac', 'dppo', 'ppo_mpc', 'all']:
            print(f"Warning: Invalid method '{method_type}'. Using 'all'.")
            method_type = 'all'
    
    include_dr = '--include-dr' in sys.argv or '--domain-rand' in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith('--input-data-root='):
            input_data_root = normalize_input_data_root(arg.split('=', 1)[1])
    
    print(f"Method filter: {method_type}")
    print(f"Include domain randomization: {include_dr}")
    print(f"Input data root: {input_data_root}")
    print()
    
    # Load data
    perf_data = load_rl_training_data(method_type, input_data_root)
    gp_mpc_data = load_gp_mpc_data()
    
    dr_data = {}
    if include_dr:
        dr_data = load_domain_randomization_convergence(method_type)
    
    if not perf_data and not gp_mpc_data:
        print(f"No training data found. Please ensure training data exists in {input_data_root / 'nominal'}")
        print("Expected directory structure:")
        print("  <input-data-root>/nominal/quadrotor_2D_attitude_<method>_data/seed<N>_*/logs/stat_eval/")
        return
    
    # Create convergence plot
    create_convergence_plot(perf_data, gp_mpc_data, dr_data, include_dr)
    
    # Print analysis
    print_sample_efficiency_analysis(perf_data, gp_mpc_data)
    
    # Note: GP-MPC analysis is now included in print_sample_efficiency_analysis

if __name__ == '__main__':
    main()
