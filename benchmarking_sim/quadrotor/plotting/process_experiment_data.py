"""
Comprehensive experiment data processor for Safe Control Gym benchmarking analysis.

This script processes experimental data for visualization and analysis, including:
• Robustness Analysis: Noise sensitivity (observation, process, parametric)
• Trajectory Analysis: Path tracking and error hull visualization  
• Generalization Analysis: Performance across different episode lengths
• Domain Randomization: RL training with environmental variations
• Convergence: GP-MPC HPO convergence curve for training plots

The script consolidates data processing that was previously scattered across multiple
files, making it easier to prepare data for all plotting scripts.

SUPPORTED CONTROLLERS:
Model-based: Linear MPC, Nonlinear MPC, GP-MPC, iLQR, LQR, Geometric Control, F-MPC
RL Methods:  PPO, SAC, DPPO, PPO-MPC (with optional domain randomization)

USAGE EXAMPLES:
# Process robustness data for all controllers and noise types
python process_experiment_data.py robustness obs_noise all
python process_experiment_data.py robustness proc_noise control  
python process_experiment_data.py robustness param rl

# Process trajectory data for path visualization
python process_experiment_data.py trajectory all
python process_experiment_data.py trajectory --controller=ppo

# Process generalization data for episode length analysis  
python process_experiment_data.py generalization all
python process_experiment_data.py generalization rl

# Process domain randomization data for RL methods
python process_experiment_data.py domain-randomization
python process_experiment_data.py domain-randomization --controller=sac

# Process convergence data for plot_convergence.py
python process_experiment_data.py convergence

# Process everything for specific controller
python process_experiment_data.py all --controller=linear_mpc_acados

COMMAND LINE SYNTAX:
python process_experiment_data.py <analysis_type> [noise_type] [method_filter] [--controller=<name>] [--input-data-root=<path>]

ARGUMENTS:
analysis_type:  'robustness', 'trajectory', 'generalization', 'domain-randomization', 'convergence', 'all'
noise_type:     'obs_noise', 'proc_noise', 'param' (required for robustness analysis)
method_filter:  'control', 'rl', 'all' (default: 'all')
--controller:   Process specific controller only (optional)
--input-data-root:
                Root for curated plotting inputs. This directory should contain
                folders such as obs_noise, proc_noise, param, generalization,
                nominal, and robustness_combo. Defaults to
                ./data/Final_june when present, otherwise ./data. Can also be
                set with SCG_PLOTTING_INPUT_ROOT.

OUTPUT:
Processed data files are saved to ./data/ directory in .npy format for use by plotting scripts.
"""

import os
import sys
from pathlib import Path
import ast
import numpy as np
import json
import pickle
from benchmarking_sim.quadrotor.benchmark_util.utils import tag_ctrl_list

# ==========================================
# CONFIGURATION
# ==========================================

# Script directory setup
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / 'data'
DEFAULT_INPUT_DATA_ROOT = DATA_DIR
DEFAULT_NAMED_INPUT_DATA_ROOT = DATA_DIR / 'Final_june'
INPUT_DATA_ROOT = None
PROCESSING_SUMMARIES = []

# Experiment configuration
MAX_SEED = 10
METRIC_FILE = 'metrics.txt'

# Failure thresholds for robustness analysis
RELATIVE_FAILURE_THRESHOLD = 200   # 200% performance degradation
ABSOLUTE_FAILURE_THRESHOLD = 0.25  # 0.25m RMSE

# Controller definitions
MODEL_BASED_CONTROLLERS = {
    'Linear MPC': 'linear_mpc_acados',
    'Nonlinear MPC': 'mpc_acados', 
    'GP-MPC': 'gpmpc_acados_TP',
    'iLQR': 'ilqr',
    'LQR': 'lqr',
    'Geometric Control': 'pid',
    'F-MPC': 'fmpc',
}

RL_CONTROLLERS = {
    'PPO': 'ppo',
    'SAC': 'sac',
    'DPPO': 'dppo',
    'PPO-MPC': 'ppo_mpc',
}

ALL_CONTROLLERS = {**MODEL_BASED_CONTROLLERS, **RL_CONTROLLERS}

RL_DATA_FOLDERS = {
    'ppo': 'quadrotor_2D_attitude_ppo_data',
    'sac': 'quadrotor_2D_attitude_sac_data',
    'dppo': 'quadrotor_2D_attitude_dppo_data',
    'ppo_mpc': 'quadrotor_2D_attitude_ppo_mpc_data',
}

# Noise scales for different analysis types
NOISE_SCALES = {
    'obs_noise': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                12, 14, 16, 18, 20, 25, 30, 35, 40,
                45, 50, 60, 70, 80, 90, 100, 110, 120,
                130, 140, 150, 160, 170, 180, 190, 200,],
    'proc_noise': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                12, 14, 16, 18, 20, 25, 30, 35, 40,
                45, 50, 60, 70, 80, 90, 100, 110, 120,
                130, 140, 150, 160, 170, 180, 190, 200,],
    'param': {
        'control': [0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
                     7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                     14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                     20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        'rl': [0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
                     7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                     14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                     20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    }
}

EPISODE_LENGTHS = [9, 10, 11, 12, 13, 14, 15]

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item') and hasattr(obj, 'shape'):  # numpy scalar or array
        if obj.shape == ():  # scalar
            return obj.item()
        else:  # array
            return obj.tolist()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def print_header(title, level=1):
    """Print formatted section headers."""
    if level == 1:
        print(f"\n{'='*80}")
        print(f"{title.upper()}")
        print(f"{'='*80}")
    elif level == 2:
        print(f"\n{'-'*60}")
        print(f"{title}")
        print(f"{'-'*60}")
    else:
        print(f"\n• {title}")

def print_summary(analysis_type, processed_count, details=None, record=True):
    """Print processing summary."""
    print_header(f"{analysis_type} Processing Complete")
    print(f"Successfully processed: {processed_count} controllers")
    if details:
        for key, value in details.items():
            print(f"{key}: {value}")
    if record:
        PROCESSING_SUMMARIES.append({
            'analysis_type': analysis_type,
            'processed_count': processed_count,
            'details': details or {},
        })

def print_final_summary_recap():
    """Replay all processing summaries at the end of the command."""
    if not PROCESSING_SUMMARIES:
        return

    print_header("Final Processing Recap")
    for summary in PROCESSING_SUMMARIES:
        print(f"\n{summary['analysis_type']}:")
        print(f"  Successfully processed: {summary['processed_count']} controllers")
        for key, value in summary['details'].items():
            print(f"  {key}: {value}")

def normalize_input_data_root(input_data_root):
    """Resolve an input root from an absolute or plotting-data-relative path."""
    root = Path(input_data_root).expanduser()
    if not root.is_absolute():
        root = DATA_DIR / root
    return root.resolve()

def default_input_data_root():
    """Return the default curated plotting input root for this workspace."""
    env_root = os.environ.get('SCG_PLOTTING_INPUT_ROOT')
    if env_root:
        return normalize_input_data_root(env_root)
    if DEFAULT_NAMED_INPUT_DATA_ROOT.exists():
        return DEFAULT_NAMED_INPUT_DATA_ROOT.resolve()
    return DEFAULT_INPUT_DATA_ROOT.resolve()

def set_input_data_root(input_data_root=None):
    """Set the curated plotting input root used by processing helpers."""
    global INPUT_DATA_ROOT
    INPUT_DATA_ROOT = (
        normalize_input_data_root(input_data_root)
        if input_data_root
        else default_input_data_root()
    )

def get_input_data_root():
    """Return the active curated plotting input root."""
    if INPUT_DATA_ROOT is None:
        set_input_data_root()
    return INPUT_DATA_ROOT

def resolve_input_data_path(*parts):
    """Resolve curated plotting inputs from the active input data root."""
    return get_input_data_root().joinpath(*parts)

def resolve_input_data_candidates(*parts):
    """Return input-root first, then data-root fallback when they differ."""
    candidates = [resolve_input_data_path(*parts)]
    data_dir_candidate = DATA_DIR.joinpath(*parts)
    if data_dir_candidate != candidates[0]:
        candidates.append(data_dir_candidate)
    return candidates

def resolve_model_based_results_path(controller_name, data_folder_name, noise_type=None):
    """Resolve curated model-based rollout inputs for a controller."""
    exp_name = noise_type or data_folder_name
    return get_input_data_root() / exp_name / controller_name

def list_run_folders(data_folder_path):
    """List run folders from either flat output or legacy temp output."""
    data_folder_path = Path(data_folder_path)
    direct_runs = [
        f for f in data_folder_path.iterdir()
        if f.is_dir() and (f / METRIC_FILE).exists()
    ]
    direct_runs.sort()
    if direct_runs:
        return direct_runs

    temp_dir = data_folder_path / 'temp'
    if temp_dir.exists():
        temp_runs = [f for f in temp_dir.iterdir() if f.is_dir()]
        temp_runs.sort()
        return temp_runs

    subfolders = [f for f in data_folder_path.iterdir() if f.is_dir()]
    subfolders.sort()
    return subfolders

# ==========================================
# DATA EXTRACTION FUNCTIONS
# ==========================================

def extract_rollouts(data_folder_path, controller_name):
    """Extract rollout data from experiment results."""
    print(f"  Extracting rollouts for {controller_name} from {data_folder_path}")
    
    if not os.path.exists(data_folder_path):
        print(f"    Warning: Data folder does not exist: {data_folder_path}")
        return None, None, None

    # Handle flat seed_timestamp folders and legacy temp/seed_timestamp folders.
    metrics = []
    traj_results = []
    timing_data = []
    
    for run_folder in list_run_folders(data_folder_path):
        metrics_file = run_folder / METRIC_FILE
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('rmse:'):  # Exact match for rmse, not exponentiated_rmse
                        rmse = float(line.split(': ')[1])
                        metrics.append(rmse)
                        break
        
        traj_files = [f for f in run_folder.iterdir() if f.name.startswith('traj_') and f.name.endswith('.npy')]
        if traj_files:
            traj_data = np.load(traj_files[0], allow_pickle=True)
            traj_results.append(traj_data)

        timing_files = [f for f in run_folder.iterdir() if 'timing' in f.name and f.name.endswith('.npy')]
        if timing_files:
            timing = np.load(timing_files[0], allow_pickle=True)
            timing_data.append(timing)
    
    return metrics, traj_results, timing_data

def extract_noise_level_data(data_folder_path, controller_name):
    """Extract data organized by noise levels from experiment results."""
    print(f"  Extracting noise-level data for {controller_name} from {data_folder_path}")
    
    if not os.path.exists(data_folder_path):
        print(f"    Warning: Data folder does not exist: {data_folder_path}")
        return {}

    noise_data = {}
    
    for run_folder in list_run_folders(data_folder_path):
        metrics_file = run_folder / METRIC_FILE
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                rmse = None
                noise_factor = None

                for line in lines:
                    if line.startswith('rmse:'):  # Exact match for rmse, not exponentiated_rmse
                        rmse = float(line.split(': ')[1])
                    elif 'noise_factor:' in line:
                        noise_factor = float(line.split(': ')[1])

                if rmse is not None and noise_factor is not None:
                    if noise_factor not in noise_data:
                        noise_data[noise_factor] = []
                    noise_data[noise_factor].append(rmse)
    
    return noise_data

# ==========================================
# ROBUSTNESS ANALYSIS FUNCTIONS
# ==========================================

def process_robustness_controller(controller_name, noise_type, is_rl=False):
    """Process robustness data for a specific controller and noise type."""
    print_header(f"Processing {controller_name} - {noise_type}", level=3)
    
    # Handle RL controllers - use nominal data for traditional robustness analysis
    if is_rl:
        return process_rl_robustness_data(controller_name, noise_type, use_domain_randomization=False)
    
    # Process model-based controllers with proper noise level extraction
    return process_model_based_robustness_data(controller_name, noise_type)

def process_model_based_robustness_data(controller_name, noise_type):
    """Process model-based controller robustness data with proper noise level separation."""
    # Check if the actual data folder exists first
    if noise_type == 'param':
        data_folder_name = f"results_param_quadrotor_2D_attitude"
    else:
        data_folder_name = f"results_{noise_type}_quadrotor_2D_attitude"
    
    data_folder_path = resolve_model_based_results_path(
        controller_name, data_folder_name, noise_type
    )
    
    if not data_folder_path.exists():
        print(f"    Warning: Data folder does not exist: {data_folder_path}")
        return None
    
    print(f"    Found data folder: {data_folder_path}")
    
    # Get all seed directories
    seed_dirs = [d for d in data_folder_path.iterdir() if d.is_dir() and d.name.startswith('seed_')]
    
    if not seed_dirs:
        print(f"    Warning: No seed directories found in {data_folder_path}")
        return None
    
    print(f"    Processing {len(seed_dirs)} seed directories")
    
    # Collect all noise level data across seeds
    all_noise_data = {}
    
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f"      Processing {seed_name}...")
        
        # Extract noise-level specific data from this seed
        noise_data = extract_noise_level_data(seed_dir, controller_name)
        
        # Merge into overall noise data
        for noise_factor, rmse_values in noise_data.items():
            if noise_factor not in all_noise_data:
                all_noise_data[noise_factor] = []
            all_noise_data[noise_factor].extend(rmse_values)
    
    if not all_noise_data:
        print(f"    Warning: No valid noise level data found for {controller_name}")
        return None
    
    # Process collected data
    noise_factors = sorted(all_noise_data.keys())
    rmse_means = []
    rmse_stds = []
    valid_noise_factors = []
    
    for noise_factor in noise_factors:
        rmse_values = all_noise_data[noise_factor]
        if len(rmse_values) > 0:
            rmse_mean = np.mean(rmse_values)
            rmse_std = np.std(rmse_values)
            rmse_means.append(rmse_mean)
            rmse_stds.append(rmse_std)
            valid_noise_factors.append(noise_factor)
            print(f"    Noise {noise_factor:6.2f}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f} ({len(rmse_values)} samples)")
    
    if not rmse_means:
        print(f"    Warning: No valid data processed for {controller_name}")
        return None
    
    # Calculate degradation relative to baseline (noise_factor = 0 or minimum)
    if valid_noise_factors[0] == 0:
        baseline_rmse = rmse_means[0]
    else:
        print(f"    Warning: No baseline (noise_factor=0) data found, using minimum RMSE")
        baseline_rmse = min(rmse_means)
    
    rmse_degradation_mean = [(rmse / baseline_rmse) * 100 for rmse in rmse_means]
    rmse_degradation_std = [(std / baseline_rmse) * 100 for std in rmse_stds]
    
    # Package results
    results = {
        'noise_factor': np.array(valid_noise_factors),
        'rmse_mean': np.array(rmse_means),
        'rmse_std': np.array(rmse_stds),
        'rmse_degradation_mean': np.array(rmse_degradation_mean),
        'rmse_degradation_std': np.array(rmse_degradation_std),
        'baseline_rmse': baseline_rmse,
        'controller': controller_name,
        'noise_type': noise_type,
        'total_samples': sum(len(all_noise_data[nf]) for nf in valid_noise_factors),
        'note': 'Model-based controller data with proper noise level separation'
    }
    
    return results

def process_rl_robustness_data(controller_name, noise_type, use_domain_randomization=False):
    """Process RL robustness data from either nominal or domain randomization structure."""
    
    if use_domain_randomization:
        print(f"    Processing RL controller using domain randomization data structure")
        data_source = 'robustness_combo'
    else:
        print(f"    Processing RL controller using nominal data structure")
        data_source = 'nominal'
    
    # Map controller names to data folder names
    folder_mapping = {
        'ppo': 'quadrotor_2D_attitude_ppo_data',
        'sac': 'quadrotor_2D_attitude_sac_data',
        'dppo': 'quadrotor_2D_attitude_dppo_data',
        'ppo_mpc': 'quadrotor_2D_attitude_ppo_mpc_data'  # if it exists
    }
    
    if controller_name not in folder_mapping:
        print(f"    Warning: Unknown RL controller: {controller_name}")
        return None
    
    controller_folder = folder_mapping[controller_name]
    data_path = resolve_input_data_path(data_source, controller_folder)
    
    print(f"    Looking in: {data_path}")
    
    if not data_path.exists():
        print(f"    Warning: RL data folder not found: {data_path}")
        return None
    
    # Get all seed directories
    seed_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('seed')]
    seed_dirs.sort()
    
    if not seed_dirs:
        print(f"    Warning: No seed directories found in {data_path}")
        return None
    
    print(f"    Found {len(seed_dirs)} seed directories")
    
    # Map noise types to file prefixes
    noise_type_mapping = {
        'obs_noise': 'ob',
        'proc_noise': 'ps', 
        'param': 'pm'
    }
    
    if noise_type not in noise_type_mapping:
        print(f"    Warning: Unknown noise type: {noise_type}")
        return None
    
    noise_prefix = noise_type_mapping[noise_type]
    
    # Collect all noise levels and their data
    noise_data = {}
    
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f"      Processing {seed_name}...")
        
        # Find all robust_metric files for this noise type
        for metric_file in seed_dir.glob(f'robust_metric_{noise_prefix}_*.npy'):
            noise_scale_str = metric_file.name.split('_')[-1].split('.')[0]
            
            try:
                noise_scale = float(noise_scale_str)
                metric_data = np.load(metric_file, allow_pickle=True).item()
                
                if noise_scale not in noise_data:
                    noise_data[noise_scale] = []
                
                # Extract RMSE data
                rmse = metric_data.get('average_rmse', np.nan)
                if not np.isnan(rmse):
                    noise_data[noise_scale].append(rmse)
                    
            except Exception as e:
                print(f"        Error loading {metric_file}: {e}")
    
    if not noise_data:
        print(f"    Warning: No valid RL robustness data found for {controller_name}")
        return None
    
    # Process collected data
    noise_factors = sorted(noise_data.keys())
    rmse_means = []
    rmse_stds = []
    valid_noise_factors = []
    
    for noise_factor in noise_factors:
        rmse_values = noise_data[noise_factor]
        if len(rmse_values) > 0:
            rmse_mean = np.mean(rmse_values)
            rmse_std = np.std(rmse_values)
            rmse_means.append(rmse_mean)
            rmse_stds.append(rmse_std)
            valid_noise_factors.append(noise_factor)
            print(f"    Noise {noise_factor:6.2f}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f} ({len(rmse_values)} samples)")
    
    if not rmse_means:
        print(f"    Warning: No valid data processed for {controller_name}")
        return None
    
    # Calculate degradation relative to baseline (noise_factor = 0 or minimum)
    if valid_noise_factors[0] == 0:
        baseline_rmse = rmse_means[0]
    else:
        print(f"    Warning: No baseline (noise_factor=0) data found, using minimum RMSE")
        baseline_rmse = min(rmse_means)
    
    rmse_degradation_mean = [(rmse / baseline_rmse) * 100 for rmse in rmse_means]
    rmse_degradation_std = [(std / baseline_rmse) * 100 for std in rmse_stds]
    
    # Package results
    results = {
        'noise_factor': np.array(valid_noise_factors),
        'rmse_mean': np.array(rmse_means),
        'rmse_std': np.array(rmse_stds),
        'rmse_degradation_mean': np.array(rmse_degradation_mean),
        'rmse_degradation_std': np.array(rmse_degradation_std),
        'baseline_rmse': baseline_rmse,
        'controller': controller_name,
        'noise_type': noise_type,
        'total_samples': sum(len(noise_data[nf]) for nf in valid_noise_factors),
        'note': 'RL controller data with proper noise level separation'
    }
    
    return results

def analyze_failure_points(processed_data, noise_type):
    """Analyze failure points for relative and absolute thresholds."""
    
    relative_failures = {}
    absolute_failures = {}
    
    print_header("Failure Analysis", level=2)
    
    # Calculate intersection testing range across all controllers (minimum range tested by all)
    max_testing_range = float('inf')
    for data in processed_data.values():
        if len(data['noise_factor']) > 0:
            max_testing_range = min(max_testing_range, np.max(data['noise_factor']))
    
    for method, data in processed_data.items():
        noise_factors = data['noise_factor']
        rmse_degradation = data['rmse_degradation_mean']
        rmse_values = data['rmse_mean']
        
        # Check relative failure (200% degradation)
        relative_failure_found = False
        for i, (factor, degradation) in enumerate(zip(noise_factors, rmse_degradation)):
            if degradation > RELATIVE_FAILURE_THRESHOLD:
                relative_failures[method] = factor
                relative_failure_found = True
                print(f"  {method:15}: RELATIVE FAILURE at {factor:6.2f} ({degradation:6.1f}% degradation)")
                break
        
        if not relative_failure_found:
            # Use max testing range when no failure occurs
            relative_failures[method] = max_testing_range
            max_degradation = np.max(rmse_degradation)
            max_factor = noise_factors[np.argmax(rmse_degradation)]
            print(f"  {method:15}: No relative failure, max {max_degradation:6.1f}% at {max_factor:6.2f}")
        
        # Check absolute failure (0.25m RMSE)
        absolute_failure_found = False
        for i, (factor, rmse) in enumerate(zip(noise_factors, rmse_values)):
            if rmse > ABSOLUTE_FAILURE_THRESHOLD:
                absolute_failures[method] = factor
                absolute_failure_found = True
                print(f"  {method:15}: ABSOLUTE FAILURE at {factor:6.2f} (RMSE = {rmse:6.3f}m)")
                break
        
        if not absolute_failure_found:
            # Use max testing range when no failure occurs
            absolute_failures[method] = max_testing_range
            max_rmse = np.max(rmse_values)
            max_factor = noise_factors[np.argmax(rmse_values)]
            print(f"  {method:15}: No absolute failure, max RMSE {max_rmse:6.3f}m at {max_factor:6.2f}")
    
    return relative_failures, absolute_failures

def process_robustness_analysis(noise_type, method_filter='all'):
    """Process robustness analysis for specified noise type and method filter."""
    print_header(f"Robustness Analysis: {noise_type.upper()}")
    
    processed_data = {}
    attempted_controllers = []
    
    # Process model-based controllers
    if method_filter in ['control', 'all']:
        print_header("Model-based Controllers", level=2)
        for display_name, folder_name in MODEL_BASED_CONTROLLERS.items():
            attempted_controllers.append(display_name)
            results = process_robustness_controller(folder_name, noise_type, is_rl=False)
            if results is not None:
                processed_data[display_name] = results
                
                # Save individual controller data
                save_path = DATA_DIR / f'{folder_name}_{noise_type}_results.npy'
                np.save(save_path, results)
                print(f"    Saved: {save_path}")
    
    # Process RL controllers  
    if method_filter in ['rl', 'all']:
        print_header("RL Controllers", level=2)
        for display_name, folder_name in RL_CONTROLLERS.items():
            attempted_controllers.append(display_name)
            results = process_robustness_controller(folder_name, noise_type, is_rl=True)
            if results is not None:
                processed_data[display_name] = results
                
                # Save individual controller data
                save_path = DATA_DIR / f'{folder_name}_{noise_type}_results.npy'
                np.save(save_path, results)
                print(f"    Saved: {save_path}")
    
    if not processed_data:
        print("  Error: No data was processed successfully")
        print_summary("Robustness Analysis", 0, {
            'Noise type': noise_type,
            'Method filter': method_filter,
            'Processed controllers': 'none',
            'Skipped controllers': ', '.join(attempted_controllers) or 'none',
        })
        return
    
    # Calculate intersection testing range (minimum range tested by all controllers - both nominal and DR)
    max_testing_range = float('inf')
    for data in processed_data.values():
        if len(data['noise_factor']) > 0:
            max_testing_range = min(max_testing_range, np.max(data['noise_factor']))
    
    # Also consider domain randomization data ranges if available
    try:
        consolidated_dr_file = DATA_DIR / 'domain_randomization_robustness.json'
        if consolidated_dr_file.exists():
            with open(consolidated_dr_file, 'r') as f:
                dr_data = json.load(f)
            
            # Calculate minimum range from domain randomization data
            for controller_name, controller_data in dr_data.items():
                if noise_type in controller_data:
                    noise_data = controller_data[noise_type]
                    if noise_data:  # Check if there's any data
                        dr_max_range = max(float(scale) for scale in noise_data.keys())
                        max_testing_range = min(max_testing_range, dr_max_range)
                        print(f"    Domain randomization {controller_name} max range: {dr_max_range}")
    
    except Exception as e:
        print(f"    Could not load domain randomization data for range calculation: {e}")
    
    print(f"    Common intersection testing range (nominal + DR): {max_testing_range}")
    
    # Analyze failure points
    relative_failures, absolute_failures = analyze_failure_points(processed_data, noise_type)
    
    # Save failure results
    save_robustness_failure_results(relative_failures, absolute_failures, noise_type, method_filter, max_testing_range)
    
    # Print summary
    print_summary("Robustness Analysis", len(processed_data), {
        'Noise type': noise_type,
        'Method filter': method_filter,
        'Processed controllers': ', '.join(processed_data.keys()) or 'none',
        'Skipped controllers': ', '.join(
            name for name in attempted_controllers if name not in processed_data
        ) or 'none',
        'Relative failures': len(relative_failures),
        'Absolute failures': len(absolute_failures),
        'Max testing range': max_testing_range
    })

def extract_domain_randomization_failure_points(dr_data, noise_type, max_testing_range):
    """Extract failure points from domain randomization data for a specific noise type."""
    
    dr_failures = {'relative': {}, 'absolute': {}}
    
    # Map controller names from DR data to display names
    controller_mapping = {
        'ppo': 'PPO',
        'sac': 'SAC', 
        'dppo': 'DPPO'
    }
    
    for controller_name, controller_data in dr_data.items():
        if controller_name in controller_mapping:
            display_name = controller_mapping[controller_name]
            
            if noise_type in controller_data:
                noise_data = controller_data[noise_type]
                
                # Sort scales as numbers for proper comparison
                sorted_scales = sorted(noise_data.keys(), key=lambda x: float(x))
                
                # Find relative failure point (200% degradation from baseline)
                baseline_rmse = None
                relative_failure_found = False
                absolute_failure_found = False
                
                for scale_str in sorted_scales:
                    scale = float(scale_str)
                    metrics = noise_data[scale_str]
                    
                    if 'rmse' in metrics:
                        rmse_data = metrics['rmse']
                        if isinstance(rmse_data, dict) and 'mean' in rmse_data:
                            rmse_value = rmse_data['mean']
                        else:
                            rmse_value = rmse_data
                        
                        # Set baseline from first (lowest noise) measurement
                        if baseline_rmse is None:
                            baseline_rmse = rmse_value
                        
                        # Check for relative failure (200% degradation)
                        if not relative_failure_found and rmse_value > (baseline_rmse * RELATIVE_FAILURE_THRESHOLD / 100):
                            dr_failures['relative'][display_name] = scale
                            relative_failure_found = True
                            print(f"  {display_name:15}: RELATIVE DR FAILURE at {scale:6.2f} (RMSE = {rmse_value:6.3f}m, baseline = {baseline_rmse:6.3f}m)")
                        
                        # Check for absolute failure (0.25m RMSE)
                        if not absolute_failure_found and rmse_value > ABSOLUTE_FAILURE_THRESHOLD:
                            dr_failures['absolute'][display_name] = scale
                            absolute_failure_found = True
                            print(f"  {display_name:15}: ABSOLUTE DR FAILURE at {scale:6.2f} (RMSE = {rmse_value:6.3f}m)")
                            
                        # If both failures found, break early
                        if relative_failure_found and absolute_failure_found:
                            break
                
                # If no failures found, use common intersection testing range (not individual max scale)
                if not relative_failure_found:
                    dr_failures['relative'][display_name] = max_testing_range
                    print(f"  {display_name:15}: No relative DR failure, using common max range {max_testing_range:6.2f}")
                
                if not absolute_failure_found:
                    dr_failures['absolute'][display_name] = max_testing_range
                    print(f"  {display_name:15}: No absolute DR failure, using common max range {max_testing_range:6.2f}")
    
    return dr_failures

def save_robustness_failure_results(relative_failures, absolute_failures, noise_type, method_filter, max_testing_range):
    """Save robustness failure results to JSON files."""
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Load domain randomization failure points if available
    dr_failures = {'relative': {}, 'absolute': {}}
    try:
        # Load domain randomization data if it exists
        consolidated_dr_file = DATA_DIR / 'domain_randomization_robustness.json'
        if consolidated_dr_file.exists():
            with open(consolidated_dr_file, 'r') as f:
                dr_data = json.load(f)
            
            print_header(f"Loading Domain Randomization Failures for {noise_type}", level=3)
            dr_failures = extract_domain_randomization_failure_points(dr_data, noise_type, max_testing_range)
            
    except Exception as e:
        print(f"  Warning: Could not load domain randomization data: {e}")
    
    # Prepare failure results with separate sections for traditional and domain randomization data
    failure_data = {
        # Traditional robustness analysis results
        'relative_failures': convert_numpy_types(relative_failures),
        'absolute_failures': convert_numpy_types(absolute_failures),
        
        # Domain randomization robustness results (separated)
        'domain_randomization_relative_failures': convert_numpy_types(dr_failures['relative']),
        'domain_randomization_absolute_failures': convert_numpy_types(dr_failures['absolute']),
        
        # Metadata
        'max_testing_range': convert_numpy_types(max_testing_range),
        'relative_threshold': RELATIVE_FAILURE_THRESHOLD,
        'absolute_threshold': ABSOLUTE_FAILURE_THRESHOLD,
        'noise_type': noise_type,
        'method_filter': method_filter
    }
    
    # Save individual noise type results
    output_file = DATA_DIR / f'failure_results_{noise_type}_{method_filter}.json'
    with open(output_file, 'w') as f:
        json.dump(failure_data, f, indent=2)
    
    print(f"  Failure results saved: {output_file}")
    
    # Load and update consolidated results
    consolidated_file = DATA_DIR / 'robustness_failure_points.json'
    
    if consolidated_file.exists():
        with open(consolidated_file, 'r') as f:
            consolidated_data = json.load(f)
        
        # Ensure the new domain randomization sections exist
        if 'domain_randomization_relative_failures' not in consolidated_data:
            consolidated_data['domain_randomization_relative_failures'] = {'obs_noise': {}, 'proc_noise': {}, 'param': {}}
        if 'domain_randomization_absolute_failures' not in consolidated_data:
            consolidated_data['domain_randomization_absolute_failures'] = {'obs_noise': {}, 'proc_noise': {}, 'param': {}}
    else:
        consolidated_data = {
            'relative_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}},
            'absolute_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}},
            'domain_randomization_relative_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}},
            'domain_randomization_absolute_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}}
        }
    
    # Update with traditional robustness results
    for method, failure_scale in relative_failures.items():
        consolidated_data['relative_failures'][noise_type][method] = convert_numpy_types(failure_scale)
    
    for method, failure_scale in absolute_failures.items():
        consolidated_data['absolute_failures'][noise_type][method] = convert_numpy_types(failure_scale)
    
    # Update with domain randomization results
    for method, failure_scale in dr_failures['relative'].items():
        consolidated_data['domain_randomization_relative_failures'][noise_type][method] = convert_numpy_types(failure_scale)
    
    for method, failure_scale in dr_failures['absolute'].items():
        consolidated_data['domain_randomization_absolute_failures'][noise_type][method] = convert_numpy_types(failure_scale)
    
    # Save consolidated data
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    print(f"  Consolidated failure results updated: {consolidated_file}")

def save_domain_randomization_robustness_data(controller_name, rob_dr_data):
    """Save domain randomization robustness data to JSON files."""
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Extract robustness data if it exists
    if 'robustness' not in rob_dr_data:
        print(f"    Warning: No robustness data found for {controller_name}")
        return
    
    robustness_data = rob_dr_data['robustness']
    
    # Convert to JSON-compatible format
    json_data = {}
    for noise_type, noise_data in robustness_data.items():
        json_data[noise_type] = {}
        for scale, metrics in noise_data.items():
            json_data[noise_type][str(scale)] = convert_numpy_types(metrics)
    
    # Save individual controller results
    output_file = DATA_DIR / f'domain_randomization_robustness_{controller_name}.json'
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"    Saved domain randomization robustness JSON: {output_file}")
    
    # Load and update consolidated domain randomization results
    consolidated_dr_file = DATA_DIR / 'domain_randomization_robustness.json'
    
    if consolidated_dr_file.exists():
        with open(consolidated_dr_file, 'r') as f:
            consolidated_dr_data = json.load(f)
    else:
        consolidated_dr_data = {}
    
    # Update with current controller's results
    consolidated_dr_data[controller_name] = json_data
    
    # Save consolidated domain randomization data
    with open(consolidated_dr_file, 'w') as f:
        json.dump(consolidated_dr_data, f, indent=2)
    
    print(f"    Consolidated domain randomization robustness updated: {consolidated_dr_file}")

def get_common_testing_range_for_noise(noise_type, dr_data):
    """Return the common testing range across processed nominal and DR data."""
    max_testing_range = float('inf')

    for controller_name in ALL_CONTROLLERS.values():
        processed_file = DATA_DIR / f'{controller_name}_{noise_type}_results.npy'
        if not processed_file.exists():
            continue

        try:
            processed_data = np.load(processed_file, allow_pickle=True).item()
            noise_factors = processed_data.get('noise_factor', [])
            if len(noise_factors) > 0:
                max_testing_range = min(max_testing_range, float(np.max(noise_factors)))
        except Exception as e:
            print(f"    Warning: Could not read {processed_file} for range calculation: {e}")

    for controller_data in dr_data.values():
        noise_data = controller_data.get(noise_type, {})
        if noise_data:
            dr_max_range = max(float(scale) for scale in noise_data.keys())
            max_testing_range = min(max_testing_range, dr_max_range)

    if max_testing_range == float('inf'):
        return 0.0
    return max_testing_range

def refresh_domain_randomization_failure_results(noise_types=None):
    """Refresh DR failure sections in robustness_failure_points.json."""
    if noise_types is None:
        noise_types = ['obs_noise', 'proc_noise', 'param']

    consolidated_dr_file = DATA_DIR / 'domain_randomization_robustness.json'
    if not consolidated_dr_file.exists():
        print(f"    No consolidated DR robustness file found: {consolidated_dr_file}")
        return

    with open(consolidated_dr_file, 'r') as f:
        dr_data = json.load(f)

    consolidated_file = DATA_DIR / 'robustness_failure_points.json'
    if consolidated_file.exists():
        with open(consolidated_file, 'r') as f:
            consolidated_data = json.load(f)
    else:
        consolidated_data = {
            'relative_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}},
            'absolute_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}},
            'domain_randomization_relative_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}},
            'domain_randomization_absolute_failures': {'obs_noise': {}, 'proc_noise': {}, 'param': {}}
        }

    consolidated_data.setdefault(
        'domain_randomization_relative_failures',
        {'obs_noise': {}, 'proc_noise': {}, 'param': {}}
    )
    consolidated_data.setdefault(
        'domain_randomization_absolute_failures',
        {'obs_noise': {}, 'proc_noise': {}, 'param': {}}
    )

    for noise_type in noise_types:
        max_testing_range = get_common_testing_range_for_noise(noise_type, dr_data)
        print_header(f"Refreshing Domain Randomization Failures for {noise_type}", level=3)
        dr_failures = extract_domain_randomization_failure_points(
            dr_data, noise_type, max_testing_range
        )

        consolidated_data['domain_randomization_relative_failures'][noise_type] = (
            convert_numpy_types(dr_failures['relative'])
        )
        consolidated_data['domain_randomization_absolute_failures'][noise_type] = (
            convert_numpy_types(dr_failures['absolute'])
        )

    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    print(f"    Refreshed DR failure results in: {consolidated_file}")

# ==========================================
# TRAJECTORY ANALYSIS FUNCTIONS
# ==========================================

def extract_trajectory_obs(data):
    """Return obs trajectories from supported trajectory result formats."""
    if isinstance(data, np.ndarray):
        if data.shape == () and hasattr(data.item(), 'keys'):
            return extract_trajectory_obs(data.item())
        return data

    if isinstance(data, dict):
        if 'obs' in data:
            return np.asarray(data['obs'])
        if 'trajs_data' in data and isinstance(data['trajs_data'], dict):
            return extract_trajectory_obs(data['trajs_data'])

    return None

def load_trajectory_npy(traj_file):
    """Load a trajectory .npy file and return obs, n_rollouts."""
    traj_data = np.load(traj_file, allow_pickle=True)
    obs_data = extract_trajectory_obs(traj_data)
    if obs_data is None:
        return None, 0
    n_rollouts = obs_data.shape[0] if obs_data.ndim > 0 else 0
    return obs_data, n_rollouts

def load_trajectory_pickle(traj_file):
    """Load a trajectory pickle file and return obs, n_rollouts."""
    with open(traj_file, 'rb') as f:
        traj_data = pickle.load(f)
    obs_data = extract_trajectory_obs(traj_data)
    if obs_data is None:
        return None, 0
    if obs_data.ndim == 2:
        obs_data = obs_data[np.newaxis, ...]
    n_rollouts = obs_data.shape[0] if obs_data.ndim > 0 else 0
    return obs_data, n_rollouts

def concatenate_trajectory_obs(obs_rollouts):
    """Concatenate trajectory arrays, padding variable rollout lengths with NaN."""
    try:
        return np.concatenate(obs_rollouts, axis=0)
    except ValueError:
        max_time = max(obs.shape[1] for obs in obs_rollouts)
        state_dim = max(obs.shape[2] for obs in obs_rollouts)
        padded_rollouts = []

        for obs in obs_rollouts:
            padded = np.full((obs.shape[0], max_time, state_dim), np.nan)
            padded[:, :obs.shape[1], :obs.shape[2]] = obs
            padded_rollouts.append(padded)

        return np.concatenate(padded_rollouts, axis=0)

def resolve_trajectory_npy_candidates(controller_name, file_prefix, episode_length):
    """Return likely aggregate trajectory .npy locations for staged datasets."""
    candidates = [
        *resolve_input_data_candidates('trajectory', 'nominal', f'traj_results_{file_prefix}_{episode_length}.npy'),
        *resolve_input_data_candidates(f'traj_results_{file_prefix}_{episode_length}.npy')
    ]

    if controller_name in RL_DATA_FOLDERS:
        controller_folder = RL_DATA_FOLDERS[controller_name]
        for data_type in ['nominal', 'generalization', 'robustness_combo']:
            candidates.extend(resolve_input_data_candidates(
                data_type, controller_folder, f'traj_results_{file_prefix}_{episode_length}.npy'
            ))

    return candidates

def resolve_trajectory_pickle_candidates(controller_name, episode_length):
    """Return per-run model-based trajectory pickle files for staged datasets."""
    candidates = []
    pickle_name = f'{controller_name}_data_quadrotor_traj_tracking.pkl'

    for episode_dir in resolve_input_data_candidates('generalization', controller_name, f'episode_{episode_length}'):
        if episode_dir.exists():
            candidates.extend(sorted(episode_dir.glob(f'*/{pickle_name}')))

    return candidates

def process_trajectory_controller(controller_name, episode_lengths=None):
    """Process trajectory data for a specific controller."""
    if episode_lengths is None:
        episode_lengths = EPISODE_LENGTHS
    
    print_header(f"Processing trajectory data: {controller_name}", level=3)
    
    trajectory_data = {}
    
    # Map controller names to file names
    file_mapping = {
        'ppo': 'ppo',
        'sac': 'sac', 
        'dppo': 'dppo',
        'ppo_mpc': 'ppo_mpc',
        'linear_mpc_acados': 'linear_mpc_acados_quadrotor_2D_attitude',
        'mpc_acados': 'mpc_acados_quadrotor_2D_attitude',
        'gpmpc_acados_TP': 'gpmpc_acados_TP_quadrotor_2D_attitude',
        'ilqr': 'ilqr_quadrotor_2D_attitude',
        'lqr': 'lqr_quadrotor_2D_attitude',
        'pid': 'pid_quadrotor_2D_attitude',
        'fmpc': 'fmpc_quadrotor_2D_attitude'
    }
    
    file_prefix = file_mapping.get(controller_name, controller_name)
    
    for episode_length in episode_lengths:
        trajectory_loaded = False
        for traj_file in resolve_trajectory_npy_candidates(controller_name, file_prefix, episode_length):
            if traj_file.exists():
                try:
                    obs_data, n_rollouts = load_trajectory_npy(traj_file)
                    if obs_data is None:
                        print(f"    Episode {episode_length}: No obs data in {traj_file}")
                        continue
                    
                    trajectory_data[f'episode_{episode_length}'] = {
                        'obs': obs_data,
                        'n_rollouts': n_rollouts,
                        'timestamp': None,
                        'source_file': str(traj_file)
                    }
                    
                    obs_shape = obs_data.shape if obs_data is not None else "No obs data"
                    print(f"    Episode {episode_length}: {obs_shape} from {traj_file.name}")
                    trajectory_loaded = True
                    break
                    
                except Exception as e:
                    print(f"    Episode {episode_length}: Error loading {traj_file} - {e}")

        if not trajectory_loaded:
            pickle_files = resolve_trajectory_pickle_candidates(controller_name, episode_length)
            obs_rollouts = []
            source_files = []
            for traj_file in pickle_files:
                try:
                    obs_data, _ = load_trajectory_pickle(traj_file)
                    if obs_data is not None:
                        obs_rollouts.append(obs_data)
                        source_files.append(str(traj_file))
                except Exception as e:
                    print(f"    Episode {episode_length}: Error loading {traj_file} - {e}")

            if obs_rollouts:
                try:
                    obs_data = concatenate_trajectory_obs(obs_rollouts)
                    trajectory_data[f'episode_{episode_length}'] = {
                        'obs': obs_data,
                        'n_rollouts': obs_data.shape[0],
                        'timestamp': None,
                        'source_file': source_files
                    }

                    print(
                        f"    Episode {episode_length}: {obs_data.shape} "
                        f"from {len(source_files)} rollout files"
                    )
                    trajectory_loaded = True
                except Exception as e:
                    print(f"    Episode {episode_length}: Error combining trajectory rollouts - {e}")
        
        if not trajectory_loaded:
            print(f"    Episode {episode_length}: No trajectory file found")
    
    return trajectory_data

def process_trajectory_analysis(method_filter='all', controller=None):
    """Process trajectory analysis for specified controllers."""
    print_header("Trajectory Analysis")
    
    if controller:
        controllers_to_process = [controller]
        print(f"  Processing specific controller: {controller}")
    else:
        if method_filter == 'control':
            controllers_to_process = list(MODEL_BASED_CONTROLLERS.values())
        elif method_filter == 'rl':
            controllers_to_process = list(RL_CONTROLLERS.values())
        else:  # 'all'
            controllers_to_process = list(ALL_CONTROLLERS.values())
        print(f"  Processing {method_filter} controllers: {len(controllers_to_process)} total")
    
    processed_count = 0
    processed_controllers = []
    skipped_controllers = []
    
    for controller_name in controllers_to_process:
        traj_data = process_trajectory_controller(controller_name)
        
        if traj_data:
            output_file = DATA_DIR / f'{controller_name}_trajectory_processed.npy'
            try:
                np.save(output_file, traj_data)
                print(f"    Saved: {output_file}")
                processed_count += 1
                processed_controllers.append(controller_name)
            except Exception as e:
                print(f"    Error saving trajectory data for {controller_name}: {e}")
                skipped_controllers.append(controller_name)
        else:
            print(f"    No trajectory data processed for {controller_name}")
            skipped_controllers.append(controller_name)
    
    print_summary("Trajectory Analysis", processed_count, {
        'Method filter': method_filter,
        'Processed controllers': ', '.join(processed_controllers) or 'none',
        'Skipped controllers': ', '.join(skipped_controllers) or 'none',
        'Episode lengths': len(EPISODE_LENGTHS)
    })

# ==========================================
# GENERALIZATION ANALYSIS FUNCTIONS
# ==========================================

def get_metric_value(metric_data, keys, default=np.nan):
    """Return the first available metric value from a list of possible keys."""
    for key in keys:
        if key in metric_data:
            return metric_data[key]
    return default

def as_scalar(value, default=np.nan):
    """Convert numpy/list scalar-like values to a float."""
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return default

def parse_metric_file(metric_file):
    """Parse key/value metrics files emitted by model-based rollouts."""
    metrics = {}
    with metric_file.open('r') as f:
        for line in f:
            if ': ' not in line:
                continue
            key, value = line.rstrip('\n').split(': ', 1)
            metrics[key] = value
    return metrics

def metric_float(metrics, key, default=np.nan):
    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return default

def metric_scalar(metrics, key, default=np.nan):
    try:
        value = ast.literal_eval(metrics[key])
        return as_scalar(value, default=default)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return default

def aggregate_model_based_generalization_results(controller_name, episode_lengths=None):
    """Build *_gen_results.npy from model-based rollout metrics in Results/generalization."""
    if episode_lengths is None:
        episode_lengths = EPISODE_LENGTHS

    results_root = resolve_model_based_results_path(
        controller_name, 'generalization', 'generalization'
    )
    if not results_root.exists():
        print(f"    Model-based generalization folder not found: {results_root}")
        return None

    results = {}
    inference_time_values = []

    for episode_length in episode_lengths:
        results_dir = results_root / f'episode_{episode_length}'
        if not results_dir.exists():
            print(f"    {controller_name} episode {episode_length}: rollout folder not found")
            return None

        rmse_values = []

        for run_folder in list_run_folders(results_dir):
            metrics_file = run_folder / METRIC_FILE
            if not metrics_file.exists():
                continue

            metrics = parse_metric_file(metrics_file)
            rmse = metric_float(metrics, 'rmse')
            inference_time = metric_scalar(metrics, 'avarage_inference_time')

            if not np.isnan(rmse):
                rmse_values.append(rmse)
            if not np.isnan(inference_time):
                inference_time_values.append(inference_time)

        if not rmse_values:
            print(f"    {controller_name} episode {episode_length}: no metrics found in {results_dir}")
            return None

        episode_key = f'_{episode_length}'
        results[episode_key] = {
            'mean_rmse': float(np.mean(rmse_values)),
            'std_rmse': float(np.std(rmse_values)),
            'num_seeds': len(rmse_values),
        }
        print(
            f"    {controller_name} episode {episode_length}: aggregated {len(rmse_values)} seeds, "
            f"RMSE = {results[episode_key]['mean_rmse']:.4f} ± {results[episode_key]['std_rmse']:.4f}"
        )

    results['inference_time'] = float(np.mean(inference_time_values)) if inference_time_values else 0.0
    results['inference_time_std'] = float(np.std(inference_time_values)) if inference_time_values else 0.0
    return results

def aggregate_gpmpc_generalization_results(episode_lengths=None, tag='hpo'):
    """Build gpmpc_acados_TP_gen_results.npy from GP-MPC rollout metrics."""
    return aggregate_model_based_generalization_results('gpmpc_acados_TP', episode_lengths)

def aggregate_rl_generalization_results(controller_name, episode_lengths=None, exp_names=None):
    """Build *_gen_results.npy for nominal RL controllers from transfer_metric files."""
    if episode_lengths is None:
        episode_lengths = EPISODE_LENGTHS
    if exp_names is None:
        exp_names = ['nominal']

    if controller_name not in RL_CONTROLLERS.values():
        return None

    controller_folder = RL_DATA_FOLDERS.get(controller_name)
    if controller_folder is None:
        print(f"    No RL data folder mapping for {controller_name}")
        return None

    data_path = None
    for exp_name in exp_names:
        candidate_path = resolve_input_data_path(exp_name, controller_folder)
        if candidate_path.exists():
            data_path = candidate_path
            break

    if data_path is None:
        data_path = resolve_input_data_path(exp_names[0], controller_folder)

    print(f"    Looking for nominal RL generalization-evaluation data in: {data_path}")

    if not data_path.exists():
        print(f"    Nominal RL data folder not found: {data_path}")
        return None

    seed_dirs = sorted(d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('seed'))
    if not seed_dirs:
        print(f"    No seed directories found in {data_path}")
        return None

    results = {}
    processed_episodes = 0

    for episode_length in episode_lengths:
        rmse_values = []
        rmse_std_values = []

        for seed_dir in seed_dirs:
            metric_file = seed_dir / f'transfer_metric_{episode_length}.npy'
            if not metric_file.exists():
                continue

            try:
                metric_data = np.load(metric_file, allow_pickle=True).item()
            except Exception as e:
                print(f"      Error loading {metric_file}: {e}")
                continue

            rmse = as_scalar(get_metric_value(metric_data, ['rmse', 'average_rmse', 'mean_rmse']))
            rmse_std = as_scalar(get_metric_value(metric_data, ['rmse_std', 'std_rmse']))

            if not np.isnan(rmse):
                rmse_values.append(rmse)
            if not np.isnan(rmse_std):
                rmse_std_values.append(rmse_std)

        if rmse_values:
            episode_key = f'_{episode_length}'
            results[episode_key] = {
                'mean_rmse': float(np.mean(rmse_values)),
                'std_rmse': float(np.mean(rmse_std_values)) if rmse_std_values else float(np.std(rmse_values)),
                'num_seeds': len(rmse_values),
            }
            processed_episodes += 1
            print(
                f"    Episode {episode_length}: aggregated {len(rmse_values)} seeds, "
                f"RMSE = {results[episode_key]['mean_rmse']:.4f} ± {results[episode_key]['std_rmse']:.4f}"
            )
        else:
            print(f"    Episode {episode_length}: no raw transfer metric data found")

    inference_time_values = []
    for seed_dir in seed_dirs:
        perf_file = seed_dir / 'perf_metric.npy'
        if not perf_file.exists():
            continue

        try:
            perf_data = np.load(perf_file, allow_pickle=True).item()
        except Exception as e:
            print(f"      Error loading {perf_file}: {e}")
            continue

        inference_time = as_scalar(
            get_metric_value(perf_data, ['average_inference_time', 'avarage_inference_time'])
        )
        if not np.isnan(inference_time):
            inference_time_values.append(inference_time)

    if inference_time_values:
        results['inference_time'] = float(np.mean(inference_time_values))
        results['inference_time_std'] = float(np.std(inference_time_values))
    else:
        results['inference_time'] = 0.0
        results['inference_time_std'] = 0.0

    if processed_episodes == 0:
        print(f"    No nominal RL generalization-evaluation data aggregated for {controller_name}")
        return None

    results['source_data_root'] = str(data_path)
    results['source_experiment'] = data_path.parent.name
    return results

def process_generalization_controller(controller_name, episode_lengths=None):
    """Process generalization data for a specific controller."""
    if episode_lengths is None:
        episode_lengths = EPISODE_LENGTHS
    
    print_header(f"Processing generalization data: {controller_name}", level=3)
    
    # Map controller names to display names and file names
    name_mapping = {
        'gpmpc_acados_TP': 'GP-MPC',
        'linear_mpc_acados': 'Linear MPC',
        'mpc_acados': 'Nonlinear MPC',
        'ilqr': 'iLQR',
        'lqr': 'LQR',
        'pid': 'Geometric Control',
        'fmpc': 'F-MPC',
        'ppo': 'PPO',
        'sac': 'SAC',
        'dppo': 'DPPO',
        'ppo_mpc': 'PPO-MPC'
    }
    
    display_name = name_mapping.get(controller_name, controller_name)
    ctrl_name = tag_ctrl_list.get(display_name, controller_name.lower())
    
    gen_file = DATA_DIR / f'{ctrl_name}_gen_results.npy'
    
    if controller_name in MODEL_BASED_CONTROLLERS.values():
        print(f"    Generating {gen_file.name} from model-based rollout metrics")
        gen_results = aggregate_model_based_generalization_results(controller_name, episode_lengths)
        if gen_results is not None:
            try:
                np.save(gen_file, gen_results)
                print(f"    Saved generated generalization file: {gen_file}")
            except Exception as e:
                print(f"    Error saving generated generalization data: {e}")
                return None
        elif gen_file.exists():
            print(f"    Raw model-based rollout data unavailable; falling back to existing file: {gen_file}")
        else:
            return None
    elif controller_name in RL_CONTROLLERS.values():
        print(f"    Generating {gen_file.name} from raw RL transfer metrics")
        gen_results = aggregate_rl_generalization_results(controller_name, episode_lengths)
        if gen_results is not None:
            try:
                np.save(gen_file, gen_results)
                print(f"    Saved generated generalization file: {gen_file}")
            except Exception as e:
                print(f"    Error saving generated generalization data: {e}")
                return None
        else:
            if gen_file.exists():
                print(f"    Raw RL data unavailable; falling back to existing file: {gen_file}")
            else:
                return None
    elif not gen_file.exists():
        print(f"    Generalization file not found: {gen_file}")
        return None
    
    try:
        gen_data = np.load(gen_file, allow_pickle=True).item()
        
        generalization_data = {
            'rmse': [],
            'rmse_std': [],
            'inference_time': gen_data.get('inference_time', 0.0),
            'inference_time_std': gen_data.get('inference_time_std', 0.0),
            'episode_lengths': episode_lengths
        }
        
        for episode_length in episode_lengths:
            episode_key = f'_{episode_length}'
            if episode_key in gen_data:
                rmse_mean = gen_data[episode_key]['mean_rmse']
                rmse_std = gen_data[episode_key]['std_rmse']
                generalization_data['rmse'].append(rmse_mean)
                generalization_data['rmse_std'].append(rmse_std)
                print(f"    Episode {episode_length}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")
            else:
                print(f"    Episode {episode_length}: No data found")
                generalization_data['rmse'].append(np.nan)
                generalization_data['rmse_std'].append(np.nan)
        
        generalization_data['rmse'] = np.array(generalization_data['rmse'])
        generalization_data['rmse_std'] = np.array(generalization_data['rmse_std'])
        
        return generalization_data
        
    except Exception as e:
        print(f"    Error loading generalization data: {e}")
        return None

def process_generalization_analysis(method_filter='all', controller=None):
    """Process generalization analysis for specified controllers."""
    print_header("Generalization Analysis")
    
    if controller:
        controllers_to_process = [controller]
        print(f"  Processing specific controller: {controller}")
    else:
        if method_filter == 'control':
            controllers_to_process = list(MODEL_BASED_CONTROLLERS.values())
        elif method_filter == 'rl':
            controllers_to_process = list(RL_CONTROLLERS.values())
        else:  # 'all'
            controllers_to_process = list(ALL_CONTROLLERS.values())
        print(f"  Processing {method_filter} controllers: {len(controllers_to_process)} total")
    
    processed_count = 0
    processed_controllers = []
    skipped_controllers = []
    
    for controller_name in controllers_to_process:
        gen_data = process_generalization_controller(controller_name)
        
        if gen_data is not None:
            output_file = DATA_DIR / f'{controller_name}_generalization_processed.npy'
            try:
                np.save(output_file, gen_data)
                print(f"    Saved: {output_file}")
                processed_count += 1
                processed_controllers.append(controller_name)
            except Exception as e:
                print(f"    Error saving generalization data for {controller_name}: {e}")
                skipped_controllers.append(controller_name)
        else:
            print(f"    No generalization data processed for {controller_name}")
            skipped_controllers.append(controller_name)
    
    print_summary("Generalization Analysis", processed_count, {
        'Method filter': method_filter,
        'Processed controllers': ', '.join(processed_controllers) or 'none',
        'Skipped controllers': ', '.join(skipped_controllers) or 'none',
        'Episode lengths': len(EPISODE_LENGTHS)
    })

# ==========================================
# DOMAIN RANDOMIZATION FUNCTIONS
# ==========================================

def process_domain_randomization_controller(controller_name, data_type='generalization'):
    """Process domain randomization data for RL controllers."""
    if controller_name not in RL_CONTROLLERS.values():
        print(f"    Domain randomization only available for RL methods, skipping {controller_name}")
        return None

    # PPO-MPC is treated as a nominal RL method, not a GEN/DR variant.
    if controller_name == 'ppo_mpc':
        print(f"    PPO-MPC doesn't have domain randomization setup, skipping")
        return None
    
    print_header(f"Processing domain randomization {data_type}: {controller_name}", level=3)
    
    # Map controller names to folder names
    folder_mapping = {
        'ppo': 'quadrotor_2D_attitude_ppo_data',
        'sac': 'quadrotor_2D_attitude_sac_data',
        'dppo': 'quadrotor_2D_attitude_dppo_data'
    }
    
    controller_folder = folder_mapping[controller_name]
    data_path = resolve_input_data_path(data_type, controller_folder)
    
    print(f"    Looking in: {data_path}")
    
    if not data_path.exists():
        print(f"    Domain randomization data folder not found: {data_path}")
        return None
    
    # Get all seed directories
    seed_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('seed')]
    seed_dirs.sort()
    
    if not seed_dirs:
        print(f"    No seed directories found in {data_path}")
        return None
    
    print(f"    Found {len(seed_dirs)} seed directories")
    
    domain_rand_data = {
        'generalization': {},
        'robustness': {
            'obs_noise': {},
            'proc_noise': {},
            'param': {}
        },
        'performance': []
    }
    
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f"      Processing {seed_name}...")
        
        try:
            # Process generalization-evaluation metrics from staged transfer files.
            for metric_file in seed_dir.glob('transfer_metric_*.npy'):
                metric_num = metric_file.name.split('_')[-1].split('.')[0]
                episode_length = int(metric_num)

                try:
                    metric_data = np.load(metric_file, allow_pickle=True).item()

                    if episode_length not in domain_rand_data['generalization']:
                        domain_rand_data['generalization'][episode_length] = {
                            'rmse': [], 'rmse_std': [], 'failure_rate': [],
                            'constraint_violation': [], 'rms_action_change': []
                        }

                    domain_rand_data['generalization'][episode_length]['rmse'].append(
                        metric_data.get('average_rmse', np.nan))
                    domain_rand_data['generalization'][episode_length]['rmse_std'].append(
                        metric_data.get('rmse_std', np.nan))
                    domain_rand_data['generalization'][episode_length]['failure_rate'].append(
                        metric_data.get('failure_rate', np.nan))
                    domain_rand_data['generalization'][episode_length]['constraint_violation'].append(
                        metric_data.get('average_constraint_violation', np.nan))
                    domain_rand_data['generalization'][episode_length]['rms_action_change'].append(
                        metric_data.get('rms_action_change', np.nan))

                except Exception as e:
                    print(f"        Error loading {metric_file}: {e}")
            
            # Process robustness metrics (robust_metric files)
            for noise_type in ['ob', 'ps', 'pm']:
                noise_type_key = {'ob': 'obs_noise', 'ps': 'proc_noise', 'pm': 'param'}[noise_type]
                
                for metric_file in seed_dir.glob(f'robust_metric_{noise_type}_*.npy'):
                    noise_scale_str = metric_file.name.split('_')[-1].split('.')[0]
                    
                    try:
                        noise_scale = float(noise_scale_str)
                        metric_data = np.load(metric_file, allow_pickle=True).item()
                        
                        if noise_scale not in domain_rand_data['robustness'][noise_type_key]:
                            domain_rand_data['robustness'][noise_type_key][noise_scale] = {
                                'rmse': [], 'rmse_std': [], 'failure_rate': [],
                                'constraint_violation': [], 'rms_action_change': []
                            }
                        
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['rmse'].append(
                            metric_data.get('average_rmse', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['rmse_std'].append(
                            metric_data.get('rmse_std', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['failure_rate'].append(
                            metric_data.get('failure_rate', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['constraint_violation'].append(
                            metric_data.get('average_constraint_violation', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['rms_action_change'].append(
                            metric_data.get('rms_action_change', np.nan))
                        
                    except Exception as e:
                        print(f"        Error loading {metric_file}: {e}")
            
            # Process performance metrics
            perf_file = seed_dir / 'perf_metric.npy'
            if perf_file.exists():
                try:
                    perf_data = np.load(perf_file, allow_pickle=True).item()
                    domain_rand_data['performance'].append(perf_data)
                except Exception as e:
                    print(f"        Error loading performance data: {e}")
        
        except Exception as e:
            print(f"      Error processing {seed_name}: {e}")
    
    # Convert lists to numpy arrays and compute statistics
    for episode_length in domain_rand_data['generalization']:
        for metric in domain_rand_data['generalization'][episode_length]:
            values = domain_rand_data['generalization'][episode_length][metric]
            domain_rand_data['generalization'][episode_length][metric] = {
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'values': np.array(values)
            }
    
    for noise_type_key in domain_rand_data['robustness']:
        for noise_scale in domain_rand_data['robustness'][noise_type_key]:
            for metric in domain_rand_data['robustness'][noise_type_key][noise_scale]:
                values = domain_rand_data['robustness'][noise_type_key][noise_scale][metric]
                domain_rand_data['robustness'][noise_type_key][noise_scale][metric] = {
                    'mean': np.nanmean(values),
                    'std': np.nanstd(values),
                    'values': np.array(values)
                }
    
    domain_rand_data['source_experiment'] = data_type
    domain_rand_data['source_data_root'] = str(data_path)
    print(f"    Successfully processed domain randomization data for {controller_name}")
    return domain_rand_data

def process_domain_randomization_analysis(controller=None):
    """Process domain randomization analysis for RL controllers."""
    print_header("Domain Randomization Analysis")
    
    if controller:
        if controller not in RL_CONTROLLERS.values():
            print(f"  Error: Domain randomization only available for RL controllers")
            print(f"  Available RL controllers: {list(RL_CONTROLLERS.values())}")
            return
        controllers_to_process = [controller]
        print(f"  Processing specific controller: {controller}")
    else:
        controllers_to_process = list(RL_CONTROLLERS.values())
        print(f"  Processing all RL controllers: {controllers_to_process}")
    
    processed_count = 0
    gen_processed_controllers = []
    dr_processed_controllers = []
    skipped_controllers = []
    
    for controller_name in controllers_to_process:
        controller_processed = False

        # Process generalization-trained RL variant.
        gen_data = process_domain_randomization_controller(controller_name, 'generalization')
        if gen_data is not None:
            output_file = DATA_DIR / f'{controller_name}_gen_generalization.npy'
            try:
                np.save(output_file, gen_data)
                print(f"    Saved GEN generalization data: {output_file}")

                legacy_output_file = DATA_DIR / f'{controller_name}_domain_rand_generalization.npy'
                np.save(legacy_output_file, gen_data)
                print(f"    Saved legacy generalization data: {legacy_output_file}")
                gen_processed_controllers.append(controller_name)
                controller_processed = True
            except Exception as e:
                print(f"    Error saving GEN generalization data: {e}")
        
        # Process robustness-combo/domain-randomized RL variant.
        dr_data = process_domain_randomization_controller(controller_name, 'robustness_combo')
        if dr_data is not None:
            try:
                generalization_output_file = DATA_DIR / f'{controller_name}_dr_generalization.npy'
                np.save(generalization_output_file, dr_data)
                print(f"    Saved DR generalization data: {generalization_output_file}")

                output_file = DATA_DIR / f'{controller_name}_domain_rand_robustness.npy'
                np.save(output_file, dr_data)
                print(f"    Saved robustness DR data: {output_file}")
                processed_count += 1
                dr_processed_controllers.append(controller_name)
                controller_processed = True
            except Exception as e:
                print(f"    Error saving robustness DR data: {e}")
            
            # Save domain randomization robustness data to JSON
            save_domain_randomization_robustness_data(controller_name, dr_data)

        if not controller_processed:
            skipped_controllers.append(controller_name)

    refresh_domain_randomization_failure_results()

    print_summary("Domain Randomization Analysis", processed_count, {
        'Data types': 'generalization + robustness',
        'GEN controllers': ', '.join(gen_processed_controllers) or 'none',
        'DR controllers': ', '.join(dr_processed_controllers) or 'none',
        'Skipped controllers': ', '.join(skipped_controllers) or 'none',
        'RL controllers only': True
    })


def process_gpmpc_convergence_data(tag='hpo'):
    """Generate GP-MPC convergence data from per-seed learning-curve CSV files."""
    print_header(f"Processing GP-MPC convergence data ({tag})", level=3)

    candidates = resolve_input_data_candidates(
        'gp_models', 'gpmpc_acados_TP', tag
    )
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        print("    GP-MPC convergence folder not found. Checked:")
        for path in candidates:
            print(f"      {path}")
        return None

    print(f"    Loading GP-MPC convergence CSVs from: {data_path}")
    seed_folders = sorted(
        [
            path for path in data_path.iterdir()
            if path.is_dir() and path.name.startswith('seed')
        ],
        key=lambda path: int(path.name.split('seed')[1].split('_')[0])
    )
    if not seed_folders:
        print(f"    No seed folders found in {data_path}")
        return None

    seed_data = []
    skipped = []
    for seed_folder in seed_folders:
        csv_file = seed_folder / 'figs' / 'rmse_error_learning_curve.csv'
        if not csv_file.exists():
            skipped.append(seed_folder.name)
            continue

        data = np.genfromtxt(csv_file, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            skipped.append(seed_folder.name)
            continue
        seed_data.append((seed_folder.name, data[:, :2]))

    if not seed_data:
        print("    No GP-MPC convergence CSVs could be loaded")
        return None

    max_epochs = max(data.shape[0] for _, data in seed_data)
    complete_seed_data = [
        data for _, data in seed_data
        if data.shape[0] == max_epochs and np.all(np.isfinite(data[:, 1]))
    ]
    early_stopped = [
        name for name, data in seed_data
        if data.shape[0] != max_epochs or not np.all(np.isfinite(data[:, 1]))
    ]

    if not complete_seed_data:
        print("    No complete GP-MPC convergence seeds found")
        return None

    merged_data = np.asarray(complete_seed_data, dtype=float)
    train_steps = np.rint(np.nanmean(merged_data[:, :, 0], axis=0)).astype(int)
    train_steps[0] = max(train_steps[0], 1)
    rmse_data = merged_data[:, :, 1]

    convergence_data = {
        'rmse': rmse_data,
        'rmse_mean': np.nanmean(rmse_data, axis=0),
        'rmse_std': np.nanstd(rmse_data, axis=0),
        'train_steps': train_steps,
        'train_steps_data': np.arange(max_epochs) * 20,
        'train_steps_seconds': train_steps / 60,
        'source_data_root': str(data_path),
        'complete_seeds': len(complete_seed_data),
        'skipped_seeds': skipped,
        'early_stopped_seeds': early_stopped,
    }

    print(
        f"    Loaded {len(complete_seed_data)} complete seeds "
        f"with {max_epochs} convergence points"
    )
    if skipped:
        print(f"    Skipped missing/invalid seeds: {', '.join(skipped)}")
    if early_stopped:
        print(f"    Ignored early-stopped seeds: {', '.join(early_stopped)}")

    return convergence_data


def process_convergence_analysis():
    """Process convergence data used by plot_convergence.py."""
    processed_count = 0
    processed = []
    skipped = []

    gpmpc_data = process_gpmpc_convergence_data('hpo')
    if gpmpc_data is not None:
        output_file = DATA_DIR / 'gpmpc_acados_TP_hpo_convergence_results.npy'
        np.save(output_file, gpmpc_data)
        print(f"    Saved GP-MPC convergence data: {output_file}")
        processed_count += 1
        processed.append('GP-MPC')
    else:
        skipped.append('GP-MPC')

    print_summary("Convergence Analysis", processed_count, {
        'Processed controllers': ', '.join(processed) or 'none',
        'Skipped controllers': ', '.join(skipped) or 'none',
    })

# ==========================================
# MAIN PROCESSING FUNCTIONS
# ==========================================

def process_single_controller(controller_name, analysis_types, noise_type=None, method_filter='all'):
    """Process specified analysis types for a single controller."""
    print_header(f"Processing {controller_name.upper()}")
    
    # Determine controller type
    is_model_based = controller_name in MODEL_BASED_CONTROLLERS.values()
    is_rl = controller_name in RL_CONTROLLERS.values()
    
    if not (is_model_based or is_rl):
        print(f"  Error: Unknown controller '{controller_name}'")
        print(f"  Available controllers: {list(ALL_CONTROLLERS.values())}")
        return
    
    # Check method filter compatibility
    if method_filter == 'control' and not is_model_based:
        print(f"  Skipping {controller_name} (not a model-based controller)")
        return
    elif method_filter == 'rl' and not is_rl:
        print(f"  Skipping {controller_name} (not an RL controller)")
        return
    
    # Process each requested analysis type
    if 'robustness' in analysis_types and noise_type:
        print(f"  Processing robustness data for {noise_type}...")
        result = process_robustness_controller(controller_name, noise_type, is_rl=is_rl)
        if result is not None:
            save_path = DATA_DIR / f'{controller_name}_{noise_type}_results.npy'
            np.save(save_path, result)
            print(f"    Saved: {save_path}")
    
    if 'trajectory' in analysis_types:
        traj_data = process_trajectory_controller(controller_name)
        if traj_data:
            save_path = DATA_DIR / f'{controller_name}_trajectory_processed.npy'
            np.save(save_path, traj_data)
            print(f"    Saved: {save_path}")
    
    if 'generalization' in analysis_types:
        gen_data = process_generalization_controller(controller_name)
        if gen_data is not None:
            save_path = DATA_DIR / f'{controller_name}_generalization_processed.npy'
            np.save(save_path, gen_data)
            print(f"    Saved: {save_path}")
    
    if 'domain-randomization' in analysis_types and is_rl:
        gen_data = process_domain_randomization_controller(controller_name, 'generalization')
        if gen_data is not None:
            save_path = DATA_DIR / f'{controller_name}_gen_generalization.npy'
            np.save(save_path, gen_data)
            print(f"    Saved: {save_path}")

            legacy_save_path = DATA_DIR / f'{controller_name}_domain_rand_generalization.npy'
            np.save(legacy_save_path, gen_data)
            print(f"    Saved: {legacy_save_path}")
        
        dr_data = process_domain_randomization_controller(controller_name, 'robustness_combo')
        if dr_data is not None:
            gen_save_path = DATA_DIR / f'{controller_name}_dr_generalization.npy'
            np.save(gen_save_path, dr_data)
            print(f"    Saved: {gen_save_path}")

            save_path = DATA_DIR / f'{controller_name}_domain_rand_robustness.npy'
            np.save(save_path, dr_data)
            print(f"    Saved: {save_path}")

            save_domain_randomization_robustness_data(controller_name, dr_data)
            refresh_domain_randomization_failure_results()

def print_usage():
    """Print usage information."""
    print(__doc__)

def parse_arguments():
    """Parse command line arguments."""
    args = sys.argv[1:]
    
    if not args or args[0] in ['-h', '--help']:
        print_usage()
        sys.exit(0)
    
    # Extract controller if specified
    controller = None
    input_data_root = None
    cleaned_args = []
    for arg in args:
        if arg.startswith('--controller='):
            controller = arg.split('=')[1]
        elif arg.startswith('--input-data-root='):
            input_data_root = arg.split('=', 1)[1]
        elif arg.startswith('--raw-results-root='):
            input_data_root = arg.split('=', 1)[1]
            print("Warning: --raw-results-root is deprecated; use --input-data-root.")
        else:
            cleaned_args.append(arg)

    set_input_data_root(input_data_root)
    
    if not cleaned_args:
        print("Error: No analysis type specified")
        print_usage()
        sys.exit(1)
    
    analysis_type = cleaned_args[0].lower()
    
    # Parse based on analysis type
    if analysis_type == 'robustness':
        if len(cleaned_args) < 2:
            print("Error: Robustness analysis requires noise type")
            print("Usage: python process_experiment_data.py robustness <noise_type> [method_filter]")
            sys.exit(1)
        
        noise_type = cleaned_args[1]
        method_filter = cleaned_args[2] if len(cleaned_args) > 2 else 'all'
        
        if noise_type not in ['obs_noise', 'proc_noise', 'param']:
            print(f"Error: Invalid noise type '{noise_type}'")
            print("Valid noise types: obs_noise, proc_noise, param")
            sys.exit(1)
        
        return analysis_type, noise_type, method_filter, controller
    
    elif analysis_type in ['trajectory', 'generalization']:
        method_filter = cleaned_args[1] if len(cleaned_args) > 1 else 'all'
        return analysis_type, None, method_filter, controller
    
    elif analysis_type == 'domain-randomization':
        return analysis_type, None, 'rl', controller

    elif analysis_type == 'convergence':
        return analysis_type, None, 'all', controller
    
    elif analysis_type == 'all':
        method_filter = cleaned_args[1] if len(cleaned_args) > 1 else 'all'
        return analysis_type, None, method_filter, controller
    
    else:
        print(f"Error: Invalid analysis type '{analysis_type}'")
        print("Valid analysis types: robustness, trajectory, generalization, domain-randomization, convergence, all")
        sys.exit(1)

def main():
    """Main processing function."""
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parse command line arguments
    analysis_type, noise_type, method_filter, controller = parse_arguments()

    print(f"Curated plotting input root: {get_input_data_root()}")
    print(f"Processed plotting output directory: {DATA_DIR}")
    
    # Validate method filter
    if method_filter not in ['control', 'rl', 'all']:
        print(f"Warning: Invalid method filter '{method_filter}'. Using 'all'.")
        method_filter = 'all'
    
    # Process based on analysis type and scope
    if analysis_type == 'convergence':
        if controller and controller != 'gpmpc_acados_TP':
            print(
                f"Warning: convergence processing is GP-MPC specific; "
                f"ignoring --controller={controller}."
            )
        process_convergence_analysis()
    elif controller:
        # Process specific controller
        if controller not in ALL_CONTROLLERS.values():
            print(f"Error: Unknown controller '{controller}'")
            print(f"Available controllers: {list(ALL_CONTROLLERS.values())}")
            sys.exit(1)
        
        if analysis_type == 'all':
            analysis_types = ['robustness', 'trajectory', 'generalization', 'domain-randomization', 'convergence']
            # For 'all' with specific controller, process all noise types for robustness
            for nt in ['obs_noise', 'proc_noise', 'param']:
                process_single_controller(controller, ['robustness'], nt, method_filter)
            # Process other analysis types
            for at in ['trajectory', 'generalization', 'domain-randomization']:
                process_single_controller(controller, [at], None, method_filter)
            process_convergence_analysis()
        else:
            analysis_types = [analysis_type]
            process_single_controller(controller, analysis_types, noise_type, method_filter)
    
    else:
        # Process all controllers for specified analysis type
        if analysis_type == 'robustness':
            process_robustness_analysis(noise_type, method_filter)
        elif analysis_type == 'trajectory':
            process_trajectory_analysis(method_filter)
        elif analysis_type == 'generalization':
            process_generalization_analysis(method_filter)
        elif analysis_type == 'domain-randomization':
            process_domain_randomization_analysis()
        elif analysis_type == 'convergence':
            process_convergence_analysis()
        elif analysis_type == 'all':
            # Process all analysis types
            for nt in ['obs_noise', 'proc_noise', 'param']:
                process_robustness_analysis(nt, method_filter)
            process_trajectory_analysis(method_filter)
            process_generalization_analysis(method_filter)
            if method_filter in ['rl', 'all']:
                process_domain_randomization_analysis()
            process_convergence_analysis()

    print_final_summary_recap()

if __name__ == '__main__':
    main()
