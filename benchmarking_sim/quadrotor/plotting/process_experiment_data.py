'''Comprehensive experiment data processor for Safe Control Gym benchmarking analysis.

This script processes experimental data for visualization and analysis, including:
• Robustness Analysis: Noise sensitivity (observation, process, parametric)
• Trajectory Analysis: Path tracking and error hull visualization
• Generalization Analysis: Performance across different episode lengths
• Domain Randomization: RL training with environmental variations

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

# Process everything for specific controller
python process_experiment_data.py all --controller=linear_mpc_acados

COMMAND LINE SYNTAX:
python process_experiment_data.py <analysis_type> [noise_type] [method_filter] [--controller=<name>]

ARGUMENTS:
analysis_type:  'robustness', 'trajectory', 'generalization', 'domain-randomization', 'all'
noise_type:     'obs_noise', 'proc_noise', 'param' (required for robustness analysis)
method_filter:  'control', 'rl', 'all' (default: 'all')
--controller:   Process specific controller only (optional)

OUTPUT:
Processed data files are saved to ../data/ directory in .npy format for use by plotting scripts.
'''

import json
import os
import sys
from pathlib import Path

import numpy as np

from benchmarking_sim.quadrotor.benchmark_util.utils import tag_ctrl_list

# ==========================================
# CONFIGURATION
# ==========================================

# Script directory setup
SCRIPT_DIR = Path(__file__).parent.resolve()
QUADROTOR_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / '../data'

# Experiment configuration
MAX_SEED = 10
METRIC_FILE = 'metrics.txt'

# Failure thresholds for robustness analysis
RELATIVE_FAILURE_THRESHOLD = 200  # 200% performance degradation
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

# Noise scales for different analysis types
NOISE_SCALES = {
    'obs_noise': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                  25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
    'proc_noise': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                   25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
    'param': {
        'control': [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
                    1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0],
        'rl': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    }
}

EPISODE_LENGTHS = [9, 10, 11, 12, 13, 14, 15]

# ==========================================
# UTILITY FUNCTIONS
# ==========================================


def convert_numpy_types(obj):
    '''Convert numpy types to Python native types for JSON serialization.'''
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
    '''Print formatted section headers.'''
    if level == 1:
        print(f'\n{"="*80}')
        print(f'{title.upper()}')
        print(f'{"="*80}')
    elif level == 2:
        print(f'\n{" - "*60}')
        print(f'{title}')
        print(f'{" - "*60}')
    else:
        print(f'\n• {title}')


def print_summary(analysis_type, processed_count, details=None):
    '''Print processing summary.'''
    print_header(f'{analysis_type} Processing Complete')
    print(f'Successfully processed: {processed_count} controllers')
    if details:
        for key, value in details.items():
            print(f'{key}: {value}')

# ==========================================
# DATA EXTRACTION FUNCTIONS
# ==========================================


def extract_rollouts(data_folder_path, controller_name):
    '''Extract rollout data from experiment results.'''
    print(f'  Extracting rollouts for {controller_name} from {data_folder_path}')

    if not os.path.exists(data_folder_path):
        print(f'    Warning: Data folder does not exist: {data_folder_path}')
        return None, None, None

    # Handle nested structure: seed_X/temp/seedX_timestamp/metrics.txt
    metrics = []
    traj_results = []
    timing_data = []

    # Check if this is a seed directory with temp subdirectory
    if data_folder_path.name.startswith('seed_'):
        temp_dir = data_folder_path / 'temp'
        if temp_dir.exists():
            # Find all experiment run folders in temp directory
            run_folders = [f for f in temp_dir.iterdir() if f.is_dir()]
            run_folders.sort()

            for run_folder in run_folders:
                # Extract metrics
                metrics_file = run_folder / METRIC_FILE
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith('rmse:'):  # Exact match for rmse, not exponentiated_rmse
                                rmse = float(line.split(': ')[1])
                                metrics.append(rmse)
                                break

                # Extract trajectory data
                traj_files = [f for f in run_folder.iterdir() if f.name.startswith('traj_') and f.name.endswith('.npy')]
                if traj_files:
                    traj_data = np.load(traj_files[0], allow_pickle=True)
                    traj_results.append(traj_data)

                # Extract timing data
                timing_files = [f for f in run_folder.iterdir() if 'timing' in f.name and f.name.endswith('.npy')]
                if timing_files:
                    timing = np.load(timing_files[0], allow_pickle=True)
                    timing_data.append(timing)
    else:
        # Original logic for flat structure
        subfolders = [f.path for f in os.scandir(data_folder_path) if f.is_dir()]
        subfolders.sort()

        for subfolder in subfolders:
            # Extract metrics
            file_path = os.path.join(subfolder, METRIC_FILE)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('rmse:'):  # Exact match for rmse, not exponentiated_rmse
                            rmse = float(line.split(': ')[1])
                            metrics.append(rmse)
                            break

            # Extract trajectory data
            traj_files = [f for f in os.listdir(subfolder) if f.startswith('traj_') and f.endswith('.npy')]
            if traj_files:
                traj_path = os.path.join(subfolder, traj_files[0])
                traj_data = np.load(traj_path, allow_pickle=True)
                traj_results.append(traj_data)

            # Extract timing data
            timing_files = [f for f in os.listdir(subfolder) if 'timing' in f and f.endswith('.npy')]
            if timing_files:
                timing_path = os.path.join(subfolder, timing_files[0])
                timing = np.load(timing_path, allow_pickle=True)
                timing_data.append(timing)

    return metrics, traj_results, timing_data


def extract_noise_level_data(data_folder_path, controller_name):
    '''Extract data organized by noise levels from experiment results.'''
    print(f'  Extracting noise-level data for {controller_name} from {data_folder_path}')

    if not os.path.exists(data_folder_path):
        print(f'    Warning: Data folder does not exist: {data_folder_path}')
        return {}

    noise_data = {}

    # Check if this is a seed directory with temp subdirectory
    if data_folder_path.name.startswith('seed_'):
        temp_dir = data_folder_path / 'temp'
        if temp_dir.exists():
            # Find all experiment run folders in temp directory
            run_folders = [f for f in temp_dir.iterdir() if f.is_dir()]
            run_folders.sort()

            for run_folder in run_folders:
                # Extract metrics and noise factor
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
    '''Process robustness data for a specific controller and noise type.'''
    print_header(f'Processing {controller_name} - {noise_type}', level=3)

    # Handle RL controllers - use nominal data for traditional robustness analysis
    if is_rl:
        return process_rl_robustness_data(controller_name, noise_type, use_domain_randomization=False)

    # Process model-based controllers with proper noise level extraction
    return process_model_based_robustness_data(controller_name, noise_type)


def process_model_based_robustness_data(controller_name, noise_type):
    '''Process model-based controller robustness data with proper noise level separation.'''
    # Check if the actual data folder exists first
    if noise_type == 'param':
        data_folder_name = 'results_param_quadrotor_2D_attitude'
    else:
        data_folder_name = f'results_{noise_type}_quadrotor_2D_attitude'

    data_folder_path = QUADROTOR_DIR / controller_name / data_folder_name

    if not data_folder_path.exists():
        print(f'    Warning: Data folder does not exist: {data_folder_path}')
        return None

    print(f'    Found data folder: {data_folder_path}')

    # Get all seed directories
    seed_dirs = [d for d in data_folder_path.iterdir() if d.is_dir() and d.name.startswith('seed_')]

    if not seed_dirs:
        print(f'    Warning: No seed directories found in {data_folder_path}')
        return None

    print(f'    Processing {len(seed_dirs)} seed directories')

    # Collect all noise level data across seeds
    all_noise_data = {}

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f'      Processing {seed_name}...')

        # Extract noise-level specific data from this seed
        noise_data = extract_noise_level_data(seed_dir, controller_name)

        # Merge into overall noise data
        for noise_factor, rmse_values in noise_data.items():
            if noise_factor not in all_noise_data:
                all_noise_data[noise_factor] = []
            all_noise_data[noise_factor].extend(rmse_values)

    if not all_noise_data:
        print(f'    Warning: No valid noise level data found for {controller_name}')
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
            print(f'    Noise {noise_factor:6.2f}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f} ({len(rmse_values)} samples)')

    if not rmse_means:
        print(f'    Warning: No valid data processed for {controller_name}')
        return None

    # Calculate degradation relative to baseline (noise_factor = 0 or minimum)
    if valid_noise_factors[0] == 0:
        baseline_rmse = rmse_means[0]
    else:
        print('    Warning: No baseline (noise_factor=0) data found, using minimum RMSE')
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
    '''Process RL robustness data from either nominal or domain randomization structure.'''

    if use_domain_randomization:
        print('    Processing RL controller using domain randomization data structure')
        data_source = 'robustness_pm'
    else:
        print('    Processing RL controller using nominal data structure')
        data_source = 'nominal'

    # Map controller names to data folder names
    folder_mapping = {
        'ppo': 'quadrotor_2D_attitude_ppo_data',
        'sac': 'quadrotor_2D_attitude_sac_data' if use_domain_randomization else 'quadrotor_2D_attitude_sac_data2',
        'dppo': 'quadrotor_2D_attitude_dppo_data',
        'ppo_mpc': 'quadrotor_2D_attitude_ppo_mpc_data'  # if it exists
    }

    if controller_name not in folder_mapping:
        print(f'    Warning: Unknown RL controller: {controller_name}')
        return None

    controller_folder = folder_mapping[controller_name]
    data_path = QUADROTOR_DIR / 'data' / data_source / controller_folder

    print(f'    Looking in: {data_path}')

    if not data_path.exists():
        print(f'    Warning: RL data folder not found: {data_path}')
        return None

    # Get all seed directories
    seed_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('seed')]
    seed_dirs.sort()

    if not seed_dirs:
        print(f'    Warning: No seed directories found in {data_path}')
        return None

    print(f'    Found {len(seed_dirs)} seed directories')

    # Map noise types to file prefixes
    noise_type_mapping = {
        'obs_noise': 'ob',
        'proc_noise': 'ps',
        'param': 'pm'
    }

    if noise_type not in noise_type_mapping:
        print(f'    Warning: Unknown noise type: {noise_type}')
        return None

    noise_prefix = noise_type_mapping[noise_type]

    # Collect all noise levels and their data
    noise_data = {}

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f'      Processing {seed_name}...')

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
                print(f'        Error loading {metric_file}: {e}')

    if not noise_data:
        print(f'    Warning: No valid RL robustness data found for {controller_name}')
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
            print(f'    Noise {noise_factor:6.2f}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f} ({len(rmse_values)} samples)')

    if not rmse_means:
        print(f'    Warning: No valid data processed for {controller_name}')
        return None

    # Calculate degradation relative to baseline (noise_factor = 0 or minimum)
    if valid_noise_factors[0] == 0:
        baseline_rmse = rmse_means[0]
    else:
        print('    Warning: No baseline (noise_factor=0) data found, using minimum RMSE')
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
    '''Analyze failure points for relative and absolute thresholds.'''

    relative_failures = {}
    absolute_failures = {}

    print_header('Failure Analysis', level=2)

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
                print(f'  {method:15}: RELATIVE FAILURE at {factor:6.2f} ({degradation:6.1f}% degradation)')
                break

        if not relative_failure_found:
            # Use max testing range when no failure occurs
            relative_failures[method] = max_testing_range
            max_degradation = np.max(rmse_degradation)
            max_factor = noise_factors[np.argmax(rmse_degradation)]
            print(f'  {method:15}: No relative failure, max {max_degradation:6.1f}% at {max_factor:6.2f}')

        # Check absolute failure (0.25m RMSE)
        absolute_failure_found = False
        for i, (factor, rmse) in enumerate(zip(noise_factors, rmse_values)):
            if rmse > ABSOLUTE_FAILURE_THRESHOLD:
                absolute_failures[method] = factor
                absolute_failure_found = True
                print(f'  {method:15}: ABSOLUTE FAILURE at {factor:6.2f} (RMSE = {rmse:6.3f}m)')
                break

        if not absolute_failure_found:
            # Use max testing range when no failure occurs
            absolute_failures[method] = max_testing_range
            max_rmse = np.max(rmse_values)
            max_factor = noise_factors[np.argmax(rmse_values)]
            print(f'  {method:15}: No absolute failure, max RMSE {max_rmse:6.3f}m at {max_factor:6.2f}')

    return relative_failures, absolute_failures


def process_robustness_analysis(noise_type, method_filter='all'):
    '''Process robustness analysis for specified noise type and method filter.'''
    print_header(f'Robustness Analysis: {noise_type.upper()}')

    processed_data = {}

    # Process model-based controllers
    if method_filter in ['control', 'all']:
        print_header('Model-based Controllers', level=2)
        for display_name, folder_name in MODEL_BASED_CONTROLLERS.items():
            results = process_robustness_controller(folder_name, noise_type, is_rl=False)
            if results is not None:
                processed_data[display_name] = results

                # Save individual controller data
                save_path = DATA_DIR / f'{folder_name}_{noise_type}_results.npy'
                np.save(save_path, results)
                print(f'    Saved: {save_path}')

    # Process RL controllers
    if method_filter in ['rl', 'all']:
        print_header('RL Controllers', level=2)
        for display_name, folder_name in RL_CONTROLLERS.items():
            results = process_robustness_controller(folder_name, noise_type, is_rl=True)
            if results is not None:
                processed_data[display_name] = results

                # Save individual controller data
                save_path = DATA_DIR / f'{folder_name}_{noise_type}_results.npy'
                np.save(save_path, results)
                print(f'    Saved: {save_path}')

    if not processed_data:
        print('  Error: No data was processed successfully')
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
                        print(f'    Domain randomization {controller_name} max range: {dr_max_range}')

    except Exception as e:
        print(f'    Could not load domain randomization data for range calculation: {e}')

    print(f'    Common intersection testing range (nominal + DR): {max_testing_range}')

    # Analyze failure points
    relative_failures, absolute_failures = analyze_failure_points(processed_data, noise_type)

    # Save failure results
    save_robustness_failure_results(relative_failures, absolute_failures, noise_type, method_filter, max_testing_range)

    # Print summary
    print_summary('Robustness Analysis', len(processed_data), {
        'Noise type': noise_type,
        'Method filter': method_filter,
        'Relative failures': len(relative_failures),
        'Absolute failures': len(absolute_failures),
        'Max testing range': max_testing_range
    })


def extract_domain_randomization_failure_points(dr_data, noise_type, max_testing_range):
    '''Extract failure points from domain randomization data for a specific noise type.'''

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
                            print(f'  {display_name:15}: RELATIVE DR FAILURE at {scale:6.2f} (RMSE = {rmse_value:6.3f}m, baseline = {baseline_rmse:6.3f}m)')

                        # Check for absolute failure (0.25m RMSE)
                        if not absolute_failure_found and rmse_value > ABSOLUTE_FAILURE_THRESHOLD:
                            dr_failures['absolute'][display_name] = scale
                            absolute_failure_found = True
                            print(f'  {display_name:15}: ABSOLUTE DR FAILURE at {scale:6.2f} (RMSE = {rmse_value:6.3f}m)')

                        # If both failures found, break early
                        if relative_failure_found and absolute_failure_found:
                            break

                # If no failures found, use common intersection testing range (not individual max scale)
                if not relative_failure_found:
                    dr_failures['relative'][display_name] = max_testing_range
                    print(f'  {display_name:15}: No relative DR failure, using common max range {max_testing_range:6.2f}')

                if not absolute_failure_found:
                    dr_failures['absolute'][display_name] = max_testing_range
                    print(f'  {display_name:15}: No absolute DR failure, using common max range {max_testing_range:6.2f}')

    return dr_failures


def save_robustness_failure_results(relative_failures, absolute_failures, noise_type, method_filter, max_testing_range):
    '''Save robustness failure results to JSON files.'''

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

            print_header(f'Loading Domain Randomization Failures for {noise_type}', level=3)
            dr_failures = extract_domain_randomization_failure_points(dr_data, noise_type, max_testing_range)

    except Exception as e:
        print(f'  Warning: Could not load domain randomization data: {e}')

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

    print(f'  Failure results saved: {output_file}')

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

    print(f'  Consolidated failure results updated: {consolidated_file}')


def save_domain_randomization_robustness_data(controller_name, rob_dr_data):
    '''Save domain randomization robustness data to JSON files.'''

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    # Extract robustness data if it exists
    if 'robustness' not in rob_dr_data:
        print(f'    Warning: No robustness data found for {controller_name}')
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

    print(f'    Saved domain randomization robustness JSON: {output_file}')

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

    print(f'    Consolidated domain randomization robustness updated: {consolidated_dr_file}')

# ==========================================
# TRAJECTORY ANALYSIS FUNCTIONS
# ==========================================


def process_trajectory_controller(controller_name, episode_lengths=None):
    '''Process trajectory data for a specific controller.'''
    if episode_lengths is None:
        episode_lengths = EPISODE_LENGTHS

    print_header(f'Processing trajectory data: {controller_name}', level=3)

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
        # Try multiple potential file locations
        potential_files = [
            DATA_DIR / 'trajectory' / 'nominal' / f'traj_results_{file_prefix}_{episode_length}.npy',
            DATA_DIR / f'traj_results_{file_prefix}_{episode_length}.npy'
        ]

        trajectory_loaded = False
        for traj_file in potential_files:
            if traj_file.exists():
                try:
                    traj_data = np.load(traj_file, allow_pickle=True)

                    # Handle different data formats
                    if isinstance(traj_data, np.ndarray) and traj_data.ndim > 0:
                        try:
                            if traj_data.size == 1 and hasattr(traj_data.item(), 'keys'):
                                # Dictionary stored in numpy array
                                data_dict = traj_data.item()
                                obs_data = data_dict.get('obs', None)
                                n_rollouts = data_dict.get('n_rollouts', 0)
                            else:
                                # Raw trajectory array
                                obs_data = traj_data
                                n_rollouts = traj_data.shape[0] if traj_data.ndim > 0 else 0
                        except Exception:
                            obs_data = traj_data
                            n_rollouts = traj_data.shape[0] if traj_data.ndim > 0 else 0
                    else:
                        obs_data = None
                        n_rollouts = 0

                    trajectory_data[f'episode_{episode_length}'] = {
                        'obs': obs_data,
                        'n_rollouts': n_rollouts,
                        'timestamp': None,
                        'source_file': str(traj_file)
                    }

                    obs_shape = obs_data.shape if obs_data is not None else 'No obs data'
                    print(f'    Episode {episode_length}: {obs_shape} from {traj_file.name}')
                    trajectory_loaded = True
                    break

                except Exception as e:
                    print(f'    Episode {episode_length}: Error loading {traj_file} - {e}')

        if not trajectory_loaded:
            print(f'    Episode {episode_length}: No trajectory file found')

    return trajectory_data


def process_trajectory_analysis(method_filter='all', controller=None):
    '''Process trajectory analysis for specified controllers.'''
    print_header('Trajectory Analysis')

    if controller:
        controllers_to_process = [controller]
        print(f'  Processing specific controller: {controller}')
    else:
        if method_filter == 'control':
            controllers_to_process = list(MODEL_BASED_CONTROLLERS.values())
        elif method_filter == 'rl':
            controllers_to_process = list(RL_CONTROLLERS.values())
        else:  # 'all'
            controllers_to_process = list(ALL_CONTROLLERS.values())
        print(f'  Processing {method_filter} controllers: {len(controllers_to_process)} total')

    processed_count = 0

    for controller_name in controllers_to_process:
        traj_data = process_trajectory_controller(controller_name)

        if traj_data:
            output_file = DATA_DIR / f'{controller_name}_trajectory_processed.npy'
            try:
                np.save(output_file, traj_data)
                print(f'    Saved: {output_file}')
                processed_count += 1
            except Exception as e:
                print(f'    Error saving trajectory data for {controller_name}: {e}')
        else:
            print(f'    No trajectory data processed for {controller_name}')

    print_summary('Trajectory Analysis', processed_count, {
        'Method filter': method_filter,
        'Episode lengths': len(EPISODE_LENGTHS)
    })

# ==========================================
# GENERALIZATION ANALYSIS FUNCTIONS
# ==========================================


def process_generalization_controller(controller_name, episode_lengths=None):
    '''Process generalization data for a specific controller.'''
    if episode_lengths is None:
        episode_lengths = EPISODE_LENGTHS

    print_header(f'Processing generalization data: {controller_name}', level=3)

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

    if not gen_file.exists():
        print(f'    Generalization file not found: {gen_file}')
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
                print(f'    Episode {episode_length}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}')
            else:
                print(f'    Episode {episode_length}: No data found')
                generalization_data['rmse'].append(np.nan)
                generalization_data['rmse_std'].append(np.nan)

        generalization_data['rmse'] = np.array(generalization_data['rmse'])
        generalization_data['rmse_std'] = np.array(generalization_data['rmse_std'])

        return generalization_data

    except Exception as e:
        print(f'    Error loading generalization data: {e}')
        return None


def process_generalization_analysis(method_filter='all', controller=None):
    '''Process generalization analysis for specified controllers.'''
    print_header('Generalization Analysis')

    if controller:
        controllers_to_process = [controller]
        print(f'  Processing specific controller: {controller}')
    else:
        if method_filter == 'control':
            controllers_to_process = list(MODEL_BASED_CONTROLLERS.values())
        elif method_filter == 'rl':
            controllers_to_process = list(RL_CONTROLLERS.values())
        else:  # 'all'
            controllers_to_process = list(ALL_CONTROLLERS.values())
        print(f'  Processing {method_filter} controllers: {len(controllers_to_process)} total')

    processed_count = 0

    for controller_name in controllers_to_process:
        gen_data = process_generalization_controller(controller_name)

        if gen_data is not None:
            output_file = DATA_DIR / f'{controller_name}_generalization_processed.npy'
            try:
                np.save(output_file, gen_data)
                print(f'    Saved: {output_file}')
                processed_count += 1
            except Exception as e:
                print(f'    Error saving generalization data for {controller_name}: {e}')
        else:
            print(f'    No generalization data processed for {controller_name}')

    print_summary('Generalization Analysis', processed_count, {
        'Method filter': method_filter,
        'Episode lengths': len(EPISODE_LENGTHS)
    })

# ==========================================
# DOMAIN RANDOMIZATION FUNCTIONS
# ==========================================


def process_domain_randomization_controller(controller_name, data_type='generalization'):
    '''Process domain randomization data for RL controllers.'''
    if controller_name not in RL_CONTROLLERS.values():
        print(f'    Domain randomization only available for RL methods, skipping {controller_name}')
        return None

    # PPO-MPC doesn't have domain randomization setup, skip it
    if controller_name == 'ppo_mpc':
        print('    PPO-MPC doesn\'t have domain randomization setup, skipping')
        return None

    print_header(f'Processing domain randomization {data_type}: {controller_name}', level=3)

    # Map controller names to folder names
    folder_mapping = {
        'ppo': 'quadrotor_2D_attitude_ppo_data',
        'sac': 'quadrotor_2D_attitude_sac_data' if data_type == 'robustness_pm' else 'quadrotor_2D_attitude_sac_data2',
        'dppo': 'quadrotor_2D_attitude_dppo_data'
    }

    controller_folder = folder_mapping[controller_name]
    data_path = QUADROTOR_DIR / 'data' / data_type / controller_folder

    print(f'    Looking in: {data_path}')

    if not data_path.exists():
        print(f'    Domain randomization data folder not found: {data_path}')
        return None

    # Get all seed directories
    seed_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('seed')]
    seed_dirs.sort()

    if not seed_dirs:
        print(f'    No seed directories found in {data_path}')
        return None

    print(f'    Found {len(seed_dirs)} seed directories')

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
        print(f'      Processing {seed_name}...')

        try:
            # Process generalization metrics (transfer_metric files)
            if data_type == 'generalization':
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
                        print(f'        Error loading {metric_file}: {e}')

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
                        print(f'        Error loading {metric_file}: {e}')

            # Process performance metrics
            perf_file = seed_dir / 'perf_metric.npy'
            if perf_file.exists():
                try:
                    perf_data = np.load(perf_file, allow_pickle=True).item()
                    domain_rand_data['performance'].append(perf_data)
                except Exception as e:
                    print(f'        Error loading performance data: {e}')

        except Exception as e:
            print(f'      Error processing {seed_name}: {e}')

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

    print(f'    Successfully processed domain randomization data for {controller_name}')
    return domain_rand_data


def process_domain_randomization_analysis(controller=None):
    '''Process domain randomization analysis for RL controllers.'''
    print_header('Domain Randomization Analysis')

    if controller:
        if controller not in RL_CONTROLLERS.values():
            print('  Error: Domain randomization only available for RL controllers')
            print(f'  Available RL controllers: {list(RL_CONTROLLERS.values())}')
            return
        controllers_to_process = [controller]
        print(f'  Processing specific controller: {controller}')
    else:
        controllers_to_process = list(RL_CONTROLLERS.values())
        print(f'  Processing all RL controllers: {controllers_to_process}')

    processed_count = 0

    for controller_name in controllers_to_process:
        # Process generalization domain randomization
        gen_dr_data = process_domain_randomization_controller(controller_name, 'generalization')
        if gen_dr_data is not None:
            output_file = DATA_DIR / f'{controller_name}_domain_rand_generalization.npy'
            try:
                np.save(output_file, gen_dr_data)
                print(f'    Saved generalization DR data: {output_file}')
            except Exception as e:
                print(f'    Error saving generalization DR data: {e}')

        # Process robustness domain randomization
        rob_dr_data = process_domain_randomization_controller(controller_name, 'robustness_pm')
        if rob_dr_data is not None:
            # Save .npy file
            output_file = DATA_DIR / f'{controller_name}_domain_rand_robustness.npy'
            try:
                np.save(output_file, rob_dr_data)
                print(f'    Saved robustness DR data: {output_file}')
                processed_count += 1
            except Exception as e:
                print(f'    Error saving robustness DR data: {e}')

            # Save domain randomization robustness data to JSON
            save_domain_randomization_robustness_data(controller_name, rob_dr_data)

    print_summary('Domain Randomization Analysis', processed_count, {
        'Data types': 'generalization + robustness',
        'RL controllers only': True
    })

# ==========================================
# MAIN PROCESSING FUNCTIONS
# ==========================================


def process_single_controller(controller_name, analysis_types, noise_type=None, method_filter='all'):
    '''Process specified analysis types for a single controller.'''
    print_header(f'Processing {controller_name.upper()}')

    # Determine controller type
    is_model_based = controller_name in MODEL_BASED_CONTROLLERS.values()
    is_rl = controller_name in RL_CONTROLLERS.values()

    if not (is_model_based or is_rl):
        print(f'  Error: Unknown controller "{controller_name}"')
        print(f'  Available controllers: {list(ALL_CONTROLLERS.values())}')
        return

    # Check method filter compatibility
    if method_filter == 'control' and not is_model_based:
        print(f'  Skipping {controller_name} (not a model-based controller)')
        return
    elif method_filter == 'rl' and not is_rl:
        print(f'  Skipping {controller_name} (not an RL controller)')
        return

    # Process each requested analysis type
    if 'robustness' in analysis_types and noise_type:
        print(f'  Processing robustness data for {noise_type}...')
        result = process_robustness_controller(controller_name, noise_type, is_rl=is_rl)
        if result is not None:
            save_path = DATA_DIR / f'{controller_name}_{noise_type}_results.npy'
            np.save(save_path, result)
            print(f'    Saved: {save_path}')

    if 'trajectory' in analysis_types:
        traj_data = process_trajectory_controller(controller_name)
        if traj_data:
            save_path = DATA_DIR / f'{controller_name}_trajectory_processed.npy'
            np.save(save_path, traj_data)
            print(f'    Saved: {save_path}')

    if 'generalization' in analysis_types:
        gen_data = process_generalization_controller(controller_name)
        if gen_data is not None:
            save_path = DATA_DIR / f'{controller_name}_generalization_processed.npy'
            np.save(save_path, gen_data)
            print(f'    Saved: {save_path}')

    if 'domain-randomization' in analysis_types and is_rl:
        # Process both generalization and robustness domain randomization
        gen_dr_data = process_domain_randomization_controller(controller_name, 'generalization')
        if gen_dr_data is not None:
            save_path = DATA_DIR / f'{controller_name}_domain_rand_generalization.npy'
            np.save(save_path, gen_dr_data)
            print(f'    Saved: {save_path}')

        rob_dr_data = process_domain_randomization_controller(controller_name, 'robustness_pm')
        if rob_dr_data is not None:
            save_path = DATA_DIR / f'{controller_name}_domain_rand_robustness.npy'
            np.save(save_path, rob_dr_data)
            print(f'    Saved: {save_path}')


def print_usage():
    '''Print usage information.'''
    print(__doc__)


def parse_arguments():
    '''Parse command line arguments.'''
    args = sys.argv[1:]

    if not args or args[0] in ['-h', '--help']:
        print_usage()
        sys.exit(0)

    # Extract controller if specified
    controller = None
    cleaned_args = []
    for arg in args:
        if arg.startswith('--controller='):
            controller = arg.split('=')[1]
        else:
            cleaned_args.append(arg)

    if not cleaned_args:
        print('Error: No analysis type specified')
        print_usage()
        sys.exit(1)

    analysis_type = cleaned_args[0].lower()

    # Parse based on analysis type
    if analysis_type == 'robustness':
        if len(cleaned_args) < 2:
            print('Error: Robustness analysis requires noise type')
            print('Usage: python process_experiment_data.py robustness <noise_type> [method_filter]')
            sys.exit(1)

        noise_type = cleaned_args[1]
        method_filter = cleaned_args[2] if len(cleaned_args) > 2 else 'all'

        if noise_type not in ['obs_noise', 'proc_noise', 'param']:
            print(f'Error: Invalid noise type "{noise_type}"')
            print('Valid noise types: obs_noise, proc_noise, param')
            sys.exit(1)

        return analysis_type, noise_type, method_filter, controller

    elif analysis_type in ['trajectory', 'generalization']:
        method_filter = cleaned_args[1] if len(cleaned_args) > 1 else 'all'
        return analysis_type, None, method_filter, controller

    elif analysis_type == 'domain-randomization':
        return analysis_type, None, 'rl', controller

    elif analysis_type == 'all':
        method_filter = cleaned_args[1] if len(cleaned_args) > 1 else 'all'
        return analysis_type, None, method_filter, controller

    else:
        print(f'Error: Invalid analysis type "{analysis_type}"')
        print('Valid analysis types: robustness, trajectory, generalization, domain-randomization, all')
        sys.exit(1)


def main():
    '''Main processing function.'''

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    # Parse command line arguments
    analysis_type, noise_type, method_filter, controller = parse_arguments()

    # Validate method filter
    if method_filter not in ['control', 'rl', 'all']:
        print(f'Warning: Invalid method filter "{method_filter}". Using "all".')
        method_filter = 'all'

    # Process based on analysis type and scope
    if controller:
        # Process specific controller
        if controller not in ALL_CONTROLLERS.values():
            print(f'Error: Unknown controller "{controller}"')
            print(f'Available controllers: {list(ALL_CONTROLLERS.values())}')
            sys.exit(1)

        if analysis_type == 'all':
            analysis_types = ['robustness', 'trajectory', 'generalization', 'domain-randomization']
            # For 'all' with specific controller, process all noise types for robustness
            for nt in ['obs_noise', 'proc_noise', 'param']:
                process_single_controller(controller, ['robustness'], nt, method_filter)
            # Process other analysis types
            for at in ['trajectory', 'generalization', 'domain-randomization']:
                process_single_controller(controller, [at], None, method_filter)
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
        elif analysis_type == 'all':
            # Process all analysis types
            for nt in ['obs_noise', 'proc_noise', 'param']:
                process_robustness_analysis(nt, method_filter)
            process_trajectory_analysis(method_filter)
            process_generalization_analysis(method_filter)
            if method_filter in ['rl', 'all']:
                process_domain_randomization_analysis()


if __name__ == '__main__':
    main()
