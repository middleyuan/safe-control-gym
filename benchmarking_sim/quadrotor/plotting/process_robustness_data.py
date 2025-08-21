"""
Comprehensive data processing script for robustness analysis.

This script combines the functionality of extract_results.py and the data processing
parts of plot_noise_all.py to extract, process, and save robustness analysis data
for both model-based controllers and RL methods. It also supports trajectory and
generalization data processing for plot_traj_hull.py and plot_generalization.py,
as well as domain randomization data for RL methods.

Usage:
    python process_robustness_data.py <noise_type> [method_type] [--controller=<name>] [--trajectory] [--generalization] [--domain-rand]
    
Arguments:
    noise_type: 'obs_noise', 'proc_noise', or 'param' (ignored if --trajectory, --generalization, or --domain-rand)
    method_type: 'control', 'rl', or 'all' (default: 'all')
    --controller: Process only specific controller (optional)
    --trajectory: Process trajectory data for plot_traj_hull.py
    --generalization: Process generalization data for plot_generalization.py
    --domain-rand: Process domain randomization data for RL methods
    
Examples:
    python process_robustness_data.py obs_noise all      # Process all methods for observation noise
    python process_robustness_data.py proc_noise control # Process only control methods for process noise
    python process_robustness_data.py param rl          # Process only RL methods for parametric noise
    python process_robustness_data.py --controller=ppo --trajectory --generalization  # Process specific controller with all data types
    python process_robustness_data.py --domain-rand     # Process domain randomization data for all RL methods
    python process_robustness_data.py --controller=ppo --domain-rand  # Process domain randomization for specific RL controller
"""

import os
import sys
from pathlib import Path
import numpy as np
import json
from benchmarking_sim.quadrotor.benchmark_util.utils import tag_ctrl_list

# script dir
script_dir = Path(__file__).parent.resolve()
quadrotor_dir = script_dir.parent

max_seed = 10
metric_name = 'metrics.txt'

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def extract_rollouts(data_folder_path, controller_name):
    """Extract rollout data from experiment results."""
    print(f"Extracting rollouts for {controller_name} from {data_folder_path}")
    
    if not os.path.exists(data_folder_path):
        print(f"Warning: data_folder_path {data_folder_path} does not exist")
        return None, None, None

    # find all the subfolders in the data_folder_path
    subfolders = [f.path for f in os.scandir(data_folder_path) if f.is_dir()]
    subfolders.sort()
    
    metrics = []
    traj_results = []
    timing_data = []
    
    for subfolder in subfolders:
        file_path = os.path.join(subfolder, 'metrics.txt')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'rmse' in line:
                        rmse = float(line.split(': ')[1])
                        metrics.append(rmse)
                        break
        
        # Load trajectory data if exists
        traj_files = [f for f in os.listdir(subfolder) if f.startswith('traj_') and f.endswith('.npy')]
        if traj_files:
            traj_path = os.path.join(subfolder, traj_files[0])
            traj_data = np.load(traj_path, allow_pickle=True)
            traj_results.append(traj_data)
        
        # Load timing data if exists
        timing_files = [f for f in os.listdir(subfolder) if 'timing' in f and f.endswith('.npy')]
        if timing_files:
            timing_path = os.path.join(subfolder, timing_files[0])
            timing = np.load(timing_path, allow_pickle=True)
            timing_data.append(timing)
    
    return metrics, traj_results, timing_data

def process_controller_noise_data(controller_name, noise_type, is_rl=False):
    """Process noise robustness data for a specific controller."""
    print(f"\nProcessing {controller_name} for {noise_type}...")
    
    # Define noise scales based on noise type
    if noise_type in ['obs_noise']:
        noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       12, 14, 16, 18, 20, 25, 30, 35, 40, 
                       45, 50, 60, 70, 80, 90, 100]
    elif noise_type == 'proc_noise':
        noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       12, 14, 16, 18, 20, 25, 30, 35, 40, 
                       45, 50, 60, 70, 80, 90, 100]
    elif noise_type == 'param':
        if is_rl:
            # RL methods typically have fewer parameter variation points
            noise_scale = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        else:
            noise_scale = [0, 0.01, 0.02, 0.05, 0.1, 
                          0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 
                          1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 
                          2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
        noise_scale.sort()
    
    rmse_data = []
    rmse_std_data = []
    valid_noise_factors = []
    
    for noise_factor in noise_scale:
        # Construct data folder path based on controller type and noise
        if is_rl:
            # RL methods typically have different folder structure
            if noise_type == 'param':
                data_folder = f"robustness/{noise_type}/domain_rand_{noise_factor}/results"
            else:
                data_folder = f"robustness/{noise_type}/{noise_factor}/results"
        else:
            # Model-based controllers
            if noise_type == 'param':
                data_folder = f"robustness/{noise_type}/domain_rand_{noise_factor}/results"
            else:
                data_folder = f"robustness/{noise_type}/{noise_factor}/results"
        
        data_folder_path = quadrotor_dir / controller_name / data_folder
        
        # Extract metrics
        metrics, _, _ = extract_rollouts(data_folder_path, controller_name)
        
        if metrics and len(metrics) > 0:
            rmse_mean = np.mean(metrics)
            rmse_std = np.std(metrics)
            rmse_data.append(rmse_mean)
            rmse_std_data.append(rmse_std)
            valid_noise_factors.append(noise_factor)
            print(f"  Noise factor {noise_factor:6.2f}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f} ({len(metrics)} samples)")
        else:
            print(f"  Noise factor {noise_factor:6.2f}: No data found")
    
    if not rmse_data:
        print(f"  Warning: No valid data found for {controller_name}")
        return None
    
    # Calculate degradation relative to baseline (noise_factor = 0)
    if valid_noise_factors[0] == 0:
        baseline_rmse = rmse_data[0]
    else:
        print(f"  Warning: No baseline (noise_factor=0) data found for {controller_name}")
        baseline_rmse = min(rmse_data)  # Use minimum as baseline
    
    rmse_degradation_mean = [(rmse / baseline_rmse) * 100 for rmse in rmse_data]
    rmse_degradation_std = [(std / baseline_rmse) * 100 for std in rmse_std_data]
    
    # Package results
    results = {
        'noise_factor': np.array(valid_noise_factors),
        'rmse_mean': np.array(rmse_data),
        'rmse_std': np.array(rmse_std_data),
        'rmse_degradation_mean': np.array(rmse_degradation_mean),
        'rmse_degradation_std': np.array(rmse_degradation_std),
        'baseline_rmse': baseline_rmse,
        'controller': controller_name,
        'noise_type': noise_type
    }
    
    return results

def process_all_controllers(noise_type, method_type='all'):
    """Process all requested controllers for the given noise type."""
    
    # Define controller mappings
    model_based_controllers = {
        'Linear MPC': 'linear_mpc_acados',
        'Nonlinear MPC': 'mpc_acados', 
        'GP-MPC': 'gpmpc_acados_TP',
        'iLQR': 'ilqr',
        'LQR': 'lqr',
        'Geometric Control': 'pid',
        'F-MPC': 'fmpc',
    }
    
    rl_controllers = {
        'PPO': 'ppo',
        'SAC': 'sac',
        'DPPO': 'dppo',
        'PPO-MPC': 'ppo_mpc',
    }
    
    processed_data = {}
    
    # Process model-based controllers
    if method_type in ['control', 'all']:
        print("\n" + "="*80)
        print("PROCESSING MODEL-BASED CONTROLLERS")
        print("="*80)
        
        for display_name, folder_name in model_based_controllers.items():
            results = process_controller_noise_data(folder_name, noise_type, is_rl=False)
            if results is not None:
                processed_data[display_name] = results
                
                # Save individual controller data
                save_path = script_dir / f'../data/{folder_name}_{noise_type}_results.npy'
                np.save(save_path, results)
                print(f"  Saved: {save_path}")
    
    # Process RL controllers  
    if method_type in ['rl', 'all']:
        print("\n" + "="*80)
        print("PROCESSING RL CONTROLLERS")
        print("="*80)
        
        for display_name, folder_name in rl_controllers.items():
            results = process_controller_noise_data(folder_name, noise_type, is_rl=True)
            if results is not None:
                processed_data[display_name] = results
                
                # Save individual controller data
                save_path = script_dir / f'../data/{folder_name}_{noise_type}_results.npy'
                np.save(save_path, results)
                print(f"  Saved: {save_path}")
    
    return processed_data

def analyze_failure_points(processed_data, noise_type):
    """Analyze failure points for relative and absolute thresholds."""
    
    relative_failure_threshold = 200  # 200% performance degradation
    absolute_failure_threshold = 0.25  # 0.25m RMSE
    
    relative_failures = {}
    absolute_failures = {}
    
    print("\n" + "="*80)
    print("FAILURE ANALYSIS")
    print("="*80)
    
    for method, data in processed_data.items():
        noise_factors = data['noise_factor']
        rmse_degradation = data['rmse_degradation_mean']
        rmse_values = data['rmse_mean']
        
        # Check relative failure (200% degradation)
        relative_failure_found = False
        for i, (factor, degradation) in enumerate(zip(noise_factors, rmse_degradation)):
            if degradation > relative_failure_threshold:
                relative_failures[method] = factor
                relative_failure_found = True
                print(f"{method:15}: RELATIVE FAILURE at {factor:6.2f} ({degradation:6.1f}% degradation)")
                break
        
        if not relative_failure_found:
            max_degradation = np.max(rmse_degradation)
            max_factor = noise_factors[np.argmax(rmse_degradation)]
            print(f"{method:15}: NO RELATIVE FAILURE, max degradation {max_degradation:6.1f}% at {max_factor:6.2f}")
        
        # Check absolute failure (0.25m RMSE)
        absolute_failure_found = False
        for i, (factor, rmse) in enumerate(zip(noise_factors, rmse_values)):
            if rmse > absolute_failure_threshold:
                absolute_failures[method] = factor
                absolute_failure_found = True
                print(f"{method:15}: ABSOLUTE FAILURE at {factor:6.2f} (RMSE = {rmse:6.3f}m)")
                break
        
        if not absolute_failure_found:
            max_rmse = np.max(rmse_values)
            max_factor = noise_factors[np.argmax(rmse_values)]
            print(f"{method:15}: NO ABSOLUTE FAILURE, max RMSE {max_rmse:6.3f}m at {max_factor:6.2f}")
    
    return relative_failures, absolute_failures

def save_failure_results(relative_failures, absolute_failures, noise_type, method_type, max_testing_range):
    """Save failure results to JSON files for use in plotting scripts."""
    
    # Create data directory if it doesn't exist
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Prepare failure results
    failure_data = {
        'relative_failures': convert_numpy_types(relative_failures),
        'absolute_failures': convert_numpy_types(absolute_failures), 
        'max_testing_range': convert_numpy_types(max_testing_range),
        'relative_threshold': 200,
        'absolute_threshold': 0.25,
        'noise_type': noise_type,
        'method_filter': method_type
    }
    
    # Save individual noise type results
    output_file = data_dir / f'failure_results_{noise_type}_{method_type}.json'
    with open(output_file, 'w') as f:
        json.dump(failure_data, f, indent=2)
    
    print(f"\nFailure results saved to: {output_file}")
    
    # Load and update consolidated results
    consolidated_file = data_dir / 'robustness_failure_points.json'
    
    if consolidated_file.exists():
        with open(consolidated_file, 'r') as f:
            consolidated_data = json.load(f)
    else:
        consolidated_data = {
            'relative_failures': {
                'obs_noise': {},
                'proc_noise': {},
                'param': {}
            },
            'absolute_failures': {
                'obs_noise': {},
                'proc_noise': {},
                'param': {}
            }
        }
    
    # Update with current results
    for method, failure_scale in relative_failures.items():
        consolidated_data['relative_failures'][noise_type][method] = convert_numpy_types(failure_scale)
    
    for method, failure_scale in absolute_failures.items():
        consolidated_data['absolute_failures'][noise_type][method] = convert_numpy_types(failure_scale)
    
    # Save consolidated data
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    print(f"Consolidated failure results updated: {consolidated_file}")

def main():
    """Main processing function."""
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    # Parse flags
    process_trajectories = '--trajectory' in args
    process_generalization = '--generalization' in args
    process_domain_rand = '--domain-rand' in args
    specific_controller = None
    
    # Extract controller name if specified
    for arg in args:
        if arg.startswith('--controller='):
            specific_controller = arg.split('=')[1]
            break
    
    # Remove flags from args
    cleaned_args = [arg for arg in args if not arg.startswith('--')]
    
    # Parse positional arguments
    noise_option = cleaned_args[0] if len(cleaned_args) > 0 and not process_trajectories and not process_generalization and not process_domain_rand else 'obs_noise'
    method_type = cleaned_args[1] if len(cleaned_args) > 1 else 'all'
    
    if method_type not in ['control', 'rl', 'all']:
        print(f"Warning: Invalid method type '{method_type}'. Using 'all'.")
        method_type = 'all'
    
    # Define all available controllers
    all_controllers = ['linear_mpc_acados', 'mpc_acados', 'gpmpc_acados_TP', 'ilqr', 'lqr', 'pid', 'fmpc', 'ppo', 'sac', 'dppo', 'ppo_mpc']
    
    if specific_controller:
        if specific_controller not in all_controllers:
            print(f"Error: Unknown controller '{specific_controller}'")
            print(f"Available controllers: {all_controllers}")
            return
        controllers_to_process = [specific_controller]
        print(f"Processing specific controller: {specific_controller}")
    else:
        if process_domain_rand:
            # For domain randomization, only process RL controllers
            controllers_to_process = ['ppo', 'sac', 'dppo']
            print(f"Processing domain randomization for RL controllers: {controllers_to_process}")
        else:
            controllers_to_process = all_controllers
            print(f"Processing all controllers")
    
    # Process each controller
    for controller in controllers_to_process:
        process_single_controller(
            controller, 
            process_robustness=not (process_trajectories or process_generalization or process_domain_rand),
            process_trajectories=process_trajectories,
            process_generalization=process_generalization,
            process_domain_rand=process_domain_rand,
            noise_type=noise_option,
            method_type=method_type
        )
    
    # If processing robustness data for all controllers, also do the consolidated processing
    if not specific_controller and not process_trajectories and not process_generalization and not process_domain_rand:
        print(f"\nProcessing {noise_option} robustness data for {method_type} methods...")
        
        # Process all controllers
        processed_data = process_all_controllers(noise_option, method_type)
        
        if not processed_data:
            print("No data was processed successfully.")
            return
        
        # Calculate maximum testing range
        max_testing_range = 0
        for data in processed_data.values():
            if len(data['noise_factor']) > 0:
                max_testing_range = max(max_testing_range, np.max(data['noise_factor']))
        
        # Analyze failure points
        relative_failures, absolute_failures = analyze_failure_points(processed_data, noise_option)
        
        # Save failure results
        save_failure_results(relative_failures, absolute_failures, noise_option, method_type, max_testing_range)
        
        print(f"\n{'='*80}")
        print("DATA PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Processed {len(processed_data)} controllers for {noise_option}")
        print(f"Relative failures: {len(relative_failures)}")
        print(f"Absolute failures: {len(absolute_failures)}")
        print(f"Maximum testing range: {max_testing_range}")
    elif process_domain_rand:
        print(f"\n{'='*80}")
        print("DOMAIN RANDOMIZATION DATA PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Processed {len(controllers_to_process)} RL controllers for domain randomization")

def process_trajectory_data(controller_name, episode_lengths=[9, 10, 11, 12, 13, 14, 15]):
    """
    Process trajectory data for a specific controller.
    
    Args:
        controller_name: Name of the controller (e.g., 'ppo', 'linear_mpc_acados')
        episode_lengths: List of episode lengths to process
    
    Returns:
        dict: Trajectory data for all episode lengths
    """
    data_dir = script_dir / '../data'
    trajectory_data = {}
    
    print(f"Processing trajectory data for {controller_name}...")
    
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
        # Try nominal trajectory data first
        traj_file = data_dir / 'trajectory' / 'nominal' / f'traj_results_{file_prefix}_{episode_length}.npy'
        
        if not traj_file.exists():
            # Try direct data directory
            traj_file = data_dir / f'traj_results_{file_prefix}_{episode_length}.npy'
        
        if traj_file.exists():
            try:
                traj_data = np.load(traj_file, allow_pickle=True)
                
                # Handle different data formats
                if isinstance(traj_data, np.ndarray) and traj_data.ndim > 0:
                    # Check if it's a structured array or contains a dictionary
                    try:
                        if traj_data.size == 1 and hasattr(traj_data.item(), 'keys'):
                            # It's a dictionary stored in numpy array
                            data_dict = traj_data.item()
                            obs_data = data_dict.get('obs', None)
                            n_rollouts = data_dict.get('n_rollouts', 0)
                        else:
                            # It's a raw trajectory array
                            obs_data = traj_data
                            n_rollouts = traj_data.shape[0] if traj_data.ndim > 0 else 0
                    except:
                        # If conversion fails, treat as raw array
                        obs_data = traj_data
                        n_rollouts = traj_data.shape[0] if traj_data.ndim > 0 else 0
                else:
                    # Handle other cases
                    obs_data = None
                    n_rollouts = 0
                
                trajectory_data[f'episode_{episode_length}'] = {
                    'obs': obs_data,
                    'n_rollouts': n_rollouts,
                    'timestamp': None
                }
                
                obs_shape = obs_data.shape if obs_data is not None else "No obs data"
                print(f"  Episode {episode_length}: {obs_shape}")
                
            except Exception as e:
                print(f"  Episode {episode_length}: Error loading - {e}")
        else:
            print(f"  Episode {episode_length}: File not found")
    
    return trajectory_data

def process_domain_randomization_data(controller_name, data_type='generalization'):
    """
    Process domain randomization data for RL controllers.
    
    Args:
        controller_name: Name of the RL controller (e.g., 'ppo', 'sac', 'dppo')
        data_type: 'generalization' or 'robustness_pm'
    
    Returns:
        dict: Domain randomization data with metrics
    """
    if controller_name not in ['ppo', 'sac', 'dppo']:
        print(f"Domain randomization only available for RL methods, skipping {controller_name}")
        return None
    
    # Map controller names to folder names
    folder_mapping = {
        'ppo': 'quadrotor_2D_attitude_ppo_data',
        'sac': 'quadrotor_2D_attitude_sac_data' if data_type == 'robustness_pm' else 'quadrotor_2D_attitude_sac_data2',
        'dppo': 'quadrotor_2D_attitude_dppo_data'
    }
    
    controller_folder = folder_mapping[controller_name]
    data_path = quadrotor_dir / 'data' / data_type / controller_folder
    
    print(f"Processing domain randomization {data_type} data for {controller_name}...")
    print(f"  Looking in: {data_path}")
    
    if not data_path.exists():
        print(f"  Domain randomization data folder not found: {data_path}")
        return None
    
    # Get all seed directories
    seed_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('seed')]
    seed_dirs.sort()
    
    if not seed_dirs:
        print(f"  No seed directories found in {data_path}")
        return None
    
    print(f"  Found {len(seed_dirs)} seed directories")
    
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
        print(f"    Processing {seed_name}...")
        
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
                                'rmse': [],
                                'rmse_std': [],
                                'failure_rate': [],
                                'constraint_violation': [],
                                'rms_action_change': []
                            }
                        
                        domain_rand_data['generalization'][episode_length]['rmse'].append(metric_data.get('average_rmse', np.nan))
                        domain_rand_data['generalization'][episode_length]['rmse_std'].append(metric_data.get('rmse_std', np.nan))
                        domain_rand_data['generalization'][episode_length]['failure_rate'].append(metric_data.get('failure_rate', np.nan))
                        domain_rand_data['generalization'][episode_length]['constraint_violation'].append(metric_data.get('average_constraint_violation', np.nan))
                        domain_rand_data['generalization'][episode_length]['rms_action_change'].append(metric_data.get('rms_action_change', np.nan))
                        
                    except Exception as e:
                        print(f"      Error loading {metric_file}: {e}")
            
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
                                'rmse': [],
                                'rmse_std': [],
                                'failure_rate': [],
                                'constraint_violation': [],
                                'rms_action_change': []
                            }
                        
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['rmse'].append(metric_data.get('average_rmse', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['rmse_std'].append(metric_data.get('rmse_std', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['failure_rate'].append(metric_data.get('failure_rate', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['constraint_violation'].append(metric_data.get('average_constraint_violation', np.nan))
                        domain_rand_data['robustness'][noise_type_key][noise_scale]['rms_action_change'].append(metric_data.get('rms_action_change', np.nan))
                        
                    except Exception as e:
                        print(f"      Error loading {metric_file}: {e}")
            
            # Process performance metrics
            perf_file = seed_dir / 'perf_metric.npy'
            if perf_file.exists():
                try:
                    perf_data = np.load(perf_file, allow_pickle=True).item()
                    domain_rand_data['performance'].append(perf_data)
                except Exception as e:
                    print(f"      Error loading performance data: {e}")
        
        except Exception as e:
            print(f"    Error processing {seed_name}: {e}")
    
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
    
    print(f"  Successfully processed domain randomization data for {controller_name}")
    return domain_rand_data

def process_generalization_data(controller_name, episode_lengths=[9, 10, 11, 12, 13, 14, 15]):
    """
    Process generalization data for a specific controller.
    
    Args:
        controller_name: Name of the controller (e.g., 'ppo', 'linear_mpc_acados')
        episode_lengths: List of episode lengths to process
    
    Returns:
        dict: Generalization metrics for all episode lengths
    """
    data_dir = script_dir / '../data'
    
    print(f"Processing generalization data for {controller_name}...")
    
    # Map controller names to file names
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
    
    gen_file = data_dir / f'{ctrl_name}_gen_results.npy'
    
    if not gen_file.exists():
        print(f"  Generalization file not found: {gen_file}")
        return None
    
    try:
        gen_data = np.load(gen_file, allow_pickle=True).item()
        
        generalization_data = {
            'rmse': [],
            'rmse_std': [],
            'inference_time': gen_data.get('inference_time', 0.0),
            'inference_time_std': gen_data.get('inference_time_std', 0.0)
        }
        
        for episode_length in episode_lengths:
            episode_key = f'_{episode_length}'
            if episode_key in gen_data:
                generalization_data['rmse'].append(gen_data[episode_key]['mean_rmse'])
                generalization_data['rmse_std'].append(gen_data[episode_key]['std_rmse'])
                print(f"  Episode {episode_length}: RMSE = {gen_data[episode_key]['mean_rmse']:.4f} ± {gen_data[episode_key]['std_rmse']:.4f}")
            else:
                print(f"  Episode {episode_length}: No data found")
                generalization_data['rmse'].append(np.nan)
                generalization_data['rmse_std'].append(np.nan)
        
        generalization_data['rmse'] = np.array(generalization_data['rmse'])
        generalization_data['rmse_std'] = np.array(generalization_data['rmse_std'])
        
        return generalization_data
        
    except Exception as e:
        print(f"  Error loading generalization data: {e}")
        return None

def save_trajectory_data(controller_name, trajectory_data):
    """Save processed trajectory data to .npy file."""
    if not trajectory_data:
        print(f"No trajectory data to save for {controller_name}")
        return
    
    output_file = script_dir / f'../data/{controller_name}_trajectory_processed.npy'
    
    try:
        np.save(output_file, trajectory_data)
        print(f"Trajectory data saved: {output_file}")
    except Exception as e:
        print(f"Error saving trajectory data for {controller_name}: {e}")

def save_generalization_data(controller_name, generalization_data):
    """Save processed generalization data to .npy file."""
    if generalization_data is None:
        print(f"No generalization data to save for {controller_name}")
        return
    
    output_file = script_dir / f'../data/{controller_name}_generalization_processed.npy'
    
    try:
        np.save(output_file, generalization_data)
        print(f"Generalization data saved: {output_file}")
    except Exception as e:
        print(f"Error saving generalization data for {controller_name}: {e}")

def process_single_controller(controller_name, process_robustness=True, process_trajectories=False, 
                            process_generalization=False, process_domain_rand=False, noise_type=None, method_type='all'):
    """
    Process all data types for a single controller.
    
    Args:
        controller_name: Name of the controller
        process_robustness: Whether to process robustness data
        process_trajectories: Whether to process trajectory data
        process_generalization: Whether to process generalization data
        process_domain_rand: Whether to process domain randomization data
        noise_type: Type of noise for robustness analysis
        method_type: Type of methods for robustness analysis
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {controller_name.upper()}")
    print(f"{'='*80}")
    
    if process_robustness and noise_type:
        # Determine if this is a model-based or RL controller
        model_based_controllers = ['linear_mpc_acados', 'mpc_acados', 'gpmpc_acados_TP', 'ilqr', 'lqr', 'pid', 'fmpc']
        rl_controllers = ['ppo', 'sac', 'dppo', 'ppo_mpc']
        
        if controller_name in model_based_controllers and method_type in ['control', 'all']:
            print(f"Processing robustness data for {controller_name} ({noise_type})...")
            result = process_controller_noise_data(controller_name, noise_type)
            if result is not None:
                save_path = script_dir / f'../data/{controller_name}_{noise_type}_results.npy'
                np.save(save_path, result)
                print(f"  Robustness data saved: {save_path}")
        elif controller_name in rl_controllers and method_type in ['rl', 'all']:
            print(f"Processing robustness data for {controller_name} ({noise_type})...")
            result = process_controller_noise_data(controller_name, noise_type)
            if result is not None:
                save_path = script_dir / f'../data/{controller_name}_{noise_type}_results.npy'
                np.save(save_path, result)
                print(f"  Robustness data saved: {save_path}")
        else:
            print(f"Skipping robustness data for {controller_name} (not in {method_type} category)")
    
    if process_trajectories:
        traj_data = process_trajectory_data(controller_name)
        save_trajectory_data(controller_name, traj_data)
    
    if process_generalization:
        gen_data = process_generalization_data(controller_name)
        save_generalization_data(controller_name, gen_data)
    
    if process_domain_rand:
        # Process domain randomization data for RL controllers
        rl_controllers = ['ppo', 'sac', 'dppo']
        if controller_name in rl_controllers:
            # Process generalization domain randomization
            gen_dr_data = process_domain_randomization_data(controller_name, 'generalization')
            if gen_dr_data is not None:
                save_path = script_dir / f'../data/{controller_name}_domain_rand_generalization.npy'
                np.save(save_path, gen_dr_data)
                print(f"  Domain randomization generalization data saved: {save_path}")
            
            # Process robustness domain randomization  
            rob_dr_data = process_domain_randomization_data(controller_name, 'robustness_pm')
            if rob_dr_data is not None:
                save_path = script_dir / f'../data/{controller_name}_domain_rand_robustness.npy'
                np.save(save_path, rob_dr_data)
                print(f"  Domain randomization robustness data saved: {save_path}")
        else:
            print(f"Domain randomization only available for RL methods, skipping {controller_name}")

if __name__ == '__main__':
    main()
