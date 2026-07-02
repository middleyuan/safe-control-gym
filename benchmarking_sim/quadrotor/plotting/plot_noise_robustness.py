"""
Plot noise robustness analysis for different controller types.
This script only handles plotting - data processing is done by process_experiment_data.py

Usage:
    python plot_noise_robustness.py <noise_type> [method_type] [--include-dr] [--dr-only]
    
Arguments:
    noise_type: 'obs_noise', 'proc_noise', 'param', or 'all'
    method_type: 'control', 'rl', or 'all' (default: 'all')
    --include-dr: Include domain randomization variants along with regular methods (optional)
    --dr-only: Plot ONLY domain randomization variants (automatically sets method_type to 'rl')
    
Examples:
    python plot_noise_robustness.py all                  # Plot all noise types for all available methods
    python plot_noise_robustness.py obs_noise all        # Plot all methods for observation noise (no DR)
    python plot_noise_robustness.py obs_noise rl --include-dr  # Plot RL methods including domain randomization
    python plot_noise_robustness.py obs_noise rl --dr-only     # Plot ONLY domain randomization variants
    python plot_noise_robustness.py proc_noise control   # Plot only control methods for process noise
    python plot_noise_robustness.py param rl            # Plot only RL methods for parametric noise

Note: Run process_experiment_data.py first to generate the required data files.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors, plotting_data_dir

# script dir
script_dir = Path(__file__).parent.resolve()
data_dir = plotting_data_dir(script_dir)
s = 2  # times std
VALID_NOISE_TYPES = ['obs_noise', 'proc_noise', 'param']
RELATIVE_FAILURE_THRESHOLD = 200
ABSOLUTE_FAILURE_THRESHOLD = 0.25
CONTROL_METHOD_NAMES = {
    'Linear MPC', 'Nonlinear MPC', 'GP-MPC', 'iLQR', 'LQR',
    'Geometric Control', 'F-MPC'
}
RL_METHOD_NAMES = {'PPO', 'SAC', 'DPPO', 'PPO-MPC'}
NOISE_DISPLAY_NAMES = {
    'obs_noise': 'Observation Noise',
    'proc_noise': 'Process Noise',
    'param': 'Parametric Uncertainty',
}

def infer_loaded_method_type(method_names, requested_method_type):
    """Infer a truthful label from the methods that were actually loaded."""
    base_methods = {method.replace(' (DR)', '') for method in method_names}
    has_control = any(method in CONTROL_METHOD_NAMES for method in base_methods)
    has_rl = any(method in RL_METHOD_NAMES for method in base_methods)

    if has_control and has_rl:
        return 'all'
    if has_rl:
        return 'rl'
    if has_control:
        return 'control'
    return requested_method_type

def method_sort_key(method_name):
    """Keep table rows in the same broad order as the plots."""
    order = [
        'Geometric Control', 'Linear MPC', 'Nonlinear MPC', 'F-MPC',
        'iLQR', 'LQR', 'GP-MPC', 'PPO', 'SAC', 'DPPO', 'PPO-MPC',
        'PPO (DR)', 'SAC (DR)', 'DPPO (DR)',
    ]
    try:
        return (0, order.index(method_name))
    except ValueError:
        return (1, method_name)

def format_noise_scale(value):
    """Format noise scales compactly for table cells."""
    if value is None or np.isnan(value):
        return 'n/a'
    if float(value).is_integer():
        return f'{int(value)}'
    return f'{value:.2f}'.rstrip('0').rstrip('.')

def format_threshold_cell(failure_scale, max_testing_range):
    """Show first failure scale, or survival through the tested range."""
    if failure_scale is None:
        return f'> {format_noise_scale(max_testing_range)}'
    return format_noise_scale(failure_scale)

def save_dataframe_table(df, output_path, title):
    """Render a DataFrame as a PNG/PDF table using the generalization-table style."""
    if df.empty:
        return

    output_path.parent.mkdir(exist_ok=True)
    fig_width = max(7, 1.5 + 1.55 * len(df.columns))
    fig_height = max(2.5, 1.0 + 0.34 * len(df.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=12)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def create_single_noise_tables(noise_data, noise_option, method_type, double_results,
                               abs_rmse_threshold_results, max_testing_range,
                               include_domain_rand=False):
    """Save relative and absolute threshold tables for one noise case."""
    loaded_method_type = infer_loaded_method_type(noise_data.keys(), method_type)
    dr_suffix = "_with_dr" if include_domain_rand and any('(DR)' in method for method in noise_data.keys()) else ""
    output_dir = script_dir / 'noise'

    methods = sorted(noise_data.keys(), key=method_sort_key)
    relative_table = {}
    absolute_table = {}

    for method in methods:
        rel_values = np.array(noise_data[method].get('rmse_degradation_mean', []), dtype=float)
        abs_values = np.array(noise_data[method].get('rmse_mean', []), dtype=float)
        relative_table[method] = {
            f'First > {RELATIVE_FAILURE_THRESHOLD}%': format_threshold_cell(
                double_results.get(method), max_testing_range
            ),
            'Max tested': format_noise_scale(max_testing_range),
            'Peak %': 'n/a' if len(rel_values) == 0 or np.all(np.isnan(rel_values)) else f'{np.nanmax(rel_values):.1f}',
        }
        absolute_table[method] = {
            f'First > {ABSOLUTE_FAILURE_THRESHOLD:.2f}m': format_threshold_cell(
                abs_rmse_threshold_results.get(method), max_testing_range
            ),
            'Max tested': format_noise_scale(max_testing_range),
            'Peak RMSE [m]': 'n/a' if len(abs_values) == 0 or np.all(np.isnan(abs_values)) else f'{np.nanmax(abs_values):.3f}',
        }

    noise_name = NOISE_DISPLAY_NAMES.get(noise_option, noise_option)
    relative_df = pd.DataFrame.from_dict(relative_table, orient='index')
    absolute_df = pd.DataFrame.from_dict(absolute_table, orient='index')

    relative_name = f"robustness_{loaded_method_type}_methods_{noise_option}_relative_table{dr_suffix}"
    absolute_name = f"robustness_{loaded_method_type}_methods_{noise_option}_absolute_table{dr_suffix}"
    output_dir.mkdir(exist_ok=True)
    relative_df.to_csv(output_dir / f'{relative_name}.csv', index_label='Method')
    absolute_df.to_csv(output_dir / f'{absolute_name}.csv', index_label='Method')
    save_dataframe_table(
        relative_df,
        output_dir / f'{relative_name}.png',
        f'{noise_name}: first noise scale above {RELATIVE_FAILURE_THRESHOLD}% relative RMSE',
    )
    save_dataframe_table(
        absolute_df,
        output_dir / f'{absolute_name}.png',
        f'{noise_name}: first noise scale above {ABSOLUTE_FAILURE_THRESHOLD:.2f}m RMSE',
    )
    print(f"Relative table saved as {output_dir / f'{relative_name}.png'}")
    print(f"Absolute table saved as {output_dir / f'{absolute_name}.png'}")

def create_combined_robustness_tables(noise_summaries, method_type, include_domain_rand=False):
    """Save combined threshold tables across obs/process/param noise."""
    summaries = {
        noise_type: summary
        for noise_type, summary in noise_summaries.items()
        if summary is not None
    }
    if not summaries:
        return

    all_methods = sorted(
        {method for summary in summaries.values() for method in summary['noise_data'].keys()},
        key=method_sort_key,
    )
    relative_table = {}
    absolute_table = {}

    for method in all_methods:
        relative_table[method] = {}
        absolute_table[method] = {}
        for noise_type in VALID_NOISE_TYPES:
            summary = summaries.get(noise_type)
            noise_name = NOISE_DISPLAY_NAMES.get(noise_type, noise_type)
            if summary is None or method not in summary['noise_data']:
                relative_table[method][noise_name] = 'n/a'
                absolute_table[method][noise_name] = 'n/a'
                continue

            max_testing_range = summary['max_testing_range']
            relative_table[method][noise_name] = format_threshold_cell(
                summary['double_results'].get(method),
                max_testing_range,
            )
            absolute_table[method][noise_name] = format_threshold_cell(
                summary['abs_rmse_threshold_results'].get(method),
                max_testing_range,
            )

    relative_df = pd.DataFrame.from_dict(relative_table, orient='index')
    absolute_df = pd.DataFrame.from_dict(absolute_table, orient='index')

    loaded_method_type = infer_loaded_method_type(all_methods, method_type)
    dr_suffix = "_with_dr" if include_domain_rand and any('(DR)' in method for method in all_methods) else ""
    output_dir = script_dir / 'noise'
    relative_name = f"robustness_{loaded_method_type}_methods_all_relative_table{dr_suffix}"
    absolute_name = f"robustness_{loaded_method_type}_methods_all_absolute_table{dr_suffix}"

    output_dir.mkdir(exist_ok=True)
    relative_df.to_csv(output_dir / f'{relative_name}.csv', index_label='Method')
    absolute_df.to_csv(output_dir / f'{absolute_name}.csv', index_label='Method')
    save_dataframe_table(
        relative_df,
        output_dir / f'{relative_name}.png',
        f'First noise scale above {RELATIVE_FAILURE_THRESHOLD}% relative RMSE',
    )
    save_dataframe_table(
        absolute_df,
        output_dir / f'{absolute_name}.png',
        f'First noise scale above {ABSOLUTE_FAILURE_THRESHOLD:.2f}m RMSE',
    )
    print(f"Combined relative table saved as {output_dir / f'{relative_name}.png'}")
    print(f"Combined absolute table saved as {output_dir / f'{absolute_name}.png'}")

def load_processed_data(noise_type, method_type='all', dr_only=False):
    """Load processed data from .npy files generated by process_experiment_data.py"""
    
    # Define controller mappings
    model_based_files = {
        'Linear MPC': 'linear_mpc_acados',
        'Nonlinear MPC': 'mpc_acados', 
        'GP-MPC': 'gpmpc_acados_TP',
        'iLQR': 'ilqr',
        'LQR': 'lqr',
        'Geometric Control': 'pid',
        'F-MPC': 'fmpc',
    }
    
    rl_files = {
        'PPO': 'ppo',
        'SAC': 'sac',
        'DPPO': 'dppo',
        'PPO-MPC': 'ppo_mpc',
    }
    
    noise_data = {}
    
    # Skip loading regular data if dr_only is True
    if not dr_only:
        # Load model-based controller data if requested
        if method_type in ['control', 'all']:
            print("Loading model-based controller data...")
            for display_name, file_name in model_based_files.items():
                file_path = data_dir / f'{file_name}_{noise_type}_results.npy'
                if file_path.exists():
                    data = np.load(file_path, allow_pickle=True).item()
                    noise_data[display_name] = data
                    print(f"  Loaded: {display_name}")
                else:
                    print(f"  Warning: {file_path} not found. Run process_experiment_data.py first.")
        
        # Load RL method data if requested
        if method_type in ['rl', 'all']:
            print("Loading RL method data...")
            for display_name, file_name in rl_files.items():
                file_path = data_dir / f'{file_name}_{noise_type}_results.npy'
                if file_path.exists():
                    data = np.load(file_path, allow_pickle=True).item()
                    noise_data[display_name] = data
                    print(f"  Loaded: {display_name}")
                else:
                    print(f"  Warning: {file_path} not found. Run process_experiment_data.py first.")
    else:
        print("Skipping regular method data (DR-only mode)")
    
    return noise_data

def load_domain_randomization_robustness_data(noise_type, method_type='all'):
    """Load domain randomization robustness data for RL methods."""
    
    noise_type_map = {
        'obs_noise': 'obs_noise',
        'proc_noise': 'proc_noise', 
        'param': 'param'
    }
    
    if noise_type not in noise_type_map:
        print(f"Unsupported noise type: {noise_type}")
        return {}
    
    rl_controllers = ['ppo', 'sac', 'dppo']
    dr_data = {}
    
    if method_type in ['rl', 'all']:
        print("Loading domain randomization robustness data...")
        for controller in rl_controllers:
            dr_file = data_dir / f'{controller}_domain_rand_robustness.npy'
            
            if dr_file.exists():
                try:
                    data = np.load(dr_file, allow_pickle=True).item()
                    
                    # Extract the specific noise type data
                    if noise_type_map[noise_type] in data['robustness']:
                        robustness_data = data['robustness'][noise_type_map[noise_type]]
                        
                        # Convert to the same format as normal robustness data
                        noise_scales = sorted(robustness_data.keys())
                        rmse_means = []
                        rmse_stds = []
                        
                        for scale in noise_scales:
                            scale_data = robustness_data[scale]
                            rmse_means.append(scale_data['rmse']['mean'])
                            rmse_stds.append(scale_data['rmse']['std'])
                        
                        # Calculate degradation relative to baseline (scale 0)
                        if noise_scales and noise_scales[0] == 0.0:
                            baseline_rmse = rmse_means[0]
                        else:
                            baseline_rmse = min(rmse_means) if rmse_means else 1.0
                        
                        rmse_degradation_mean = [(rmse / baseline_rmse) * 100 for rmse in rmse_means]
                        rmse_degradation_std = [(std / baseline_rmse) * 100 for std in rmse_stds]
                        
                        # Create display name
                        display_name = f"{controller.upper()} (DR)"
                        
                        dr_data[display_name] = {
                            'noise_factor': np.array(noise_scales),
                            'rmse_mean': np.array(rmse_means),
                            'rmse_std': np.array(rmse_stds),
                            'rmse_degradation_mean': np.array(rmse_degradation_mean),
                            'rmse_degradation_std': np.array(rmse_degradation_std),
                            'baseline_rmse': baseline_rmse,
                            'controller': f"{controller}_domain_rand",
                            'noise_type': noise_type
                        }
                        
                        print(f"  Loaded: {display_name}")
                    else:
                        print(f"  No {noise_type} data found in {dr_file}")
                        
                except Exception as e:
                    print(f"  Error loading {dr_file}: {e}")
            else:
                print(f"  Domain randomization file not found: {dr_file}")
    
    return dr_data

def print_failure_analysis(noise_data, noise_type):
    """Print failure analysis from loaded data."""
    
    double_results = {}
    abs_rmse_threshold_results = {}
    
    # Calculate maximum testing range
    max_testing_range = 0
    for data in noise_data.values():
        if len(data['noise_factor']) > 0:
            max_testing_range = max(max_testing_range, np.max(data['noise_factor']))
    
    print("\n" + "="*80)
    print("TESTING RANGE ANALYSIS")
    print("="*80)
    
    for method in noise_data.keys():
        x_data = np.array(noise_data[method]['noise_factor'])
        y_mean = np.array(noise_data[method]['rmse_degradation_mean'])
        valid_mask = ~np.isnan(y_mean)
        
        if np.any(valid_mask):
            x_valid = x_data[valid_mask]
            min_range = np.min(x_valid)
            max_range = np.max(x_valid)
            print(f"{method:15}: Range [{min_range:6.2f}, {max_range:6.2f}] - {len(x_valid)} valid points")
    
    print(f"\nMaximum testing range: {max_testing_range:6.2f}")
    
    print("\n" + "="*80)
    print(f"RELATIVE PERFORMANCE ANALYSIS ({RELATIVE_FAILURE_THRESHOLD}% Degradation Threshold)")
    print("="*80)
    
    for method in noise_data.keys():
        x_data = np.array(noise_data[method]['noise_factor']) 
        y_mean = np.array(noise_data[method]['rmse_degradation_mean'])
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y_mean)
        if np.any(valid_mask):
            x_valid = x_data[valid_mask]
            y_mean_valid = y_mean[valid_mask]
            
            # Find when RMSE degradation exceeds threshold
            failure_found = False
            for i, scale in enumerate(x_valid):
                if y_mean_valid[i] > RELATIVE_FAILURE_THRESHOLD:
                    print(f"{method:15}: FAILS at noise scale {scale:6.2f} (degradation = {y_mean_valid[i]:6.1f}%)")
                    double_results[method] = scale
                    failure_found = True
                    break
            
            if not failure_found:
                max_degradation = np.max(y_mean_valid)
                max_scale = x_valid[np.argmax(y_mean_valid)]
                print(f"{method:15}: NO FAILURE, max degradation = {max_degradation:6.1f}% at scale {max_scale:6.2f} (tested up to {np.max(x_valid):6.2f})")
        else:
            print(f"{method:15}: NO VALID DATA")
    
    print("\n" + "="*80)
    print(f"ABSOLUTE PERFORMANCE ANALYSIS ({ABSOLUTE_FAILURE_THRESHOLD:.2f}m RMSE Threshold)")
    print("="*80)
    
    for method in noise_data.keys():
        if 'rmse_mean' in noise_data[method] and 'rmse_std' in noise_data[method]:
            x_data = np.array(noise_data[method]['noise_factor'])
            y_mean = np.array(noise_data[method]['rmse_mean'])
            
            # Filter out NaN values
            valid_mask = ~np.isnan(y_mean)
            if np.any(valid_mask):
                x_valid = x_data[valid_mask]
                y_mean_valid = y_mean[valid_mask]
                
                # Check when absolute rmse exceeds threshold
                failure_found = False
                for i, scale in enumerate(x_valid):
                    if y_mean_valid[i] > ABSOLUTE_FAILURE_THRESHOLD:
                        print(f"{method:15}: FAILS at noise scale {scale:6.2f} (RMSE = {y_mean_valid[i]:6.3f}m)")
                        abs_rmse_threshold_results[method] = scale
                        failure_found = True
                        break
                
                if not failure_found:
                    max_rmse = np.max(y_mean_valid)
                    max_scale = x_valid[np.argmax(y_mean_valid)]
                    print(f"{method:15}: NO FAILURE, max RMSE = {max_rmse:6.3f}m at scale {max_scale:6.2f} (tested up to {np.max(x_valid):6.2f})")
            else:
                print(f"{method:15}: NO VALID DATA")
        else:
            print(f"{method:15}: MISSING RMSE DATA - skipping absolute analysis")
    
    return double_results, abs_rmse_threshold_results, max_testing_range

def create_robustness_plots(noise_data, noise_option, method_type, double_results, abs_rmse_threshold_results, include_domain_rand=False):
    """Create robustness plots."""
    
    # Determine noise scale for plotting
    if noise_option in ['obs_noise']:
        noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        12, 14, 16, 18, 20, 25, 30, 35, 40,
                        45, 50, 60, 70, 80, 90, 100, 110, 120,
                        130, 140, 150, 160, 170, 180, 190, 200,]
    elif noise_option == 'proc_noise':
        noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        12, 14, 16, 18, 20, 25, 30, 35, 40,
                        45, 50, 60, 70, 80, 90, 100, 110, 120,
                        130, 140, 150, 160, 170, 180, 190, 200,]
    elif noise_option == 'param':
        noise_scale = [0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
                     7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                     14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                     20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
        noise_scale.sort()
    
    # Determine legend columns based on number of methods
    num_methods = len(noise_data)
    if num_methods <= 4:
        legend_cols = 2
    elif num_methods <= 7:
        legend_cols = 3
    else:
        legend_cols = 4
    
    # Create relative performance plot
    fig = plt.figure(figsize=(8, 2))
    
    for method in noise_data.keys():
        x_data = np.array(noise_data[method]['noise_factor'])
        y_mean = np.array(noise_data[method]['rmse_degradation_mean'])
        y_std = np.array(noise_data[method]['rmse_degradation_std'])
        
        # Use dashed line for domain randomization variants
        linestyle = '--' if '(DR)' in method else '-'
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y_mean)
        if np.any(valid_mask):
            x_valid = x_data[valid_mask]
            y_mean_valid = y_mean[valid_mask]
            y_std_valid = y_std[valid_mask]
            
            plt.plot(x_valid, y_mean_valid, label=method, color=plot_colors[method], linestyle=linestyle)
            plt.fill_between(x_valid, 
                           y_mean_valid - s * y_std_valid,  
                           y_mean_valid + s * y_std_valid, color=plot_colors[method], alpha=0.1)
    
    plt.xlabel("Noise Scale")
    plt.ylabel("Relative Performance %")
    
    if noise_option == 'obs_noise':
        plt.legend(ncol=legend_cols, fontsize=9)
        plt.ylim(0, 1200)
        plt.xlim(0, 200)
        plt.title("Performance Degradation with Observation Noise")
        for method in double_results.keys():
            plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
    elif noise_option == 'proc_noise':
        plt.legend(ncol=legend_cols, loc='upper right', fontsize=9)
        plt.title("Performance Degradation with Process Noise")
        for method in double_results.keys():
            plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
        plt.xlim(0, 100)
    elif noise_option == 'param':
        plt.title("Performance Degradation with Parametric Uncertainty")
        plt.xlabel("Randomization scale")
        plt.ylim(0, 2000)
        plt.xlim(0, 20)
        plt.legend(ncol=legend_cols, loc='upper right', fontsize=9)
        for method in double_results.keys():
            plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
    
    plt.plot(noise_scale, [200]*len(noise_scale), 
             color='grey', linestyle='-.', label='Relative Perf=200%')
    
    loaded_method_type = infer_loaded_method_type(noise_data.keys(), method_type)
    dr_suffix = "_with_dr" if include_domain_rand and any('(DR)' in method for method in noise_data.keys()) else ""
    plot_save_name = f"robustness_{loaded_method_type}_methods_{noise_option}{dr_suffix}"
    plot_save_path = script_dir / 'noise' / f'{plot_save_name}.png'
    plot_save_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_save_path, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(plot_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Relative performance plot saved as {plot_save_path}")
    
    # Create absolute performance plot
    fig_abs = plt.figure(figsize=(8, 3))
    abs_rmse_threshold = ABSOLUTE_FAILURE_THRESHOLD
    
    for method in noise_data.keys():
        if 'rmse_mean' in noise_data[method] and 'rmse_std' in noise_data[method]:
            x_data = np.array(noise_data[method]['noise_factor'])
            y_mean = np.array(noise_data[method]['rmse_mean'])
            y_std = np.array(noise_data[method]['rmse_std'])
            
            # Use dashed line for domain randomization variants
            linestyle = '--' if '(DR)' in method else '-'
            
            # Filter out NaN values
            valid_mask = ~np.isnan(y_mean)
            if np.any(valid_mask):
                x_valid = x_data[valid_mask]
                y_mean_valid = y_mean[valid_mask]
                y_std_valid = y_std[valid_mask]
                
                plt.plot(x_valid, y_mean_valid, label=method, color=plot_colors[method], linestyle=linestyle)
                plt.fill_between(x_valid,
                               y_mean_valid - s * y_std_valid,
                               y_mean_valid + s * y_std_valid,
                               color=plot_colors[method], alpha=0.1)
    
    plt.xlabel("Noise Scale")
    plt.ylabel("RMSE [m]")
    
    if noise_option == 'obs_noise':
        plt.legend(ncol=legend_cols, fontsize=9)
        plt.xlim(0, 200)
        plt.ylim(0, 0.3)
        plt.title("Absolute Performance with Observation Noise")
        for method in abs_rmse_threshold_results.keys():
            plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
    elif noise_option == 'proc_noise':
        plt.legend(ncol=legend_cols, loc='upper left', fontsize=9)
        plt.ylim(0, 1.5)
        plt.xlim(0, 100)
        plt.title("Absolute Performance with Process Noise")
        for method in abs_rmse_threshold_results.keys():
            plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
    elif noise_option == 'param':
        plt.title("Absolute Performance with Parametric Uncertainty")
        plt.xlabel("Randomization scale")
        plt.xlim(0, 20)
        plt.ylim(0, 0.5)
        plt.legend(ncol=legend_cols, loc='upper left', fontsize=9)
        for method in abs_rmse_threshold_results.keys():
            plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
    
    plt.plot(noise_scale, [abs_rmse_threshold]*len(noise_scale), 
             color='grey', linestyle='-.', label=f'RMSE={abs_rmse_threshold}')
    
    plot_save_name_abs = f"robustness_{loaded_method_type}_methods_{noise_option}_absolute{dr_suffix}"
    plot_save_path_abs = script_dir / 'noise' / f'{plot_save_name_abs}.png'
    plt.savefig(plot_save_path_abs, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(plot_save_path_abs.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Absolute performance plot saved as {plot_save_path_abs}")

def create_ridge_plot(noise_data, noise_option, method_type, include_domain_rand=False):
    """Create a ridge plot showing RMSE distributions."""
    
    # Calculate actual data bounds for each controller first
    controller_bounds = {}
    all_rmse_values = []
    
    for method in noise_data.keys():
        if 'rmse_mean' in noise_data[method]:
            # Get only actual RMSE values (no synthetic data)
            rmse_values = np.array(noise_data[method]['rmse_mean'])
            valid_rmse = [val for val in rmse_values if not (np.isnan(val) or np.isinf(val))]
            
            if valid_rmse:
                controller_bounds[method] = {
                    'min': min(valid_rmse),
                    'max': max(valid_rmse),
                    'values': valid_rmse
                }
                all_rmse_values.extend(valid_rmse)
    
    if not all_rmse_values:
        print("No valid data for ridge plot")
        return None
    
    # Calculate global RMSE range from actual data only
    global_rmse_min = max(0, min(all_rmse_values) * 0.95)  # Small padding
    global_rmse_max = max(all_rmse_values) * 1.05  # Small padding
    
    print(f"Global RMSE range: [{global_rmse_min:.4f}, {global_rmse_max:.4f}]")
    
    # Create ridge plot
    fig, axes = plt.subplots(len(controller_bounds), 1, figsize=(8, len(controller_bounds) * 1.5), 
                            sharex=True, gridspec_kw={'hspace': 0.1})
    
    if len(controller_bounds) == 1:
        axes = [axes]
    
    for i, (method, bounds) in enumerate(controller_bounds.items()):
        ax = axes[i]
        
        # Add shaded gray areas where this controller has no data
        if bounds['min'] > global_rmse_min:
            ax.axvspan(global_rmse_min, bounds['min'], alpha=0.2, color='gray', zorder=0)
        
        if bounds['max'] < global_rmse_max:
            ax.axvspan(bounds['max'], global_rmse_max, alpha=0.2, color='gray', zorder=0)
        
        # Create KDE only within the actual data bounds
        rmse_values = np.array(bounds['values'])
        
        if len(rmse_values) > 1:
            # Use scipy's gaussian_kde for better control
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(rmse_values)
            
            # Evaluate KDE only within actual data range
            x_range = np.linspace(bounds['min'], bounds['max'], 200)
            density = kde(x_range)
            
            # Create filled area plot constrained to actual data bounds
            ax.fill_between(x_range, 0, density, 
                           color=plot_colors.get(method, 'blue'), alpha=0.7, zorder=2)
            ax.plot(x_range, density, 
                   color=plot_colors.get(method, 'blue'), linewidth=1.5, zorder=3)
            
        elif len(rmse_values) == 1:
            # Single point - show as a vertical line
            ax.axvline(x=rmse_values[0], color=plot_colors.get(method, 'blue'), 
                      linewidth=3, alpha=0.8, zorder=2)
        
        # Customize this subplot
        ax.set_ylim(0, None)
        ax.set_xlim(global_rmse_min, global_rmse_max)
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add controller name
        ax.text(0.02, 0.8, method, transform=ax.transAxes, 
               fontsize=12, ha='left', va='top', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Configure the bottom axis
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('RMSE [m]', fontsize=12)
    
    # Set title
    if noise_option == 'obs_noise':
        title = 'RMSE Distribution Across Observation Noise Levels'
    elif noise_option == 'proc_noise':
        title = 'RMSE Distribution Across Process Noise Levels'
    elif noise_option == 'param':
        title = 'RMSE Distribution Across Parametric Uncertainty Levels'
    else:
        title = f'RMSE Distribution Across {noise_option} Levels'
    
    fig.suptitle(title, fontsize=14, y=0.95)
    
    # Save ridge plot
    dr_suffix = "_with_dr" if include_domain_rand and any('(DR)' in method for method in noise_data.keys()) else ""
    ridge_save_name = f"ridge_plot_rmse_{method_type}_{noise_option}{dr_suffix}"
    ridge_save_path = script_dir / 'noise' / f'{ridge_save_name}.png'
    ridge_save_path.parent.mkdir(exist_ok=True)
    plt.savefig(ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Ridge plot saved as {ridge_save_path}")
    plt.close()
    
    return controller_bounds

def print_summary_report(noise_data, double_results, abs_rmse_threshold_results, max_testing_range, noise_option, method_type):
    """Print a comprehensive summary of the analysis."""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    # Noise type info
    if noise_option == 'obs_noise':
        noise_description = "Observation Noise"
        noise_unit = ""
    elif noise_option == 'proc_noise':
        noise_description = "Process Noise"
        noise_unit = ""
    elif noise_option == 'param':
        noise_description = "Parametric Uncertainty"
        noise_unit = ""
    else:
        noise_description = noise_option
        noise_unit = ""
    
    print(f"Noise Type: {noise_description}")
    loaded_method_type = infer_loaded_method_type(noise_data.keys(), method_type)
    print(f"Method Filter: {method_type.upper()}")
    if loaded_method_type != method_type:
        print(f"Methods Plotted: {loaded_method_type.upper()} (based on available data)")
    print(f"Controllers Analyzed: {len(noise_data)}")
    print(f"Maximum Testing Range: {max_testing_range:6.2f}{noise_unit}")
    
    # Relative performance summary
    print(f"\nRelative Performance ({RELATIVE_FAILURE_THRESHOLD}% degradation threshold):")
    if double_results:
        print("  Controllers that FAILED:")
        for method, failure_scale in sorted(double_results.items(), key=lambda x: x[1]):
            print(f"    {method:15}: {failure_scale:6.2f}{noise_unit}")
        
        robust_methods = set(noise_data.keys()) - set(double_results.keys())
        if robust_methods:
            print("  Controllers that SURVIVED:")
            for method in sorted(robust_methods):
                print(f"    {method:15}: No failure up to {max_testing_range:6.2f}{noise_unit}")
    else:
        print("  NO FAILURES detected for any controller")
    
    # Absolute performance summary
    if any('rmse_mean' in data for data in noise_data.values()):
        print(f"\nAbsolute Performance ({ABSOLUTE_FAILURE_THRESHOLD:.2f}m RMSE threshold):")
        if abs_rmse_threshold_results:
            print("  Controllers that FAILED:")
            for method, failure_scale in sorted(abs_rmse_threshold_results.items(), key=lambda x: x[1]):
                print(f"    {method:15}: {failure_scale:6.2f}{noise_unit}")
            
            robust_methods_abs = set(noise_data.keys()) - set(abs_rmse_threshold_results.keys())
            # Only include methods that have RMSE data
            robust_methods_abs = {m for m in robust_methods_abs if 'rmse_mean' in noise_data[m]}
            if robust_methods_abs:
                print("  Controllers that SURVIVED:")
                for method in sorted(robust_methods_abs):
                    print(f"    {method:15}: No failure up to {max_testing_range:6.2f}{noise_unit}")
        else:
            print("  NO FAILURES detected for any controller")
    
    print("\n" + "="*80)

def plot_single_noise(noise_option, method_type, include_domain_rand=False, dr_only=False):
    """Load data and create plots for one noise type."""
    print('noise_option', noise_option)
    print('method_type', method_type)
    print('include_domain_rand', include_domain_rand)
    print('dr_only', dr_only)
    
    # Load processed data
    noise_data = load_processed_data(noise_option, method_type, dr_only=dr_only)
    
    # Load domain randomization data (only if requested)
    dr_data = {}
    if include_domain_rand:
        if dr_only:
            print("Loading ONLY domain randomization variants...")
        else:
            print("Including domain randomization variants...")
        dr_data = load_domain_randomization_robustness_data(noise_option, method_type)
    else:
        print("Excluding domain randomization variants (use --include-dr to include)")
    
    # Combine regular and domain randomization data
    all_data = {**noise_data, **dr_data}
    
    if not all_data:
        print(f"No data files found for {noise_option}. Please run process_experiment_data.py first.")
        return False
    
    if dr_only:
        print(f"Loaded {len(dr_data)} domain randomization methods")
    else:
        print(f"Loaded {len(noise_data)} regular methods and {len(dr_data)} domain randomization methods")
    
    # Create extended plot colors for domain randomization variants
    global plot_colors
    extended_plot_colors = plot_colors.copy()
    if include_domain_rand:
        for method_name in dr_data.keys():
            if '(DR)' in method_name:
                base_method = method_name.replace(' (DR)', '')
                if base_method in plot_colors:
                    extended_plot_colors[method_name] = plot_colors[base_method]
    
    # Temporarily update global plot_colors
    original_plot_colors = plot_colors
    plot_colors = extended_plot_colors
    
    try:
        # Print failure analysis
        double_results, abs_rmse_threshold_results, max_testing_range = print_failure_analysis(all_data, noise_option)
        
        # Create plots (pass appropriate flags for file naming)
        create_robustness_plots(all_data, noise_option, method_type, double_results, abs_rmse_threshold_results, include_domain_rand or dr_only)

        # Create threshold summary tables for this noise type.
        create_single_noise_tables(
            all_data,
            noise_option,
            method_type,
            double_results,
            abs_rmse_threshold_results,
            max_testing_range,
            include_domain_rand or dr_only,
        )
        
        # # Create ridge plot (pass appropriate flags for file naming)
        # create_ridge_plot(all_data, noise_option, method_type, include_domain_rand or dr_only)
        
        # Print summary report
        print_summary_report(all_data, double_results, abs_rmse_threshold_results, max_testing_range, noise_option, method_type)
    finally:
        # Restore original plot_colors
        plot_colors = original_plot_colors
    
    return {
        'noise_data': all_data,
        'double_results': double_results,
        'abs_rmse_threshold_results': abs_rmse_threshold_results,
        'max_testing_range': max_testing_range,
    }

def main():
    """Main plotting function."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        noise_option = sys.argv[1].lower()
    else:
        noise_option = 'obs_noise'
    
    if len(sys.argv) > 2:
        method_type = sys.argv[2].lower()
        if method_type not in ['control', 'rl', 'all']:
            print(f"Warning: Invalid method type '{method_type}'. Using 'all'.")
            method_type = 'all'
    else:
        method_type = 'all'
    
    # Check for domain randomization flags
    include_domain_rand = '--include-dr' in sys.argv or '--domain-rand' in sys.argv
    dr_only = '--dr-only' in sys.argv
    
    # If dr_only is True, automatically include domain randomization and force method_type to rl
    if dr_only:
        include_domain_rand = True
        if method_type not in ['rl', 'all']:
            print("DR-only mode requires RL methods. Setting method_type to 'rl'.")
            method_type = 'rl'
    
    if noise_option == 'all':
        success_count = 0
        noise_summaries = {}
        for noise_type in VALID_NOISE_TYPES:
            print("\n" + "#" * 80)
            print(f"PLOTTING {noise_type.upper()}")
            print("#" * 80)
            summary = plot_single_noise(noise_type, method_type, include_domain_rand, dr_only)
            if summary:
                noise_summaries[noise_type] = summary
                success_count += 1
        if success_count == 0:
            print("No data files found for any noise type. Please run process_experiment_data.py first.")
        else:
            create_combined_robustness_tables(
                noise_summaries,
                method_type,
                include_domain_rand or dr_only,
            )
        return
    
    if noise_option not in VALID_NOISE_TYPES:
        print(f"Invalid noise type '{noise_option}'. Use one of {VALID_NOISE_TYPES}, or 'all'.")
        return
    
    plot_single_noise(noise_option, method_type, include_domain_rand, dr_only)

if __name__ == '__main__':
    main()
