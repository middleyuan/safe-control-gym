"""
Plot noise robustness analysis for different controller types.

Usage:
    python plot_noise_all.py <noise_type> [method_type]
    
Arguments:
    noise_type: 'obs_noise', 'proc_noise', or 'param'
    method_type: 'control', 'rl', or 'all' (default: 'all')
    
Examples:
    python plot_noise_all.py obs_noise all        # Plot all methods for observation noise
    python plot_noise_all.py proc_noise control   # Plot only control methods for process noise
    python plot_noise_all.py param rl            # Plot only RL methods for parametric noise
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, tag_ctrl_list

# script dir
script_dir = Path(__file__).parent.resolve()

max_seed = 10
metric_name = 'metrics.txt'
s = 2 # times std


if len(sys.argv) > 1:
    noise_option = sys.argv[1]
else:
    noise_option = 'obs_noise'
    # noise_option = 'proc_noise'
    # noise_option = 'param'

# Check for method type filter (control, rl, or all)
if len(sys.argv) > 2:
    method_type = sys.argv[2].lower()
    if method_type not in ['control', 'rl', 'all']:
        print(f"Warning: Invalid method type '{method_type}'. Using 'all'.")
        method_type = 'all'
else:
    method_type = 'all'

print('noise_option', noise_option)
print('method_type', method_type)

# Initialize empty noise_data dictionary
noise_data = {}

# Load model-based controller data if requested
if method_type in ['control', 'all']:
    print("Loading model-based controller data...")
    linear_mpc_data = np.load(script_dir / f'../data/linear_mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
    mpc_data = np.load(script_dir / f'../data/mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
    gpmpc_data = np.load(script_dir / f'../data/gpmpc_acados_TP_{noise_option}_results.npy', allow_pickle=True).item()
    ilqr_data = np.load(script_dir / f'../data/ilqr_{noise_option}_results.npy', allow_pickle=True).item()
    lqr_data = np.load(script_dir / f'../data/lqr_{noise_option}_results.npy', allow_pickle=True).item()
    pid_data = np.load(script_dir / f'../data/pid_{noise_option}_results.npy', allow_pickle=True).item()
    fmpc_data = np.load(script_dir / f'../data/fmpc_{noise_option}_results.npy', allow_pickle=True).item()
    
    # Add model-based controllers to noise_data
    noise_data.update({
        'Linear MPC': linear_mpc_data, 
        'Nonlinear MPC': mpc_data, 
        'GP-MPC': gpmpc_data,
        'iLQR': ilqr_data,
        'LQR': lqr_data,
        'Geometric Control': pid_data,
        'F-MPC': fmpc_data,
    })

# Load RL method data if requested
if method_type in ['rl', 'all']:
    print("Loading RL method data...")
    ppo_data = np.load(script_dir / f'../data/ppo_{noise_option}_results.npy', allow_pickle=True).item()
    sac_data = np.load(script_dir / f'../data/sac_{noise_option}_results.npy', allow_pickle=True).item()
    dppo_data = np.load(script_dir / f'../data/dppo_{noise_option}_results.npy', allow_pickle=True).item()
    ppo_mpc_data = np.load(script_dir / f'../data/ppo_mpc_{noise_option}_results.npy', allow_pickle=True).item()
    
    # Add RL controllers to noise_data
    noise_data.update({
        'PPO': ppo_data,
        'SAC': sac_data,
        'DPPO': dppo_data,
        'PPO-MPC': ppo_mpc_data,
    })

# Determine legend columns based on number of methods
num_methods = len(noise_data)
if num_methods <= 4:
    legend_cols = 2
elif num_methods <= 7:
    legend_cols = 3
else:
    legend_cols = 4

print(f"Loaded {num_methods} methods, using {legend_cols} legend columns")
    
# Define noise scales based on noise type (use the model-based controller scales for consistency)
if noise_option in ['obs_noise']:
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,\
                         12, 14, 16, 18, 20, 25, 30, 35, 40, \
                         45, 50, 60, 70, 80, 90, 100]
elif noise_option == 'proc_noise':
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,\
                    12, 14, 16, 18, 20, 25, 30, 35, 40, \
                    45, 50, 60, 70, 80, 90, 100]
elif noise_option == 'param':
    noise_scale = [0, 0.01, 0.02, 0.05, 0.1, \
                     0.2, 0.4, 0.6, 0.8, 1.0, 1.2, \
                     1.4, 1.6, 1.8, 2.0, 2.2, 2.4, \
                     2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
    noise_scale.sort()

# Function to interpolate or align data to common noise scale
def align_data_to_scale(data, target_scale):
    """Align data to target noise scale by interpolation or subsetting."""
    original_scale = data['noise_factor']
    original_rmse_deg_mean = data['rmse_degradation_mean']
    original_rmse_deg_std = data['rmse_degradation_std']
    original_rmse_mean = data.get('rmse_mean', None)
    original_rmse_std = data.get('rmse_std', None)
    
    # Find common indices
    aligned_rmse_deg_mean = []
    aligned_rmse_deg_std = []
    aligned_rmse_mean = [] if original_rmse_mean is not None else None
    aligned_rmse_std = [] if original_rmse_std is not None else None
    
    for target_val in target_scale:
        # Find closest match in original scale
        if target_val in original_scale:
            idx = list(original_scale).index(target_val)
            aligned_rmse_deg_mean.append(original_rmse_deg_mean[idx])
            aligned_rmse_deg_std.append(original_rmse_deg_std[idx])
            if original_rmse_mean is not None:
                aligned_rmse_mean.append(original_rmse_mean[idx])
                aligned_rmse_std.append(original_rmse_std[idx])
        else:
            # Use interpolation or NaN for missing values
            aligned_rmse_deg_mean.append(np.nan)
            aligned_rmse_deg_std.append(np.nan)
            if original_rmse_mean is not None:
                aligned_rmse_mean.append(np.nan)
                aligned_rmse_std.append(np.nan)
    
    aligned_data = data.copy()
    aligned_data['rmse_degradation_mean'] = np.array(aligned_rmse_deg_mean)
    aligned_data['rmse_degradation_std'] = np.array(aligned_rmse_deg_std)
    if original_rmse_mean is not None:
        aligned_data['rmse_mean'] = np.array(aligned_rmse_mean)
        aligned_data['rmse_std'] = np.array(aligned_rmse_std)
    
    return aligned_data

# Align all data to common noise scale
for method in noise_data.keys():
    noise_data[method] = align_data_to_scale(noise_data[method], noise_scale)

# Analyze testing ranges and find intersection
def analyze_testing_ranges():
    """Analyze and report the testing ranges for each controller."""
    print("\n" + "="*80)
    print("TESTING RANGE ANALYSIS")
    print("="*80)
    
    controller_ranges = {}
    for method in noise_data.keys():
        x_data = np.array(noise_scale)
        y_mean = noise_data[method]['rmse_degradation_mean']
        valid_mask = ~np.isnan(y_mean)
        
        if np.any(valid_mask):
            x_valid = x_data[valid_mask]
            min_range = np.min(x_valid)
            max_range = np.max(x_valid)
            controller_ranges[method] = {'min': min_range, 'max': max_range, 'valid_points': len(x_valid)}
            print(f"{method:15}: Range [{min_range:6.2f}, {max_range:6.2f}] - {len(x_valid)} valid points")
    
    # Find intersection of all ranges
    if controller_ranges:
        intersection_min = max([r['min'] for r in controller_ranges.values()])
        intersection_max = min([r['max'] for r in controller_ranges.values()])
        print(f"\nCommon range intersection: [{intersection_min:6.2f}, {intersection_max:6.2f}]")
        print(f"Maximum testing range: {max([r['max'] for r in controller_ranges.values()]):6.2f}")
    
    return controller_ranges

controller_ranges = analyze_testing_ranges()
max_testing_range = max([r['max'] for r in controller_ranges.values()]) if controller_ranges else max(noise_scale)

fig = plt.figure(figsize=(8, 3))

# plot rmse degradation for all methods
double_results = {}
relative_failure_threshold = 200  # 200% performance degradation

print("\n" + "="*80)
print("RELATIVE PERFORMANCE ANALYSIS (200% Degradation Threshold)")
print("="*80)

for method in noise_data.keys():
    # Get aligned data and filter out NaN values
    x_data = np.array(noise_scale)
    y_mean = noise_data[method]['rmse_degradation_mean']
    y_std = noise_data[method]['rmse_degradation_std']
    
    # Filter out NaN values
    valid_mask = ~np.isnan(y_mean)
    if np.any(valid_mask):
        x_valid = x_data[valid_mask]
        y_mean_valid = y_mean[valid_mask]
        y_std_valid = y_std[valid_mask]
        
        plt.plot(x_valid, y_mean_valid, label=method, color=plot_colors[method])
        plt.fill_between(x_valid, 
                         y_mean_valid - s * y_std_valid,  
                         y_mean_valid + s * y_std_valid, color=plot_colors[method], alpha=0.1)
        
        # Find when RMSE degradation exceeds threshold using valid data
        failure_found = False
        for i, scale in enumerate(x_valid):
            if y_mean_valid[i] > relative_failure_threshold:
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
    
plt.xlabel("Noise Scale")
plt.ylabel("Relative Performance %")
if noise_option == 'obs_noise':
    plt.legend(ncol=legend_cols, fontsize=9)
    plt.ylim(0, 1200)
    plt.xlim(0, 100)
    plt.title("Performance Degradation with Observation Noise")
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'proc_noise':
    plt.legend(ncol=legend_cols, loc='upper right', fontsize=9)
    plt.title("Performance Degradation with Process Noise")
    plt.plot(noise_scale, [200]*len(noise_scale), \
             color='grey', linestyle='-.', label='Relative Perf=200%')
    # plot vertical line at 200% performance
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'param':
    plt.title("Performance Degradation with Parametric Uncertainty")
    plt.xlabel("Randomization scale")
    plt.ylim(0, 2000)
    plt.xlim(0, 5)
    plt.legend(ncol=legend_cols, loc='upper right', fontsize=9)
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
# plot an empty line for the legend
plt.plot(0, np.nan, linestyle='--', color='grey', label='Relative Perf=200%')

plt.plot(noise_scale, [200]*len(noise_scale), \
            color='grey', linestyle='-.', label='Relative Perf=200%')
plot_save_name = f"robustness_{method_type}_methods_{noise_option}"
plot_save_path = script_dir / 'noise' / f'{plot_save_name}.png'
plt.savefig(plot_save_path,bbox_inches="tight", pad_inches=0.1)
plt.savefig(plot_save_path.with_suffix('.pdf'),bbox_inches="tight", pad_inches=0.1)
print(f"plots saved as {plot_save_path}")

# Absolute performance degradation
fig_abs = plt.figure(figsize=(8, 3))

# abs_rmse_threshold = 0.1
abs_rmse_threshold = 0.25
abs_rmse_threshold_results = {}

print("\n" + "="*80)
print(f"ABSOLUTE PERFORMANCE ANALYSIS ({abs_rmse_threshold:.2f}m RMSE Threshold)")
print("="*80)

for method in noise_data.keys():
    if 'rmse_mean' in noise_data[method] and 'rmse_std' in noise_data[method]:
        # Get aligned data and filter out NaN values
        x_data = np.array(noise_scale)
        y_mean = noise_data[method]['rmse_mean']
        y_std = noise_data[method]['rmse_std']
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y_mean)
        if np.any(valid_mask):
            x_valid = x_data[valid_mask]
            y_mean_valid = y_mean[valid_mask]
            y_std_valid = y_std[valid_mask]
            
            plt.plot(x_valid, y_mean_valid, label=method, color=plot_colors[method])
            plt.fill_between(x_valid,
                             y_mean_valid - s * y_std_valid,
                             y_mean_valid + s * y_std_valid,
                             color=plot_colors[method], alpha=0.1)
            
            # Check when absolute rmse exceeds threshold using valid data
            failure_found = False
            for i, scale in enumerate(x_valid):
                if y_mean_valid[i] > abs_rmse_threshold:
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

plt.xlabel("Noise Scale")
plt.ylabel("RMSE [m]")
if noise_option == 'obs_noise':
    plt.legend(ncol=legend_cols, fontsize=9)
    plt.xlim(0, 100)
    plt.ylim(0, 0.3)
    plt.title("Absolute Performance with Observation Noise")
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'proc_noise':
    plt.legend(ncol=legend_cols, loc='upper left', fontsize=9)
    plt.ylim(0, 1.5)
    plt.title("Absolute Performance with Process Noise")
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'param':
    plt.title("Absolute Performance with Parametric Uncertainty")
    plt.xlabel("Randomization scale")
    plt.xlim(0, 5)
    plt.ylim(0, 0.5)
    plt.legend(ncol=legend_cols, loc='upper left', fontsize=9)
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])

plt.plot(noise_scale, [abs_rmse_threshold]*len(noise_scale), \
            color='grey', linestyle='-.', label=f'RMSE={abs_rmse_threshold}')

plot_save_name_abs = f"robustness_{method_type}_methods_{noise_option}_absolute"
plot_save_path_abs = script_dir / 'noise' / f'{plot_save_name_abs}.png'
plt.savefig(plot_save_path_abs, bbox_inches="tight", pad_inches=0.1)
plt.savefig(plot_save_path_abs.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
print(f"plots saved as {plot_save_path_abs}")

def create_ridge_plot():
    """Create a ridge plot showing RMSE distributions for each controller using only actual data."""
    
    # Calculate actual data bounds for each controller first
    controller_bounds = {}
    all_rmse_values = []
    
    for method in noise_data.keys():
        if 'rmse_mean' in noise_data[method]:
            # Get only actual RMSE values (no synthetic data)
            rmse_values = noise_data[method]['rmse_mean']
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
        # Left side: from global min to controller min
        if bounds['min'] > global_rmse_min:
            ax.axvspan(global_rmse_min, bounds['min'], alpha=0.2, color='gray', zorder=0)
        
        # Right side: from controller max to global max  
        if bounds['max'] < global_rmse_max:
            ax.axvspan(bounds['max'], global_rmse_max, alpha=0.2, color='gray', zorder=0)
        
        # Create KDE only within the actual data bounds (no extrapolation)
        rmse_values = np.array(bounds['values'])
        
        if len(rmse_values) > 1:
            # Use scipy's gaussian_kde for better control
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(rmse_values)
            
            # Evaluate KDE only within actual data range (no extrapolation)
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
        
        # Add mean line
        # mean_rmse = np.mean(rmse_values)
        # ax.axvline(x=mean_rmse, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=4)
        
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
        
        # # Add statistics
        # ax.text(0.98, 0.8, f'μ={mean_rmse:.3f}\nσ={np.std(rmse_values):.3f}', 
        #        transform=ax.transAxes, fontsize=10, ha='right', va='top',
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # print(f"Controller {method}: RMSE range [{bounds['min']:.3f}, {bounds['max']:.3f}], mean={mean_rmse:.3f}")
    
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
    ridge_save_name = f"ridge_plot_rmse_{method_type}_{noise_option}"
    ridge_save_path = script_dir / 'noise' / f'{ridge_save_name}.png'
    plt.savefig(ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Ridge plot saved as {ridge_save_path}")
    plt.close()
    
    return controller_bounds

# Create ridge plot
controller_bounds = create_ridge_plot()

# Final summary report
def print_summary_report():
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
    print(f"Method Filter: {method_type.upper()}")
    print(f"Controllers Analyzed: {len(noise_data)}")
    print(f"Maximum Testing Range: {max_testing_range:6.2f}{noise_unit}")
    
    # Relative performance summary
    print(f"\nRelative Performance ({relative_failure_threshold}% degradation threshold):")
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
        print(f"\nAbsolute Performance ({abs_rmse_threshold:.2f}m RMSE threshold):")
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

print_summary_report()

# Save failure results for use in plot_radar.py
def save_failure_results():
    """Save failure results to JSON files for use in plot_radar.py."""
    import json
    
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
    
    # Create data directory if it doesn't exist
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Prepare failure results for different noise types
    failure_data = {
        'relative_failures': {},  # 200% degradation threshold failures
        'absolute_failures': {},  # 0.25m RMSE threshold failures
        'max_testing_range': convert_numpy_types(max_testing_range),
        'relative_threshold': convert_numpy_types(relative_failure_threshold),
        'absolute_threshold': convert_numpy_types(abs_rmse_threshold),
        'noise_type': noise_option,
        'method_filter': method_type
    }
    
    # Add relative failure results (200% degradation threshold)
    for method, failure_scale in double_results.items():
        failure_data['relative_failures'][method] = convert_numpy_types(failure_scale)
    
    # Add absolute failure results (0.25m RMSE threshold)  
    for method, failure_scale in abs_rmse_threshold_results.items():
        failure_data['absolute_failures'][method] = convert_numpy_types(failure_scale)
    
    # Add non-failure information (methods that survived up to max testing range)
    robust_methods_relative = set(noise_data.keys()) - set(double_results.keys())
    robust_methods_absolute = set(noise_data.keys()) - set(abs_rmse_threshold_results.keys())
    # Only include methods that have RMSE data for absolute analysis
    robust_methods_absolute = {m for m in robust_methods_absolute if 'rmse_mean' in noise_data[m]}
    
    failure_data['robust_methods_relative'] = list(robust_methods_relative)
    failure_data['robust_methods_absolute'] = list(robust_methods_absolute)
    
    # Save to JSON file
    output_file = data_dir / f'failure_results_{noise_option}_{method_type}.json'
    with open(output_file, 'w') as f:
        json.dump(failure_data, f, indent=2)
    
    print(f"\nFailure results saved to: {output_file}")
    
    # Also save a consolidated file for radar plot usage
    # This maps noise types to method failure points
    consolidated_file = data_dir / 'robustness_failure_points.json'
    
    # Load existing data if file exists
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
    for method, failure_scale in double_results.items():
        consolidated_data['relative_failures'][noise_option][method] = convert_numpy_types(failure_scale)
    
    for method, failure_scale in abs_rmse_threshold_results.items():
        consolidated_data['absolute_failures'][noise_option][method] = convert_numpy_types(failure_scale)
    
    # Save consolidated data
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    print(f"Consolidated failure results updated: {consolidated_file}")

save_failure_results()
