"""
Custom ridge plot generator for arbitrary controller selection.

This script allows you to generate ridge plots with any combination of controllers,
including domain randomization variants.

Usage:
    python plot_custom_ridge.py <noise_type> <controller1> <controller2> ... [--include-dr controller_dr1 ...]
    
Arguments:
    noise_type: 'obs_noise', 'proc_noise', or 'param'
    controllers: Space-separated list of controller names
    --include-dr: Optional flag followed by space-separated list of controllers to include DR variants
    
Available Controllers:
    Model-based: linear_mpc_acados, mpc_acados, gpmpc_acados_TP, ilqr, lqr, pid, fmpc
    RL: ppo, sac, dppo, ppo_mpc
    
Examples:
    python plot_custom_ridge.py obs_noise lqr fmpc mpc_acados sac --include-dr sac
    python plot_custom_ridge.py param ppo sac dppo --include-dr ppo sac dppo
    python plot_custom_ridge.py proc_noise linear_mpc_acados gpmpc_acados_TP pid
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors

# script dir
script_dir = Path(__file__).parent.resolve()

def load_controller_data(controller_name, noise_type, include_dr=False):
    """Load data for a specific controller."""
    
    # Controller name mappings for display
    display_name_map = {
        'linear_mpc_acados': 'Linear MPC',
        'mpc_acados': 'Nonlinear MPC', 
        'gpmpc_acados_TP': 'GP-MPC',
        'ilqr': 'iLQR',
        'lqr': 'LQR',
        'pid': 'Geometric Control',
        'fmpc': 'F-MPC',
        'ppo': 'PPO',
        'sac': 'SAC',
        'dppo': 'DPPO',
        'ppo_mpc': 'PPO-MPC',
    }
    
    controller_data = {}
    
    # Load regular controller data
    file_path = script_dir / f'../data/{controller_name}_{noise_type}_results.npy'
    if file_path.exists():
        try:
            data = np.load(file_path, allow_pickle=True).item()
            display_name = display_name_map.get(controller_name, controller_name.upper())
            controller_data[display_name] = data
            print(f"  Loaded: {display_name}")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    else:
        print(f"  Warning: {file_path} not found")
    
    # Load domain randomization data if requested and available
    if include_dr and controller_name in ['ppo', 'sac', 'dppo']:
        dr_file = script_dir / f'../data/{controller_name}_domain_rand_robustness.npy'
        if dr_file.exists():
            try:
                dr_data = np.load(dr_file, allow_pickle=True).item()
                
                # Map noise type names
                noise_type_map = {
                    'obs_noise': 'obs_noise',
                    'proc_noise': 'proc_noise', 
                    'param': 'param'
                }
                
                if noise_type_map[noise_type] in dr_data['robustness']:
                    robustness_data = dr_data['robustness'][noise_type_map[noise_type]]
                    
                    # Convert to the same format as normal robustness data
                    noise_scales = sorted(robustness_data.keys())
                    rmse_means = []
                    rmse_stds = []
                    
                    for scale in noise_scales:
                        scale_data = robustness_data[scale]
                        rmse_means.append(scale_data['rmse']['mean'])
                        rmse_stds.append(scale_data['rmse']['std'])
                    
                    # Calculate degradation relative to baseline
                    if noise_scales and noise_scales[0] == 0.0:
                        baseline_rmse = rmse_means[0]
                    else:
                        baseline_rmse = min(rmse_means) if rmse_means else 1.0
                    
                    rmse_degradation_mean = [(rmse / baseline_rmse) * 100 for rmse in rmse_means]
                    rmse_degradation_std = [(std / baseline_rmse) * 100 for std in rmse_stds]
                    
                    display_name = display_name_map.get(controller_name, controller_name.upper()) + " (DR)"
                    
                    controller_data[display_name] = {
                        'noise_factor': np.array(noise_scales),
                        'rmse_mean': np.array(rmse_means),
                        'rmse_std': np.array(rmse_stds),
                        'rmse_degradation_mean': np.array(rmse_degradation_mean),
                        'rmse_degradation_std': np.array(rmse_degradation_std),
                        'baseline_rmse': baseline_rmse,
                        'controller': f"{controller_name}_domain_rand",
                        'noise_type': noise_type
                    }
                    
                    print(f"  Loaded: {display_name}")
                else:
                    print(f"  No {noise_type} data found in DR file for {controller_name}")
            except Exception as e:
                print(f"  Error loading DR data for {controller_name}: {e}")
        else:
            print(f"  Warning: DR file not found for {controller_name}")
    
    return controller_data

def create_custom_ridge_plot(noise_data, noise_type, controller_list):
    """Create a custom ridge plot for selected controllers."""
    
    if not noise_data:
        print("No valid data for ridge plot")
        return None
    
    # Calculate actual data bounds for each controller
    controller_bounds = {}
    all_rmse_values = []
    
    for method in noise_data.keys():
        if 'rmse_mean' in noise_data[method]:
            # Get only actual RMSE values
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
    global_rmse_min = max(0, min(all_rmse_values) * 0.95)
    global_rmse_max = max(all_rmse_values) * 1.05
    
    print(f"Global RMSE range: [{global_rmse_min:.4f}, {global_rmse_max:.4f}]")
    
    # Create ridge plot
    fig, axes = plt.subplots(len(controller_bounds), 1, figsize=(8, len(controller_bounds) * 1.5), 
                            sharex=True, gridspec_kw={'hspace': 0.1})
    
    if len(controller_bounds) == 1:
        axes = [axes]
    
    # Extend plot colors for DR variants
    extended_colors = plot_colors.copy()
    for method in controller_bounds.keys():
        if '(DR)' in method:
            base_method = method.replace(' (DR)', '')
            if base_method in plot_colors:
                extended_colors[method] = plot_colors[base_method]
    
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
            
            # Create evaluation points only within data range
            x_eval = np.linspace(bounds['min'], bounds['max'], 200)
            y_eval = kde(x_eval)
            
            # Normalize to fit in the subplot height
            y_eval = y_eval / np.max(y_eval) * 0.8
            
            # Plot the KDE curve
            color = extended_colors.get(method, 'blue')
            linestyle = '--' if '(DR)' in method else '-'
            ax.fill_between(x_eval, 0, y_eval, alpha=0.6, color=color)
            ax.plot(x_eval, y_eval, color=color, linewidth=2, linestyle=linestyle)
            
        elif len(rmse_values) == 1:
            # Single point - draw a vertical line
            color = extended_colors.get(method, 'blue')
            ax.axvline(x=rmse_values[0], color=color, linewidth=3, alpha=0.8)
        
        # Set controller name on y-axis
        ax.set_ylabel(method, rotation=0, ha='right', va='center', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Print range info
        print(f"{method:15}: Range [{bounds['min']:.4f}, {bounds['max']:.4f}] - {len(bounds['values'])} points")
    
    # Set common x-label and title
    axes[-1].set_xlabel('RMSE (m)', fontsize=12)
    
    # Create title based on noise type
    noise_title_map = {
        'obs_noise': 'Observation Noise',
        'proc_noise': 'Process Noise',
        'param': 'Parametric Uncertainty'
    }
    
    title = f'RMSE Distribution Ridge Plot - {noise_title_map.get(noise_type, noise_type)}'
    fig.suptitle(title, fontsize=14, y=0.95)
    
    # Set global x-limits
    for ax in axes:
        ax.set_xlim(global_rmse_min, global_rmse_max)
    
    # Save ridge plot
    controller_names_str = "_".join([name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for name in controller_bounds.keys()])
    ridge_save_name = f"custom_ridge_{noise_type}_{controller_names_str}"
    ridge_save_path = script_dir / 'custom_plots' / f'{ridge_save_name}.png'
    ridge_save_path.parent.mkdir(exist_ok=True)
    plt.savefig(ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Custom ridge plot saved as {ridge_save_path}")
    plt.show()
    
    return controller_bounds

def main():
    """Main function to parse arguments and generate custom ridge plot."""
    
    if len(sys.argv) < 3:
        print("Usage: python plot_custom_ridge.py <noise_type> <controller1> <controller2> ... [--include-dr controller_dr1 ...]")
        print("Example: python plot_custom_ridge.py obs_noise lqr fmpc mpc_acados sac --include-dr sac")
        return
    
    # Parse arguments
    args = sys.argv[1:]
    noise_type = args[0]
    
    if noise_type not in ['obs_noise', 'proc_noise', 'param']:
        print(f"Error: Invalid noise type '{noise_type}'. Must be one of: obs_noise, proc_noise, param")
        return
    
    # Find --include-dr flag
    dr_flag_idx = None
    if '--include-dr' in args:
        dr_flag_idx = args.index('--include-dr')
    
    # Extract controller lists
    if dr_flag_idx is not None:
        controllers = args[1:dr_flag_idx]
        dr_controllers = args[dr_flag_idx+1:]
    else:
        controllers = args[1:]
        dr_controllers = []
    
    if not controllers:
        print("Error: No controllers specified")
        return
    
    print(f"Noise type: {noise_type}")
    print(f"Controllers: {controllers}")
    print(f"DR Controllers: {dr_controllers}")
    
    # Load data for all requested controllers
    all_data = {}
    
    print("\nLoading controller data...")
    for controller in controllers:
        include_dr = controller in dr_controllers
        controller_data = load_controller_data(controller, noise_type, include_dr)
        all_data.update(controller_data)
    
    if not all_data:
        print("No data loaded. Please check controller names and ensure data files exist.")
        print("Run process_robustness_data.py first to generate the required data files.")
        return
    
    print(f"\nLoaded {len(all_data)} controller variants")
    
    # Generate custom ridge plot
    create_custom_ridge_plot(all_data, noise_type, controllers)

if __name__ == '__main__':
    main()
