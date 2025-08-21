"""
Plot generalization analysis for different controller types.

Usage:
    python plot_generalization.py [option] [--include-dr]
    
Arguments:
    option: 'all', 'control', or 'rl' (default: 'all')
    --include-dr: Include domain randomization variants (optional)
    
Examples:
    python plot_generalization.py all           # Plot all methods (no DR)
    python plot_generalization.py rl --include-dr  # Plot RL methods including domain randomization
    python plot_generalization.py control       # Plot only control methods
    python plot_generalization.py rl           # Plot only RL methods
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, tag_ctrl_list, load_metric
import sys

def load_domain_randomization_data(script_dir, method_name):
    """Load domain randomization generalization data for RL methods."""
    method_map = {
        'PPO': 'ppo',
        'SAC': 'sac', 
        'DPPO': 'dppo'
    }
    
    if method_name not in method_map:
        return None
    
    data_file = script_dir / f'../data/{method_map[method_name]}_domain_rand_generalization.npy'
    
    if not data_file.exists():
        print(f"Domain randomization file not found: {data_file}")
        return None
    
    try:
        data = np.load(data_file, allow_pickle=True).item()
        episode_lengths = [9, 10, 11, 12, 13, 14, 15]
        
        rmse_values = []
        rmse_std_values = []
        
        for episode_len in episode_lengths:
            if episode_len in data['generalization']:
                rmse_values.append(data['generalization'][episode_len]['rmse']['mean'])
                rmse_std_values.append(data['generalization'][episode_len]['rmse']['std'])
            else:
                rmse_values.append(np.nan)
                rmse_std_values.append(np.nan)
        
        return {
            'rmse': np.array(rmse_values),
            'rmse_std': np.array(rmse_std_values),
            'inference_time': 0.0,  # placeholder
            'inference_time_std': 0.0  # placeholder
        }
    except Exception as e:
        print(f"Error loading domain randomization data for {method_name}: {e}")
        return None

################ plot options ################

# Read option from command line arguments, otherwise use default
if len(sys.argv) > 1:
    option = sys.argv[1]
else:
    option = 'all'  # default option
    # option = 'control'
    # options = 'rl'

# Check for domain randomization flag
include_domain_rand = '--include-dr' in sys.argv or '--domain-rand' in sys.argv

assert option in ['all', 'control', 'rl'], f"Invalid option: {option}. Must be one of ['all', 'control', 'rl']"
##############################################
control_list = [
    'Geometric Control', 'Linear MPC', 'Nonlinear MPC',
    'F-MPC', 'iLQR', 'LQR', 'GP-MPC',
]
rl_list = ['PPO', 'SAC', 'DPPO', 'PPO-MPC']

script_dir = Path.cwd()
# Create the 'generalization' folder if it doesn't exist
output_dir = script_dir / "generalization"
output_dir.mkdir(exist_ok=True)

# Load domain randomization data for RL methods first (only if requested)
domain_rand_metrics = {}
if include_domain_rand:
    print("Including domain randomization variants...")
    for method in ['PPO', 'SAC', 'DPPO']:
        dr_data = load_domain_randomization_data(script_dir, method)
        if dr_data is not None:
            domain_rand_metrics[f'{method} (DR)'] = dr_data
            print(f"Loaded domain randomization data for {method}")
        else:
            print(f"No domain randomization data found for {method}")
else:
    print("Excluding domain randomization variants (use --include-dr to include)")

# Add domain randomization variants for RL methods (only if data was loaded)
rl_dr_list = []
if include_domain_rand:
    for method in ['PPO', 'SAC', 'DPPO']:
        if f'{method} (DR)' in domain_rand_metrics:
            rl_dr_list.append(f'{method} (DR)')

if option == 'control':
    plot_list = control_list
elif option == 'rl':
    plot_list = rl_list + rl_dr_list
elif option == 'all':
    plot_list = control_list + rl_list + rl_dr_list

# Update plot colors for domain randomization variants (use same color as nominal)
extended_plot_colors = plot_colors.copy()
if include_domain_rand:
    for method in ['PPO', 'SAC', 'DPPO']:
        if f'{method} (DR)' in domain_rand_metrics:
            extended_plot_colors[f'{method} (DR)'] = plot_colors[method]

# Define the episode lengths and periods
episode_len_list = [9, 10, 11, 12, 13, 14, 15]
episode_period_list =  [episode_len_list[i]/2 for i in range(len(episode_len_list))]

# Initialize the transfer_metric dictionary
transfer_metric = {}
transfer_metric = load_metric(script_dir, transfer_metric, 'PPO')
transfer_metric = load_metric(script_dir, transfer_metric, 'SAC')
transfer_metric = load_metric(script_dir, transfer_metric, 'DPPO')
transfer_metric = load_metric(script_dir, transfer_metric, 'PPO-MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'iLQR')
transfer_metric = load_metric(script_dir, transfer_metric, 'F-MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Nonlinear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Linear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Geometric Control')
transfer_metric = load_metric(script_dir, transfer_metric, 'LQR')
tag = ''
transfer_metric = load_metric(script_dir, transfer_metric, 'GP-MPC', tag=tag)

# Add domain randomization metrics to transfer_metric
transfer_metric.update(domain_rand_metrics)

# Ensure plotting order: largest episode period on the left, smallest on the right
episode_len_list = [9, 10, 11, 12, 13, 14, 15]
episode_period_list = [T / 2 for T in episode_len_list]
# Sort in descending order for plotting (largest period left)
episode_period_sorted_idx = np.argsort(episode_period_list)[::-1]
episode_period_list = [episode_period_list[i] for i in episode_period_sorted_idx]
episode_len_list = [episode_len_list[i] for i in episode_period_sorted_idx]

# Reorder transfer_metric arrays to match the plotting order
for method in plot_list:
    transfer_metric[method]['rmse'] = transfer_metric[method]['rmse'][episode_period_sorted_idx]
    transfer_metric[method]['rmse_std'] = transfer_metric[method]['rmse_std'][episode_period_sorted_idx]

# Set seaborn style
sns.set_theme(style="whitegrid")

# plot generalization curves
fig, ax = plt.subplots(figsize=(8, 3))
for method in plot_list:
    # Use dashed line for domain randomization variants
    linestyle = '--' if '(DR)' in method else '-'
    color = extended_plot_colors.get(method, extended_plot_colors.get(method.replace(' (DR)', ''), 'blue'))
    
    ax.plot(episode_period_list, 
            transfer_metric[method]['rmse'], 
            label=method, color=color, linestyle=linestyle)
    ax.fill_between(episode_period_list, 
                    transfer_metric[method]['rmse'] - transfer_metric[method]['rmse_std'],  
                    transfer_metric[method]['rmse'] + transfer_metric[method]['rmse_std'], 
                    color=color, alpha=0.1)
ax.axvline(x=5.5, linestyle='-.', color='gray')
ax.text(0.81, 0.75, "Nominal Task", transform=ax.transAxes, fontsize=10, verticalalignment='center', horizontalalignment='right')

# Only create inset for 'all' and 'control' options
if option in ['all', 'control']:
    # Create a zoomed-in inset plot
    fill_alpha = 0.2
    ax_inset = inset_axes(ax, width="40%", height="50%", loc='upper left', 
                        bbox_to_anchor=(0.1, -0.05, 1, 1), bbox_transform=ax.transAxes)
    ax_inset.set_facecolor((0.5, 0.5, 0.5, fill_alpha))  # Set background to transparent gray
    for method in plot_list:
        linestyle = '--' if '(DR)' in method else '-'
        color = extended_plot_colors.get(method, extended_plot_colors.get(method.replace(' (DR)', ''), 'blue'))
        
        ax_inset.plot(episode_period_list, 
                    transfer_metric[method]['rmse'], 
                    color=color, linestyle=linestyle)
        ax_inset.fill_between(episode_period_list, 
                            transfer_metric[method]['rmse'] - transfer_metric[method]['rmse_std'],  
                            transfer_metric[method]['rmse'] + transfer_metric[method]['rmse_std'], 
                            color=color, alpha=0.1)
    mark_inset(ax, ax_inset, loc1=1, loc2=3, 
                fc="gray", ec="0.3", alpha=fill_alpha)
    ax_inset.set_ylim(0, 0.045)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.invert_xaxis()

ax.set_ylim(0, 0.17)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
ax.invert_xaxis()
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Figure-Eight Trajectory Period (s)")
ax.set_ylabel("RMSE [m]")

# Save the plot to the 'generalization' folder in both PDF and PNG formats
dr_suffix = "_with_dr" if include_domain_rand and rl_dr_list else ""
fig.savefig(output_dir / f"generalization_curve_{option}{dr_suffix}.pdf", bbox_inches="tight", pad_inches=0.1)
fig.savefig(output_dir / f"generalization_curve_{option}{dr_suffix}.png", bbox_inches="tight", pad_inches=0.1)

# Convert the transfer metric to a table
transfer_metric_table = {}
for method in plot_list:
    transfer_metric_table[method] = {}
    for i, T in enumerate(episode_len_list):
        transfer_metric_table[method][T] = f"{transfer_metric[method]['rmse'][i]:.3f} +/- {transfer_metric[method]['rmse_std'][i]:.3f}"

# Create a pandas DataFrame
df = pd.DataFrame(transfer_metric_table)

# Save the DataFrame to a CSV file in the 'generalization' folder
df.to_csv(output_dir / f"transfer_metric_table_{option}{dr_suffix}.csv", index_label="Episode Length")

# Save the DataFrame to a PNG file in the 'generalization' folder
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))
fig.savefig(output_dir / f"transfer_metric_table_{option}{dr_suffix}.png", bbox_inches='tight')
plt.close(fig)

# Print the path where the plots and table are saved
print(f"Plots and table saved in: {output_dir.resolve()}")