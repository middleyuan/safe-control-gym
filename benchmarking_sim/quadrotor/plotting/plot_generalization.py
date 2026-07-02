"""
Plot generalization analysis for different controller types.

Usage:
    python plot_generalization.py [option] [--include-dr] [--use-dr] [--include-gen] [--use-gen]
    
Arguments:
    option: 'all', 'control', 'rl', 'dr', or 'gen' (default: 'all')
    --include-dr: Add robustness-combo DR variants alongside nominal RL methods
    --use-dr: Replace nominal PPO/SAC/DPPO curves with robustness-combo DR variants
    --include-gen: Add generalization-trained variants alongside nominal RL methods
    --use-gen: Replace nominal PPO/SAC/DPPO curves with generalization-trained variants
    
Examples:
    python plot_generalization.py all           # Plot all methods (no DR)
    python plot_generalization.py rl --include-dr  # Plot RL methods including DR variants
    python plot_generalization.py all --use-dr  # Plot all methods, using DR variants where available
    python plot_generalization.py all --use-gen  # Plot all methods, using GEN variants where available
    python plot_generalization.py control       # Plot only control methods
    python plot_generalization.py rl           # Plot only RL methods
    python plot_generalization.py dr           # Plot only DR methods
    python plot_generalization.py gen          # Plot only GEN methods
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, load_metric, plotting_data_dir
import sys

def load_variant_generalization_data(script_dir, method_name, variant):
    """Load GEN or DR generalization-evaluation data for RL methods."""
    method_map = {
        'PPO': 'ppo',
        'SAC': 'sac', 
        'DPPO': 'dppo'
    }
    
    if method_name not in method_map:
        return None
    
    data_dir = plotting_data_dir(script_dir)
    data_file = data_dir / f'{method_map[method_name]}_{variant}_generalization.npy'

    # Backward-compatible fallback for older processed GEN files.
    if variant == 'gen' and not data_file.exists():
        data_file = data_dir / f'{method_map[method_name]}_domain_rand_generalization.npy'
    
    if not data_file.exists():
        print(f"{variant.upper()} generalization file not found: {data_file}")
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
            'inference_time': 0.0,
            'inference_time_std': 0.0,
            'source_data_root': data.get('source_data_root'),
            'source_experiment': data.get('source_experiment', variant)
        }
    except Exception as e:
        print(f"Error loading {variant.upper()} generalization data for {method_name}: {e}")
        return None

def variant_label(method_name, variant):
    """Return the displayed method label for a plotted variant."""
    return f'{method_name} ({variant.upper()})'

def base_method_name(method_name):
    """Strip plotted variant suffixes from a method label."""
    return (
        method_name
        .replace(' (DR)', '')
        .replace(' (GEN)', '')
    )

def method_linestyle(method_name):
    """Use distinct line styles for nominal, GEN, and DR curves."""
    if '(DR)' in method_name:
        return '--'
    if '(GEN)' in method_name:
        return ':'
    return '-'

################ plot options ################

# Read option from command line arguments, otherwise use default
if len(sys.argv) > 1:
    option = sys.argv[1]
else:
    option = 'all'  # default option
    # option = 'control'
    # options = 'rl'

# Check for RL variant flags.
include_dr = '--include-dr' in sys.argv or '--domain-rand' in sys.argv
use_dr = '--use-dr' in sys.argv
include_gen = '--include-gen' in sys.argv
use_gen = '--use-gen' in sys.argv

if use_dr and use_gen:
    raise ValueError("Use only one replacement flag: --use-dr or --use-gen.")

if option == 'dr' or use_dr:
    include_dr = True
if option == 'gen' or use_gen:
    include_gen = True

assert option in ['all', 'control', 'rl', 'dr', 'gen'], (
    f"Invalid option: {option}. Must be one of ['all', 'control', 'rl', 'dr', 'gen']"
)
##############################################
control_list = [
    'Geometric Control', 'Linear MPC', 'Nonlinear MPC',
    'F-MPC', 'iLQR', 'LQR', 'GP-MPC',
]
rl_list = ['PPO', 'SAC', 'DPPO', 'PPO-MPC']

script_dir = Path(__file__).parent.resolve()
# Create the 'generalization' folder if it doesn't exist
output_dir = script_dir / "generalization"
output_dir.mkdir(exist_ok=True)

# Load requested RL variants.
variant_metrics = {}
variant_lists = {'gen': [], 'dr': []}
requested_variants = []
if include_gen:
    requested_variants.append('gen')
if include_dr:
    requested_variants.append('dr')

if requested_variants:
    for variant in requested_variants:
        print(f"Including {variant.upper()} variants...")
        for method in ['PPO', 'SAC', 'DPPO']:
            variant_data = load_variant_generalization_data(script_dir, method, variant)
            if variant_data is not None:
                label = variant_label(method, variant)
                variant_metrics[label] = variant_data
                variant_lists[variant].append(label)
                print(f"Loaded {variant.upper()} data for {method}")
            else:
                print(f"No {variant.upper()} data found for {method}")
else:
    print("Excluding RL variants (use --include-gen, --use-gen, --include-dr, or --use-dr)")

replacement_variant = 'gen' if use_gen else 'dr' if use_dr else None
replacement_labels = variant_lists[replacement_variant] if replacement_variant else []
replacement_bases = {
    base_method_name(label) for label in replacement_labels
}
nominal_rl_list = [
    method for method in rl_list
    if method not in replacement_bases
]

additive_variant_labels = []
for variant in ['gen', 'dr']:
    if variant == replacement_variant:
        continue
    additive_variant_labels.extend(variant_lists[variant])

rl_plot_list = nominal_rl_list + replacement_labels + additive_variant_labels

if option == 'control':
    plot_list = control_list
elif option == 'rl':
    plot_list = rl_plot_list
elif option == 'dr':
    plot_list = variant_lists['dr']
elif option == 'gen':
    plot_list = variant_lists['gen']
elif option == 'all':
    plot_list = control_list + rl_plot_list

# Update plot colors for variants (use same color as nominal method).
extended_plot_colors = plot_colors.copy()
for label in variant_metrics:
    extended_plot_colors[label] = plot_colors[base_method_name(label)]

# Define the episode lengths and periods
episode_len_list = [9, 10, 11, 12, 13, 14, 15]
episode_period_list =  [episode_len_list[i]/2 for i in range(len(episode_len_list))]

# Initialize the transfer_metric dictionary
transfer_metric = {}
for method in plot_list:
    if '(DR)' in method or '(GEN)' in method:
        continue

    try:
        transfer_metric = load_metric(script_dir, transfer_metric, method)
    except FileNotFoundError as e:
        print(f"Warning: skipping {method}; generalization data file not found: {e.filename}")

# Add variant metrics to transfer_metric.
transfer_metric.update(variant_metrics)
plot_list = [method for method in plot_list if method in transfer_metric]

if not plot_list:
    raise RuntimeError("No generalization data found for the selected plotting option.")

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
    linestyle = method_linestyle(method)
    color = extended_plot_colors.get(method, extended_plot_colors.get(base_method_name(method), 'blue'))
    
    ax.plot(episode_period_list, 
            transfer_metric[method]['rmse'], 
            label=method, color=color, linestyle=linestyle)
    ax.fill_between(episode_period_list, 
                    transfer_metric[method]['rmse'] - transfer_metric[method]['rmse_std'],  
                    transfer_metric[method]['rmse'] + transfer_metric[method]['rmse_std'], 
                    color=color, alpha=0.1)
ax.axvline(x=5.5, linestyle='-.', color='gray')
ax.text(0.81, 0.75, "Nominal Task", transform=ax.transAxes, fontsize=10, verticalalignment='center', horizontalalignment='right')

# Add additional vertical lines for DR option
if option == 'dr':
    ax.axvline(x=7.5, linestyle='-.', color='gray')
    ax.axvline(x=4.5, linestyle='-.', color='gray')

# Only create inset for 'all' and 'control' options (not for 'dr' only)
if option in ['all', 'control']:
    # Create a zoomed-in inset plot
    fill_alpha = 0.2
    ax_inset = inset_axes(ax, width="40%", height="50%", loc='upper left', 
                        bbox_to_anchor=(0.1, -0.05, 1, 1), bbox_transform=ax.transAxes)
    ax_inset.set_facecolor((0.5, 0.5, 0.5, fill_alpha))  # Set background to transparent gray
    for method in plot_list:
        linestyle = method_linestyle(method)
        color = extended_plot_colors.get(method, extended_plot_colors.get(base_method_name(method), 'blue'))
        
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

# Add legend for variant-only options since there will be fewer methods to display.
if option in ['dr', 'gen']:
    ax.legend(loc='best')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Figure-Eight Trajectory Period (s)")
ax.set_ylabel("RMSE [m]")

suffix_parts = []
if use_gen and variant_lists['gen']:
    suffix_parts.append('use_gen')
elif include_gen and variant_lists['gen']:
    suffix_parts.append('with_gen')

if use_dr and variant_lists['dr']:
    suffix_parts.append('use_dr')
elif include_dr and variant_lists['dr']:
    suffix_parts.append('with_dr')

if option == 'gen' and suffix_parts == ['with_gen']:
    suffix_parts = []
elif option == 'dr' and suffix_parts == ['with_dr']:
    suffix_parts = []

variant_suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
fig.savefig(output_dir / f"generalization_curve_{option}{variant_suffix}.pdf", bbox_inches="tight", pad_inches=0.1)
fig.savefig(output_dir / f"generalization_curve_{option}{variant_suffix}.png", bbox_inches="tight", pad_inches=0.1)

# Convert the transfer metric to a table
transfer_metric_table = {}
for method in plot_list:
    transfer_metric_table[method] = {}
    for i, T in enumerate(episode_len_list):
        transfer_metric_table[method][T] = f"{transfer_metric[method]['rmse'][i]:.3f} +/- {transfer_metric[method]['rmse_std'][i]:.3f}"

# Create a pandas DataFrame
df = pd.DataFrame(transfer_metric_table)

# Save the DataFrame to a CSV file in the 'generalization' folder
df.to_csv(output_dir / f"transfer_metric_table_{option}{variant_suffix}.csv", index_label="Episode Length")

# Save the DataFrame to a PNG file in the 'generalization' folder
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))
fig.savefig(output_dir / f"transfer_metric_table_{option}{variant_suffix}.png", bbox_inches='tight')
plt.close(fig)

# Print the path where the plots and table are saved
print(f"Plots and table saved in: {output_dir.resolve()}")
