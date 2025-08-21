import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, tag_ctrl_list, load_metric
import sys

################ plot options ################

# Read option from command line arguments, otherwise use default
if len(sys.argv) > 1:
    option = sys.argv[1]
else:
    option = 'all'  # default option
    # option = 'control'
    # options = 'rl'

assert option in ['all', 'control', 'rl'], f"Invalid option: {option}. Must be one of ['all', 'control', 'rl']"
##############################################
control_list = [
    'Geometric Control', 'Linear MPC', 'Nonlinear MPC',
    'F-MPC', 'iLQR', 'LQR', 'GP-MPC',
]
rl_list = ['PPO', 'SAC', 'DPPO', 'PPO-MPC']

if option == 'control':
    plot_list = control_list
elif option == 'rl':
    plot_list = rl_list
elif option == 'all':
    plot_list = control_list + rl_list

# plot_list = [
#     'PPO', 'SAC', 'DPPO', 
#     'PPO-MPC',
#     'GP-MPC', 
#     'F-MPC', 'iLQR', 'Nonlinear MPC', 
#     'Geometric Control', 'LQR', 'Linear MPC', 
# ]

script_dir = Path.cwd()
# Create the 'generalization' folder if it doesn't exist
output_dir = script_dir / "generalization"
output_dir.mkdir(exist_ok=True)

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
    ax.plot(episode_period_list, 
            transfer_metric[method]['rmse'], 
            label=method, color=plot_colors[method])
    ax.fill_between(episode_period_list, 
                    transfer_metric[method]['rmse'] - transfer_metric[method]['rmse_std'],  
                    transfer_metric[method]['rmse'] + transfer_metric[method]['rmse_std'], 
                    color=plot_colors[method], alpha=0.1)
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
        ax_inset.plot(episode_period_list, 
                    transfer_metric[method]['rmse'], 
                    color=plot_colors[method])
        ax_inset.fill_between(episode_period_list, 
                            transfer_metric[method]['rmse'] - transfer_metric[method]['rmse_std'],  
                            transfer_metric[method]['rmse'] + transfer_metric[method]['rmse_std'], 
                            color=plot_colors[method], alpha=0.1)
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
fig.savefig(output_dir / f"generalization_curve_{option}.pdf", bbox_inches="tight", pad_inches=0.1)
fig.savefig(output_dir / f"generalization_curve_{option}.png", bbox_inches="tight", pad_inches=0.1)

# Convert the transfer metric to a table
transfer_metric_table = {}
for method in plot_list:
    transfer_metric_table[method] = {}
    for i, T in enumerate(episode_len_list):
        transfer_metric_table[method][T] = f"{transfer_metric[method]['rmse'][i]:.3f} +/- {transfer_metric[method]['rmse_std'][i]:.3f}"

# Create a pandas DataFrame
df = pd.DataFrame(transfer_metric_table)

# Save the DataFrame to a CSV file in the 'generalization' folder
df.to_csv(output_dir / f"transfer_metric_table_{option}.csv", index_label="Episode Length")

# Save the DataFrame to a PNG file in the 'generalization' folder
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))
fig.savefig(output_dir / f"transfer_metric_table_{option}.png", bbox_inches='tight')
plt.close(fig)

# Print the path where the plots and table are saved
print(f"Plots and table saved in: {output_dir.resolve()}")