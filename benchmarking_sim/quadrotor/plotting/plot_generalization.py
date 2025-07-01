import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, tag_ctrl_list, load_metric

################ plot options ################

option = 'all'
# option = 'control'
# option = 'rl'
assert option in ['all', 'control', 'rl', ] 
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
transfer_metric = {
'PPO': {'rmse': np.array([0.1518289 , 0.04699026, 0.01248575, 0.04337291, 0.07692969, 0.10662454, 0.13239991]),
  'rmse_std': np.array([0.00188754, 0.0011886 , 0.00071831, 0.00107574, 0.00108538, 0.00117313, 0.00103038])},
 'SAC': {'rmse': np.array([0.13106732, 0.04412564, 0.04392989, 0.05757863, 0.07185517, 0.08548335, 0.09768431]),  
         'rmse_std': np.array([0.00451328, 0.0007881 , 0.00079868, 0.00086922, 0.00076229, 0.0006071 , 0.00066663])}, 
 'DPPO': {'rmse': np.array([0.16318646, 0.05295693, 0.02341874, 0.04777906, 0.08090912, 0.11122228, 0.1399848 ]),  
          'rmse_std': np.array([0.00500729, 0.00245919, 0.00125781, 0.00147439, 0.00164736, 0.00200121, 0.00172556])}, 
 'PPO-MPC': {'rmse': np.array([0.02657339, 0.01117535, 0.00869317, 0.0078968 , 0.00635579, 0.00604681, 0.00595736]),  
             'rmse_std': np.array([0.00253017, 0.00121169, 0.00088094, 0.0006619, 0.0005072, 0.00043874, 0.00037119])}}

transfer_metric = load_metric(script_dir, transfer_metric, 'iLQR')
transfer_metric = load_metric(script_dir, transfer_metric, 'F-MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Nonlinear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Linear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Geometric Control')
transfer_metric = load_metric(script_dir, transfer_metric, 'LQR')
tag = ''
transfer_metric = load_metric(script_dir, transfer_metric, 'GP-MPC', tag=tag)

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
ax_inset.set_ylim(0, 0.045)
ax_inset.invert_xaxis()
ax_inset.tick_params(axis='both', which='major', labelsize=8)
mark_inset(ax, ax_inset, loc1=2, loc2=4, 
            fc="gray", ec="0.3", alpha=fill_alpha)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.invert_xaxis()
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