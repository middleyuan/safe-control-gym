'''Plot the pareto front for the hpo history'''
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
ctrl = 'lqr'
tag = 'basic'
script_dir = os.path.dirname(__file__)
if ctrl == 'pid':
    data_dir = os.path.join(script_dir,
                            f'../../examples/hpo/hpo/{ctrl}/basic/seed1_Mar-18-13-13-16_a0c7dff/hpo/trials.csv')
elif ctrl == 'lqr':
    data_dir = os.path.join(script_dir,
                            f'../../examples/hpo/hpo/{ctrl}/basic/seed1_Mar-18-13-24-43_a0c7dff/hpo/trials.csv')
    if tag == 'cross':
        data_dir = os.path.join(script_dir,
                                f'../../examples/hpo/hpo/{ctrl}/cross/seed1_Mar-18-14-05-07_a0c7dff/hpo/trials.csv')
data = pd.read_csv(data_dir)

# The first three columns are idx, exponentiated_rmse, exponentiated_rms_action_change
exponetiated_rmse = data['exponentiated_rmse']
exponentiated_rms_action_change = data['exponentiated_rms_action_change']
print('exponetiated_rmse:', exponetiated_rmse)
print('exponentiated_rms_action_change:', exponentiated_rms_action_change)

# Convert to rmse and rms_action_change
rmse = -np.log(exponetiated_rmse)
rms_action_change = -np.log(exponentiated_rms_action_change)
print('RMSE:', rmse)
print('RMS Action Change:', rms_action_change)

#########################################################################
# Get the pareto front
pareto_front = []
for i in range(len(rmse)):
    dominated = False
    for j in range(len(rmse)):
        if (rmse[j] <= rmse[i] and rms_action_change[j] <= rms_action_change[i]) and (rmse[j] < rmse[i] or rms_action_change[j] < rms_action_change[i]):
            dominated = True
            break
    if not dominated:
        pareto_front.append(i)

sort_by_rmse = np.argsort(rmse[pareto_front])
pareto_front = [pareto_front[i] for i in sort_by_rmse]
print('Pareto front (rmse increase):', pareto_front)
#########################################################################

# Find the best with 0.55 * exponentiated_rmse + 0.45 * exponentiated_rms_action_change
combined = 0.55 * exponetiated_rmse + 0.45 * exponentiated_rms_action_change
best = np.argmax(combined)
print('Best:', best)

# Plot the pareto front
plt.figure()
plt.scatter(rmse, rms_action_change, c='blue')
plt.scatter(rmse[pareto_front], rms_action_change[pareto_front], c='red')

# Plot the solution with best 0.55 * exponentiated_rmse + 0.45 * exponentiated_rms_action_change
plt.scatter(rmse[best], rms_action_change[best], c='green', marker='x')

# Plot a line connecting the points in the pareto front
for i in range(len(pareto_front) - 1):
    plt.plot([rmse[pareto_front[i]], rmse[pareto_front[i + 1]]],
             [rms_action_change[pareto_front[i]], rms_action_change[pareto_front[i + 1]]],
             c='red')

# Plot the index next to each point
for i in range(len(rmse)):
    plt.text(rmse[i], rms_action_change[i], str(i))
plt.xlim(0, 0.5)

plt.xlabel('RMSE [m]')
plt.ylabel('RMS Action Change')
plt.title('HPO History and Pareto Front')
plt.legend(['All', 'Pareto Front', 'Best'])

plt.savefig(f'pareto_front_{ctrl}_{tag}.png')
