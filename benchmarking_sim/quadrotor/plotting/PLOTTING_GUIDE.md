# Plotting Guide for Safe Control Gym Benchmarking

This guide provides command-line instructions for generating various types of plots for the Safe Control Gym quadrotor benchmarking analysis.

## Prerequisites

Before running any plotting commands, ensure you have processed the data:

```bash
# Process robustness data for all controllers and noise types
python process_experiment_data.py robustness obs_noise all
python process_experiment_data.py robustness proc_noise all
python process_experiment_data.py robustness param all

# Process trajectory data for path visualization
python process_experiment_data.py trajectory all

# Process generalization data for episode length analysis
python process_experiment_data.py generalization all

# Process domain randomization data for RL methods
python process_experiment_data.py domain-randomization

# Process everything at once (recommended for comprehensive analysis)
python process_experiment_data.py all
```

---

## 1. Generalization Plots

### Control + RL Methods
```bash
python plot_generalization.py all
```

### RL Methods Only
```bash
python plot_generalization.py rl
```

### Control Methods Only
```bash
python plot_generalization.py control
```

### RL Methods with Domain Randomization
```bash
python plot_generalization.py rl --include-dr
```

---

## 2. Robustness Plots

### Control Methods
```bash
# Observation noise
python plot_noise_robustness.py obs_noise control

# Process noise
python plot_noise_robustness.py proc_noise control

# Parametric uncertainty
python plot_noise_robustness.py param control
```

### RL Methods
```bash
# Observation noise
python plot_noise_robustness.py obs_noise rl

# Process noise
python plot_noise_robustness.py proc_noise rl

# Parametric uncertainty
python plot_noise_robustness.py param rl
```

### RL Methods with Domain Randomization
```bash
# Include both regular and DR variants
python plot_noise_robustness.py obs_noise rl --include-dr
python plot_noise_robustness.py proc_noise rl --include-dr
python plot_noise_robustness.py param rl --include-dr

# Plot ONLY domain randomization variants
python plot_noise_robustness.py obs_noise rl --dr-only
python plot_noise_robustness.py proc_noise rl --dr-only
python plot_noise_robustness.py param rl --dr-only
```

### All Methods Combined
```bash
# All methods for each noise type
python plot_noise_robustness.py obs_noise all
python plot_noise_robustness.py proc_noise all
python plot_noise_robustness.py param all

# All methods including domain randomization
python plot_noise_robustness.py obs_noise all --include-dr
python plot_noise_robustness.py proc_noise all --include-dr
python plot_noise_robustness.py param all --include-dr
```

---

## 3. Custom Ridge Plots

### Basic Ridge Plots
```bash
# Compare specific controllers for observation noise
python plot_custom_ridge_levels.py obs_noise lqr fmpc mpc_acados sac

# Compare RL methods for process noise
python plot_custom_ridge_levels.py proc_noise ppo sac dppo

# Compare control methods for parametric uncertainty
python plot_custom_ridge_levels.py param lqr ilqr linear_mpc_acados
```

### Ridge Plots with Domain Randomization
```bash
# Include domain randomization for specific RL controllers
python plot_custom_ridge_levels.py obs_noise lqr fmpc mpc_acados sac --include-dr sac
python plot_custom_ridge_levels.py proc_noise ppo sac dppo --include-dr ppo sac dppo
python plot_custom_ridge_levels.py param sac dppo --include-dr sac dppo
```

### Comprehensive Comparisons
```bash
# All main controller types with DR for SAC
python plot_custom_ridge_levels.py obs_noise lqr linear_mpc_acados mpc_acados gpmpc_acados_TP sac --include-dr sac

# RL methods comparison with domain randomization
python plot_custom_ridge_levels.py obs_noise ppo sac dppo ppo_mpc --include-dr ppo sac dppo
```

---

## 4. Trajectory/Path with Error Plots

### RL Methods
```bash
python3 plot_traj_hull.py rl 9
python3 plot_traj_hull.py rl 11
python3 plot_traj_hull.py rl 15
```

### Control Methods
```bash
python3 plot_traj_hull.py mb 9
python3 plot_traj_hull.py mb 11
python3 plot_traj_hull.py mb 15
```

---

## 6. Convergence

```bash
# RL training convergence analysis
python plot_convergence.py all                    # All RL methods
python plot_convergence.py ppo                    # PPO only
python plot_convergence.py sac                    # SAC only
python plot_convergence.py dppo                   # DPPO only
python plot_convergence.py ppo_mpc                # PPO-MPC only
python plot_convergence.py all --include-dr       # Include domain randomization
```

---

## Available Controllers

### Model-based Controllers
- `linear_mpc_acados` - Linear MPC
- `mpc_acados` - Nonlinear MPC
- `gpmpc_acados_TP` - GP-MPC
- `ilqr` - iLQR
- `lqr` - LQR
- `pid` - Geometric Control
- `fmpc` - F-MPC

### RL Controllers
- `ppo` - Proximal Policy Optimization
- `sac` - Soft Actor-Critic
- `dppo` - Distributed PPO
- `ppo_mpc` - PPO with MPC

### Noise Types
- `obs_noise` - Observation noise
- `proc_noise` - Process noise
- `param` - Parametric uncertainty

---

## Output Locations

All plots are saved in the following directories:

- **Robustness plots**: `noise/`
- **Generalization plots**: `generalization/`
- **Trajectory plots**: `trajectory/`
- **Custom ridge plots**: `custom_plots/`
- **Convergence plots**: `convergence/`

Each plot is saved in both PNG and PDF formats for publication use.

---

## Common Flags

- `--include-dr`: Include domain randomization variants along with regular methods
- `--dr-only`: Plot only domain randomization variants (RL methods only)
- `control`: Filter to model-based controllers only
- `rl`: Filter to reinforcement learning methods only
- `all`: Include both control and RL methods

---

## Examples for Paper Figures

### Figure 1: Robustness Comparison
```bash
python plot_noise_robustness.py obs_noise all --include-dr
```

### Figure 2: Generalization Analysis
```bash
python plot_generalization.py all --include-dr
```

### Figure 3: Ridge Plot Comparison
```bash
python plot_custom_ridge_levels.py obs_noise lqr mpc_acados sac --include-dr sac
```

### Figure 4: Trajectory Visualization
```bash
python plot_traj_hull.py all --include-dr
```

### Figure 5: Training Convergence
```bash
python plot_convergence.py all --include-dr
```

---

## Troubleshooting

1. **No data files found**: Run the data processing script first:
   ```bash
   python process_experiment_data.py all
   ```
2. **Missing specific data type**: Process the required analysis:
   ```bash
   python process_experiment_data.py robustness obs_noise all
   python process_experiment_data.py trajectory all
   python process_experiment_data.py generalization all
   python process_experiment_data.py domain-randomization
   ```
3. **Missing controllers**: Check that all required data files exist in `../data/`
4. **Plot formatting issues**: Ensure matplotlib and seaborn are up to date

For additional help, check the individual script documentation with `python <script_name>.py --help`.

## Data Processing Reference

The `process_experiment_data.py` script supports the following analysis types:

### Comprehensive Processing
```bash
# Process all data types for all controllers (recommended)
python process_experiment_data.py all

# Process all data types for specific method category
python process_experiment_data.py all control  # Model-based only
python process_experiment_data.py all rl       # RL methods only
```

### Specific Analysis Types
```bash
# Robustness analysis (requires noise type)
python process_experiment_data.py robustness obs_noise all
python process_experiment_data.py robustness proc_noise control
python process_experiment_data.py robustness param rl

# Trajectory analysis
python process_experiment_data.py trajectory all
python process_experiment_data.py trajectory control

# Generalization analysis
python process_experiment_data.py generalization all
python process_experiment_data.py generalization rl

# Domain randomization (RL methods only)
python process_experiment_data.py domain-randomization
```

### Controller-Specific Processing
```bash
# Process specific controller for all analysis types
python process_experiment_data.py all --controller=linear_mpc_acados
python process_experiment_data.py all --controller=ppo

# Process specific controller for specific analysis
python process_experiment_data.py trajectory --controller=sac
python process_experiment_data.py robustness obs_noise --controller=gpmpc_acados_TP
```
