"""Defines the hyperparameter search space for each algorithm."""

from typing import Any, Dict

# PPO_dict = {
#     'hidden_dim': {'values': [8, 16, 32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
#     'gae_lambda': {'values': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
#     'clip_param': {'values': [0.1, 0.2, 0.3, 0.4], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
#     'opt_epochs': {'values': [1, 5, 10, 20, 25], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'mini_batch_size': {'values': [32, 64, 128, 256], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'max_env_steps': {'values': [30000, 72000, 114000, 156000, 216000, 336000, 456000, 540000, 660000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'activation': {'values': ['tanh', 'relu', 'leaky_relu'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
#     'target_kl': {'values': [0.00000001, 0.8], 'scale': 'uniform', 'type': float, 'cat': 'float'},
#     'entropy_coef': {'values': [0.00000001, 0.1], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'actor_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'critic_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'rollout_batch_size': {'values': [2, 3, 4, 5, 8], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'rew_state_weight': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
#     'rew_act_weight': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
# }
PPO_dict = {
    # 'hidden_dim': {'values': [8, 16, 32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'gae_lambda': {'values': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'clip_param': {'values': [0.1, 0.2, 0.3, 0.4, 0.5], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'opt_epochs': {'values': [5, 10, 20, 25, 30, 35], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'mini_batch_size': {'values': [32, 64, 128, 256], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'max_env_steps': {'values': [30000, 72000, 114000, 156000, 216000, 336000, 456000, 540000, 660000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'activation': {'values': ['tanh', 'relu', 'leaky_relu'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
    'target_kl': {'values': [0.00000001, 0.8], 'scale': 'uniform', 'type': float, 'cat': 'float'},
    'entropy_coef': {'values': [0.00000001, 0.1], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'exploration_init': {'values': [1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    # 'actor_lr': {'values': [1e-5, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},  # log-scaled
    # 'critic_lr': {'values': [1e-5, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},  # log-scaled
    'actor_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'critic_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'rollout_batch_size': {'values': [2, 3, 4, 5, 8, 10, 15], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'rew_state_weight': {'values': [0.0001, 0.01, 0.1, 1., 3., 5., 10., 15., 20.], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
    # 'rew_act_weight': {'values': [0.0001, 0.01, 0.1, 1., 3., 5., 10., 15., 20.], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
    'rew_state_weight': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'rew_act_weight': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

PPO_SF_dict = {
    # ppo related
    'hidden_dim': {'values': [8, 16, 32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'gae_lambda': {'values': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'clip_param': {'values': [0.1, 0.2, 0.3, 0.4], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'opt_epochs': {'values': [1, 5, 10, 20, 25], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'mini_batch_size': {'values': [32, 64, 128, 256], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'max_env_steps': {'values': [30000, 72000, 114000, 156000, 216000, 336000, 456000, 540000, 660000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'activation': {'values': ['tanh', 'relu', 'leaky_relu'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
    'target_kl': {'values': [0.00000001, 0.8], 'scale': 'uniform', 'type': float, 'cat': 'float'},
    'entropy_coef': {'values': [0.00000001, 0.1], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'actor_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'critic_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'rollout_batch_size': {'values': [1], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'rew_state_weight': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'rew_act_weight': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'sf_penalty': {'values': [0.001, 10.0], 'scale': 'uniform', 'type': float, 'cat': 'float'},
    # mpc related
    'horizon': {'values': [15, 20, 25, 30, 35, 40, 45, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
	'slack_cost': {'values': [10.0, 1000.0], 'scale': 'uniform', 'type': float, 'cat': 'float'},
	'max_w': {'values': [0.0, 0.01], 'scale': 'uniform', 'type': float, 'cat': 'float'}
}

# SAC_dict = {
#     'hidden_dim': {'values': [8, 16, 32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
#     'train_interval': {'values': [10, 100, 200, 500, 1000, 1500, 2000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'train_batch_size': {'values': [32, 64, 128, 256, 512, 1024], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'max_env_steps': {'values': [30000, 72000, 114000, 156000, 216000, 336000, 456000, 540000, 660000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'warm_up_steps': {'values': [500, 1000, 2000, 4000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'max_buffer_size': {'values': [20000, 50000, 100000, 150000, 200000, 350000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'activation': {'values': ['tanh', 'relu', 'leaky_relu'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
#     'tau': {'values': [0.005, 1.0], 'scale': 'uniform', 'type': float, 'cat': 'float'},
#     'init_temperature': {'values': [0.01, 1], 'scale': 'uniform', 'type': float, 'cat': 'float'},
#     'actor_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'critic_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'entropy_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'rew_state_weight': {'values': [0.001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
#     'rew_act_weight': {'values': [0.001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
# }
SAC_dict = {
    # 'hidden_dim': {'values': [8, 16, 32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'train_interval': {'values': [100, 200, 500, 1000, 1500, 2000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'train_batch_size': {'values': [64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'max_env_steps': {'values': [30000, 72000, 114000, 156000, 216000, 336000, 456000, 540000, 660000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'warm_up_steps': {'values': [500, 1000, 2000, 4000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'max_buffer_size': {'values': [20000, 50000, 100000, 150000, 200000, 350000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'activation': {'values': ['tanh', 'relu', 'leaky_relu'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
    'tau': {'values': [0.005, 1.0], 'scale': 'uniform', 'type': float, 'cat': 'float'},
    'init_temperature': {'values': [0.01, 1], 'scale': 'uniform', 'type': float, 'cat': 'float'},
    # 'actor_lr': {'values': [1e-5, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},  # log-scaled
    # 'critic_lr': {'values': [1e-5, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},  # log-scaled
    # 'entropy_lr': {'values': [1e-5, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},  # log-scaled
    'actor_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'critic_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'entropy_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    # 'rew_state_weight': {'values': [0.0001, 0.01, 0.1, 1., 3., 5., 10., 15., 20.], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
    # 'rew_act_weight': {'values': [0.0001, 0.01, 0.1, 1., 3., 5., 10., 15., 20.], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
    'rew_state_weight': {'values': [0.001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'rew_act_weight': {'values': [0.001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

# DPPO_dict = {
#     'hidden_dim': {'values': [8, 16, 32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
#     'gae_lambda': {'values': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
#     'clip_param': {'values': [0.1, 0.2, 0.3, 0.4], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
#     'opt_epochs': {'values': [1, 5, 10, 20, 25], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'mini_batch_size': {'values': [32, 64, 128, 256], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'max_env_steps': {'values': [30000, 72000, 114000, 156000, 216000, 336000, 456000, 540000, 660000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'activation': {'values': ['tanh', 'relu', 'leaky_relu'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
#     'target_kl': {'values': [0.00000001, 0.8], 'scale': 'uniform', 'type': float, 'cat': 'float'},
#     'entropy_coef': {'values': [0.00000001, 0.1], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'actor_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'critic_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
#     'rollout_batch_size': {'values': [2, 3, 4, 5, 8], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'quantile_count': {'values': [32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
#     'rew_state_weight': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
#     'rew_act_weight': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
# }
DPPO_dict = {
    # 'hidden_dim': {'values': [8, 16, 32, 64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'gae_lambda': {'values': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'clip_param': {'values': [0.1, 0.2, 0.3, 0.4, 0.5], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'opt_epochs': {'values': [5, 10, 20, 25,  30, 35], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'mini_batch_size': {'values': [32, 64, 128, 256], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'max_env_steps': {'values': [30000, 72000, 114000, 156000, 216000, 336000, 456000, 540000, 660000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'activation': {'values': ['tanh', 'relu', 'leaky_relu'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
    'target_kl': {'values': [0.00000001, 0.8], 'scale': 'uniform', 'type': float, 'cat': 'float'},
    'entropy_coef': {'values': [0.00000001, 0.1], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'exploration_init': {'values': [1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'beta': {'values': [-1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    # 'actor_lr': {'values': [1e-5, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},  # log-scaled
    # 'critic_lr': {'values': [1e-5, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},  # log-scaled
    'actor_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'critic_lr': {'values': [1e-5, 1e-2], 'scale': 'log', 'type': float, 'cat': 'float'},  # log-scaled
    'rollout_batch_size': {'values': [2, 3, 4, 5, 8, 10, 15], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'quantile_count': {'values': [64, 128, 256, 512], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'rew_state_weight': {'values': [0.0001, 0.01, 0.1, 1., 3., 5., 10., 15., 20.], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
    # 'rew_act_weight': {'values': [0.0001, 0.01, 0.1, 1., 3., 5., 10., 15., 20.], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
    'rew_state_weight': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'rew_act_weight': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

GPMPC_dict = {
    'horizon': {'values': [15, 20, 25, 30, 35, 40, 45, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'n_ind_points': {'values': [30, 40, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'num_epochs': {'values': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'num_samples': {'values': [20, 30, 40, 50, 60, 70, 80], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'optimization_iterations': {'values': [500, 1000, 1500, 2000, 2500, 3000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},  # type belongs to int due to 1 DoF
    'kernel': {'values': ['Matern', 'RBF'], 'scale': 'uniform', 'type': str, 'cat': 'categorical'},
    'learning_rate': {'values': [5e-4, 0.5], 'scale': 'log', 'type': float, 'cat': 'float'},  # type belongs to float due to 1 DoF
    'q_mpc': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_mpc': {'values': [0.0001, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

GPMPC_TP_dict = {
    'horizon': {'values': [15, 20, 25, 30, 35, 40, 45, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'n_ind_points': {'values': [30, 40, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'num_epochs': {'values': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'num_samples': {'values': [20, 30, 40, 50, 60, 70, 80], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'optimization_iterations': {'values': [500, 1000, 1500, 2000, 2500, 3000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},  # type belongs to int due to 1 DoF
    'learning_rate': {'values': [5e-4, 0.5], 'scale': 'log', 'type': float, 'cat': 'float'},  # type belongs to float due to 1 DoF
    'q_mpc': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_mpc': {'values': [1, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

LMPC_dict = {
    'horizon': {'values': [15, 20, 25, 30, 35, 40, 45, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'q_mpc': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_mpc': {'values': [1, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

MPC_dict = {
    'horizon': {'values': [15, 20, 25, 30, 35, 40, 45, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'q_mpc': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_mpc': {'values': [1, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

FMPC_dict = {
    'horizon': {'values': [15, 20, 25, 30, 35, 40, 45, 50], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    # 'q_mpc': {'values': [0.0001, 70], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    # 'r_mpc': {'values': [1e-7, 1e-4], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'q_mpc': {'values': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
    'r_mpc': {'values': [1e-10, 1e-8, 1e-6, 1e-4], 'scale': 'uniform', 'type': list, 'cat': 'discrete'},
}

LQR_dict = {
    'q_lqr': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_lqr': {'values': [1, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

iLQR_dict = {
    'max_iterations': {'values': [5, 10, 15, 20], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'lamb_factor': {'values': [5, 10, 15, 20, 25], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'lamb_max': {'values': [1000, 1500, 2000, 2500, 3000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'epsilon': {'values': [0.0001, 0.0005, 0.001, 0.005, 0.01], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'q_lqr': {'values': [0.0001, 20], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_lqr': {'values': [1, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

iLQR_SF_dict = {
    'max_iterations': {'values': [5, 10, 15, 20], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'lamb_factor': {'values': [5, 10, 15], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'lamb_max': {'values': [1000, 1500, 2000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'epsilon': {'values': [0.01, 0.005, 0.001], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'q_lqr': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_lqr': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'horizon': {'values': [10, 15, 20, 25, 30, 35, 40], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'n_samples': {'values': [400, 600, 800, 1000], 'scale': 'uniform', 'type': int, 'cat': 'discrete'},
    'tau': {'values': [0.95, 0.99], 'scale': 'uniform', 'type': float, 'cat': 'discrete'},
    'q_lin': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'r_lin': {'values': [0.0001, 15], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}

PID_dict = {
    'p_coeff_for': {'values': [0.001, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'i_coeff_for': {'values': [0.001, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
    'd_coeff_for': {'values': [0.001, 5], 'scale': 'uniform', 'type': list, 'cat': 'float'},
}


HYPERPARAMS_DICT = {
    'ppo': PPO_dict,
    'ppo_mpsf': PPO_SF_dict,
    'dppo': DPPO_dict,
    'sac': SAC_dict,
    'gp_mpc': GPMPC_dict,
    'gpmpc_acados': GPMPC_dict,
    'gpmpc_acados_TP': GPMPC_TP_dict,
    'fmpc': FMPC_dict,
    'linear_mpc': LMPC_dict,
    'linear_mpc_acados': LMPC_dict,
    'mpc_acados': MPC_dict,
    'lqr': LQR_dict,
    'ilqr': iLQR_dict,
    'ilqr_sf': iLQR_SF_dict,
    'pid': PID_dict,
}


def is_log_scale(param: Dict[str, Any]) -> bool:
    """Check if the hyperparameter log scale.

    args:
        param (dict): the hyperparameter dictionary

    returns:
        bool: True if the hyperparameter is log-scaled, False otherwise

    """

    return param['scale'] == 'log'
