""" Utils for Optuna hyperparameter optimization. """

from typing import Any, Dict

import optuna

from safe_control_gym.hyperparameters.hpo_search_space import (GPMPC_dict, GPMPC_TP_dict, LMPC_dict, MPC_dict, PPO_dict,
                                                               DPPO_dict, SAC_dict, LQR_dict, iLQR_dict, iLQR_SF_dict,
                                                               PID_dict, FMPC_dict, is_log_scale)


def ppo_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space

    """

    # TODO: conditional hyperparameters

    # model args
    hidden_dim = trial.suggest_categorical('hidden_dim', PPO_dict['hidden_dim']['values'])
    activation = trial.suggest_categorical('activation', PPO_dict['activation']['values'])

    # loss args
    gamma = trial.suggest_categorical('gamma', PPO_dict['gamma']['values'])
    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    gae_lambda = trial.suggest_categorical('gae_lambda', PPO_dict['gae_lambda']['values'])
    clip_param = trial.suggest_categorical('clip_param', PPO_dict['clip_param']['values'])
    # Limit the KL divergence between updates
    target_kl = trial.suggest_float('target_kl', PPO_dict['target_kl']['values'][0], PPO_dict['target_kl']['values'][1], log=is_log_scale(PPO_dict['target_kl']))
    # Entropy coefficient for the loss calculation
    entropy_coef = trial.suggest_float('entropy_coef', PPO_dict['entropy_coef']['values'][0], PPO_dict['entropy_coef']['values'][1], log=is_log_scale(PPO_dict['entropy_coef']))

    # optim args
    opt_epochs = trial.suggest_categorical('opt_epochs', PPO_dict['opt_epochs']['values'])
    mini_batch_size = trial.suggest_categorical('mini_batch_size', PPO_dict['mini_batch_size']['values'])
    actor_lr = trial.suggest_float('actor_lr', PPO_dict['actor_lr']['values'][0], PPO_dict['actor_lr']['values'][1], log=is_log_scale(PPO_dict['actor_lr']))
    critic_lr = trial.suggest_float('critic_lr', PPO_dict['critic_lr']['values'][0], PPO_dict['critic_lr']['values'][1], log=is_log_scale(PPO_dict['critic_lr']))
    rollout_batch_size = trial.suggest_categorical('rollout_batch_size', PPO_dict['rollout_batch_size']['values'])
    max_env_steps = trial.suggest_categorical('max_env_steps', PPO_dict['max_env_steps']['values'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'rew_state_weight_{i}', PPO_dict['rew_state_weight']['values'][0], PPO_dict['rew_state_weight']['values'][1], log=is_log_scale(PPO_dict['rew_state_weight']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'rew_act_weight_{i}', PPO_dict['rew_act_weight']['values'][0], PPO_dict['rew_act_weight']['values'][1], log=is_log_scale(PPO_dict['rew_act_weight']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'hidden_dim': hidden_dim,
        'activation': activation,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_param': clip_param,
        'target_kl': target_kl,
        'entropy_coef': entropy_coef,
        'opt_epochs': opt_epochs,
        'mini_batch_size': mini_batch_size,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'rollout_batch_size': rollout_batch_size,
        'max_env_steps': max_env_steps,
        'rew_state_weight': state_weight,
        'rew_act_weight': action_weight,
    }

    return hps_suggestion

def dppo_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for DPPO hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space

    """
    
    # model args
    hidden_dim = trial.suggest_categorical('hidden_dim', DPPO_dict['hidden_dim']['values'])
    activation = trial.suggest_categorical('activation', DPPO_dict['activation']['values'])

    # loss args
    gamma = trial.suggest_categorical('gamma', DPPO_dict['gamma']['values'])
    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    gae_lambda = trial.suggest_categorical('gae_lambda', DPPO_dict['gae_lambda']['values'])
    clip_param = trial.suggest_categorical('clip_param', DPPO_dict['clip_param']['values'])
    # Limit the KL divergence between updates
    target_kl = trial.suggest_float('target_kl', DPPO_dict['target_kl']['values'][0], DPPO_dict['target_kl']['values'][1], log=is_log_scale(DPPO_dict['target_kl']))
    # Entropy coefficient for the loss calculation
    entropy_coef = trial.suggest_float('entropy_coef', DPPO_dict['entropy_coef']['values'][0], DPPO_dict['entropy_coef']['values'][1], log=is_log_scale(DPPO_dict['entropy_coef']))
    # quntile count for quantile regression
    quantile_count = trial.suggest_categorical('quantile_count', DPPO_dict['quantile_count']['values'])

    # optim args
    opt_epochs = trial.suggest_categorical('opt_epochs', DPPO_dict['opt_epochs']['values'])
    mini_batch_size = trial.suggest_categorical('mini_batch_size', DPPO_dict['mini_batch_size']['values'])
    actor_lr = trial.suggest_float('actor_lr', DPPO_dict['actor_lr']['values'][0], DPPO_dict['actor_lr']['values'][1], log=is_log_scale(DPPO_dict['actor_lr']))
    critic_lr = trial.suggest_float('critic_lr', DPPO_dict['critic_lr']['values'][0], DPPO_dict['critic_lr']['values'][1], log=is_log_scale(DPPO_dict['critic_lr']))
    rollout_batch_size = trial.suggest_categorical('rollout_batch_size', DPPO_dict['rollout_batch_size']['values'])
    max_env_steps = trial.suggest_categorical('max_env_steps', DPPO_dict['max_env_steps']['values'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'rew_state_weight_{i}', PPO_dict['rew_state_weight']['values'][0], PPO_dict['rew_state_weight']['values'][1], log=is_log_scale(PPO_dict['rew_state_weight']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'rew_act_weight_{i}', PPO_dict['rew_act_weight']['values'][0], PPO_dict['rew_act_weight']['values'][1], log=is_log_scale(PPO_dict['rew_act_weight']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'hidden_dim': hidden_dim,
        'activation': activation,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_param': clip_param,
        'target_kl': target_kl,
        'entropy_coef': entropy_coef,
        'opt_epochs': opt_epochs,
        'mini_batch_size': mini_batch_size,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'rollout_batch_size': rollout_batch_size,
        'max_env_steps': max_env_steps,
        'quantile_count': quantile_count,
        'rew_state_weight': state_weight,
        'rew_act_weight': action_weight,
    }

    return hps_suggestion

def sac_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for SAC hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    # TODO: conditional hyperparameters

    # model args
    hidden_dim = trial.suggest_categorical('hidden_dim', SAC_dict['hidden_dim']['values'])
    activation = trial.suggest_categorical('activation', SAC_dict['activation']['values'])

    # loss args
    gamma = trial.suggest_categorical('gamma', SAC_dict['gamma']['values'])
    tau = trial.suggest_float('tau', SAC_dict['tau']['values'][0], SAC_dict['tau']['values'][1], log=is_log_scale(SAC_dict['tau']))

    # optim args
    train_interval = trial.suggest_categorical('train_interval', SAC_dict['train_interval']['values'])
    train_batch_size = trial.suggest_categorical('train_batch_size', SAC_dict['train_batch_size']['values'])
    actor_lr = trial.suggest_float('actor_lr', SAC_dict['actor_lr']['values'][0], SAC_dict['actor_lr']['values'][1], log=is_log_scale(SAC_dict['actor_lr']))
    critic_lr = trial.suggest_float('critic_lr', SAC_dict['critic_lr']['values'][0], SAC_dict['critic_lr']['values'][1], log=is_log_scale(SAC_dict['critic_lr']))

    max_env_steps = trial.suggest_categorical('max_env_steps', SAC_dict['max_env_steps']['values'])
    warm_up_steps = trial.suggest_categorical('warm_up_steps', SAC_dict['warm_up_steps']['values'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'rew_state_weight_{i}', SAC_dict['rew_state_weight']['values'][0], SAC_dict['rew_state_weight']['values'][1], log=is_log_scale(SAC_dict['rew_state_weight']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'rew_act_weight_{i}', SAC_dict['rew_act_weight']['values'][0], SAC_dict['rew_act_weight']['values'][1], log=is_log_scale(SAC_dict['rew_act_weight']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'hidden_dim': hidden_dim,
        'activation': activation,
        'gamma': gamma,
        'tau': tau,
        'train_interval': train_interval,
        'train_batch_size': train_batch_size,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'max_env_steps': max_env_steps,
        'warm_up_steps': warm_up_steps,
        'rew_state_weight': state_weight,
        'rew_act_weight': action_weight,
    }

    return hps_suggestion


def gpmpc_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    horizon = trial.suggest_categorical('horizon', GPMPC_dict['horizon']['values'])
    kernel = trial.suggest_categorical('kernel', GPMPC_dict['kernel']['values'])
    n_ind_points = trial.suggest_categorical('n_ind_points', GPMPC_dict['n_ind_points']['values'])
    num_epochs = trial.suggest_categorical('num_epochs', GPMPC_dict['num_epochs']['values'])
    num_samples = trial.suggest_categorical('num_samples', GPMPC_dict['num_samples']['values'])

    optimization_iterations = trial.suggest_categorical('optimization_iterations', GPMPC_dict['optimization_iterations']['values'])
    learning_rate = trial.suggest_float('learning_rate', GPMPC_dict['learning_rate']['values'][0], GPMPC_dict['learning_rate']['values'][1], log=is_log_scale(GPMPC_dict['learning_rate']))

    # objective
    state_weight = [
        trial.suggest_float(f'q_mpc_{i}', GPMPC_dict['q_mpc']['values'][0], GPMPC_dict['q_mpc']['values'][1], log=is_log_scale(GPMPC_dict['q_mpc']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_mpc_{i}', GPMPC_dict['r_mpc']['values'][0], GPMPC_dict['r_mpc']['values'][1], log=is_log_scale(GPMPC_dict['r_mpc']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'horizon': horizon,
        'kernel': kernel,
        'n_ind_points': n_ind_points,
        'num_epochs': num_epochs,
        'num_samples': num_samples,
        'optimization_iterations': optimization_iterations,
        'learning_rate': learning_rate,
        'q_mpc': state_weight,
        'r_mpc': action_weight,
    }

    return hps_suggestion

def gpmpc_tp_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    horizon = trial.suggest_categorical('horizon', GPMPC_TP_dict['horizon']['values'])
    n_ind_points = trial.suggest_categorical('n_ind_points', GPMPC_TP_dict['n_ind_points']['values'])
    num_epochs = trial.suggest_categorical('num_epochs', GPMPC_TP_dict['num_epochs']['values'])
    num_samples = trial.suggest_categorical('num_samples', GPMPC_TP_dict['num_samples']['values'])

    optimization_iterations = trial.suggest_categorical('optimization_iterations', GPMPC_TP_dict['optimization_iterations']['values'])
    learning_rate = trial.suggest_float('learning_rate', GPMPC_TP_dict['learning_rate']['values'][0], GPMPC_TP_dict['learning_rate']['values'][1], log=is_log_scale(GPMPC_TP_dict['learning_rate']))

    # objective
    state_weight = [
        trial.suggest_float(f'q_mpc_{i}', GPMPC_TP_dict['q_mpc']['values'][0], GPMPC_TP_dict['q_mpc']['values'][1], log=is_log_scale(GPMPC_TP_dict['q_mpc']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_mpc_{i}', GPMPC_TP_dict['r_mpc']['values'][0], GPMPC_TP_dict['r_mpc']['values'][1], log=is_log_scale(GPMPC_TP_dict['r_mpc']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'horizon': horizon,
        'n_ind_points': n_ind_points,
        'num_epochs': num_epochs,
        'num_samples': num_samples,
        'optimization_iterations': optimization_iterations,
        'learning_rate': learning_rate,
        'q_mpc': state_weight,
        'r_mpc': action_weight,
    }

    return hps_suggestion


def lmpc_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for LMPC hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    horizon = trial.suggest_categorical('horizon', LMPC_dict['horizon']['values'])

    # objective
    state_weight = [
        trial.suggest_float(f'q_mpc_{i}', LMPC_dict['q_mpc']['values'][0], LMPC_dict['q_mpc']['values'][1], log=is_log_scale(LMPC_dict['q_mpc']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_mpc_{i}', LMPC_dict['r_mpc']['values'][0], LMPC_dict['r_mpc']['values'][1], log=is_log_scale(LMPC_dict['r_mpc']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'horizon': horizon,
        'q_mpc': state_weight,
        'r_mpc': action_weight,
    }

    return hps_suggestion


def mpc_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for MPC hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    horizon = trial.suggest_categorical('horizon', MPC_dict['horizon']['values'])

    # objective
    state_weight = [
        trial.suggest_float(f'q_mpc_{i}', MPC_dict['q_mpc']['values'][0], MPC_dict['q_mpc']['values'][1], log=is_log_scale(MPC_dict['q_mpc']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_mpc_{i}', MPC_dict['r_mpc']['values'][0], MPC_dict['r_mpc']['values'][1], log=is_log_scale(MPC_dict['r_mpc']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'horizon': horizon,
        'q_mpc': state_weight,
        'r_mpc': action_weight,
    }

    return hps_suggestion

def fmpc_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for FMPC hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    horizon = trial.suggest_categorical('horizon', FMPC_dict['horizon']['values'])

    # hard coded dim ro Q and R
    if state_dim == 6:
        q_dim = 8
        r_dim = 2
    else:
        raise ValueError('Only 6D state space is supported for now.')
    # objective
    state_weight = [
        trial.suggest_float(f'q_mpc_{i}', FMPC_dict['q_mpc']['values'][0], FMPC_dict['q_mpc']['values'][1], log=is_log_scale(FMPC_dict['q_mpc']))
        for i in range(q_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_mpc_{i}', FMPC_dict['r_mpc']['values'][0], FMPC_dict['r_mpc']['values'][1], log=is_log_scale(FMPC_dict['r_mpc']))
        for i in range(r_dim)
    ]

    hps_suggestion = {
        'horizon': horizon,
        'q_mpc': state_weight,
        'r_mpc': action_weight,
    }

    return hps_suggestion

def lqr_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for LQR hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    # cost parameters
    state_weight = [
        trial.suggest_float(f'q_lqr_{i}', LQR_dict['q_lqr']['values'][0], LQR_dict['q_lqr']['values'][1], log=is_log_scale(LQR_dict['q_lqr']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_lqr_{i}', LQR_dict['r_lqr']['values'][0], LQR_dict['r_lqr']['values'][1], log=is_log_scale(LQR_dict['r_lqr']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'q_lqr': state_weight,
        'r_lqr': action_weight,
    }

    return hps_suggestion


def ilqr_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for iLQR hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    max_iterations = trial.suggest_categorical('max_iterations', iLQR_dict['max_iterations']['values'])
    lamb_factor = trial.suggest_categorical('lamb_factor', iLQR_dict['lamb_factor']['values'])
    lamb_max = trial.suggest_categorical('lamb_max', iLQR_dict['lamb_max']['values'])
    epsilon = trial.suggest_categorical('epsilon', iLQR_dict['epsilon']['values'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'q_lqr_{i}', iLQR_dict['q_lqr']['values'][0], iLQR_dict['q_lqr']['values'][1], log=is_log_scale(iLQR_dict['q_lqr']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_lqr_{i}', iLQR_dict['r_lqr']['values'][0], iLQR_dict['r_lqr']['values'][1], log=is_log_scale(iLQR_dict['r_lqr']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'max_iterations': max_iterations,
        'lamb_factor': lamb_factor,
        'lamb_max': lamb_max,
        'epsilon': epsilon,
        'q_lqr': state_weight,
        'r_lqr': action_weight,
    }

    return hps_suggestion


def ilqr_sf_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for iLQR hyperparameters with safety filter.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    max_iterations = trial.suggest_categorical('max_iterations', iLQR_SF_dict['max_iterations']['values'])
    lamb_factor = trial.suggest_categorical('lamb_factor', iLQR_SF_dict['lamb_factor']['values'])
    lamb_max = trial.suggest_categorical('lamb_max', iLQR_SF_dict['lamb_max']['values'])
    epsilon = trial.suggest_categorical('epsilon', iLQR_SF_dict['epsilon']['values'])

    # safety filter
    horizon = trial.suggest_categorical('horizon', iLQR_SF_dict['horizon']['values'])
    n_samples = trial.suggest_categorical('n_samples', iLQR_SF_dict['n_samples']['values'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'q_lqr_{i}', iLQR_SF_dict['q_lqr']['values'][0], iLQR_SF_dict['q_lqr']['values'][1], log=is_log_scale(iLQR_SF_dict['q_lqr']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_lqr_{i}', iLQR_SF_dict['r_lqr']['values'][0], iLQR_SF_dict['r_lqr']['values'][1], log=is_log_scale(iLQR_SF_dict['r_lqr']))
        for i in range(action_dim)
    ]

    # safety filter
    tau = trial.suggest_float('tau', iLQR_SF_dict['tau']['values'][0], iLQR_SF_dict['tau']['values'][1], log=is_log_scale(iLQR_SF_dict['tau']))
    sf_state_weight = [
        trial.suggest_float(f'q_lin_{i}', iLQR_SF_dict['q_lin']['values'][0], iLQR_SF_dict['q_lin']['values'][1], log=is_log_scale(iLQR_SF_dict['q_lin']))
        for i in range(state_dim)
    ]
    sf_action_weight = [
        trial.suggest_float(f'r_lin_{i}', iLQR_SF_dict['r_lin']['values'][0], iLQR_SF_dict['r_lin']['values'][1], log=is_log_scale(iLQR_SF_dict['r_lin']))
        for i in range(action_dim)
    ]
    hps_suggestion = {
        'max_iterations': max_iterations,
        'lamb_factor': lamb_factor,
        'lamb_max': lamb_max,
        'epsilon': epsilon,
        'horizon': horizon,
        'n_samples': n_samples,
        'q_lqr': state_weight,
        'r_lqr': action_weight,
        'tau': tau,
        'q_lin': sf_state_weight,
        'r_lin': sf_action_weight,
    }

    return hps_suggestion

def pid_sampler(trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for PID hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
        state_dim: dimension of the state space
        action_dim: dimension of the action space
    """

    p_coeff_for = [
        trial.suggest_float(f'p_coeff_for_{i}', PID_dict['p_coeff_for']['values'][0], PID_dict['p_coeff_for']['values'][1], log=is_log_scale(PID_dict['p_coeff_for']))
        for i in range(3)
    ]
    i_coeff_for = [
        trial.suggest_float(f'i_coeff_for_{i}', PID_dict['i_coeff_for']['values'][0], PID_dict['i_coeff_for']['values'][1], log=is_log_scale(PID_dict['i_coeff_for']))
        for i in range(3)
    ]
    d_coeff_for = [
        trial.suggest_float(f'd_coeff_for_{i}', PID_dict['d_coeff_for']['values'][0], PID_dict['d_coeff_for']['values'][1], log=is_log_scale(PID_dict['d_coeff_for']))
        for i in range(3)
    ]
    p_coeff_tor = [
        trial.suggest_float(f'p_coeff_tor_{i}', PID_dict['p_coeff_tor']['values'][0], PID_dict['p_coeff_tor']['values'][1], log=is_log_scale(PID_dict['p_coeff_tor']))
        for i in range(3)
    ]
    i_coeff_tor = [
        trial.suggest_float(f'i_coeff_tor_{i}', PID_dict['i_coeff_tor']['values'][0], PID_dict['i_coeff_tor']['values'][1], log=is_log_scale(PID_dict['i_coeff_tor']))
        for i in range(3)
    ]
    d_coeff_tor = [
        trial.suggest_float(f'd_coeff_tor_{i}', PID_dict['d_coeff_tor']['values'][0], PID_dict['d_coeff_tor']['values'][1], log=is_log_scale(PID_dict['d_coeff_tor']))
        for i in range(3)
    ]

    hps_suggestion = {
        'p_coeff_for': p_coeff_for,
        'i_coeff_for': i_coeff_for,
        'd_coeff_for': d_coeff_for,
        'p_coeff_tor': p_coeff_tor,
        'i_coeff_tor': i_coeff_tor,
        'd_coeff_tor': d_coeff_tor,
    }

    return hps_suggestion

def get_distributions(hyperparameter_dict, hps_config):
    """
    Generalized function to create a mapping of Optuna distributions based on a hyperparameter dictionary.

    Args:
        hyperparameter_dict (dict): Dictionary containing hyperparameter definitions.
        hps_config (dict): Dictionary containing warmup hyperparameter configurations.

    Returns:
        dict: A dictionary mapping hyperparameter names to their respective Optuna distributions.
    """
    distributions = {}

    for hp_name, hp_info in hyperparameter_dict.items():
        hp_values = hp_info['values']
        scale = hp_info['scale']
        is_list = hp_info['type'] == list
        cat = hp_info['cat']

        if cat == 'float':
            if scale == 'uniform':
                if is_list:
                    for i in range(len(hps_config[hp_name])):
                        distributions[f'{hp_name}_{i}'] = optuna.distributions.UniformDistribution(hp_values[0], hp_values[1])
                else:
                    distributions[hp_name] = optuna.distributions.UniformDistribution(hp_values[0], hp_values[1])
            elif scale == 'log':
                if is_list:
                    for i in range(len(hps_config[hp_name])):
                        distributions[f'{hp_name}_{i}'] = optuna.distributions.LogUniformDistribution(hp_values[0], hp_values[1])
                else:
                    distributions[hp_name] = optuna.distributions.LogUniformDistribution(hp_values[0], hp_values[1])
            else:
                raise ValueError('Invalid scale')

        elif cat == 'discrete' or cat == 'categorical':
            distributions[hp_name] = optuna.distributions.CategoricalDistribution(hp_values)
        else:
            raise ValueError('Invalid hyperparameter category')
    
    return distributions


HYPERPARAMS_SAMPLER = {
    'ppo': ppo_sampler,
    'dppo': dppo_sampler,
    'sac': sac_sampler,
    'gp_mpc': gpmpc_sampler,
    'gpmpc_acados': gpmpc_sampler,
    'gpmpc_acados_TP': gpmpc_tp_sampler,
    'linear_mpc': lmpc_sampler,
    'mpc_acados': mpc_sampler,
    'fmpc': fmpc_sampler,
    'lqr': lqr_sampler,
    'ilqr': ilqr_sampler,
    'ilqr_sf': ilqr_sf_sampler,
    'pid': pid_sampler,
}
