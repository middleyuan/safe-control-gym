'''Template hyperparameter optimization/hyperparameter evaluation script.'''

from safe_control_gym.hyperparameters.optuna.hpo_optuna import HPO_Optuna
from safe_control_gym.hyperparameters.vizier.hpo_vizier import HPO_Vizier
from safe_control_gym.hyperparameters.hpo_eval import HPOEval
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import set_device_from_config, set_dir_from_config, set_seed_from_config


def hpo(config):
    '''Hyperparameter optimization.

    Usage:
        * to start HPO, use with `--func hpo`.
    '''

    # initialize safety filter
    if 'safety_filter' not in config:
        config.safety_filter = None
        config.sf_config = None

    # change the cost function for rl methods
    if config.algo == 'ppo' or config.algo == 'sac' or config.algo == 'dppo':
        config.task_config.cost = 'rl_reward'
        config.task_config.obs_goal_horizon = 1
        config.task_config.normalized_rl_action_space = False
        if 'disturbances' in config.task_config:
            if config.task_config.disturbances is not None:
                # raise ValueError('Double check with this setup.')
                config.task_config.disturbances.observation[0]['std'] += [0, 0, 0, 0, 0, 0]
        config.algo_config.log_interval = 10000000
        config.algo_config.eval_interval = 10000000
        if config.algo == 'ppo' and config.safety_filter == 'nl_mpsc':
            config.sf_config.cost_function='one_step_cost'
            config.sf_config.soften_constraints = True
            config.algo_config.rollout_batch_size = 1
            config.algo_config.filter_train_actions = False
            config.algo_config.penalize_sf_diff = False
            config.algo_config.sf_penalty = 0.03
            # config.algo_config.training = True
    elif config.algo == 'fmpc' or config.algo == 'gp_mpc' or config.algo == 'gpmpc_acados' or config.algo == 'gpmpc_acados_TP' or config.algo == 'linear_mpc' or config.algo == 'linear_mpc_acados' or config.algo == 'mpc_acados':
        pass
    elif config.algo == 'pid' or config.algo == 'lqr' or config.algo == 'ilqr':
        pass
    else:
        raise ValueError('Only ppo, sac, dppo, fmpc, gp_mpc, gpmpc_acados, linear_mpc, mpc_acados, ilqr, lqr, pid are supported for now.')

    # Experiment setup.
    set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # HPO
    if config.sampler == 'optuna':
        hpo = HPO_Optuna(config.hpo_config,
                         config.task_config,
                         config.algo_config,
                         config.algo,
                         config.task,
                         config.output_dir,
                         config.safety_filter,
                         config.sf_config,
                         config.load_study,
                         config.resume
                         )
    elif config.sampler == 'vizier':
        hpo = HPO_Vizier(config.hpo_config,
                         config.task_config,
                         config.algo_config,
                         config.algo,
                         config.task,
                         config.output_dir,
                         config.safety_filter,
                         config.sf_config,
                         config.load_study,
                         config.resume
                         )
    else:
        raise ValueError('Only optuna and vizier are supported for now.')

    hpo.hyperparameter_optimization()
    print('Hyperparameter optimization done.')

def eval(config):
    '''Hyperparameter evaluation.

    Usage:
        * to evaluate hyperparameters, use with `--func eval`.
    '''

    # initialize safety filter
    if 'safety_filter' not in config:
        config.safety_filter = None
        config.sf_config = None
    
    # change the cost function for rl methods
    if config.algo == 'ppo' or config.algo == 'sac' or config.algo == 'dppo':
        config.task_config.cost = 'rl_reward'
        config.task_config.obs_goal_horizon = 1
        config.task_config.normalized_rl_action_space = False
        if 'disturbances' in config.task_config:
            if config.task_config.disturbances is not None:
                # raise ValueError('Double check with this setup.')
                config.task_config.disturbances.observation[0]['std'] += [0, 0, 0, 0, 0, 0]
        config.algo_config.log_interval = 10000000
        config.algo_config.eval_interval = 10000000
        if config.algo == 'ppo' and config.safety_filter == 'nl_mpsc':
            config.sf_config.cost_function='one_step_cost'
            config.sf_config.soften_constraints = True
            config.algo_config.rollout_batch_size = 1
            config.algo_config.filter_train_actions = False
            config.algo_config.penalize_sf_diff = False
            config.algo_config.sf_penalty = 0.03
            # config.algo_config.training = True
    elif config.algo == 'fmpc' or config.algo == 'gp_mpc' or config.algo == 'gpmpc_acados' or config.algo == 'gpmpc_acados_TP' or config.algo == 'linear_mpc' or config.algo == 'linear_mpc_acados' or config.algo == 'mpc_acados':
        pass
    elif config.algo == 'pid' or config.algo == 'lqr' or config.algo == 'ilqr':
        pass
    else:
        raise ValueError('Only ppo, sac, dppo, fmpc, gp_mpc, gpmpc_acados, linear_mpc, mpc_acados, ilqr, lqr, pid are supported for now.')

    appended_folder = config.overrides[0].split('quadrotor_2D_attitude_tracking')[-1].split('.')[0]
    config.output_dir += f'/{appended_folder}'
    
    hpo_eval = HPOEval(config.hpo_config,
                       config.task_config,
                       config.algo_config,
                       config.algo,
                       config.task,
                       config.output_dir,
                       config.safety_filter,
                       config.sf_config,
                       )
    hpo_eval.hp_evaluation()

MAIN_FUNCS = {'hpo': hpo, 'eval': eval}

if __name__ == '__main__':
    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='hpo', help='main function to run.')
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='optuna', help='which package to use in HPO.')
    fac.add_argument('--resume', type=int, default=False, help='if to resume HPO.') # this is defined as int to be able to pass from bash command
    # merge config
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception('Main function {} not supported.'.format(config.func))
    func(config)
