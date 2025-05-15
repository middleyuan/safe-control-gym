import os
import sys

import pytest

from examples.hpo.hpo_experiment import hpo
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.hyperparameters.hpo_search_space import HYPERPARAMS_DICT


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ilqr', 'gp_mpc', 'ppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc'])
@pytest.mark.parametrize('SAMPLER', ['optuna', 'vizier'])
def test_hpo_cartpole(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER, SAMPLER):
    '''Test HPO for one single trial.
        (create a study from scratch)
    '''

    # output_dir
    output_dir = f'./examples/hpo/results/{ALGO}'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # delete database
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db')
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db-journal'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db-journal')
    if os.path.exists(f'{ALGO}_hpo_endpoint.yaml'):
        os.system(f'rm {ALGO}_hpo_endpoint.yaml')

    if ALGO == 'ilqr':
        PRIOR = '100'
    elif ALGO == 'gp_mpc' or ALGO == 'gpmpc_acados':
        PRIOR = '200'

    # check if the config file exists
    TASK_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{SYS}_{TASK}.yaml'
    ALGO_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    HPO_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{ALGO}_{SYS}_hpo.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'
    assert os.path.exists(HPO_CONFIG_PATH),  f'{HPO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            pytest.skip('SAFETY_FILTER is only supported for ilqr')
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST = 'one_step_cost'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                            SAFETY_FILTER_CONFIG_PATH,
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='optuna', help='which sampler to use in HPO.')
    config = fac.merge()
    config.hpo_config.trials = 1
    config.hpo_config.repetitions = 1
    config.sampler = SAMPLER

    # hyperparameter configurations (to make tests faster)
    # should belong to the distributions defined in the search space
    if 'optimization_iterations' in config.hpo_config.hps_config:
        d = len(config.hpo_config.hps_config.optimization_iterations)
        config.hpo_config.hps_config.optimization_iterations = [HYPERPARAMS_DICT[ALGO]['optimization_iterations']['values'][0]] * d
    if 'num_epochs' in config.hpo_config.hps_config:
        config.hpo_config.hps_config.num_epochs = HYPERPARAMS_DICT[ALGO]['num_epochs']['values'][0]
    if 'max_env_steps' in config.hpo_config.hps_config:
        config.hpo_config.hps_config.max_env_steps = HYPERPARAMS_DICT[ALGO]['max_env_steps']['values'][0]

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # delete database
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db')
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db-journal'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db-journal')
    if os.path.exists(f'{ALGO}_hpo_endpoint.yaml'):
        os.system(f'rm {ALGO}_hpo_endpoint.yaml')


@pytest.mark.parametrize('SYS', ['quadrotor_2D_attitude', 'quadrotor_3D_attitude'])
@pytest.mark.parametrize('TASK', ['tracking'])
@pytest.mark.parametrize('ALGO', ['pid', 'lqr', 'ilqr', 'gp_mpc', 'gpmpc_acados', 'gpmpc_acados_TP', 'fmpc', 'linear_mpc', 'linear_mpc_acados', 'mpc_acados', 'ppo', 'sac', 'dppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc', 'nl_mpsc'])
@pytest.mark.parametrize('SAMPLER', ['optuna', 'vizier'])
def test_hpo_quadrotor(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER, SAMPLER):
    '''Test HPO for one single trial.
        (create a study from scratch)
    '''

    # output_dir
    output_dir = f'./examples/hpo/results/{ALGO}'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # delete database
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db')
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db-journal'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db-journal')
    if os.path.exists(f'{ALGO}_hpo_endpoint.yaml'):
        os.system(f'rm {ALGO}_hpo_endpoint.yaml')

    SYS_NAME = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' or SYS == 'quadrotor_3D_attitude' else SYS

    if ALGO == 'fmpc' or ALGO == 'pid' or ALGO == 'lqr' or ALGO == 'ilqr' or ALGO == 'gpmpc_acados' or ALGO == 'linear_mpc' or ALGO == 'linear_mpc_acados' or ALGO == 'mpc_acados' or ALGO == 'gpmpc_acados_TP':
        PRIOR = '100'
    elif ALGO == 'gp_mpc':
        PRIOR = '200'

    # check if the config file exists
    TASK_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{SYS}_{TASK}.yaml'
    ALGO_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    HPO_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{ALGO}_{SYS}_hpo.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'
    assert os.path.exists(HPO_CONFIG_PATH),  f'{HPO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            pytest.skip('SAFETY_FILTER is only supported for ilqr')
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST = 'one_step_cost'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                            SAFETY_FILTER_CONFIG_PATH,
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]
    elif SAFETY_FILTER == 'nl_mpsc':
        if ALGO != 'ppo':
            pytest.skip('SAFETY_FILTER is only supported for ppo')
            raise ValueError('SAFETY_FILTER is only supported for ppo')
        SAFETY_FILTER_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{SAFETY_FILTER}_{SYS}.yaml'
        TASK_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/ppo_mpsc_{SYS}_{TASK}_basic.yaml'
        ALGO_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/ppo_mpsc_{SYS}_{TASK}_{PRIOR}.yaml'
        HPO_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/ppo_mpsc_{SYS}_hpo.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
        assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'
        assert os.path.exists(HPO_CONFIG_PATH),  f'{HPO_CONFIG_PATH} does not exist'
        MPSC_COST='one_step_cost'
        FILTER=True
        SF_PEN=0.03
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                            SAFETY_FILTER_CONFIG_PATH,
                        '--kv_overrides', 
                            f'sf_config.cost_function={MPSC_COST}',
                            'sf_config.soften_constraints=True',
                            f'algo_config.filter_train_actions={FILTER}',
                            f'algo_config.penalize_sf_diff={FILTER}',
                            f'algo_config.sf_penalty={SF_PEN}',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='optuna', help='which sampler to use in HPO.')
    fac.add_argument('--resume', type=bool, default=False, help='if to resume HPO.')
    config = fac.merge()
    config.hpo_config.trials = 1
    config.hpo_config.repetitions = 1
    config.sampler = SAMPLER

    # hyperparameter configurations (to make tests faster)
    # should belong to the distributions defined in the search space
    if 'optimization_iterations' in config.hpo_config.hps_config:
        d = len(config.hpo_config.hps_config.optimization_iterations)
        config.hpo_config.hps_config.optimization_iterations = [100] * d
    if 'num_epochs' in config.hpo_config.hps_config:
        config.hpo_config.hps_config.num_epochs = 2
    if 'max_env_steps' in config.hpo_config.hps_config:
        config.hpo_config.hps_config.max_env_steps = 1000

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # delete database
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db')
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db-journal'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db-journal')
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}_endpoint.yaml'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}_endpoint.yaml')
