'''Base hyerparameter optimization class.'''


import os
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import colorsys

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.hyperparameters.hpo_search_space import HYPERPARAMS_DICT
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs

def interpolate_color(base_hex, light_hex, num_seeds, max_seeds):
    # Convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    # Convert RGB to HEX
    def rgb_to_hex(rgb):
        return '#' + ''.join(f'{int(c * 255):02x}' for c in rgb)
    
    # Convert RGB to HLS
    base_rgb = hex_to_rgb(base_hex)
    light_rgb = hex_to_rgb(light_hex)
    base_hls = colorsys.rgb_to_hls(*base_rgb)
    light_hls = colorsys.rgb_to_hls(*light_rgb)
    
    # Interpolate lightness
    ratio = min(num_seeds / max_seeds, 1.0)  # Clamp to [0, 1]
    interpolated_lightness = base_hls[1] + (light_hls[1] - base_hls[1]) * ratio
    
    # Keep hue and saturation constant, change lightness
    interpolated_hls = (base_hls[0], interpolated_lightness, base_hls[2])
    interpolated_rgb = colorsys.hls_to_rgb(*interpolated_hls)
    
    return rgb_to_hex(interpolated_rgb)


class BaseHPO(ABC):

    def __init__(self,
                 hpo_config,
                 task_config,
                 algo_config,
                 algo='ilqr',
                 task='stabilization',
                 output_dir='./results',
                 safety_filter=None,
                 sf_config=None,
                 load_study=False,
                 resume=False):
        """
        Base class for Hyperparameter Optimization (HPO).

        Args:
            hpo_config (dict): Configuration specific to hyperparameter optimization.
            task_config (dict): Configuration for the task.
            algo_config (dict): Algorithm configuration.
            algo (str): Algorithm name.
            task (str): The task/environment the agent will interact with.
            output_dir (str): Directory where results and models will be saved.
            safety_filter (str): Safety filter to be applied (optional).
            sf_config: Safety filter configuration (optional).
            load_study (bool): Load existing study if True.
            resume (bool): Resume existing trials if True.
        """
        self.algo = algo
        self.task = task
        self.output_dir = output_dir
        self.exp_name = output_dir.split('/')[-2]
        self.task_config = task_config
        self.hpo_config = hpo_config
        self.algo_config = algo_config
        self.safety_filter = safety_filter
        self.sf_config = sf_config
        assert self.safety_filter is None or self.sf_config is not None, 'Safety filter config must be provided if safety filter is not None'
        if self.safety_filter == 'linear_mpsc' and self.algo == 'ilqr':
            self.search_space_key = 'ilqr_sf'
        elif self.safety_filter == 'nl_mpsc' and self.algo == 'ppo':
            self.search_space_key = 'ppo_mpsf'
            self.study_name = algo + '_mpsf_hpo' + f'_{self.exp_name}'
        else:
            self.study_name = algo + '_hpo' + f'_{self.exp_name}'
            self.search_space_key = self.algo
        self.logger = ExperimentLogger(output_dir)
        self.load_study = load_study
        # set up hp initializations
        if f'{self.exp_name}_init' in self.hpo_config:
            self.hps_config = self.hpo_config[f'{self.exp_name}_init']
        else:
            self.hps_config = hpo_config.hps_config
        self.n_episodes = hpo_config.n_episodes
        self.objective_bounds = hpo_config.objective_bounds

        env_func = partial(make, self.task, output_dir=self.output_dir, **self.task_config)
        self.env = env_func()

        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        self.resume = resume

        self.append_hps_config()
        self.check_hyperparmeter_config()

        assert len(hpo_config.objective) == len(hpo_config.direction), 'objective and direction must have the same length'

    def append_hps_config(self):
        """
        Append hyperparameters (self.hps_config) if safety filter is not None.

        """
        if self.safety_filter is not None:
            for hp in HYPERPARAMS_DICT[self.search_space_key]:
                if hp in self.sf_config:
                    self.hps_config[hp] = self.sf_config[hp]

    def remove_umoptimized_hps(self, params):
        ''' Remove unoptimized hyperparameters from the sampled hyperparameters (one may wants to speficify hps in self.hps_config
            but does not want them to be optimized).'''
        
        for hp in list(params.keys()):
            # Handle multidimensional hyperparameters (e.g., q_mpc_0, q_mpc_1)
            base_param, index_str = hp.rsplit('_', 1) if '_' in hp else (hp, '')
            if index_str.isdigit():
                if base_param not in HYPERPARAMS_DICT[self.search_space_key]:
                    del params[hp]
            else:
                if hp not in HYPERPARAMS_DICT[self.search_space_key]:
                    del params[hp]

        return params
    
    def add_unoptimized_hps(self, params):
        ''' Add unoptimized hyperparameters to the sampled hyperparameters for saving purposes.'''

        for hp in self.hps_config:
            if hp not in params:
                params[hp] = self.hps_config[hp]

        return params

    def special_handle(self, param_name, param_value):
        """
        Special handling for specific hyperparameters, e.g., learning_rate and optimization_iterations, which
        have list types in configs but only one DoF in the search space. Special handling can be added here in
        the future if needed.

        Args:
            param_name (str): Name of the hyperparameter.
            param_value (Any): Sampled value of the hyperparameter.

        Returns:
            Valid (bool): True if the hyperparameter is valid, False otherwise.
            param_value (Any): If valid, sampled value of the hyperparameter cast to the appropriate type based on self.hps_config.
        """

        # special cases: learning_rate and optimization_iterations for gp_mpc
        valid = False
        if self.algo == 'gp_mpc' or self.algo == 'gpmpc_acados' or self.algo == 'gpmpc_acados_TP':
            if param_name == 'learning_rate' or param_name == 'optimization_iterations':
                if type(param_value) is not type(self.hps_config[param_name]):
                    param_value = len(self.hps_config[param_name]) * [type(self.hps_config[param_name][0])(param_value)]
                if type(param_value) is type(self.hps_config[param_name]):
                    valid = True
                return valid, param_value

        return valid, param_value

    def check_hyperparmeter_config(self):
        """
        Check if the hyperparameter configuration (self.hps_config) is valid, e.g., if types match what defined in hpo_search_space.py.
        """
        valid = True
        for param in self.hps_config:
            if param in HYPERPARAMS_DICT[self.search_space_key]:
                if HYPERPARAMS_DICT[self.search_space_key][param]['type'] is not type(self.hps_config[param]):
                    valid = False
                    valid, _ = self.special_handle(param, self.hps_config[param])
                    assert valid, f'Hyperparameter {param} should be of type {HYPERPARAMS_DICT[self.search_space_key][param]["type"]}'

    @abstractmethod
    def setup_problem(self):
        """
        Setup hyperparameter optimization, e.g., search space, study, algorithm, etc.
        Needs to be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def warm_start(self, params):
        """
        Warm start the study.

        Args:
            params (dict): Specified hyperparameters.
            objective (float): Objective value.
        """
        raise NotImplementedError
    
    @abstractmethod
    def resume_trials(self):
        """
        Resume existing trials.
        """
        raise NotImplementedError

    def cast_to_original_type_from_config(self, param_name, param_value):
        """
        Cast the parameter to its original type based on the existing task, algo, or safty filter config.

        Args:
            param_name (str): Name of the hyperparameter.
            param_value (Any): Sampled value of the hyperparameter.

        Returns:
            Any: The parameter value cast to the appropriate type.
        """
        # Check if the parameter exists in task_config, algo_config, or sf_config
        if param_name in self.task_config:
            current_value = self.task_config[param_name]
            valid, res_value = self.special_handle(param_name, param_value)
        elif param_name in self.algo_config:
            current_value = self.algo_config[param_name]
            valid, res_value = self.special_handle(param_name, param_value)
        elif self.safety_filter is not None and param_name in self.sf_config:
            current_value = self.sf_config[param_name]
            valid, res_value = self.special_handle(param_name, param_value)
        else:
            raise ValueError(f'Unknown parameter {param_name} - cannot cast to original type from config')

        if valid:
            return res_value
        else:
            return type(current_value)(param_value)

    def cast_to_original_type_from_hyperparams_dict(self, param_name, param_value):
        """
        Cast the parameter to its original type based on HYPERPARAMS_DICT.

        Args:
            param_name (str): Name of the hyperparameter.
            param_value (Any): Sampled value of the hyperparameter.

        Returns:
            Any: The parameter value cast to the appropriate type.
        """
        return HYPERPARAMS_DICT[self.search_space_key][param_name]['type'](param_value)

    def param_to_config(self, params):
        """
        Convert sampled hyperparameters to configurations (self.task_config, self.algo_config, and self.sf_config).

        Args:
            params (dict): Sampled hyperparameters.

        """
        # Special handling for iLQR: set reward weights to match LQR weights
        if self.algo == 'ilqr':
            # Look for q_lqr and r_lqr parameters (either single values or multi-dimensional)
            q_lqr_params = {}
            r_lqr_params = {}
            
            for param_name, param_value in params.items():
                if param_name.startswith('q_lqr_'):
                    # Multi-dimensional parameter (e.g., q_lqr_0, q_lqr_1)
                    base_param, index_str = param_name.rsplit('_', 1)
                    if index_str.isdigit():
                        index = int(index_str)
                        q_lqr_params[index] = param_value
                elif param_name.startswith('r_lqr_'):
                    # Multi-dimensional parameter (e.g., r_lqr_0, r_lqr_1)
                    base_param, index_str = param_name.rsplit('_', 1)
                    if index_str.isdigit():
                        index = int(index_str)
                        r_lqr_params[index] = param_value
            
            # Set task reward weights to match LQR weights
            if q_lqr_params:
                if hasattr(self.task_config, 'rew_state_weight'):
                    for index, value in q_lqr_params.items():
                        if index < len(self.task_config.rew_state_weight):
                            self.task_config.rew_state_weight[index] = value

            if r_lqr_params:
                if hasattr(self.task_config, 'rew_act_weight'):
                    for index, value in r_lqr_params.items():
                        if index < len(self.task_config.rew_act_weight):
                            self.task_config.rew_act_weight[index] = value

        # Iterate through the params dictionary
        for param_name, param_value in params.items():
            # Handle multidimensional hyperparameters (e.g., q_mpc_0, q_mpc_1)
            base_param, index_str = param_name.rsplit('_', 1) if '_' in param_name else (None, '')
            if index_str.isdigit():
                if (base_param in self.algo_config or base_param in self.task_config or (self.safety_filter is not None and base_param in self.sf_config)):
                    # If base parameter exists in algo_config as a list/array
                    index = int(index_str)
                    
                    # Special handling for PID 2D constraint before setting the parameter
                    if (self.algo == 'pid' and self.task == 'quadrotor_2D_attitude' and 
                        base_param in ['p_coeff_for', 'i_coeff_for', 'd_coeff_for', 'p_coeff_tor', 'i_coeff_tor', 'd_coeff_tor'] and 
                        index == 1 and base_param in self.algo_config):
                        # For y-axis (index 1), use the x-axis value (index 0) if it exists in params
                        x_axis_param = f'{base_param}_0'
                        if x_axis_param in params:
                            param_value = params[x_axis_param]
                    
                    if base_param in self.algo_config:
                        self.algo_config[base_param][index] = param_value
                    elif base_param in self.task_config:
                        self.task_config[base_param][index] = param_value
                    elif base_param in self.sf_config:
                        self.sf_config[base_param][index] = param_value
            # Handle the mapping based on the parameter name
            elif param_name in self.algo_config:
                # If the param is related to the algorithm, update the algo_config
                self.algo_config[param_name] = self.cast_to_original_type_from_config(param_name, param_value)
            elif param_name in self.task_config:
                # If the param is related to the task, update the task_config
                self.task_config[param_name] = self.cast_to_original_type_from_config(param_name, param_value)
            elif self.safety_filter is not None and param_name in self.sf_config:
                # If the param is related to the safety filter, update the sf_config
                self.sf_config[param_name] = self.cast_to_original_type_from_config(param_name, param_value)
            else:
                # If the parameter doesn't map to a known config, log or handle it
                print(f'Warning: Unknown parameter {param_name} - not mapped to any configuration')

    def config_to_param(self, params):
        """
        Convert configuration to hyperparameters (mainly to handle multidimensional hyperparameters) as the input to add_trial.

        Args:
            params (dict): User specified hyperparameters (self.hps_config).

        Returns:
            dict: Hyperparameter representation in the problem.
        """
        params = deepcopy(params)
        for param in list(params.keys()):
            is_list = isinstance(params[param], list)
            if param in HYPERPARAMS_DICT[self.search_space_key]:
                if is_list:
                    # multi-dimensional hyperparameters
                    if list == HYPERPARAMS_DICT[self.search_space_key][param]['type']:
                        for i, value in enumerate(params[param]):
                            new_param = f'{param}_{i}'
                            params[new_param] = value
                        del params[param]
                    # single-dimensional hyperparameters but in list format
                    else:
                        params[param] = params[param][0]

        return params

    def post_process_best_hyperparams(self, params):
        """
        Post-process the best hyperparameters after optimization (mainly to handle multidimensional hyperparameters).

        Args:
            params (dict): Best hyperparameters.

        Returns:
            params (dict): Post-processed hyperparameters.
        """
        aggregated_params = {}
        for param_name, param_value in params.items():
            # Handle multidimensional hyperparameters (e.g., q_mpc_0, q_mpc_1)
            base_param, index_str = param_name.rsplit('_', 1) if '_' in param_name else (None, '')
            if index_str.isdigit():
                if base_param in self.algo_config or base_param in self.task_config or (self.safety_filter is not None and base_param in self.sf_config):
                    # If base parameter exists in algo_config as a list/array
                    index = int(index_str)
                    if base_param in aggregated_params:
                        aggregated_params[base_param][index] = param_value
                    else:
                        aggregated_params[base_param] = {index: param_value}

        for param_name, param_value in aggregated_params.items():
            for hp in list(params.keys()):  # Create a list of keys to iterate over
                if param_name in hp:
                    del params[hp]
            params[param_name] = [param_value[i] for i in range(len(param_value))]

        for hp in params:
            params[hp] = self.cast_to_original_type_from_config(hp, params[hp])

        return params

    def apply_constraints_to_params(self, params):
        """
        Apply constraint logic to parameters for logging/saving purposes.
        This ensures that saved trial parameters reflect the actual constraints used.
        
        Args:
            params (dict): Original hyperparameters.
            
        Returns:
            dict: Constrained hyperparameters.
        """
        # Make a copy to avoid modifying the original
        constrained_params = deepcopy(params)
        
        # Apply PID 2D constraint for saving purposes
        if self.algo == 'pid' and self.task == 'quadrotor_2D_attitude':
            # PID parameters that need 2D constraint: p_coeff_for, i_coeff_for, d_coeff_for
            pid_params = ['p_coeff_for', 'i_coeff_for', 'd_coeff_for', 'p_coeff_tor', 'i_coeff_tor', 'd_coeff_tor']
            
            for pid_param in pid_params:
                if pid_param in constrained_params and isinstance(constrained_params[pid_param], list):
                    if len(constrained_params[pid_param]) >= 2:
                        # Set y-axis (index 1) equal to x-axis (index 0)
                        constrained_params[pid_param][1] = constrained_params[pid_param][0]
        
        return constrained_params

    def evaluate(self, params, seed_list=None):
        """
        Evaluation of hyperparameters.

        Args:
            params (dict): Hyperparameters to be evaluated.
            seed_list (list): List of seeds for evaluation.
        Returns:
            Sampled objective value (list)
        """
        if seed_list is not None:
            assert len(seed_list) == self.hpo_config.repetitions, 'Number of seeds should be equal to the number of repetitions'
        sampled_hyperparams = params

        seeds, trajs_data_list, metrics_list = [], [], []
        returns = { obj: [] for obj in self.hpo_config.objective }
        for i in range(self.hpo_config.repetitions):

            seed = np.random.randint(0, 10000) if seed_list is None else seed_list[i]

            # update the agent config with sample candidate hyperparameters
            # pass the hyperparameters to config
            self.param_to_config(sampled_hyperparams)

            seeds.append(seed)
            self.logger.info('Sample hyperparameters: {}'.format(sampled_hyperparams))
            self.logger.info('Seeds: {}'.format(seeds))

            try:
                self.env_func = partial(make, self.task, output_dir=self.output_dir, **self.task_config)
                # using deepcopy(self.algo_config) prevent setting being overwritten
                self.agent = make(self.algo,
                                  self.env_func,
                                  training=True,
                                  checkpoint_path=os.path.join(self.output_dir, 'model_latest.pt'),
                                  output_dir=os.path.join(self.output_dir, 'hpo'),
                                  use_gpu=self.hpo_config.use_gpu,
                                  seed=seed,
                                  **deepcopy(self.algo_config))

                self.agent.reset()
                seed += 110
                eval_env = self.env_func(seed=seed)
                # Setup safety filter
                if self.safety_filter is not None:
                    env_func_filter = partial(make,
                                              self.task,
                                              **self.task_config)
                    safety_filter = make(self.safety_filter,
                                         env_func_filter,
                                         **self.sf_config)
                    safety_filter.reset()
                    # try:
                    #     safety_filter.learn()
                    # except Exception as e:
                    #     self.logger.info(f'Exception occurs when constructing safety filter: {e}')
                    #     self.logger.info('Safety filter config: {}'.format(self.sf_config))
                    #     self.logger.std_out_logger.logger.exception('Full exception traceback')
                    #     self.agent.close()
                    #     del self.agent
                    #     del self.env_func
                    #     return self.none_handler()
                    # mkdirs(f'{self.output_dir}/models/')
                    # safety_filter.save(path=f'{self.output_dir}/models/{self.safety_filter}.pkl')
                    self.agent.safety_filter = safety_filter
                    experiment = BaseExperiment(eval_env, self.agent, safety_filter=safety_filter)
                else:
                    experiment = BaseExperiment(eval_env, self.agent)
            except Exception as e:
                # catch exception
                self.logger.info(f'Exception occurs when constructing agent: {e}')
                self.logger.std_out_logger.logger.exception('Full exception traceback')
                if hasattr(self, 'agent'):
                    self.agent.close()
                    del self.agent
                del self.env_func
                return self.none_handler()

            # return objective estimate
            try:
                # self.agent.learn()
                experiment.launch_training()
            except Exception as e:
                # catch the NaN generated by the sampler
                self.agent.close()
                del self.agent
                del self.env_func
                self.logger.info(f'Exception occurs during learning: {e}')
                self.logger.std_out_logger.logger.exception('Full exception traceback')
                print(e)
                print('Sampled hyperparameters:')
                print(sampled_hyperparams)
                return self.none_handler()

            # TODO: add n_episondes to the config
            try:
                if self.algo == 'ppo' and self.safety_filter == 'nl_mpsc':
                    self.sf_config.cost_function = 'precomputed_cost'
                    self.sf_config.mpsc_cost_horizon = 25
                    self.sf_config.decay_factor = 1
                    self.sf_config.max_w = 0.0
                    self.sf_config.slack_cost = 1000.0
                    self.task_config.done_on_violation = False
                    self.task_config.randomized_init = False
                    env_func = partial(make, self.task, output_dir=self.output_dir, **self.task_config)
                    env = env_func()
                    self.task_config.constraints[0].upper_bounds = [0.899, 1.99, 1.449, 1.99, 0.749, 2.99]
                    self.task_config.constraints[0].lower_bounds = [-0.899, -1.99, 0.551, -1.99, -0.749, -2.99]
                    self.task_config.constraints[1].upper_bounds = [0.59, 0.436]
                    self.task_config.constraints[1].lower_bounds = [0.113, -0.436]
                    env_func = partial(make, self.task, output_dir=self.output_dir, **self.task_config)
                    agent = make(self.algo,
                                  env_func,
                                  training=False,
                                  checkpoint_path=os.path.join(self.output_dir, 'model_latest.pt'),
                                  output_dir=os.path.join(self.output_dir, 'hpo'),
                                  use_gpu=self.hpo_config.use_gpu,
                                  seed=seed,
                                  **deepcopy(self.algo_config))
                    agent.load(os.path.join(self.output_dir, 'model_latest.pt'))
                    sf = make(self.safety_filter,
                                        env_func,
                                        **self.sf_config)
                    sf.reset()
                    if self.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
                        sf.cost_function.uncertified_controller = self.agent
                        sf.cost_function.output_dir = '.'
                    exp = BaseExperiment(env, agent, safety_filter=sf)
                    trajs_data, metrics = exp.run_evaluation(n_episodes=self.n_episodes, n_steps=None, done_on_max_steps=True)
                    exp.close()
                    sf.close()
                else:
                    trajs_data, metrics = experiment.run_evaluation(n_episodes=self.n_episodes, n_steps=None, done_on_max_steps=True)
            except Exception as e:
                self.agent.close()
                # delete instances
                del self.agent
                del self.env_func
                del experiment
                self.logger.info(f'Exception occurs during evaluation: {e}')
                self.logger.std_out_logger.logger.exception('Full exception traceback')
                print(e)
                print('Sampled hyperparameters:')
                print(sampled_hyperparams)
                return self.none_handler()

            # at the moment, only single-objective optimization is supported
            trajs_data_list.append(trajs_data)
            metrics_list.append(metrics)
            for obj in self.hpo_config.objective:
                obj_value = metrics[obj]
                returns[obj].append(obj_value)
            self.logger.info('Sampled objectives: {}'.format(returns))

            self.agent.close()
            # delete instances
            del self.agent
            del self.env_func
        
        self.trajs_data_list = trajs_data_list
        self.metrics_list = metrics

        return returns

    def none_handler(self):
        """
        Assign worse objective values (based on objective bound) to None returns.
        """
        returns = { obj: [] for obj in self.hpo_config.objective }
        for obj in self.hpo_config.objective:
            if self.hpo_config.direction[0] == 'maximize':
                returns[obj].append(self.objective_bounds[0][0])
            else:
                returns[obj].append(self.objective_bounds[0][1])
        return returns

    @abstractmethod
    def hyperparameter_optimization(self):
        """
        Hyperparameter optimization loop. Should be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self):
        """
        Save checkpoints, results, and logs during optimization.
        """
        raise NotImplementedError
    
    def plot_results(self, trajs_data_list, metrics, output_dir, tag):
        '''Plot the evaluation results, overlaying all trajectories.

        Args:
            trajs_data_list (list): List of seeds, where each seed is a dictionary with keys:
                'obs': Array of shape (num_episodes, ).
                'current_physical_action': Array of shape (num_episodes, ).
            metrics (list): List of metrics for each seed.
            output_dir (str): Output directory for plots.
            tag (str): Other info for the plot.
        '''
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Number of seeds
        num_seeds = len(trajs_data_list)

        # metrics
        returns_mean = np.mean(metrics['rmse'])
        action_change = np.mean(metrics['rms_action_change'])

        # Flatten data: extract episodes from each seed
        state_traj = np.vstack([seed_data['obs'] for seed_data in trajs_data_list])
        action_traj = np.vstack([seed_data['current_clipped_action'] for seed_data in trajs_data_list]) 

        # determin the state index
        if self.env.state_dim == 6:
            x_idx = 0
            y_idx = None
            z_idx = 2
            thrust_idx = 0
            pitch_idx = 1
        elif self.env.state_dim == 12:
            x_idx = 0
            y_idx = 2
            z_idx = 4
            thrust_idx = 0
            pitch_idx = 2
        else:
            raise ValueError('State dimension not supported')

        # Total number of episodes
        num_episodes = state_traj.shape[0]
        episodes_per_seed = state_traj.shape[0] // num_seeds  # Assuming equal episodes per seed

        alpha_values = np.linspace(0.65, 1.0, num_seeds)  # Transparency range

        thrust_color = '#1b9e77'  # Green
        thrust_light_color = '#a6dbb0'  # Light green
        pitch_color = '#7570b3'  # Purple
        pitch_light_color = '#c7c1e4'  # Light purple

        # Plot all action trajectories in one plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for ep_idx in range(num_episodes):
            seed_idx = ep_idx // episodes_per_seed  # Determine which seed this episode belongs to
            color = interpolate_color(thrust_color, thrust_light_color, seed_idx, num_seeds)
            ax.plot(action_traj[ep_idx, :, thrust_idx], label=f'Seed {seed_idx + 1} Thrust' if ep_idx % episodes_per_seed == 0 else None, color=color, lw=2)
            color = interpolate_color(pitch_color, pitch_light_color, seed_idx, num_seeds)
            ax.plot(action_traj[ep_idx, :, pitch_idx], label=f'Seed {seed_idx + 1} Pitch' if ep_idx % episodes_per_seed == 0 else None, color=color, lw=2, linestyle='--')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Action')
        ax.set_title(f'Action Trajectories w/ RMS: {action_change:.4f} {tag}')
        ax.legend(loc='best', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'action_trajectories_{tag}.png'))
        plt.close()

        # Plot all state trajectories in x-z plane in one plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for ep_idx in range(num_episodes):
            seed_idx = ep_idx // episodes_per_seed  # Determine which seed this episode belongs to
            alpha = alpha_values[seed_idx]
            ax.plot(state_traj[ep_idx, :, x_idx], state_traj[ep_idx, :, z_idx],
                    label=f'Seed {seed_idx + 1}' if ep_idx % episodes_per_seed == 0 else None, color='blue', lw=2, alpha=alpha)
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$z$ [m]')
        ax.set_title(f'State Paths w/ RMSE: {returns_mean:.4f} {tag}')
        ax.legend(loc='best', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'state_paths_xz_{tag}.png'))
        plt.close()

        if y_idx is not None:
            # Plot all state trajectories in x-y plane in one plot
            fig, ax = plt.subplots(figsize=(8, 4))
            for ep_idx in range(num_episodes):
                seed_idx = ep_idx // episodes_per_seed
                alpha = alpha_values[seed_idx]
                ax.plot(state_traj[ep_idx, :, x_idx], state_traj[ep_idx, :, y_idx],
                        label=f'Seed {seed_idx + 1}' if ep_idx % episodes_per_seed == 0 else None, color='blue', lw=2, alpha=alpha)
            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel('$y$ [m]')
            ax.set_title(f'State Paths w/ RMSE: {returns_mean:.4f} {tag}')
            ax.legend(loc='best', fontsize='small', ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'state_paths_xy_{tag}.png'))
            plt.close()

    def plot_results_grid(self, trajs_dict, metrics_dict, output_dir):
        """
        Plot evaluation results in a grid layout.
        Each row corresponds to a trajectory type (state or action),
        and each column corresponds to a hyperparameter set.

        Args:
            trajs_dict (dict): Dictionary where keys are tags (e.g., 'handtuned hps', 'vizier hps') 
                            and values are corresponding trajs_data_list.
            metrics_dict (dict): Dictionary where keys are tags (e.g., 'handtuned hps', 'vizier hps')
            output_dir (str): Output directory for plots.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a figure with a grid layout: 2 rows (state and action) and N columns (tags)
        num_tags = len(trajs_dict)
        fig, axes = plt.subplots(2, num_tags, figsize=(5 * num_tags, 8))
        
        if num_tags == 1:  # Handle case where there is only one tag (axes won't be a 2D array)
            axes = np.expand_dims(axes, axis=-1)

        # Define a base color for each hyperparameter set
        color_palette = plt.get_cmap('tab10')
        hps_colors = {tag: color_palette(i) for i, tag in enumerate(self.hp_eval_list)}

        # Iterate over tags and data
        for col_idx, (tag, trajs_data_list) in enumerate(trajs_dict.items()):
            # Number of seeds
            num_seeds = len(trajs_data_list)

            # Flatten data: extract episodes from each seed
            state_traj = np.vstack([seed_data['obs'] for seed_data in trajs_data_list])
            action_traj = np.vstack([seed_data['current_clipped_action'] for seed_data in trajs_data_list])

            # determin the state index
            if self.env.state_dim == 6:
                x_idx = 0
                z_idx = 2
                thrust_idx = 0
                pitch_idx = 1
            elif self.env.state_dim == 12:
                x_idx = 0
                z_idx = 4
                thrust_idx = 0
                pitch_idx = 2
            else:
                raise ValueError('State dimension not supported')

            # Total number of episodes
            num_episodes = state_traj.shape[0]
            episodes_per_seed = state_traj.shape[0] // num_seeds  # Assuming equal episodes per seed

            # Define a base color and varying transparency levels
            base_color = hps_colors.get(tag, 'gray')
            alpha_values = np.linspace(0.65, 1.0, num_seeds)  # Transparency range
            thrust_color = '#1b9e77'  # Green
            thrust_light_color = '#a6dbb0'  # Light green
            pitch_color = '#7570b3'  # Purple
            pitch_light_color = '#c7c1e4'  # Light purple

            metrics = metrics_dict[tag]
            action_change = np.mean(metrics['rms_action_change'])

            # Plot action trajectories
            ax = axes[0, col_idx]
            for ep_idx in range(num_episodes):
                seed_idx = ep_idx // episodes_per_seed  # Determine which seed this episode belongs to
                alpha = alpha_values[seed_idx]
                color = interpolate_color(thrust_color, thrust_light_color, seed_idx, num_seeds)
                ax.plot(action_traj[ep_idx, :, thrust_idx], label=f'Seed {seed_idx + 1} Thrust' if ep_idx % episodes_per_seed == 0 else None,
                        color=color, lw=2, alpha=alpha)
                color = interpolate_color(pitch_color, pitch_light_color, seed_idx, num_seeds)
                ax.plot(action_traj[ep_idx, :, pitch_idx], label=f'Seed {seed_idx + 1} Pitch' if ep_idx % episodes_per_seed == 0 else None,
                        color=color, lw=2, linestyle='--', alpha=alpha)
            ax.set_xlabel('Time step')
            ax.set_ylabel('Action')
            ax.set_title(f'Action Trajectories w/ RMS: {action_change:.4f} ({tag})')
            ax.legend(loc='best', fontsize='small', ncol=2)

            # metrics
            returns_mean = np.mean(metrics['rmse'])

            # Plot state trajectories in x-z plane
            ax = axes[1, col_idx]
            for ep_idx in range(num_episodes):
                seed_idx = ep_idx // episodes_per_seed  # Determine which seed this episode belongs to
                alpha = alpha_values[seed_idx]
                ax.plot(state_traj[ep_idx, :, x_idx], state_traj[ep_idx, :, z_idx],
                        label=f'Seed {seed_idx + 1}' if ep_idx % episodes_per_seed == 0 else None,
                        color=base_color, lw=2, alpha=alpha)
            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel('$z$ [m]')
            ax.set_title(f'State Paths w/ RMSE: {returns_mean:.4f} ({tag})')
            ax.legend(loc='best', fontsize='small', ncol=2)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_evaluation_results.png'))
        plt.close()
