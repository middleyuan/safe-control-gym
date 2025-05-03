"""Evaluation of hyperparameter optimization results."""

import numpy as np
from matplotlib.colors import to_rgba

from safe_control_gym.hyperparameters.base_hpo import BaseHPO

class HPOEval(BaseHPO):

    def __init__(self,
                 hpo_config,
                 task_config,
                 algo_config,
                 algo='ilqr',
                 task='stabilization',
                 output_dir='./results',
                 safety_filter=None,
                 sf_config=None):
        '''Hyperparameter optimization evaluation class.

        Args:
            hpo_config (dict): HPO configuration.
            task_config (dict): Task configuration.
            algo_config (dict): Algorithm configuration.
            algo (str): Algorithm name.
            task (str): Task name.
            output_dir (str): Output directory.
            safety_filter (str): Safety filter name.
            sf_config (dict): Safety filter configuration.
        '''
        super(HPOEval, self).__init__(hpo_config, task_config, algo_config, algo, task, output_dir, safety_filter, sf_config, False, False)

        self.hp_eval_list = self.hpo_config.hp_eval_list

    def interpolate_color(self, base_color, light_color, index, total):
        '''
        Interpolates between base_color and light_color based on index.
        '''
        base_rgba = np.array(to_rgba(base_color))
        light_rgba = np.array(to_rgba(light_color))
        return tuple((1 - index / (total - 1)) * base_rgba + (index / (total - 1)) * light_rgba)

    def hp_evaluation(self):
        '''Evaluate the handtuned/optimized hyperparameters and make the comparison.'''

        trajs_dict = {}
        metrics_dict = {}

        # evaluate hyperparameters specified in hp_eval_list
        for hp_name in self.hp_eval_list:
            hp_eval = self.hpo_config[hp_name]
            hps = self.config_to_param(hp_eval)
            np.random.seed(self.hpo_config.seed)
            self.evaluate(hps, seed_list=[num for num in range(self.hpo_config.repetitions)])
            trajs_data_list = self.trajs_data_list
            metrics_list = self.metrics_list
            self.plot_results(trajs_data_list, metrics_list, self.output_dir, f'({hp_name})')

            trajs_dict[hp_name] = trajs_data_list
            metrics_dict[hp_name] = metrics_list        

        # Remove None values
        trajs_dict = {k: v for k, v in trajs_dict.items() if v is not None}
        metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}

        self.plot_results_grid(trajs_dict, metrics_dict, self.output_dir)

    def setup_problem(self):
        """
        Dummy function.
        """
        raise NotImplementedError

    def warm_start(self, params):
        """
        Dummy function.
        """
        raise NotImplementedError
    
    def resume_trials(self):
        """
        Dummy function.
        """
        raise NotImplementedError

    def hyperparameter_optimization(self):
        """
        Dummy function.
        """
        raise NotImplementedError

    def checkpoint(self):
        """
        Dummy function.
        """
        raise NotImplementedError