""" The implementation of HPO class using Vizier

Reference:
    * https://oss-vizier.readthedocs.io/en/latest/
    * https://arxiv.org/pdf/0912.3995

"""

import csv
import os
import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import yaml, json
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import servers

from safe_control_gym.hyperparameters.base_hpo import BaseHPO
from safe_control_gym.hyperparameters.hpo_search_space import HYPERPARAMS_DICT
from safe_control_gym.hyperparameters.hpo_utils import get_smallest_and_latest_seed_folder
matplotlib.use('Agg')

class HPO_Vizier(BaseHPO):

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
        Hyperparameter Optimization (HPO) class using package Vizier.

        Args:
            hpo_config: Configuration specific to hyperparameter optimization.
            task_config: Configuration for the task.
            algo_config: Algorithm configuration.
            algo (str): Algorithm name.
            task (str): The task/environment the agent will interact with.
            output_dir (str): Directory where results and models will be saved.
            safety_filter (str): Safety filter to be applied (optional).
            sf_config: Safety filter configuration (optional).
            load_study (bool): Load existing study if True.
            resume (bool): if resume from a trial file.
        """
        super().__init__(hpo_config, task_config, algo_config, algo, task, output_dir, safety_filter, sf_config, load_study, resume)

        self.client_id = f'client_{os.getpid()}'  # use process id as client id
        self.setup_problem()

    def setup_problem(self):
        """ Setup hyperparameter optimization, e.g., search space, study, algorithm, etc. """

        # define the problem statement
        self.problem = vz.ProblemStatement()

        # define the search space
        self.search_space = HYPERPARAMS_DICT[self.search_space_key]

        for hp_name, hp_info in self.search_space.items():
            hp_values = hp_info['values']
            scale = hp_info['scale']
            is_list = hp_info['type'] == list
            cat = hp_info['cat']

            if cat == 'float':
                if scale == 'uniform':
                    if is_list:
                        for i in range(len(self.hps_config[hp_name])):
                            self.problem.search_space.root.add_float_param(f'{hp_name}_{i}', hp_values[0], hp_values[1])
                    else:
                        self.problem.search_space.root.add_float_param(hp_name, hp_values[0], hp_values[1])
                elif scale == 'log':
                    if is_list:
                        for i in range(len(self.hps_config[hp_name])):
                            self.problem.search_space.root.add_float_param(f'{hp_name}_{i}', hp_values[0], hp_values[1], scale_type=vz.ScaleType.LOG)
                    else:
                        self.problem.search_space.root.add_float_param(hp_name, hp_values[0], hp_values[1], scale_type=vz.ScaleType.LOG)
                else:
                    raise ValueError('Invalid scale')

            elif cat == 'discrete':
                if scale == 'uniform':
                    if is_list:
                        for i in range(len(self.hps_config[hp_name])):
                            self.problem.search_space.root.add_discrete_param(f'{hp_name}_{i}', hp_values)
                    else:
                        self.problem.search_space.root.add_discrete_param(hp_name, hp_values)
                elif scale == 'log':
                    if is_list:
                        for i in range(len(self.hps_config[hp_name])):
                            self.problem.search_space.root.add_discrete_param(f'{hp_name}_{i}', hp_values, scale_type=vz.ScaleType.LOG)
                    else:
                        self.problem.search_space.root.add_discrete_param(hp_name, hp_values, scale_type=vz.ScaleType.LOG)
                else:
                    raise ValueError('Invalid scale')

            elif cat == 'categorical':
                self.problem.search_space.root.add_categorical_param(hp_name, hp_values)
            else:
                raise ValueError('Invalid hyperparameter category')

        # Set optimization direction based on objective and direction from the HPO config
        for objective, direction in zip(self.hpo_config.objective, self.hpo_config.direction):
            if direction == 'maximize':
                self.problem.metric_information.append(
                    vz.MetricInformation(name=objective, goal=vz.ObjectiveMetricGoal.MAXIMIZE))
            elif direction == 'minimize':
                self.problem.metric_information.append(
                    vz.MetricInformation(name=objective, goal=vz.ObjectiveMetricGoal.MINIMIZE))
            else:
                raise ValueError('Invalid direction, must be either maximize or minimize')

    def hyperparameter_optimization(self) -> None:
        """ Hyperparameter optimization.
        """
        if self.load_study:
            # try to load the study from the endpoint file periodically
            while not os.path.exists(f'{self.study_name}_vizier_endpoint.yaml'):
                self.logger.info('Endpoint file not found. Waiting for the endpoint file to be created.')
                time.sleep(10)
            with open(f'{self.study_name}_vizier_endpoint.yaml', 'r') as config_file:
                endpoint = yaml.safe_load(config_file)['endpoint']
            clients.environment_variables.server_endpoint = endpoint
            study_config = vz.StudyConfig.from_problem(self.problem)
            study_config.algorithm = 'GAUSSIAN_PROCESS_BANDIT'
            self.study_client = clients.Study.from_study_config(study_config, owner='owner', study_id=self.study_name)
            self.study_client = clients.Study.from_resource_name(self.study_client.resource_name)
        else:
            server = servers.DefaultVizierServer(database_url=f'sqlite:///{self.study_name}_vizier.db')
            clients.environment_variables.server_endpoint = server.endpoint
            endpoint = server.endpoint
            if os.path.exists(f'{self.study_name}_vizier_endpoint.yaml'):
                os.remove(f'{self.study_name}_vizier_endpoint.yaml')
            with open(f'{self.study_name}_vizier_endpoint.yaml', 'w') as config_file:
                yaml.dump({'endpoint': endpoint}, config_file, default_flow_style=False)

            study_config = vz.StudyConfig.from_problem(self.problem)
            study_config.algorithm = 'GAUSSIAN_PROCESS_BANDIT'
            self.study_client = clients.Study.from_study_config(study_config, owner='owner', study_id=self.study_name)
            self.resume_trials() if self.resume else self.warm_start(self.config_to_param(self.hps_config))

        existing_trials = 0
        while existing_trials < self.hpo_config.trials:
            # get suggested hyperparameters
            suggestions = self.study_client.suggest(count=1, client_id=self.client_id)
            # suggestions = self.study_client.suggest(count=1)

            for suggestion in suggestions:
                if suggestion.id > self.hpo_config.trials:
                    self.logger.info(f'Trial {suggestion.id} is deleted as it exceeds the maximum number of trials.')
                    suggestion.delete()
                    existing_trials = suggestion.id
                    break
                self.logger.info(f'Hyperparameter optimization trial {suggestion.id}/{self.hpo_config.trials}')
                existing_trials = suggestion.id
                # evaluate the suggested hyperparameters
                materialized_suggestion = suggestion.materialize()
                suggested_params = {key: val.value for key, val in materialized_suggestion.parameters._items.items()}
                res = self.evaluate(suggested_params, seed_list=[num for num in range(self.hpo_config.repetitions)])
                if res != self.none_handler():
                    trajs_data_list = self.trajs_data_list
                    metrics_list = self.metrics_list
                    try:
                        self.plot_results(trajs_data_list, metrics_list, self.output_dir, f'(trial_{suggestion.id})')
                    except Exception as e:
                        self.logger.info('Error plotting results: {}'.format(e))
                        self.logger.std_out_logger.logger.exception('Full exception traceback')
                objective_values = {obj: np.mean(res[obj]) for obj in self.hpo_config.objective}
                self.logger.info(f'Returns: {objective_values}')
                final_measurement = vz.Measurement(objective_values)
                self.objective_value = objective_values
                # wandb.log({f'{self.hpo_config.objective[0]}': objective_value})
                suggestion.complete(final_measurement)

            if existing_trials > 0:
                self.checkpoint()

        if self.load_study is False:

            completed_trial_filter = vz.TrialFilter(status=[vz.TrialStatus.COMPLETED])
            finished_trials = len(list(self.study_client.trials(trial_filter=completed_trial_filter).get()))
            # wait until other clients to finish
            while finished_trials < self.hpo_config.trials:
                self.logger.info(f'Waiting for other clients to finish remaining trials: {self.hpo_config.trials - finished_trials}')
                finished_trials = len(list(self.study_client.trials(trial_filter=completed_trial_filter).get()))
                # sleep for 10 seconds
                time.sleep(10)

            self.logger.info(f'Have finished trials: {finished_trials}/{self.hpo_config.trials}')

            self.checkpoint()

            self.logger.info('Deleting server.')
            del server

        self.logger.close()

    def warm_start(self, params):
        """
        Warm start the study.

        Args:
            params (dict): Specified hyperparameters to be evaluated.
        """
        if hasattr(self, 'study_client'):
            res = self.evaluate(params, seed_list=[num for num in range(self.hpo_config.repetitions)])
            if res != self.none_handler():
                trajs_data_list = self.trajs_data_list
                metrics_list = self.metrics_list
                try:
                    self.plot_results(trajs_data_list, metrics_list, self.output_dir, '(warmstart)')
                except Exception as e:
                    self.logger.info('Error plotting results: {}'.format(e))
                    self.logger.std_out_logger.logger.exception('Full exception traceback')
            objective_values = {obj: np.mean(res[obj]) for obj in self.hpo_config.objective}
            params = self.remove_umoptimized_hps(params)
            trial = vz.Trial(parameters=params, final_measurement=vz.Measurement(objective_values))
            self.study_client._add_trial(trial)
            self.warmstart_trial_value = res

    def resume_trials(self):
        """
        Resume trials from a trial file.
        """
        def helper(s):
            try:
                return json.loads(s)  
            except:
                return float(s)
        # get previous and lastest seed folder
        try:
            folder_path = get_smallest_and_latest_seed_folder(self.output_dir)
            csv_file = os.path.join(folder_path, 'hpo', 'trials.csv')
            data = pd.read_csv(csv_file)
            length = len(data)
            for i in range(length):
                config = {key: helper(data[key].iloc[i]) for key in self.hps_config.keys()}
                params = self.config_to_param(config)
                objective_values = {obj: data[obj].iloc[i] for obj in self.hpo_config.objective}
                trial = vz.Trial(parameters=params, final_measurement=vz.Measurement(objective_values))
                self.study_client._add_trial(trial)
                self.logger.info(f'Resume trial {i} with hyperparameters: {params}')
                self.logger.info(f'Returns: {objective_values}')
        except:
            self.logger.info('No trial file found to resume')

    def checkpoint(self):
        """
        Save checkpoints, results, and logs during hyperparameter optimization.
        Supports logging and visualizing multiple optimization objectives.
        """
        output_dir = os.path.join(self.output_dir, 'hpo')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save warmstart trial value if exists
        if hasattr(self, 'warmstart_trial_value'):
            with open(f'{output_dir}/warmstart_trial_value.txt', 'w') as f:
                f.write(str(self.warmstart_trial_value))
        
        # Filter completed trials
        completed_trial_filter = vz.TrialFilter(status=[vz.TrialStatus.COMPLETED])
        all_trials = [tc.materialize() for tc in self.study_client.trials(trial_filter=completed_trial_filter)]
        
        try:
            # Handle optimal trials for hyperparameter optimization
            optimal_trials = list(self.study_client.optimal_trials())
            
            # Save hyperparameters for each optimal trial
            for optimal_trial in optimal_trials:
                optimal_trial = optimal_trial.materialize()
                
                # Extract parameters
                params = {key: val.value for key, val in optimal_trial.parameters._items.items()}
                params = self.post_process_best_hyperparams(params)
                params = self.add_unoptimized_hps(params)
                
                # Create filename with multiple objective values
                objective_values = [
                    f"{objective}_{optimal_trial.final_measurement.metrics[objective].value:.4f}"
                    for objective in self.hpo_config.objective
                ]
                filename = f'{output_dir}/hyperparameters_trial{optimal_trial.id}_' + '_'.join(objective_values) + '.yaml'
                
                with open(filename, 'w') as f:
                    yaml.dump(params, f, default_flow_style=False)
        
        except Exception as e:
            print(e)
            print('Saving hyperparameters failed')
        
        try:
            # Visualization for hyperparameter optimization
            plt.figure(figsize=(12, 5))
            
            # Create subplots for each objective
            num_objectives = len(self.hpo_config.objective)
            for obj_idx, objective in enumerate(self.hpo_config.objective, 1):
                plt.subplot(1, num_objectives, obj_idx)
                
                # Scatter plot of trials for this objective
                trial_i = [t.id - 1 for t in all_trials]
                trial_ys = [t.final_measurement.metrics[objective].value for t in all_trials]
                plt.scatter(trial_i, trial_ys, label='trials', marker='o', color='blue')
                
                if num_objectives == 1:
                    # Mark optimal trials
                    optimal_trial_i = [t.id - 1 for t in optimal_trials]
                    optimal_trial_ys = [t.final_measurement.metrics[objective].value for t in optimal_trials]
                    plt.scatter(optimal_trial_i, optimal_trial_ys, label='optimal', marker='x', color='green', s=100)
                
                plt.title(f'Optimization History: {objective}')
                plt.xlabel('Trial')
                plt.ylabel(f'{objective} Value')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir + '/optimization_history.png')
            plt.close()
            
            # Collect trial data for CSV
            trial_data = []
            parameter_keys = set()
            
            for t in all_trials:
                trial_number = t.id - 1
                # Collect all objective values
                trial_objective_values = {
                    objective: t.final_measurement.metrics[objective].value 
                    for objective in self.hpo_config.objective
                }
                
                # Extract parameters for each trial
                trial_params = {key: val.value for key, val in t.parameters._items.items()}
                trial_params = self.post_process_best_hyperparams(trial_params)
                trial_params = self.add_unoptimized_hps(trial_params)
                parameter_keys.update(trial_params.keys())
                
                trial_data.append((trial_number, trial_objective_values, trial_params))
            
            # Convert set to sorted list for consistent CSV header
            parameter_keys = sorted(list(parameter_keys))
            
            # Save to CSV file
            csv_file = 'trials.csv'
            with open(output_dir + '/' + csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Create header: number, objective values, then parameters
                header = ['number'] + self.hpo_config.objective + parameter_keys
                writer.writerow(header)
                
                # Write trial data
                for trial_number, objective_values, trial_params in trial_data:
                    # Ensure objectives and parameters are in consistent order
                    row_values = [trial_number]
                    row_values.extend([objective_values.get(obj, '') for obj in self.hpo_config.objective])
                    row_values.extend([json.dumps(trial_params.get(key, '')) for key in parameter_keys])
                    
                    writer.writerow(row_values)
        
        except Exception as e:
            print(e)
            print('Saving hyperparameter optimization history failed')
