""" The implementation of HPO class

Reference: 
    * stable baselines3 https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
    * Optuna: https://optuna.org

Reruirement:
    * python -m pip install pymysql

"""
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from copy import deepcopy

import optuna
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from optuna_dashboard import run_server
from optuna.trial import FrozenTrial

from safe_control_gym.hyperparameters.hpo_sampler import HYPERPARAMS_SAMPLER, HYPERPARAMS_DICT
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import save_video, mkdirs
from safe_control_gym.utils.logging import ExperimentLogger



class HPO(object):

    def __init__(self, algo, task, sampler, load_study, output_dir, task_config, hpo_config, **algo_config):
        """ Hyperparameter optimization class
        
        args:
            algo: algo name
            env_func: environment that the agent will interact with
            output_dir: output directory
            hpo_config: hyperparameter optimization configuration
            algo_config: algorithm configuration

        """

        self.algo = algo
        self.study_name = algo + "_hpo"
        self.task = task
        self.load_study = load_study
        self.task_config = task_config
        self.hpo_config = hpo_config
        self.hps_config = hpo_config.hps_config
        self.output_dir = output_dir
        self.algo_config = algo_config
        self.logger = ExperimentLogger(output_dir, log_file_out=False)
        self.total_runs = 0
        # init sampler
        if sampler == "RandomSampler":
            self.sampler = RandomSampler(seed=self.hpo_config.seed)
        elif sampler == "TPESampler":
            self.sampler = TPESampler(seed=self.hpo_config.seed)
        else:
            raise ValueError("Unknown sampler.")
        # check if config.hpo_config.prior is defined
        if hasattr(self.hpo_config, 'prior'):
            self.prior = self.hpo_config.prior
        else:
            self.prior = False
        assert len(hpo_config.objective) == len(hpo_config.direction), "objective and direction must have the same length"
    
    def objective(self, trial: optuna.Trial) -> float:
        """ The stochastic objective function for a HPO tool to optimize over
        
        args:
            trial: A single call of the objective function

        """

        # sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](self.hps_config, trial, self.prior)

        # log trial number
        self.logger.info("Trial number: {}".format(trial.number))

        # flag for increasing runs
        increase_runs = True
        first_iteration = True

        # do repetition
        returns, efficiencies, seeds = [], [], []
        while increase_runs:
            increase_runs = False
            if first_iteration:
                Gs_rew = np.inf
                Gs_eff = np.inf
            for i in range(self.hpo_config.repetitions):

                seed = np.random.randint(0, 10000)
                # update the agent config with sample candidate hyperparameters
                # new agent with the new hps
                for hp in sampled_hyperparams:
                    self.algo_config[hp] = sampled_hyperparams[hp]

                seeds.append(seed)
                self.logger.info("Sample hyperparameters: {}".format(sampled_hyperparams))
                self.logger.info("Seeds: {}".format(seeds))

                try:
                    self.env_func = partial(make, self.task, output_dir=self.output_dir, **self.task_config)
                    # using deepcopy(self.algo_config) prevent setting being overwritten
                    self.agent = make(self.algo,
                                        self.env_func,
                                        training=True,
                                        checkpoint_path=os.path.join(self.output_dir, "model_latest.pt"),
                                        output_dir=os.path.join(self.output_dir, "hpo"),
                                        use_gpu=self.hpo_config.use_gpu,
                                        seed=seed,
                                        **deepcopy(self.algo_config))

                    self.agent.reset()
                except Exception as e:
                    # catch exception
                    self.logger.info(f'Exception occurs when constructing agent: {e}')

                # return objective estimate
                # TODO: report intermediate results to Optuna for pruning
                try:
                    self.agent._learn()
                    self.total_runs += 1

                except Exception as e:
                    # catch the NaN generated by the sampler
                    self.agent.close()
                    del self.agent
                    del self.env_func
                    self.logger.info(f'Exception occurs during learning: {e}')
                    print(e)
                    print("Sampled hyperparameters:")
                    print(sampled_hyperparams)
                    returns.append(0.0)
                    efficiencies.append(0.0)
                    break

                # TODO: multiple evaluation
                avg_return = self.agent._run()

                # learning curve
                if hasattr(self.agent, 'learning_curve'):
                    mean_returns, steps = self.agent.learning_curve['mean_returns'], self.agent.learning_curve['steps']
                    # find the index of the mean_returns that is 70% of the arv_return
                    idx = np.where(mean_returns >= 0.7*avg_return)[0]
                    if len(idx) > 0:
                        efficiency = mean_returns[idx[0]] / steps[idx[0]]
                    else:
                        efficiency = 0.0
                else:
                    efficiency = 0.0

                returns.append(avg_return)
                efficiencies.append(efficiency)
                self.logger.info("Sampled rewards: {}".format(returns))

                self.agent.close()
                # delete instances
                del self.agent
                del self.env_func

            Gss_rew = self._compute_cvar(np.array(returns), self.hpo_config.alpha)
            Gss_eff = self._compute_cvar(np.array(efficiencies), self.hpo_config.alpha)

            # if the current objective is better than the best objective, trigger more runs to avoid maximization bias
            if self.hpo_config.warm_trials < len(self.study.trials) and self.hpo_config.dynamical_runs:
                if Gss_rew > self.study.best_value or first_iteration == False:
                    if abs(Gs_rew - Gss_rew) > self.hpo_config.approximation_threshold:
                        increase_runs = True
                        first_iteration = False
                        Gs_rew, Gs_eff = Gss_rew, Gss_eff
                        self.logger.info("Trigger more runs")
                    else:
                        increase_runs = False
        
        return_cvar = Gss_rew
        efficiency_cvar = Gss_eff

        self.logger.info("CVaR of returns: {}".format(return_cvar))

        if 'performance' in self.hpo_config.objective and 'efficiency' not in self.hpo_config.objective:
            return return_cvar
        elif 'efficiency' in self.hpo_config.objective and 'performance' not in self.hpo_config.objective:
            return efficiency_cvar
        elif 'performance' in self.hpo_config.objective and 'efficiency' in self.hpo_config.objective:
            return return_cvar, efficiency_cvar
        else:
            raise ValueError("Objective must be performance, efficiency or both")

    
    def hyperparameter_optimization(self) -> None:
        
        if self.load_study:
            self.study = optuna.load_study(study_name=self.study_name, storage="mysql+pymysql://optuna@localhost/{}".format(self.study_name))
        else:
            # single-objective optimization
            if len(self.hpo_config.direction) == 1:
                self.study = optuna.create_study(
                                                direction=self.hpo_config.direction[0],
                                                sampler=self.sampler,
                                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                                                study_name=self.study_name,
                                                storage="mysql+pymysql://optuna@localhost/{}".format(self.study_name),
                                                load_if_exists=self.hpo_config.load_if_exists
                                                )
            # multi-objective optimization
            else:
                self.study = optuna.create_study(
                                                directions=self.hpo_config.direction,
                                                sampler=self.sampler,
                                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                                                study_name=self.study_name,
                                                storage="mysql+pymysql://optuna@localhost/{}".format(self.study_name),
                                                load_if_exists=self.hpo_config.load_if_exists
                                                )
            
        
        self.study.optimize(self.objective,
                            catch=(RuntimeError,),
                            callbacks=[MaxTrialsCallback(self.hpo_config.trials, states=(TrialState.COMPLETE,))],
                            )

        output_dir = os.path.join(self.output_dir, "hpo")
        # save meta data
        self.study.trials_dataframe().to_csv(output_dir+"/trials.csv")

        # save top-n best hyperparameters
        if len(self.hpo_config.direction) == 1:
            trials = self.study.trials
            if self.hpo_config.direction[0] == "minimize":
                trials.sort(key=self._value_key)
            else:
                trials.sort(key=self._value_key, reverse=True)
            for i in range(min(self.hpo_config.save_n_best_hps, len(self.study.trials))):
                params = trials[i].params
                with open(f"{output_dir}/hyperparameters_{trials[i].value:.4f}.yaml", "w")as f:
                    yaml.dump(params, f, default_flow_style=False)
                if self.hpo_config.perturb_hps:
                    self._perturb_hps(params, f"{output_dir}/hyperparameters_{trials[i].value:.4f}")
        else:
            best_trials = self.study.best_trials
            for i in range(len(self.study.best_trials)):
                params = best_trials[i].params
                with open(f"{output_dir}/best_hyperparameters_[{best_trials[i].values[0]:.4f},{best_trials[i].values[1]:.4f}].yaml", "w")as f:
                    yaml.dump(params, f, default_flow_style=False)
                if self.hpo_config.perturb_hps:
                    self._perturb_hps(params, f"{output_dir}/best_hyperparameters_[{best_trials[i].values[0]:.4f},{best_trials[i].values[1]:.4f}]")

        # dashboard
        if self.hpo_config.dashboard:
            run_server("sqlite:///{}.db".format(self.study_name))

        # save plot
        try:
            if len(self.hpo_config.objective) == 1:
                plot_param_importances(self.study)
                plt.tight_layout()
                plt.savefig(output_dir+"/param_importances.png")
                #plt.show()
                plt.close()
                plot_optimization_history(self.study)
                plt.tight_layout()
                plt.savefig(output_dir+"/optimization_history.png")
                #plt.show()
                plt.close()
            else:
                for i in range(len(self.hpo_config.objective)):
                    plot_param_importances(self.study, target=lambda t: t.values[i])
                    plt.tight_layout()
                    plt.savefig(output_dir+"/param_importances_{}.png".format(self.hpo_config.objective[i]))
                    #plt.show()
                    plt.close()
                    plot_optimization_history(self.study, target=lambda t: t.values[i])
                    plt.tight_layout()
                    plt.savefig(output_dir+"/optimization_history_{}.png".format(self.hpo_config.objective[i]))
                    #plt.show()
                    plt.close()
        except Exception as e:
            print(e)
            print("Plotting failed.")

        self.logger.info("Total runs: {}".format(self.total_runs))
        self.logger.close()

        return
    
    def _value_key(self, trial: FrozenTrial) -> float:
        """ Returns value of trial object for sorting

        """
        if trial.value is None:
            if self.hpo_config.direction[0] == "minimize":
                return float("inf")
            else:
                return float("-inf")
        else:
            return trial.value
    def _compute_cvar(self, returns: np.ndarray, alpha: float = 0.2) -> float:
        """ Compute CVaR

        """
        assert returns.ndim == 1, "returns must be 1D array"
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        VaR_idx = int(alpha * n)
        if VaR_idx == 0:
            VaR_idx = 1
        
        if self.hpo_config.direction[0] == "minimize":
            CVaR = sorted_returns[-VaR_idx:].mean()
        else:
            CVaR = sorted_returns[:VaR_idx].mean()

        return CVaR
    
    def _perturb_hps(self, hp: dict = {}, output_dir: str = '', hp_path: str = '') -> None:
        """ Perturb hyperparameters

        """

        assert (len(hp) > 0 and output_dir != '') or hp_path != '', "Either hp or hp_path must be given"

        hps_dict = HYPERPARAMS_DICT[self.algo]

        if hp_path == '': # After finishing HPO, will enter here if perturb_hps is True
            # make a folder named after the given hp config
            mkdirs(output_dir)

            # perturb each hyperparameter
            for key in hps_dict['categorical']:
                if key in hp:
                    mkdirs(f"{output_dir}/{key}")
                    tmp_hp = deepcopy(hp)
                    interval = sorted(hps_dict['categorical'][key])
                    try:
                        if isinstance(hp[key], list):
                            id = np.argwhere(np.array(interval) == hp[key][0])[0][0]
                        else:
                            id = np.argwhere(np.array(interval) == hp[key])[0][0]
                    except:
                        if isinstance(hp[key], list):
                            id = np.argmin(np.abs(np.array(interval) - hp[key][0]))
                        else:
                            id = np.argmin(np.abs(np.array(interval) - hp[key]))
                    if isinstance(hp[key], str): # real categorical type, e.g., activation function
                        for perturbation in hps_dict['categorical'][key]:
                            mkdirs(f"{output_dir}/{key}/{perturbation}")
                            tmp_hp[key] = perturbation
                            with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                                yaml.dump(tmp_hp, f, default_flow_style=False)
                    else:
                        if isinstance(hp[key], float): # float
                            if id < len(interval)-1 and id > 0:
                                dx = (interval[id+1] - interval[id-1]) / self.hpo_config.divisor
                            elif id == 0:
                                dx = (interval[id+1] - interval[id]) / self.hpo_config.divisor * 2
                            else: # id == len(interval)-1
                                dx = (interval[id] - interval[id-1]) / self.hpo_config.divisor * 2
                        elif isinstance(hp[key], int): # integer
                            if key == 'max_env_steps': # special treatment for max_env_steps because of the vec env.
                                dx = self.algo_config['rollout_batch_size']  * hp['rollout_steps']
                            else:
                                dx = 1
                        elif isinstance(hp[key], list): # list
                            if isinstance(hp[key][0], float): # float
                                if id < len(interval)-1 and id > 0:
                                    dx = (interval[id+1] - interval[id-1]) / self.hpo_config.divisor
                                elif id == 0:
                                    dx = (interval[id+1] - interval[id]) / self.hpo_config.divisor * 2
                                else: # id == len(interval)-1
                                    dx = (interval[id] - interval[id-1]) / self.hpo_config.divisor * 2
                            elif isinstance(hp[key][0], int): # integer
                                    dx = 1
                        else:
                            raise ValueError("Unknown hyperparameter type.")
                        # pertub the hyperparameter by dx on right including the optimized one
                        for i in range(self.hpo_config.side_perturb_num+1):
                            perturbation = interval[id] + (i)*dx
                            while perturbation > max(interval):
                                if isinstance(hp[key], int):
                                    break
                                elif isinstance(hp[key], list) and isinstance(hp[key][0], int):
                                    break
                                dx = dx/10
                                perturbation = interval[id] + (i)*dx
                            mkdirs(f"{output_dir}/{key}/{perturbation}")
                            if isinstance(hp[key], list):
                                tmp_hp[key] = [perturbation]*len(hp[key])
                            else:
                                tmp_hp[key] = perturbation
                            with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                                yaml.dump(tmp_hp, f, default_flow_style=False)
                        # pertub the hyperparameter by dx on left
                        for i in range(self.hpo_config.side_perturb_num):
                            perturbation = interval[id] - (i+1)*dx
                            while perturbation < min(interval):
                                if isinstance(hp[key], int):
                                    break
                                elif isinstance(hp[key], list) and isinstance(hp[key][0], int):
                                    break
                                dx = dx/10
                                perturbation = interval[id] - (i+1)*dx
                            mkdirs(f"{output_dir}/{key}/{perturbation}")
                            if isinstance(hp[key], list):
                                tmp_hp[key] = [perturbation]*len(hp[key])
                            else:
                                tmp_hp[key] = perturbation
                            with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                                yaml.dump(tmp_hp, f, default_flow_style=False)
               
            for key in hps_dict['float']:
                if key in hp:
                    mkdirs(f"{output_dir}/{key}")
                    tmp_hp = deepcopy(hp)
                    interval = sorted(hps_dict['float'][key])
                    dx = (max(interval) - min(interval)) / self.hpo_config.divisor / 10

                    # pertub the hyperparameter by dx on right
                    for i in range(self.hpo_config.side_perturb_num+1):
                        if isinstance(hp[key], list):
                            perturbation = hp[key][0] + (i)*dx
                        else:
                            perturbation = hp[key] + (i)*dx
                        while perturbation > max(interval):
                            dx = dx/10
                            perturbation = hp[key] + (i)*dx
                        mkdirs(f"{output_dir}/{key}/{perturbation}")
                        tmp_hp[key] = perturbation
                        with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                            yaml.dump(tmp_hp, f, default_flow_style=False)
                    # pertub the hyperparameter by dx on left
                    for i in range(self.hpo_config.side_perturb_num):
                        if isinstance(hp[key], list):
                            perturbation = hp[key][0] - (i+1)*dx
                        else:
                            perturbation = hp[key] - (i+1)*dx
                        
                        while perturbation < min(interval):
                            dx = dx/10
                            if isinstance(hp[key], list):
                                perturbation = hp[key][0] - (i+1)*dx
                            else:
                                perturbation = hp[key] - (i+1)*dx
                        mkdirs(f"{output_dir}/{key}/{perturbation}")
                        if isinstance(hp[key], list):
                            tmp_hp[key] = [perturbation]*len(hp[key])
                        else:
                            tmp_hp[key] = perturbation
                        with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                            yaml.dump(tmp_hp, f, default_flow_style=False)
        else: # Without doing HPO, will enter here if perturb_hps is True
            # perturb each hyperparameter
            output_dir = hp_path.split(".yaml")[0]
            mkdirs(output_dir)
            with open(hp_path, "r") as f:
                hp = yaml.load(f, Loader=yaml.FullLoader)
            for key in hps_dict['categorical']:
                if key in hp:
                    mkdirs(f"{output_dir}/{key}")
                    tmp_hp = deepcopy(hp)
                    interval = sorted(hps_dict['categorical'][key])
                    try:
                        if isinstance(hp[key], list):
                            id = np.argwhere(np.array(interval) == hp[key][0])[0][0]
                        else:
                            id = np.argwhere(np.array(interval) == hp[key])[0][0]
                    except:
                        if isinstance(hp[key], list):
                            id = np.argmin(np.abs(np.array(interval) - hp[key][0]))
                        else:
                            id = np.argmin(np.abs(np.array(interval) - hp[key]))
                    if isinstance(hp[key], str): # real categorical type, e.g., activation function
                        for perturbation in hps_dict['categorical'][key]:
                            mkdirs(f"{output_dir}/{key}/{perturbation}")
                            tmp_hp[key] = perturbation
                            with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                                yaml.dump(tmp_hp, f, default_flow_style=False)
                    else:
                        if isinstance(hp[key], float): # float
                            if id < len(interval)-1 and id > 0:
                                dx = (interval[id+1] - interval[id-1]) / self.hpo_config.divisor
                            elif id == 0:
                                dx = (interval[id+1] - interval[id]) / self.hpo_config.divisor * 2
                            else: # id == len(interval)-1
                                dx = (interval[id] - interval[id-1]) / self.hpo_config.divisor * 2
                        elif isinstance(hp[key], int): # integer
                            if key == 'max_env_steps': # special treatment for max_env_steps because of the vec env.
                                dx = self.algo_config['rollout_batch_size'] * hp['rollout_steps']
                            else:
                                dx = 1
                        elif isinstance(hp[key], list): # list
                            if isinstance(hp[key][0], float): # float
                                if id < len(interval)-1 and id > 0:
                                    dx = (interval[id+1] - interval[id-1]) / self.hpo_config.divisor
                                elif id == 0:
                                    dx = (interval[id+1] - interval[id]) / self.hpo_config.divisor * 2
                                else: # id == len(interval)-1
                                    dx = (interval[id] - interval[id-1]) / self.hpo_config.divisor * 2
                            elif isinstance(hp[key][0], int): # integer
                                    dx = 1
                        else:
                            raise ValueError("Unknown hyperparameter type.")
                        # pertub the hyperparameter by dx on right including the optimized one
                        for i in range(self.hpo_config.side_perturb_num+1):
                            perturbation = interval[id] + (i)*dx
                            while perturbation > max(interval):
                                if isinstance(hp[key], int):
                                    break
                                elif isinstance(hp[key], list) and isinstance(hp[key][0], int):
                                    break
                                dx = dx/10
                                perturbation = interval[id] + (i)*dx
                            mkdirs(f"{output_dir}/{key}/{perturbation}")
                            if isinstance(hp[key], list):
                                tmp_hp[key] = [perturbation]*len(hp[key])
                            else:
                                tmp_hp[key] = perturbation
                            with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                                yaml.dump(tmp_hp, f, default_flow_style=False)
                        # pertub the hyperparameter by dx on left
                        for i in range(self.hpo_config.side_perturb_num):
                            perturbation = interval[id] - (i+1)*dx
                            while perturbation < min(interval):
                                if isinstance(hp[key], int):
                                    break
                                elif isinstance(hp[key], list) and isinstance(hp[key][0], int):
                                    break
                                dx = dx/10
                                perturbation = interval[id] - (i+1)*dx
                            mkdirs(f"{output_dir}/{key}/{perturbation}")
                            if isinstance(hp[key], list):
                                tmp_hp[key] = [perturbation]*len(hp[key])
                            else:
                                tmp_hp[key] = perturbation
                            with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                                yaml.dump(tmp_hp, f, default_flow_style=False)
               
            for key in hps_dict['float']:
                if key in hp:
                    mkdirs(f"{output_dir}/{key}")
                    tmp_hp = deepcopy(hp)
                    interval = sorted(hps_dict['float'][key])
                    dx = (max(interval) - min(interval)) / self.hpo_config.divisor / 10

                    # pertub the hyperparameter by dx on right
                    for i in range(self.hpo_config.side_perturb_num+1):
                        if isinstance(hp[key], list):
                            perturbation = hp[key][0] + (i)*dx
                        else:
                            perturbation = hp[key] + (i)*dx
                        while perturbation > max(interval):
                            dx = dx/10
                            perturbation = hp[key] + (i)*dx
                        mkdirs(f"{output_dir}/{key}/{perturbation}")
                        if isinstance(hp[key], list):
                            tmp_hp[key] = [perturbation]*len(hp[key])
                        else:
                            tmp_hp[key] = perturbation
                        with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                            yaml.dump(tmp_hp, f, default_flow_style=False)
                    # pertub the hyperparameter by dx on left
                    for i in range(self.hpo_config.side_perturb_num):
                        if isinstance(hp[key], list):
                            perturbation = hp[key][0] - (i+1)*dx
                        else:
                            perturbation = hp[key] - (i+1)*dx
                        
                        while perturbation < min(interval):
                            dx = dx/10
                            if isinstance(hp[key], list):
                                perturbation = hp[key][0] - (i+1)*dx
                            else:
                                perturbation = hp[key] - (i+1)*dx
                        mkdirs(f"{output_dir}/{key}/{perturbation}")
                        if isinstance(hp[key], list):
                            tmp_hp[key] = [perturbation]*len(hp[key])
                        else:
                            tmp_hp[key] = perturbation
                        with open(f"{output_dir}/{key}/{perturbation}/{key}_{perturbation}.yaml", "w")as f:
                            yaml.dump(tmp_hp, f, default_flow_style=False)