

import os
import time
from datetime import datetime

import casadi as cs
import gpytorch
import matplotlib.pyplot as plt
import munch
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.signal import butter, filtfilt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from termcolor import colored

from safe_control_gym.controllers.mpc.gp_utils import (GaussianProcess, ZeroMeanIndependentGPModel,
                                                       covMatern52_single, covSE_single, kmeans_centriods)
from safe_control_gym.controllers.mpc.gpmpc_base import GPMPC
from safe_control_gym.controllers.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.utils import timing


class GPMPC_ACADOS_TRPY(GPMPC):
    '''Implements a GP-MPC controller with Acados optimization.'''

    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            constraint_tol: float = 1e-8,
            additional_constraints: list = None,
            soft_constraints: dict = None,
            warmstart: bool = True,
            train_iterations: int = None,
            test_data_ratio: float = 0.2,
            overwrite_saved_data: bool = True,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            n_ind_points: int = 30,
            inducing_point_selection_method='kmeans',
            recalc_inducing_points_at_every_step=False,
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            online_learning: bool = False,
            prior_info: dict = None,
            sparse_gp: bool = False,
            # inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = 'results/temp',
            compute_ipopt_initial_guess: bool = True,
            use_RTI: bool = False,
            use_linear_prior: bool = True,
            train_env_rand_info: dict = None,
            obs_noise_std: float = 0.005,
            act_noise_std: float = 0.005,
            param_noise_std: list = None,
            **kwargs
    ):
        super().__init__(
            env_func=env_func,
            seed=seed,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            constraint_tol=constraint_tol,
            additional_constraints=additional_constraints,
            soft_constraints=soft_constraints,
            warmstart=warmstart,
            train_iterations=train_iterations,
            test_data_ratio=test_data_ratio,
            overwrite_saved_data=overwrite_saved_data,
            optimization_iterations=optimization_iterations,
            learning_rate=learning_rate,
            normalize_training_data=normalize_training_data,
            use_gpu=use_gpu,
            gp_model_path=gp_model_path,
            prob=prob,
            initial_rollout_std=initial_rollout_std,
            input_mask=input_mask,
            target_mask=target_mask,
            gp_approx=gp_approx,
            sparse_gp=sparse_gp,
            n_ind_points=n_ind_points,
            inducing_point_selection_method=inducing_point_selection_method,
            recalc_inducing_points_at_every_step=recalc_inducing_points_at_every_step,
            online_learning=online_learning,
            prior_info=prior_info,
            prior_param_coeff=prior_param_coeff,
            terminate_run_on_done=terminate_run_on_done,
            output_dir=output_dir,
            obs_noise_std=obs_noise_std,
            act_noise_std=act_noise_std,
            **kwargs)
        self.param_noise_std = param_noise_std
        self.input_mask = None
        self.target_mask = None
        self.train_env_rand_info = train_env_rand_info
        self.rand_hist = {'task_rand': [], 'domain_rand': []}

        # MPC params
        # self.use_linear_prior = use_linear_prior
        self.compute_ipopt_initial_guess = False
        self.use_RTI = use_RTI

        if hasattr(self, 'prior_ctrl'):
            self.prior_ctrl.close()

        self.prior_ctrl = MPC_ACADOS(
            env_func=self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            use_gpu=use_gpu,
            seed=seed,
            use_RTI=use_RTI,
            prior_info=prior_info,
        )
        self.prior_ctrl.reset()
        self.prior_dynamics_func = self.prior_ctrl.dynamics_func
        self.prior_dynamics_func_c = self.prior_ctrl.model.fc_func

        self.x_guess = None
        self.u_guess = None
        self.x_prev = None
        self.u_prev = None
        # print('prior_info[prior_prop]', prior_info['prior_prop'])
        self.state_labels = self.env.STATE_LABELS.copy()
        self.action_labels = self.env.ACTION_LABELS.copy()
        # self.uncertain_dim = ['x_dot', 'y_dot', 'z_dot',
        #                       'p', 'q', 'r']
        # self.uncertain_dim_idx = [self.state_labels.index(i) for i in self.uncertain_dim]
        self.uncertain_dim = ['p', 'q', 'r', 'force_motor']
        self.uncertain_dim_idx = [self.state_labels.index(i) for i in self.uncertain_dim]
        self.x_dot_idx = self.state_labels.index('x_dot')
        self.y_dot_idx = self.state_labels.index('y_dot')
        self.z_dot_idx = self.state_labels.index('z_dot')
        self.theta_idx = self.state_labels.index('theta')
        self.theta_dot_idx = self.state_labels.index('p')
        self.phi_idx = self.state_labels.index('phi')
        self.phi_dot_idx = self.state_labels.index('p')
        self.psi_idx = self.state_labels.index('psi')
        self.psi_dot_idx = self.state_labels.index('r')
        self.tau_idx = self.state_labels.index('force_motor')
        self.T_cmd_idx = self.action_labels.index('T_c')
        self.theta_cmd_idx = self.action_labels.index('R_c')
        self.phi_cmd_idx = self.action_labels.index('P_c')
        self.psi_cmd_idx = self.action_labels.index('Y_c')
        self.Bd = np.eye(self.model.nx)[:, self.uncertain_dim_idx]
        self.input_mask = None
        self.target_mask = None
        self.rand_hist = {'task_rand': [], 'domain_rand': []}
        self.new_GP_model = False
        # self.param_noise_std = param_noise_std

        # Store the noise variances for use in GP training
        self.thrust_noise_var = None
        self.roll_noise_var = None
        self.pitch_noise_var = None
        self.yaw_noise_var = None

    def preprocess_training_data(self,
                                 x_seq,
                                 u_seq,
                                 x_next_seq
                                 ):
        '''Converts trajectory data for GP trianing.

        Args:
            x_seq (list): state sequence of np.array (nx,).
            u_seq (list): action sequence of np.array (nu,).
            x_next_seq (list): next state sequence of np.array (nx,).

        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).
        '''
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
        # it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        dt = 1 / 60
        # x_pred_seq = self.prior_dynamics_func(x0=x_seq.T, p=u_seq.T)['xf'].toarray()
        # T_prior_data = x_seq[:, self.tau_idx].reshape(-1, 1)
        # numerical differentiation
        x_dot_seq = [(x_next_seq[i, :] - x_seq[i, :]) / dt for i in range(x_seq.shape[0])]
        x_dot_seq = np.array(x_dot_seq)

        # Apply low-pass filter to x_dot_seq
        if x_dot_seq.shape[0] > 6:  # Need at least 6 samples for filtfilt to work properly
            # Design Butterworth low-pass filter
            fs = 1 / dt  # Sampling frequency (60 Hz)
            cutoff = 10.0  # Cutoff frequency in Hz
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(2, normal_cutoff, btype='low', analog=False)

            # Apply filter to each column of x_dot_seq
            x_dot_seq_filtered = np.zeros_like(x_dot_seq)
            for i in range(x_dot_seq.shape[1]):
                x_dot_seq_filtered[:, i] = filtfilt(b, a, x_dot_seq[:, i])
            x_dot_seq = x_dot_seq_filtered

        # T_true_data = np.sqrt((x_dot_seq[:, self.z_dot_idx] +  g) ** 2
        #                     + (x_dot_seq[:, self.x_dot_idx] ** 2)
        #                     + (x_dot_seq[:, self.y_dot_idx] ** 2)).reshape(-1, 1)
        T_true_data = x_dot_seq[:, self.tau_idx].reshape(-1, 1)
        T_prior = self.prior_dynamics_func_c(x=x_seq.T, u=u_seq.T)['f'].toarray()[self.tau_idx, :].reshape(-1, 1)
        targets_T = (T_true_data - T_prior).reshape(-1, 1)
        input_T = np.concatenate([x_seq[:, self.tau_idx].reshape(-1, 1),
                                  u_seq[:, self.T_cmd_idx].reshape(-1, 1)], axis=1)

        theta_true = x_dot_seq[:, self.theta_idx]
        theta_prior = self.prior_dynamics_func_c(x=x_seq.T, u=u_seq.T)['f'].toarray()[self.theta_idx, :]
        targets_theta = (theta_true - theta_prior).reshape(-1, 1)
        input_theta = np.concatenate([x_seq[:, self.theta_idx].reshape(-1, 1),
                                      x_seq[:, self.theta_dot_idx].reshape(-1, 1),
                                      u_seq[:, self.theta_cmd_idx].reshape(-1, 1)], axis=1)

        phi_true = x_dot_seq[:, self.phi_idx]
        phi_prior = self.prior_dynamics_func_c(x=x_seq.T, u=u_seq.T)['f'].toarray()[self.phi_idx, :]
        targets_phi = (phi_true - phi_prior).reshape(-1, 1)
        input_phi = np.concatenate([x_seq[:, self.phi_idx].reshape(-1, 1),
                                    x_seq[:, self.phi_dot_idx].reshape(-1, 1),
                                    u_seq[:, self.phi_cmd_idx].reshape(-1, 1)], axis=1)

        psi_true = x_dot_seq[:, self.psi_idx]
        psi_prior = self.prior_dynamics_func_c(x=x_seq.T, u=u_seq.T)['f'].toarray()[self.psi_idx, :]
        targets_psi = (psi_true - psi_prior).reshape(-1, 1)
        input_psi = np.concatenate([x_seq[:, self.psi_idx].reshape(-1, 1),
                                    x_seq[:, self.psi_dot_idx].reshape(-1, 1),
                                    u_seq[:, self.psi_cmd_idx].reshape(-1, 1)], axis=1)

        # train_input = np.concatenate([input_T, input_phi, input_theta], axis=1)
        # train_output = np.concatenate([targets_T, targets_phi, targets_theta], axis=1)
        train_input = np.concatenate([input_T, input_phi, input_theta, input_psi], axis=1)
        train_output = np.concatenate([targets_T, targets_phi, targets_theta, targets_psi], axis=1)

        # # Estimate the noise propagated into T, R, P, Y
        # # Noise variance calculation based on numerical differentiation and control inputs
        # var_x_ddot = 2*self.obs_noise_std[self.x_dot_idx]**2/dt**2 if hasattr(self, 'obs_noise_std') else 1e-6
        # var_y_ddot = 2*self.obs_noise_std[self.y_dot_idx]**2/dt**2 if hasattr(self, 'obs_noise_std') else 1e-6
        # var_z_ddot = 2*self.obs_noise_std[self.z_dot_idx]**2/dt**2 if hasattr(self, 'obs_noise_std') else 1e-6

        # # Thrust (T) noise variance - from force_motor dynamics
        # T_cmd = u_seq[:, self.T_cmd_idx]
        # thrust_noise_var = self.act_noise_std[0]**2 if hasattr(self, 'act_noise_std') else 1e-6

        # # Roll (R) noise variance - from roll dynamics
        # roll_noise_var = 2*self.obs_noise_std[self.phi_dot_idx]**2/dt**2 if hasattr(self, 'obs_noise_std') else 1e-6
        # roll_noise_var += self.act_noise_std[1]**2 if hasattr(self, 'act_noise_std') else 0

        # # Pitch (P) noise variance - from pitch dynamics
        # pitch_noise_var = 2*self.obs_noise_std[self.theta_dot_idx]**2/dt**2 if hasattr(self, 'obs_noise_std') else 1e-6
        # pitch_noise_var += self.act_noise_std[2]**2 if hasattr(self, 'act_noise_std') else 0

        # # Yaw (Y) noise variance - from yaw dynamics
        # yaw_noise_var = 2*self.obs_noise_std[self.psi_dot_idx]**2/dt**2 if hasattr(self, 'obs_noise_std') else 1e-6
        # yaw_noise_var += self.act_noise_std[3]**2 if hasattr(self, 'act_noise_std') else 0

        thrust_noise_var = 0.3
        pitch_noise_var = 2
        roll_noise_var = 2
        yaw_noise_var = 2

        # Store the noise variances for use in GP training
        self.thrust_noise_var = np.array(np.max(thrust_noise_var))
        self.roll_noise_var = np.array(np.max(roll_noise_var))
        self.pitch_noise_var = np.array(np.max(pitch_noise_var))
        self.yaw_noise_var = np.array(np.max(yaw_noise_var))

        return train_input, train_output

    def learn(self, env=None):
        '''Performs multiple epochs learning.
        '''

        train_runs = {0: {}}
        test_runs = {0: {}}

        # epoch seed factor
        np.random.seed(self.seed)
        epoch_seeds = np.random.randint(1000, size=self.num_epochs, dtype=int) * self.seed
        epoch_seeds = [int(seed) for seed in epoch_seeds]

        if self.same_train_initial_state:
            train_env = self.env_func(seed=epoch_seeds[0])
            train_env.action_space.seed(epoch_seeds[0])
            train_envs = [train_env] * self.num_epochs
        else:
            train_envs = []
            for epoch in range(self.num_epochs):
                train_envs.append(self.env_func(seed=epoch_seeds[epoch]))
                train_envs[epoch].action_space.seed(epoch_seeds[epoch])

        test_envs = []
        if self.same_test_initial_state:
            for epoch in range(self.num_epochs):
                test_envs.append(self.env_func(seed=epoch_seeds[epoch]))
                test_envs[epoch].action_space.seed(epoch_seeds[epoch])
        else:
            test_env = self.env_func(seed=epoch_seeds[0])
            test_env.action_space.seed(epoch_seeds[0])
            test_envs = [test_env] * self.num_epochs

        for env in train_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]
        for env in test_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]

        # creating train and test experiments
        train_experiments = [BaseExperiment(env=env, ctrl=self, reset_when_created=False) for env in train_envs[1:]]
        test_experiments = [BaseExperiment(env=env, ctrl=self, reset_when_created=False) for env in test_envs[1:]]
        # first experiments are for the prior
        train_experiments.insert(0, BaseExperiment(env=train_envs[0], ctrl=self.prior_ctrl, reset_when_created=False))
        test_experiments.insert(0, BaseExperiment(env=test_envs[0], ctrl=self.prior_ctrl, reset_when_created=False))

        for episode in range(self.num_train_episodes_per_epoch):
            self.env = train_envs[0]
            run_results = train_experiments[0].run_evaluation(n_episodes=1)
            train_runs[0].update({episode: munch.munchify(run_results)})
        for test_ep in range(self.num_test_episodes_per_epoch):
            self.env = test_envs[0]
            run_results = test_experiments[0].run_evaluation(n_episodes=1)
            test_runs[0].update({test_ep: munch.munchify(run_results)})

        training_results = None
        for epoch in range(1, self.num_epochs):
            # only take data from the last episode from the last epoch
            episode_length = train_runs[epoch - 1][self.num_train_episodes_per_epoch - 1][0]['obs'][0].shape[0]
            if True:
                x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, self.num_samples, train_envs[epoch - 1].np_random)
            else:
                x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, self.num_samples)
            train_inputs, train_targets = self.preprocess_training_data(x_seq, actions, x_next_seq)  # np.ndarray
            training_results = self.train_gp(input_data=train_inputs, target_data=train_targets)

            if self.plot_trained_gp:
                self.plot_gp_TRPY(train_inputs, train_targets, title=f'epoch_{epoch}_train', output_dir=self.output_dir)

            # Test new policy.
            test_runs[epoch] = {}
            for test_ep in range(self.num_test_episodes_per_epoch):
                self.x_prev = test_runs[epoch - 1][episode][0]['obs'][0][:self.T + 1, :].T
                self.u_prev = test_runs[epoch - 1][episode][0]['action'][0][:self.T, :].T
                self.env = test_envs[epoch]
                run_results = test_experiments[epoch].run_evaluation(n_episodes=1)
                test_runs[epoch].update({test_ep: munch.munchify(run_results)})

            x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(test_runs, epoch - 1, episode_length)
            train_inputs, train_targets = self.preprocess_training_data(x_seq, actions, x_next_seq)  # np.ndarray

            if self.plot_trained_gp:
                self.plot_gp_TRPY(train_inputs, train_targets, title=f'epoch_{epoch}_test', output_dir=self.output_dir)

                # Use the test run data for open-loop prediction evaluation
                test_episode_data = test_runs[epoch - 1][0][0]  # First test episode from previous epoch
                x0 = test_episode_data['obs'][0][0, :]  # Initial state
                u_seq = test_episode_data['action'][0]  # Control sequence
                x_true = test_episode_data['obs'][0]  # True trajectory

                self.plot_open_loop_prediction(
                    x0=x0,
                    u_seq=u_seq,
                    x_true=x_true,
                    title=f'epoch_{epoch}_open_loop_eval',
                    output_dir=self.output_dir
                )

            # gather training data
            train_runs[epoch] = {}
            for episode in range(self.num_train_episodes_per_epoch):
                self.x_prev = train_runs[epoch - 1][episode][0]['obs'][0][:self.T + 1, :].T
                self.u_prev = train_runs[epoch - 1][episode][0]['action'][0][:self.T, :].T
                self.env = train_envs[epoch]
                run_results = train_experiments[epoch].run_evaluation(n_episodes=1)
                train_runs[epoch].update({episode: munch.munchify(run_results)})

            # lengthscale, outputscale, noise, kern = self.gaussian_process.get_hyperparameters(as_numpy=True)
            # compute the condition number of the kernel matrix
            # self.rand_hist['task_rand'].append(train_envs[epoch].episode_len)
            self.rand_hist['task_rand'].append(test_experiments[epoch].env.episode_len)
            domain_rand_info = {}
            for keys, values in test_experiments[epoch].env.disturbances.items():
                if keys == 'downwash':
                    domain_rand_info[keys] = test_experiments[epoch].env.dw_model.pos
                else:
                    domain_rand_info[keys] = values.disturbances[0].std
            env_dyn_params = {}
            env_dyn_params['prop_values'] = train_experiments[epoch].env.last_prop_values
            domain_rand_info['env_dyn_params'] = env_dyn_params
            self.rand_hist['domain_rand'].append(domain_rand_info)
            # TODO: fix data logging
            np.savez(os.path.join(self.output_dir, 'epoch_data'),
                     data_inputs=training_results['train_inputs'],
                     data_targets=training_results['train_targets'],
                     train_runs=train_runs,
                     test_runs=test_runs,
                     num_epochs=self.num_epochs,
                     num_train_episodes_per_epoch=self.num_train_episodes_per_epoch,
                     num_test_episodes_per_epoch=self.num_test_episodes_per_epoch,
                     num_samples=self.num_samples,
                     # trajectory=self.trajectory,
                     # ctrl_freq=self.config.task_config.ctrl_freq,
                     # lengthscales=lengthscale,
                     # outputscale=outputscale,
                     # noise=noise,
                     # kern=kern,
                     train_data=self.train_data,
                     test_data=self.test_data,
                     )

        if training_results:
            np.savez(os.path.join(self.output_dir, 'data'),
                     data_inputs=training_results['train_inputs'],
                     data_targets=training_results['train_targets'])

        # close environments
        for experiment in train_experiments:
            experiment.env.close()
        for experiment in test_experiments:
            experiment.env.close()
        # delete c_generated_code folder and acados_ocp_solver.json files
        os.system(f'rm -rf {self.output_dir}/*c_generated_code*')
        os.system(f'rm -rf {self.output_dir}/*acados_ocp_solver*')

        self.train_runs = train_runs
        self.test_runs = test_runs

        return train_runs, test_runs

    def load(self, model_path):
        '''Load the model from a file.

        Args:
            model_path (str): Path to the model to load.
        '''
        data = np.load(f'{model_path}/data.npz')
        gp_model_path_T = f'{model_path}/best_model_T.pth'
        gp_model_path_R = f'{model_path}/best_model_R.pth'
        gp_model_path_P = f'{model_path}/best_model_P.pth'
        gp_model_path_Y = f'{model_path}/best_model_Y.pth'

        gp_model_path = [gp_model_path_T,
                         gp_model_path_R,
                         gp_model_path_P,
                         gp_model_path_Y]
        self.train_gp(input_data=data['data_inputs'],
                      target_data=data['data_targets'],
                      gp_model=gp_model_path)

    @timing
    def train_gp(self,
                 input_data, target_data,
                 gp_model=None,
                 overwrite_saved_data: bool = None,
                 train_hardware_data: bool = False,
                 ):
        '''Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            gp_model (str): if not None, this is the path to pretrained models to use instead of training new ones.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.
            train_hardware_data (bool): True to train on hardware data. If true, will load the data and perform training.
        Returns:
            training_results (dict): Dictionary of the training results.
        '''
        if gp_model is None and not train_hardware_data:
            gp_model = self.gp_model_path
        if overwrite_saved_data is None:
            overwrite_saved_data = self.overwrite_saved_data
        self.reset()
        train_inputs = input_data
        train_targets = target_data
        if (self.data_inputs is None and self.data_targets is None) or overwrite_saved_data:
            self.data_inputs = train_inputs
            self.data_targets = train_targets
        else:
            self.data_inputs = np.vstack((self.data_inputs, train_inputs))
            self.data_targets = np.vstack((self.data_targets, train_targets))

        total_input_data = self.data_inputs.shape[0]
        # If validation set is desired.
        if self.test_data_ratio > 0 and self.test_data_ratio is not None:
            train_idx, test_idx = train_test_split(
                list(range(total_input_data)),
                test_size=self.test_data_ratio,
                random_state=self.seed
            )

        else:
            # Otherwise, just copy the training data into the test data.
            train_idx = list(range(total_input_data))
            test_idx = list(range(total_input_data))

        train_inputs = self.data_inputs[train_idx, :]
        train_targets = self.data_targets[train_idx, :]
        self.train_data = {'train_inputs': train_inputs, 'train_targets': train_targets}
        test_inputs = self.data_inputs[test_idx, :]
        test_targets = self.data_targets[test_idx, :]
        self.test_data = {'test_inputs': test_inputs, 'test_targets': test_targets}

        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()

        # seperate the data for T R and P
        T_data_idx = [0, 1]
        train_input_T = train_inputs_tensor[:, T_data_idx].reshape(-1, 2)
        train_target_T = train_targets_tensor[:, T_data_idx[0]].reshape(-1)
        test_inputs_T = test_inputs_tensor[:, T_data_idx].reshape(-1, 2)
        test_targets_T = test_targets_tensor[:, T_data_idx[0]].reshape(-1)

        R_data_idx = [2, 3, 4]
        train_input_R = train_inputs_tensor[:, R_data_idx].reshape(-1, 3)
        test_inputs_R = test_inputs_tensor[:, R_data_idx].reshape(-1, 3)
        train_target_R = train_targets_tensor[:, 1].reshape(-1)
        test_targets_R = test_targets_tensor[:, 1].reshape(-1)

        P_data_idx = [5, 6, 7]
        train_input_P = train_inputs_tensor[:, P_data_idx].reshape(-1, 3)
        test_inputs_P = test_inputs_tensor[:, P_data_idx].reshape(-1, 3)
        train_target_P = train_targets_tensor[:, 2].reshape(-1)
        test_targets_P = test_targets_tensor[:, 2].reshape(-1)

        Y_data_idx = [8, 9, 10]
        train_input_Y = train_inputs_tensor[:, Y_data_idx].reshape(-1, 3)
        test_inputs_Y = test_inputs_tensor[:, Y_data_idx].reshape(-1, 3)
        train_target_Y = train_targets_tensor[:, 3].reshape(-1)
        test_targets_Y = test_targets_tensor[:, 3].reshape(-1)

        # Define likelihood.
        likelihood_T = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        likelihood_R = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        likelihood_P = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        likelihood_Y = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()

        GP_T = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_T,
            kernel=self.kernel,
        )
        GP_R = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_R,
            kernel=self.kernel,
        )

        GP_P = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_P,
            kernel=self.kernel,
        )

        GP_Y = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_Y,
            kernel=self.kernel,
        )

        if gp_model:
            print(colored(f'Loaded pretrained model from {self.gp_model_path}', 'green'))
            GP_T.init_with_hyperparam(train_input_T, train_target_T, gp_model[0])
            GP_R.init_with_hyperparam(train_input_R, train_target_R, gp_model[1])
            GP_P.init_with_hyperparam(train_input_P, train_target_P, gp_model[2])
            GP_Y.init_with_hyperparam(train_input_Y, train_target_Y, gp_model[3])
        else:
            GP_T.train(train_input_T, train_target_T, test_inputs_T, test_targets_T,
                       n_train=self.optimization_iterations[0], learning_rate=self.learning_rate[0],
                       gpu=self.use_gpu, fname=os.path.join(self.output_dir, 'best_model_T.pth'),
                       init_noise_var=self.thrust_noise_var if hasattr(self, 'thrust_noise_var') else None)
            GP_R.train(train_input_R, train_target_R, test_inputs_R, test_targets_R,
                       n_train=self.optimization_iterations[1], learning_rate=self.learning_rate[1],
                       gpu=self.use_gpu, fname=os.path.join(self.output_dir, 'best_model_R.pth'),
                       init_noise_var=self.roll_noise_var if hasattr(self, 'roll_noise_var') else None)
            GP_P.train(train_input_P, train_target_P, test_inputs_P, test_targets_P,
                       n_train=self.optimization_iterations[2], learning_rate=self.learning_rate[2],
                       gpu=self.use_gpu, fname=os.path.join(self.output_dir, 'best_model_P.pth'),
                       init_noise_var=self.pitch_noise_var if hasattr(self, 'pitch_noise_var') else None)
            GP_Y.train(train_input_Y, train_target_Y, test_inputs_Y, test_targets_Y,
                       n_train=self.optimization_iterations[3], learning_rate=self.learning_rate[3],
                       gpu=self.use_gpu, fname=os.path.join(self.output_dir, 'best_model_Y.pth'),
                       init_noise_var=self.yaw_noise_var if hasattr(self, 'yaw_noise_var') else None)

        self.new_GP_model = True
        self.gaussian_process = [GP_T, GP_R, GP_P, GP_Y]
        self.reset()
        # if self.train_data['train_targets'].shape[0] <= self.n_ind_points:
        #    n_ind_points = self.train_data['train_targets'].shape[0]
        # else:
        #    n_ind_points = self.n_ind_points
        # self.set_gp_dynamics_func(n_ind_points)
        # self.setup_gp_optimizer(n_ind_points)
        # Collect training results.
        training_results = {}
        training_results['train_targets'] = train_targets
        training_results['train_inputs'] = train_inputs
        return training_results

    def setup_acados_model(self, n_ind_points) -> AcadosModel:

        # setup GP related
        self.inverse_cdf = scipy.stats.norm.ppf(1 - (1 / self.model.nx - (self.prob + 1) / (2 * self.model.nx)))
        self.create_sparse_GP_machinery(n_ind_points)

        # setup acados model
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        acados_model.name = self.env.NAME + '_' + current_time

        z = cs.vertcat(acados_model.x, acados_model.u)  # GP prediction point

        '''
        z_ind should be of shape (n_ind_points, z.shape[0]) or (n_ind_points, len(self.input_mask))
        mean_post_factor should be of shape (len(self.target_mask), n_ind_points)
        Here we create the corresponding parameters since acados supports only 1D parameters
        '''
        # define the dynamics
        T_pred_point = z[[self.tau_idx, self.T_cmd_idx + self.model.nx]]
        R_pred_point = z[[self.phi_idx, self.phi_dot_idx, self.phi_cmd_idx + self.model.nx]]
        P_pred_point = z[[self.theta_idx, self.theta_dot_idx, self.theta_cmd_idx + self.model.nx]]
        Y_pred_point = z[[self.psi_idx, self.psi_dot_idx, self.psi_cmd_idx + self.model.nx]]
        if self.sparse_gp:
            # sparse GP inducing points
            z_ind = cs.MX.sym('z_ind', n_ind_points, 11)
            mean_post_factor = cs.MX.sym('mean_post_factor', 4, n_ind_points)
            acados_model.p = cs.vertcat(cs.reshape(z_ind, -1, 1), cs.reshape(mean_post_factor, -1, 1))
            T_pred = cs.sum2(self.K_z_zind_func_T(z1=T_pred_point, z2=z_ind)['K'] * mean_post_factor[0, :])
            R_pred = cs.sum2(self.K_z_zind_func_R(z1=R_pred_point, z2=z_ind)['K'] * mean_post_factor[1, :])
            P_pred = cs.sum2(self.K_z_zind_func_P(z1=P_pred_point, z2=z_ind)['K'] * mean_post_factor[2, :])
            Y_pred = cs.sum2(self.K_z_zind_func_Y(z1=Y_pred_point, z2=z_ind)['K'] * mean_post_factor[3, :])
            # self.sparse_gp_func = cs.Function('sparse_func',
            #                                   [acados_model.x, acados_model.u, z_ind, mean_post_factor], [f_disc])
        else:
            GP_T, GP_R, GP_P, GP_Y = self.gaussian_process
            T_pred = GP_T.casadi_predict(z=T_pred_point)['mean']
            P_pred = GP_P.casadi_predict(z=P_pred_point)['mean']
            R_pred = GP_R.casadi_predict(z=R_pred_point)['mean']
            Y_pred = GP_Y.casadi_predict(z=Y_pred_point)['mean']
        f_cont = self.prior_dynamics_func_c(x=acados_model.x, u=acados_model.u)['f']\
            + cs.vertcat(0, 0,
                         0, 0,
                         0, 0,
                         0, 0, 0,
                         R_pred, P_pred, Y_pred, T_pred)
        self.f_cont_func = cs.Function('f_cont_func', [acados_model.x, acados_model.u, acados_model.p], [f_cont])
        acados_model.f_expl_expr = f_cont

        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model
        # T_cmd = cs.MX.sym('T_cmd')
        # theta_cmd = cs.MX.sym('theta_cmd')
        # theta = cs.MX.sym('theta')
        # theta_dot = cs.MX.sym('theta_dot')
        # T_true_func = self.env.T_mapping_func
        # T_prior_func = self.prior_ctrl.env.T_mapping_func
        # T_res = T_true_func(T_cmd) - T_prior_func(T_cmd)
        # self.T_res_func = cs.Function('T_res_func', [T_cmd], [T_res])
        # R_true_func = self.env.R_mapping_func
        # R_prior_func = self.prior_ctrl.env.R_mapping_func
        # R_res = R_true_func(theta, theta_dot, theta_cmd) - R_prior_func(theta, theta_dot, theta_cmd)
        # self.R_res_func = cs.Function('R_res_func', [theta, theta_dot, theta_cmd], [R_res])
        # P_true_func = self.env.P_mapping_func
        # P_prior_func = self.prior_ctrl.env.P_mapping_func
        # P_res = P_true_func(theta, theta_dot, theta_cmd) - P_prior_func(theta, theta_dot, theta_cmd)
        # self.P_res_func = cs.Function('P_res_func', [theta, theta_dot, theta_cmd], [P_res])

    def setup_acados_optimizer(self, n_ind_points):
        print('=================Setting up GPMPC acados optimizer=================')
        # before_optimizer_setup = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

        # set dimensions
        ocp.dims.N = self.T  # prediction horizon

        # set cost
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        # cost weight matrices
        ocp.cost.W = scipy.linalg.block_diag(self.Q / self.dt, self.R / self.dt)
        ocp.cost.W_e = self.P if hasattr(self, 'P') else self.Q
        # ocp.cost.W_e = self.Q

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Constraints

        # for state_constraint in self.constraints.state_constraints:
        #     if isinstance(state_constraint, BoundedConstraint):
        #         ocp.constraints.lbx = state_constraint.lower_bounds
        #         ocp.constraints.ubx = state_constraint.upper_bounds
        #         ocp.constraints.idxbx = np.arange(nx)
        #         ocp.constraints.lbx_e = state_constraint.lower_bounds
        #         ocp.constraints.ubx_e = state_constraint.upper_bounds
        #         ocp.constraints.idxbx_e = np.arange(nx)
        #     else:
        #         raise ValueError('Constraint type not supported. Support only for BoundedConstraint and descendants. Check constraints.py.')
        # for input_constraint in self.constraints.input_constraints:
        #     if isinstance(input_constraint, BoundedConstraint):
        #         ocp.constraints.lbu = input_constraint.lower_bounds
        #         ocp.constraints.ubu = input_constraint.upper_bounds
        #         ocp.constraints.idxbu = np.arange(nu)
        #     else:
        #         raise ValueError('Constraint type not supported. Support only for BoundedConstraint and descendants. Check constraints.py.')

        # general constraint expressions
        state_constraint_expr_list = []
        input_constraint_expr_list = []
        state_tighten_list = []
        input_tighten_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(ocp.model.x))
            # chance state constraint tightening
            state_tighten_list.append(cs.MX.sym(f'state_tighten_{sc_i}', state_constraint(ocp.model.x).shape[0], 1))
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(ocp.model.u))
            # chance input constraint tightening
            input_tighten_list.append(cs.MX.sym(f'input_tighten_{ic_i}', input_constraint(ocp.model.u).shape[0], 1))

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list)  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(ocp, h0_expr, h_expr, he_expr, state_tighten_list, input_tighten_list)
        # pass the tightening variables to the ocp object as parameters
        tighten_param = cs.vertcat(*state_tighten_list, *input_tighten_list)
        if self.sparse_gp:
            ocp.model.p = cs.vertcat(ocp.model.p, tighten_param)
        else:
            ocp.model.p = tighten_param
        ocp.parameter_values = np.zeros((ocp.model.p.shape[0], ))  # dummy values
        # slack costs for nonlinear constraints
        if self.gp_soft_constraints:
            # slack variables for all constraints
            ocp.constraints.Jsh_0 = np.eye(h0_expr.shape[0])
            ocp.constraints.Jsh = np.eye(h_expr.shape[0])
            ocp.constraints.Jsh_e = np.eye(he_expr.shape[0])
            # slack penalty (TODO: using the value specified in the config)
            L2_pen = self.gp_soft_constraints_coeff
            L1_pen = self.gp_soft_constraints_coeff
            ocp.cost.zl_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.zu_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zl_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.zl = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.zu = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.Zu_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.zl_e = L1_pen * np.ones(he_expr.shape[0])
            ocp.cost.zu_e = L1_pen * np.ones(he_expr.shape[0])

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.integrator_type = 'ERK'

        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 25 if not self.use_RTI else 1
        # ocp.solver_options.qp_solver_iter_max = 10
        # ocp.solver_options.qp_tol = 1e-4
        # ocp.solver_options.tol = 1e-4
        # ocp.solver_options.as_rti_level = 0 if not self.use_RTI else 4
        # ocp.solver_options.as_rti_iter = 1 if not self.use_RTI else 1

        # ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH' if not self.use_RTI else 'MERIT_BACKTRACKING'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        # prediction horizon
        ocp.solver_options.tf = self.T * self.dt

        # c code generation
        # NOTE: when using GP-MPC, a separated directory is needed;
        # otherwise, Acados solver can read the wrong c code
        # get current time in yy-mm-dd-hh-mm-ss format
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        ocp.code_export_directory = self.output_dir + f'/gpmpc_c_generated_code_{current_time}'
        # ocp.code_export_directory = self.output_dir + '/gpmpc_c_generated_code'

        self.ocp = ocp
        self.opti_dict = {'n_ind_points': n_ind_points}
        # compute sparse GP values
        # the actual values will be set in select_action_with_gp
        self.mean_post_factor_val_all, self.z_ind_val_all = self.precompute_mean_post_factor_all_data()
        if self.sparse_gp:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.mean_post_factor_val = mean_post_factor_val
            self.z_ind_val = z_ind_val
        else:
            mean_post_factor_val, z_ind_val = self.precompute_mean_post_factor_all_data()
            self.mean_post_factor_val = mean_post_factor_val
            self.z_ind_val = z_ind_val

    def processing_acados_constraints_expression(self, ocp: AcadosOcp, h0_expr, h_expr, he_expr,
                                                 state_tighten_list, input_tighten_list) -> AcadosOcp:
        '''Preprocess the constraints to be compatible with acados.
            Args:
                h0_expr (casadi expression): initial state constraints
                h_expr (casadi expression): state and input constraints
                he_expr (casadi expression): terminal state constraints
                state_tighten_list (list): list of casadi SX variables for state constraint tightening
                input_tighten_list (list): list of casadi SX variables for input constraint tightening
            Returns:
                ocp (AcadosOcp): acados ocp object with constraints set

        Note:
        all constraints in safe-control-gym are defined as g(x, u) <= constraint_tol
        However, acados requires the constraints to be defined as lb <= g(x, u) <= ub
        Thus, a large negative number (-1e8) is used as the lower bound.
        See: https://github.com/acados/acados/issues/650

        An alternative way to set the constraints is to use bounded constraints of acados:
        # bounded input constraints
        idxbu = np.where(np.sum(self.env.constraints.input_constraints[0].constraint_filter, axis=0) != 0)[0]
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds
        ocp.constraints.idxbu = idxbu # active constraints dimension
        '''
        # NOTE: only the upper bound is tightened due to constraint are defined in the
        # form of g(x, u) <= constraint_tol in safe-control-gym

        # lambda functions to set the upper and lower bounds of the chance constraints
        def constraint_ub_chance(constraint):
            return -self.constraint_tol * np.ones(constraint.shape)

        def constraint_lb_chance(constraint):
            return -1e8 * np.ones(constraint.shape)
        state_tighten_var = cs.vertcat(*state_tighten_list)
        input_tighten_var = cs.vertcat(*input_tighten_list)

        ub = {'h': constraint_ub_chance(h_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
              'h0': constraint_ub_chance(h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
              'he': constraint_ub_chance(he_expr - state_tighten_var)}
        lb = {'h': constraint_lb_chance(h_expr),
              'h0': constraint_lb_chance(h0_expr),
              'he': constraint_lb_chance(he_expr)}

        # make sure all the ub and lb are 1D casaadi SX variables
        # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
        for key in ub.keys():
            ub[key] = ub[key].flatten() if ub[key].ndim != 1 else ub[key]
            lb[key] = lb[key].flatten() if lb[key].ndim != 1 else lb[key]
        # check ub and lb dimensions
        for key in ub.keys():
            assert ub[key].ndim == 1, f'ub[{key}] is not 1D numpy array'
            assert lb[key].ndim == 1, f'lb[{key}] is not 1D numpy array'
        assert ub['h'].shape == lb['h'].shape, 'h_ub and h_lb have different shapes'

        # pass the constraints to the ocp object
        ocp.model.con_h_expr_0 = h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr = h_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr_e = he_expr - state_tighten_var
        ocp.dims.nh_0, ocp.dims.nh, ocp.dims.nh_e = \
            h0_expr.shape[0], h_expr.shape[0], he_expr.shape[0]
        # assign constraints upper and lower bounds
        ocp.constraints.uh_0 = ub['h0']
        ocp.constraints.lh_0 = lb['h0']
        ocp.constraints.uh = ub['h']
        ocp.constraints.lh = lb['h']
        ocp.constraints.uh_e = ub['he']
        ocp.constraints.lh_e = lb['he']

        return ocp

    def select_action(self, obs, info=None):
        time_before = time.time()
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            action = self.select_action_with_gp(obs)
        time_after = time.time()
        self.results_dict['runtime'].append(time_after - time_before)
        self.last_obs = obs
        self.last_action = action

        return action

    def select_action_with_gp(self, obs):
        time_before = time.time()
        nx, nu = self.model.nx, self.model.nu
        # TODO: replace this with something safer
        n_ind_points = self.opti_dict['n_ind_points']

        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, 'lbx', obs)
        self.acados_ocp_solver.set(0, 'ubx', obs)

        # compute the sparse GP values
        if self.recalc_inducing_points_at_every_step:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.results_dict['inducing_points'].append(z_ind_val)
        else:
            # use the precomputed values
            mean_post_factor_val = self.mean_post_factor_val
            z_ind_val = self.z_ind_val
            self.results_dict['inducing_points'] = [z_ind_val]

        # Set the probabilistic state and input constraint set limits.
        # Tightening at the first step is possible if self.compute_initial_guess is used
        state_constraint_set_prev, input_constraint_set_prev = self.precompute_probabilistic_limits()

        # set acados parameters
        if self.sparse_gp:
            # sparse GP parameters
            assert z_ind_val.shape == (n_ind_points, 11)
            assert mean_post_factor_val.shape == (4, n_ind_points)
            # casadi use column major order, while np uses row major order by default
            # Thus, Fortran order (column major) is used to reshape the arrays
            z_ind_val = z_ind_val.reshape(-1, 1, order='F')
            mean_post_factor_val = mean_post_factor_val.reshape(-1, 1, order='F')
            dyn_value = np.concatenate((z_ind_val, mean_post_factor_val)).reshape(-1)
            # tighten constraints
            for idx in range(self.T):
                # tighten initial and path constraints
                state_constraint_set = state_constraint_set_prev[0][:, idx]
                input_constraint_set = input_constraint_set_prev[0][:, idx]
                tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
                # set the parameter values
                parameter_values = np.concatenate((dyn_value, tighten_value))
                # parameter_values = dyn_value
                # check the shapes
                assert self.ocp.model.p.shape[0] == parameter_values.shape[0], \
                    f'parameter_values.shape: {parameter_values.shape}; model.p.shape: {self.ocp.model.p.shape}'
                self.acados_ocp_solver.set(idx, 'p', parameter_values)
            # tighten terminal state constraints
            tighten_value = np.concatenate((state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,))))
            # set the parameter values
            parameter_values = np.concatenate((dyn_value, tighten_value))
            # parameter_values = dyn_value
            self.acados_ocp_solver.set(self.T, 'p', parameter_values)
        else:
            pass

        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1
        y_ref = np.concatenate((goal_states[:, :-1], np.repeat(self.U_EQ.reshape(-1, 1), self.T, axis=1)), axis=0)
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, 'yref', y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, 'yref', y_ref_e)

        # solve the optimization problem
        if self.use_RTI:
            # preparation phase
            self.acados_ocp_solver.options_set('rti_phase', 1)
            status = self.acados_ocp_solver.solve()

            # feedback phase
            self.acados_ocp_solver.options_set('rti_phase', 2)
            status = self.acados_ocp_solver.solve()
        else:
            status = self.acados_ocp_solver.solve()
        if status not in [0, 2]:
            self.acados_ocp_solver.print_statistics()
            print(colored(f'acados returned status {status}. ', 'red'))

        action = self.acados_ocp_solver.get(0, 'u')
        # get the open-loop solution
        if self.x_prev is None and self.u_prev is None:
            self.x_prev = np.zeros((nx, self.T + 1))
            self.u_prev = np.zeros((nu, self.T))
        if self.u_prev is not None and nu == 1:
            self.u_prev = self.u_prev.reshape((1, -1))

        for i in range(self.T + 1):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
        for i in range(self.T):
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
        if nu == 1:
            self.u_prev = self.u_prev.flatten()
        self.x_guess = self.x_prev
        self.u_guess = self.u_prev

        time_after = time.time()
        # print(f'gpmpc acados sol time: {time_after - time_before:.3f}; sol status {status}; nlp iter {self.acados_ocp_solver.get_stats('sqp_iter')}; qp iter {self.acados_ocp_solver.get_stats('qp_iter')}')
        if time_after - time_before > 1 / 60:
            print(colored(f'========= Warning: GPMPC ACADOS took {time_after - time_before:.3f} seconds =========', 'yellow'))
        self.results_dict['inference_time'].append(self.acados_ocp_solver.get_stats('time_tot'))

        if hasattr(self, 'K'):
            action += self.K @ (self.x_prev[:, 0] - obs)

        return action

    def precompute_mean_post_factor_all_data(self):
        '''If the number of data points is less than the number of inducing points, use all the data
        as kernel points.
        '''
        dim_gp_outputs = len(self.gaussian_process)
        n_training_samples = self.train_data['train_targets'].shape[0]
        inputs = self.train_data['train_inputs']
        targets = self.train_data['train_targets']
        mean_post_factor = np.zeros((dim_gp_outputs, n_training_samples))
        for i in range(dim_gp_outputs):
            K_z_z = self.gaussian_process[i].model.K_plus_noise_inv
            mean_post_factor[i] = K_z_z.detach().numpy() @ targets[:, i]

        return mean_post_factor, inputs

    def precompute_sparse_gp_values(self, n_ind_points):
        '''Uses the last MPC solution to precomupte values associated with the FITC GP approximation.

        Args:
            n_ind_points (int): Number of inducing points.
        '''
        n_data_points = self.train_data['train_targets'].shape[0]
        dim_gp_outputs = len(self.gaussian_process)
        inputs = self.train_data['train_inputs']
        targets = self.train_data['train_targets']

        # Get the inducing points.
        if False and self.x_prev is not None and self.u_prev is not None:
            # Use the previous MPC solution as in Hewing 2019.
            z_prev = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))
            z_prev = z_prev[:, self.input_mask]
            inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points - self.T, replace=False)
            # z_ind = self.data_inputs[inds][:, self.input_mask]
            z_ind = np.vstack((z_prev, inputs[inds][:, self.input_mask]))
        else:
            # If there is no previous solution. Choose T random training set points.
            if self.inducing_point_selection_method == 'kmeans':
                centroids = kmeans_centriods(n_ind_points, inputs, rand_state=self.seed)
                contiguous_masked_inputs = np.ascontiguousarray(inputs)  # required for version sklearn later than 1.0.2
                inds, _ = pairwise_distances_argmin_min(centroids, contiguous_masked_inputs)
                z_ind = inputs[inds]
            elif self.inducing_point_selection_method == 'random':
                inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points, replace=False)
                z_ind = inputs[inds]
            else:
                raise ValueError('[Error]: gp_mpc.precompute_sparse_gp_values: Only \'kmeans\' or \'random\' allowed.')

        use_pinv = True

        # Define GP models and their corresponding data indices
        gp_models = self.gaussian_process
        data_indices = {
            0: [0],        # T: tau index
            1: [1, 2, 3],  # R: phi, phi_dot, phi_cmd indices
            2: [4, 5, 6],  # P: theta, theta_dot, theta_cmd indices
            3: [7, 8, 9]   # Y: psi, psi_dot, psi_cmd indices
        }

        # Initialize tensors
        K_zind_zind = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points)).double()
        K_zind_zind_inv = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points)).double()
        K_x_zind = torch.zeros((dim_gp_outputs, n_data_points, n_ind_points)).double()
        K_plus_noise = torch.zeros((dim_gp_outputs, n_data_points, n_data_points)).double()
        mean_post_factor = torch.zeros((dim_gp_outputs, n_ind_points)).double()
        Sigma_inv = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points)).double()

        # Process each GP model
        for i, gp_model in enumerate(gp_models):
            idx = data_indices[i]

            # Compute covariance matrices
            K_zind_zind_i = gp_model.model.covar_module(torch.from_numpy(z_ind[:, idx]).double())
            K_zind_zind[i] = K_zind_zind_i.evaluate().detach()

            # Compute inverse
            if use_pinv:
                K_zind_zind_inv[i] = torch.pinverse(K_zind_zind_i.evaluate().detach())
            else:
                K_zind_zind_inv[i] = K_zind_zind_i.inv_matmul(torch.eye(n_ind_points).double()).detach()

            # Cross-covariance
            K_x_zind[i] = gp_model.model.covar_module(
                torch.from_numpy(inputs[:, idx]).double(),
                torch.from_numpy(z_ind[:, idx]).double()
            ).evaluate().detach()

            # Noise matrix
            K_plus_noise[i] = gp_model.model.K_plus_noise.detach()

            # Compute Q_X_X
            if use_pinv:
                Q_X_X_i = K_x_zind[i] @ K_zind_zind_inv[i] @ K_x_zind[i].T
            else:
                Q_X_X_i = K_x_zind[i] @ torch.linalg.solve(K_zind_zind[i], K_x_zind[i].T)

            # Compute Gamma and Gamma_inv
            Gamma_i = torch.diagonal(K_plus_noise[i] - Q_X_X_i)
            Gamma_inv_i = torch.diag_embed(1 / Gamma_i)

            # Compute Sigma_inv
            Sigma_inv[i] = K_zind_zind[i] + K_x_zind[i].T @ Gamma_inv_i @ K_x_zind[i]

            # Compute mean posterior factor
            target_tensor = torch.from_numpy(targets[:, i]).double()
            if use_pinv:
                Sigma_i = torch.pinverse(Sigma_inv[i])
                mean_post_factor[i] = Sigma_i @ K_x_zind[i].T @ Gamma_inv_i @ target_tensor
            else:
                mean_post_factor[i] = torch.linalg.solve(
                    Sigma_inv[i],
                    K_x_zind[i].T @ Gamma_inv_i @ target_tensor
                )
        return mean_post_factor.detach().numpy(), Sigma_inv.detach().numpy(), K_zind_zind_inv.detach().numpy(), z_ind

    def create_sparse_GP_machinery(self, n_ind_points):
        '''This setups the gaussian process approximations for FITC formulation.'''
        T_data_idx = [0, 1]
        R_data_idx = [2, 3, 4]
        P_data_idx = [5, 6, 7]
        Y_data_idx = [8, 9, 10]

        GP_T = self.gaussian_process[0]
        GP_R = self.gaussian_process[1]
        GP_P = self.gaussian_process[2]
        GP_Y = self.gaussian_process[3]

        lengthscales_T = GP_T.model.covar_module.base_kernel.lengthscale.detach().numpy()
        lengthscales_R = GP_R.model.covar_module.base_kernel.lengthscale.detach().numpy()
        lengthscales_P = GP_P.model.covar_module.base_kernel.lengthscale.detach().numpy()
        lengthscales_Y = GP_Y.model.covar_module.base_kernel.lengthscale.detach().numpy()
        signal_var_T = GP_T.model.covar_module.outputscale.detach().numpy()
        signal_var_R = GP_R.model.covar_module.outputscale.detach().numpy()
        signal_var_P = GP_P.model.covar_module.outputscale.detach().numpy()
        signal_var_Y = GP_Y.model.covar_module.outputscale.detach().numpy()
        noise_var_T = GP_T.likelihood.noise.detach().numpy()
        noise_var_R = GP_R.likelihood.noise.detach().numpy()
        noise_var_P = GP_P.likelihood.noise.detach().numpy()
        noise_var_Y = GP_Y.likelihood.noise.detach().numpy()
        gp_K_plus_noise_T = GP_T.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_R = GP_R.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_P = GP_P.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_Y = GP_Y.model.K_plus_noise.detach().numpy()

        # Stacking
        lengthscales = np.vstack((lengthscales_T, lengthscales_R, lengthscales_P, lengthscales_Y))
        signal_var = np.array([signal_var_T, signal_var_R, signal_var_P, signal_var_Y])
        noise_var = np.array([noise_var_T, noise_var_R, noise_var_P, noise_var_Y])
        gp_K_plus_noise = np.zeros((len(self.gaussian_process),
                                    gp_K_plus_noise_T.shape[0],
                                    gp_K_plus_noise_T.shape[1]))
        gp_K_plus_noise[0] = gp_K_plus_noise_T
        gp_K_plus_noise[1] = gp_K_plus_noise_R
        gp_K_plus_noise[2] = gp_K_plus_noise_P
        gp_K_plus_noise[3] = gp_K_plus_noise_Y

        self.length_scales = lengthscales.squeeze()
        self.signal_var = signal_var.squeeze()
        self.noise_var = noise_var.squeeze()
        self.gp_K_plus_noise = gp_K_plus_noise
        Nx = self.train_data['train_inputs'].shape[1]

        # Create CasADI function for computing the kernel K_z_zind with parameters for z, z_ind, length scales and signal variance.
        # We need the CasADI version of this so that it can by symbolically differentiated in in the MPC optimization.
        z1_T = cs.SX.sym('z1', 2)
        z2_T = cs.SX.sym('z2', 2)
        ell_s_T = cs.SX.sym('ell', 1)
        sf2_s_T = cs.SX.sym('sf2')
        z1_R = cs.SX.sym('z1', 3)
        z2_R = cs.SX.sym('z2', 3)
        ell_s_R = cs.SX.sym('ell', 1)
        sf2_s_R = cs.SX.sym('sf2')
        z1_P = cs.SX.sym('z1', 3)
        z2_P = cs.SX.sym('z2', 3)
        ell_s_P = cs.SX.sym('ell', 1)
        sf2_s_P = cs.SX.sym('sf2')
        z1_Y = cs.SX.sym('z1', 3)
        z2_Y = cs.SX.sym('z2', 3)
        ell_s_Y = cs.SX.sym('ell', 1)
        sf2_s_Y = cs.SX.sym('sf2')
        z_ind = cs.SX.sym('z_ind', n_ind_points, Nx)
        ks_T = cs.SX.zeros(1, n_ind_points)  # kernel vector
        ks_R = cs.SX.zeros(1, n_ind_points)  # kernel vector
        ks_P = cs.SX.zeros(1, n_ind_points)  # kernel vector
        ks_Y = cs.SX.zeros(1, n_ind_points)  # kernel vector

        # Create CasADI kernel functions based on GP kernel types
        if GP_T.kernel == 'RBF_single':
            covFunc_T = cs.Function('covSE', [z1_T, z2_T, ell_s_T, sf2_s_T],
                                    [covSE_single(z1_T, z2_T, ell_s_T, sf2_s_T)])
        elif GP_T.kernel == 'Matern_single':
            covFunc_T = cs.Function('covMatern', [z1_T, z2_T, ell_s_T, sf2_s_T],
                                    [covMatern52_single(z1_T, z2_T, ell_s_T, sf2_s_T)])
        else:
            # Default to RBF_single for backward compatibility
            covFunc_T = cs.Function('covSE', [z1_T, z2_T, ell_s_T, sf2_s_T],
                                    [covSE_single(z1_T, z2_T, ell_s_T, sf2_s_T)])

        if GP_R.kernel == 'RBF_single':
            covFunc_R = cs.Function('covSE', [z1_R, z2_R, ell_s_R, sf2_s_R],
                                    [covSE_single(z1_R, z2_R, ell_s_R, sf2_s_R)])
        elif GP_R.kernel == 'Matern_single':
            covFunc_R = cs.Function('covMatern', [z1_R, z2_R, ell_s_R, sf2_s_R],
                                    [covMatern52_single(z1_R, z2_R, ell_s_R, sf2_s_R)])
        else:
            # Default to RBF_single for backward compatibility
            covFunc_R = cs.Function('covSE', [z1_R, z2_R, ell_s_R, sf2_s_R],
                                    [covSE_single(z1_R, z2_R, ell_s_R, sf2_s_R)])

        if GP_P.kernel == 'RBF_single':
            covFunc_P = cs.Function('covSE', [z1_P, z2_P, ell_s_P, sf2_s_P],
                                    [covSE_single(z1_P, z2_P, ell_s_P, sf2_s_P)])
        elif GP_P.kernel == 'Matern_single':
            covFunc_P = cs.Function('covMatern', [z1_P, z2_P, ell_s_P, sf2_s_P],
                                    [covMatern52_single(z1_P, z2_P, ell_s_P, sf2_s_P)])
        else:
            # Default to RBF_single for backward compatibility
            covFunc_P = cs.Function('covSE', [z1_P, z2_P, ell_s_P, sf2_s_P],
                                    [covSE_single(z1_P, z2_P, ell_s_P, sf2_s_P)])

        if GP_Y.kernel == 'RBF_single':
            covFunc_Y = cs.Function('covSE', [z1_Y, z2_Y, ell_s_Y, sf2_s_Y],
                                    [covSE_single(z1_Y, z2_Y, ell_s_Y, sf2_s_Y)])
        elif GP_Y.kernel == 'Matern_single':
            covFunc_Y = cs.Function('covMatern', [z1_Y, z2_Y, ell_s_Y, sf2_s_Y],
                                    [covMatern52_single(z1_Y, z2_Y, ell_s_Y, sf2_s_Y)])
        else:
            # Default to RBF_single for backward compatibility
            covFunc_Y = cs.Function('covSE', [z1_Y, z2_Y, ell_s_Y, sf2_s_Y],
                                    [covSE_single(z1_Y, z2_Y, ell_s_Y, sf2_s_Y)])

        for i in range(n_ind_points):
            ks_T[i] = covFunc_T(z1_T, z_ind[i, T_data_idx], ell_s_T, sf2_s_T)
            ks_R[i] = covFunc_R(z1_R, z_ind[i, R_data_idx], ell_s_R, sf2_s_R)
            ks_P[i] = covFunc_P(z1_P, z_ind[i, P_data_idx], ell_s_P, sf2_s_P)
            ks_Y[i] = covFunc_Y(z1_Y, z_ind[i, Y_data_idx], ell_s_Y, sf2_s_Y)
        ks_func_T = cs.Function('K_s', [z1_T, z_ind, ell_s_T, sf2_s_T], [ks_T])
        ks_func_R = cs.Function('K_s', [z1_R, z_ind, ell_s_R, sf2_s_R], [ks_R])
        ks_func_P = cs.Function('K_s', [z1_P, z_ind, ell_s_P, sf2_s_P], [ks_P])
        ks_func_Y = cs.Function('K_s', [z1_Y, z_ind, ell_s_Y, sf2_s_Y], [ks_Y])

        K_z_zind_T = ks_func_T(z1_T, z_ind, self.length_scales[0], self.signal_var[0])
        K_z_zind_R = ks_func_R(z1_R, z_ind, self.length_scales[1], self.signal_var[1])
        K_z_zind_P = ks_func_P(z1_P, z_ind, self.length_scales[2], self.signal_var[2])
        K_z_zind_Y = ks_func_Y(z1_Y, z_ind, self.length_scales[3], self.signal_var[3])
        self.K_z_zind_func_T = cs.Function('K_z_zind', [z1_T, z_ind], [K_z_zind_T], ['z1', 'z2'], ['K'])
        self.K_z_zind_func_R = cs.Function('K_z_zind', [z1_R, z_ind], [K_z_zind_R], ['z1', 'z2'], ['K'])
        self.K_z_zind_func_P = cs.Function('K_z_zind', [z1_P, z_ind], [K_z_zind_P], ['z1', 'z2'], ['K'])
        self.K_z_zind_func_Y = cs.Function('K_z_zind', [z1_Y, z_ind], [K_z_zind_Y], ['z1', 'z2'], ['K'])

    def precompute_probabilistic_limits(self,
                                        print_sets=False
                                        ):
        '''This updates the constraint value limits to account for the uncertainty in the dynamics rollout.

        Args:
            print_sets (bool): True to print out the sets for debugging purposes.
        '''

        # ======== TODO: should the noise propagation be updated? ========
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        state_covariances = np.zeros((self.T + 1, nx, nx))
        input_covariances = np.zeros((self.T, nu, nu))
        # Initilize lists for the tightening of each constraint.
        state_constraint_set = []
        for state_constraint in self.constraints.state_constraints:
            state_constraint_set.append(np.zeros((state_constraint.num_constraints, T + 1)))
        input_constraint_set = []
        for input_constraint in self.constraints.input_constraints:
            input_constraint_set.append(np.zeros((input_constraint.num_constraints, T)))
        if self.x_prev is not None and self.u_prev is not None:
            # cov_x = np.zeros((nx, nx))
            cov_x = np.diag([self.initial_rollout_std**2] * nx)
            if nu == 1:
                z_batch = np.hstack((self.x_prev[:, :-1].T, self.u_prev.reshape(1, -1).T))  # (T, input_dim)
            else:
                z_batch = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))  # (T, input_dim)

            # Compute the covariance of the dynamics at each time step.
            # _, cov_d_tensor_batch = self.gaussian_process.predict(z_batch, return_pred=False)
            GP_T = self.gaussian_process[0]
            GP_R = self.gaussian_process[1]
            GP_P = self.gaussian_process[2]
            GP_Y = self.gaussian_process[3]
            T_pred_point_batch = z_batch[:, [self.tau_idx, self.model.nx + self.T_cmd_idx]]
            R_pred_point_batch = z_batch[:, [self.phi_idx, self.phi_dot_idx,
                                             self.model.nx + self.phi_cmd_idx]]
            P_pred_point_batch = z_batch[:, [self.theta_idx, self.theta_dot_idx,
                                             self.model.nx + self.theta_cmd_idx]]
            Y_pred_point_batch = z_batch[:, [self.psi_idx, self.psi_dot_idx,
                                             self.model.nx + self.psi_cmd_idx]]
            cov_d_batch_T = np.diag(GP_T.predict(T_pred_point_batch, return_pred=False)[1])
            cov_d_batch_R = np.diag(GP_R.predict(R_pred_point_batch, return_pred=False)[1])
            cov_d_batch_P = np.diag(GP_P.predict(P_pred_point_batch, return_pred=False)[1])
            cov_d_batch_Y = np.diag(GP_Y.predict(Y_pred_point_batch, return_pred=False)[1])
            num_batch = z_batch.shape[0]

            cov_d_batch = np.zeros((num_batch, 4, 4))
            cov_d_batch[:, 0, 0] = cov_d_batch_R
            cov_d_batch[:, 1, 1] = cov_d_batch_P
            cov_d_batch[:, 2, 2] = cov_d_batch_Y
            cov_d_batch[:, 3, 3] = cov_d_batch_T
            cov_noise_T = GP_T.likelihood.noise.detach().numpy()
            cov_noise_R = GP_R.likelihood.noise.detach().numpy()
            cov_noise_P = GP_P.likelihood.noise.detach().numpy()
            cov_noise_Y = GP_Y.likelihood.noise.detach().numpy()
            cov_noise_batch = np.zeros((num_batch, 4, 4))
            cov_noise_batch[:, 0, 0] = cov_noise_R
            cov_noise_batch[:, 1, 1] = cov_noise_P
            cov_noise_batch[:, 2, 2] = cov_noise_Y
            cov_noise_batch[:, 3, 3] = cov_noise_T
            # discretize (?)
            cov_noise_batch = cov_noise_batch * self.dt**2
            cov_d_batch = cov_d_batch * self.dt**2

            for i in range(T):
                state_covariances[i] = cov_x
                cov_u = self.lqr_gain @ cov_x @ self.lqr_gain.T
                input_covariances[i] = cov_u
                cov_xu = cov_x @ self.lqr_gain.T
                if self.gp_approx == 'taylor':
                    raise NotImplementedError('Taylor GP approximation is currently not working.')
                elif self.gp_approx == 'mean_eq':
                    cov_d = cov_d_batch[i, :, :]
                    cov_noise = cov_noise_batch[i, :, :]
                    cov_d = cov_d + cov_noise
                else:
                    raise NotImplementedError('gp_approx method is incorrect or not implemented')
                # Loop through input constraints and tighten by the required ammount.
                for ui, input_constraint in enumerate(self.constraints.input_constraints):
                    input_constraint_set[ui][:, i] = -1 * self.inverse_cdf * \
                        np.absolute(input_constraint.A) @ np.sqrt(np.diag(cov_u))
                for si, state_constraint in enumerate(self.constraints.state_constraints):
                    state_constraint_set[si][:, i] = -1 * self.inverse_cdf * \
                        np.absolute(state_constraint.A) @ np.sqrt(np.diag(cov_x))
                if self.gp_approx == 'taylor':
                    raise NotImplementedError('Taylor GP rollout not implemented.')
                elif self.gp_approx == 'mean_eq':
                    # Compute the next step propogated state covariance using mean equivilence.
                    cov_x = self.discrete_dfdx @ cov_x @ self.discrete_dfdx.T + \
                        self.discrete_dfdx @ cov_xu @ self.discrete_dfdu.T + \
                        self.discrete_dfdu @ cov_xu.T @ self.discrete_dfdx.T + \
                        self.discrete_dfdu @ cov_u @ self.discrete_dfdu.T + \
                        self.Bd @ cov_d @ self.Bd.T
                else:
                    raise NotImplementedError('gp_approx method is incorrect or not implemented')
            # Update Final covariance.
            for si, state_constraint in enumerate(self.constraints.state_constraints):
                state_constraint_set[si][:, -1] = -1 * self.inverse_cdf * \
                    np.absolute(state_constraint.A) @ np.sqrt(np.diag(cov_x))
            state_covariances[-1] = cov_x
        if print_sets:
            print('Probabilistic State Constraint values along Horizon:')
            print(state_constraint_set)
            print('Probabilistic Input Constraint values along Horizon:')
            print(input_constraint_set)
        self.results_dict['input_constraint_set'].append(input_constraint_set)
        self.results_dict['state_constraint_set'].append(state_constraint_set)
        self.results_dict['state_horizon_cov'].append(state_covariances)
        self.results_dict['input_horizon_cov'].append(input_covariances)
        return state_constraint_set, input_constraint_set

    @timing
    def reset(self):
        print(colored('Resetting the GPMPC controller.', 'green'))
        '''Reset the controller before running.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        self.setup_prior_dynamics()
        self.prior_ctrl.reset()
        self.setup_prior_dynamics()
        if self.gaussian_process is not None:
            # sparse GP
            if self.sparse_gp and self.train_data['train_targets'].shape[0] <= self.n_ind_points:
                n_ind_points = self.train_data['train_targets'].shape[0]
            elif self.sparse_gp:
                n_ind_points = self.n_ind_points
            else:
                n_ind_points = self.train_data['train_targets'].shape[0]

            # explicitly clear the previously generated c code, ocp and solver
            # otherwise the number of parameters will be incorrect
            # TODO: find a better way to handle this
            if self.new_GP_model:
                self.acados_model = None
                self.ocp = None
                self.acados_ocp_solver = None
                # reinitialize the acados model and solver
                self.setup_acados_model(n_ind_points)
                self.setup_acados_optimizer(n_ind_points)
                # get time in $ymd_HMS format
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.acados_ocp_solver = AcadosOcpSolver(self.ocp,
                                                         self.output_dir + f'/gpmpc_acados_ocp_solver_{current_time}.json')
                self.new_GP_model = False
            else:
                self.acados_ocp_solver.reset()  # Reset the solver to clear previous results.

        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.x_guess = None
        self.u_guess = None

    def plot_gp_TRPY(self, train_inputs, train_targets, title=None, output_dir=None):
        num_data = train_inputs.shape[0]
        t = np.arange(num_data)

        # Convert inputs to torch tensors if needed
        if not isinstance(train_inputs, torch.Tensor):
            train_inputs_tensor = torch.Tensor(train_inputs).double()
        else:
            train_inputs_tensor = train_inputs.double()

        if not isinstance(train_targets, torch.Tensor):
            train_targets_tensor = torch.Tensor(train_targets).double()
        else:
            train_targets_tensor = train_targets.double()

        # Define GP models, their names, and corresponding data indices
        # These should match the indices used during training
        gp_models = self.gaussian_process
        gp_names = ['T', 'R', 'P', 'Y']
        gp_units = ['[$m/s^2$]', '[$rad/s^2$]', '[$rad/s^2$]', '[$rad/s^2$]']
        gp_colors = ['blue', 'red', 'green', 'orange']
        data_indices = {
            0: [0, 1],      # T: indices [0, 1] (2 inputs)
            1: [2, 3, 4],   # R: indices [2, 3, 4] (3 inputs)
            2: [5, 6, 7],   # P: indices [5, 6, 7] (3 inputs)
            3: [8, 9, 10]   # Y: indices [8, 9, 10] (3 inputs)
        }

        # Prepare data for all GP models
        gp_data = {}
        for i, (gp_model, name) in enumerate(zip(gp_models, gp_names)):
            idx = data_indices[i]
            input_data = train_inputs_tensor[:, idx]

            # Get GP predictions
            mean, _, preds = gp_model.predict(input_data)
            lower, upper = preds.confidence_region()

            # Convert to numpy
            mean_np = mean.numpy()
            lower_np = lower.numpy()
            upper_np = upper.numpy()

            # Calculate percentage within 2-sigma
            target_data = train_targets_tensor[:, i].numpy()
            num_within_2std = np.sum((target_data > lower_np) & (target_data < upper_np))
            percentage_within_2std = num_within_2std / num_data * 100

            gp_data[i] = {
                'mean': mean_np,
                'lower': lower_np,
                'upper': upper_np,
                'target': target_data,
                'input': input_data.numpy(),
                'percentage': percentage_within_2std,
                'name': name,
                'unit': gp_units[i],
                'color': gp_colors[i]
            }

        # Create subplots for all GP models
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Plot each GP model
        for i in range(4):
            data = gp_data[i]
            ax = axes[i]

            # Time series plot
            ax.scatter(t, data['target'], label='Target', color='gray', alpha=0.6)
            ax.plot(t, data['mean'], label='GP mean', color=data['color'], linewidth=2)
            ax.fill_between(t, data['lower'], data['upper'], alpha=0.3,
                            color=data['color'], label='2-$\\sigma$')

            ax.set_ylabel(f'{data["name"]} residual {data["unit"]}')
            ax.set_xlabel('data points')
            ax.set_title(f'{data["name"]} residual, {data["percentage"]:.2f}% within 2-$\\sigma$')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt_title = f'GP_validation_TRPY_{title}'
        plt.suptitle(plt_title, fontsize=16)
        fig.tight_layout()
        file_name = f'{plt_title}.png'
        fig.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
        print(f'Plot saved at {os.path.join(output_dir, file_name)}')
        plt.close()

    def plot_open_loop_prediction(self,
                                  x0,
                                  u_seq,
                                  x_true,
                                  title=None,
                                  output_dir=None):
        '''
        Open-loop prediction plots comparing GP dynamics predictions with ground truth.
        Integrates the continuous-time dynamics f_cont_func and compares the predicted
        force_motor state (last dimension) with the ground truth force_motor state.
        '''
        if output_dir is None:
            output_dir = self.output_dir
        if title is None:
            title = 'open_loop_prediction'

        # Convert inputs to numpy arrays if needed
        if not isinstance(x0, np.ndarray):
            x0 = np.array(x0)
        if not isinstance(u_seq, np.ndarray):
            u_seq = np.array(u_seq)
        if not isinstance(x_true, np.ndarray):
            x_true = np.array(x_true)

        # Get trajectory length
        horizon = u_seq.shape[0]
        dt = 1 / 60  # Assuming 60Hz simulation
        time_steps = np.arange(horizon + 1) * dt

        # Initialize arrays for predictions
        x_pred = np.zeros((horizon + 1, x0.shape[0]))
        f_cont_pred = np.zeros((horizon, x0.shape[0]))
        force_motor_pred = np.zeros(horizon + 1)  # Predicted force_motor state (integrated)
        force_motor_true = np.zeros(horizon + 1)  # True force_motor state

        # Set initial condition
        x_pred[0, :] = x0
        force_motor_pred[0] = x0[-1]  # Initial force_motor state
        force_motor_true[0] = x_true[0, -1]  # True initial force_motor state

        # Run open-loop prediction
        for k in range(horizon):
            # Current state and control
            x_k = x_pred[k, :]
            u_k = u_seq[k, :]

            # Predict dynamics using f_cont_func (continuous time)
            # Note: f_cont_func expects [x, u, p] where p is parameters (empty for now)
            f_cont_k = self.f_cont_func(x_k, u_k, [])
            f_cont_pred[k, :] = np.array(f_cont_k).flatten()

            # Integrate dynamics for next state prediction (Euler integration)
            x_pred[k + 1, :] = x_k + dt * np.array(f_cont_k).flatten()

            # Extract integrated force_motor state prediction (last dimension of integrated state)
            force_motor_pred[k + 1] = x_pred[k + 1, -1]

            # Ground truth force_motor state (last dimension of x_true)
            if k + 1 < x_true.shape[0]:
                force_motor_true[k + 1] = x_true[k + 1, -1]
            else:
                force_motor_true[k + 1] = force_motor_true[k]  # Use previous value for last step

        # Create the comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Force motor state prediction vs ground truth
        axes[0, 0].plot(time_steps, force_motor_pred, 'b-', linewidth=2, label='GP Predicted force_motor')
        axes[0, 0].plot(time_steps, force_motor_true, 'r--', linewidth=2, label='Ground Truth force_motor')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('Force Motor [N]')
        axes[0, 0].set_title('Force Motor State: GP Prediction vs Ground Truth')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Force motor prediction error
        force_motor_error = force_motor_pred - force_motor_true
        axes[0, 1].plot(time_steps, force_motor_error, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Force Motor Error [N]')
        axes[0, 1].set_title('Force Motor State Prediction Error')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: State trajectory comparison (position)
        axes[1, 0].plot(time_steps, x_pred[:, 0], 'b-', linewidth=2, label='GP Predicted x')
        axes[1, 0].plot(time_steps, x_true[:len(time_steps), 0], 'r--', linewidth=2, label='Ground Truth x')
        axes[1, 0].plot(time_steps, x_pred[:, 1], 'c-', linewidth=2, label='GP Predicted y')
        axes[1, 0].plot(time_steps, x_true[:len(time_steps), 1], 'm--', linewidth=2, label='Ground Truth y')
        axes[1, 0].plot(time_steps, x_pred[:, 2], 'y-', linewidth=2, label='GP Predicted z')
        axes[1, 0].plot(time_steps, x_true[:len(time_steps), 2], 'k--', linewidth=2, label='Ground Truth z')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Position [m]')
        axes[1, 0].set_title('Position Trajectories')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Overall state prediction error (RMS)
        state_errors = np.sqrt(np.mean((x_pred[:len(x_true), :] - x_true[:len(x_pred), :])**2, axis=1))
        axes[1, 1].plot(time_steps[:len(state_errors)], state_errors, 'k-', linewidth=2)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('RMS State Error')
        axes[1, 1].set_title('Overall State Prediction Error')
        axes[1, 1].grid(True, alpha=0.3)

        # Add overall title and save
        plt.suptitle(f'Open-Loop GP Dynamics Prediction - {title}', fontsize=16)
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, f'open_loop_prediction_{title}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Open-loop prediction plot saved at {plot_path}')

        # Print some statistics
        force_motor_rmse = np.sqrt(np.mean(force_motor_error**2))
        force_motor_mae = np.mean(np.abs(force_motor_error))
        print(f'Force Motor Prediction RMSE: {force_motor_rmse:.6f}')
        print(f'Force Motor Prediction MAE: {force_motor_mae:.6f}')

        plt.close()

        return {
            'force_motor_pred': force_motor_pred,
            'force_motor_true': force_motor_true,
            'force_motor_error': force_motor_error,
            'force_motor_rmse': force_motor_rmse,
            'force_motor_mae': force_motor_mae,
            'x_pred': x_pred,
            'f_cont_pred': f_cont_pred
        }
