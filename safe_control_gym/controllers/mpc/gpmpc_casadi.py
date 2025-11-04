'''Model Predictive Control with a Gaussian Process model.

Based on:
    * L. Hewing, J. Kabzan and M. N. Zeilinger, 'Cautious Model Predictive Control Using Gaussian Process Regression,'
     in IEEE Transactions on Control Systems Technology, vol. 28, no. 6, pp. 2736-2743, Nov. 2020, doi: 10.1109/TCST.2019.2949757.

Implementation details:
    1. The previous time step MPC solution is used to compute the set constraints and GP dynamics rollout.
       Here, the dynamics are rolled out using the Mean Equivelence method, the fastest, but least accurate.
    2. The GP is approximated using the Fully Independent Training Conditional (FITC) outlined in
        * J. Quinonero-Candela, C. E. Rasmussen, and R. Herbrich, “A unifying view of sparse approximate Gaussian process regression,”
          Journal of Machine Learning Research, vol. 6, pp. 1935–1959, 2005.
          https://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
        * E. Snelson and Z. Ghahramani, “Sparse gaussian processes using pseudo-inputs,” in Advances in Neural Information Processing
          Systems, Y. Weiss, B. Scholkopf, and J. C. Platt, Eds., 2006, pp. 1257–1264.
       and the inducing points are the previous MPC solution.
    3. Each dimension of the learned error dynamics is an independent Zero Mean SE Kernel GP.
'''
import time
from copy import deepcopy

import numpy as np

from safe_control_gym.controllers.mpc.gpmpc_base import GPMPC
from safe_control_gym.controllers.mpc.linear_mpc import MPC, LinearMPC
from safe_control_gym.envs.benchmark_env import Task


class GPMPC_CASADI(GPMPC):
    '''MPC with Gaussian Process as dynamics residual.'''

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
            kernel: str = 'Matern',
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            sparse_gp: bool = False,
            n_ind_points: int = 150,
            inducing_point_selection_method: str = 'kmeans',
            recalc_inducing_points_at_every_step: bool = False,
            online_learning: bool = False,
            prior_info: dict = None,
            # inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = 'results/temp',
            use_linear_prior: bool = True,
            plot_trained_gp: bool = False,
            **kwargs
    ):
        '''Initialize GP-MPC.

        Args:
            env_func (gym.Env): functionalized initialization of the environment.
            seed (int): random seed.
            horizon (int): MPC planning horizon.
            Q, R (np.array): cost weight matrix.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            train_iterations (int): the number of training examples to use for each dimension of the GP.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.
            optimization_iterations (list): the number of optimization iterations for each dimension of the GP.
            learning_rate (list): the learning rate for training each dimension of the GP.
            normalize_training_data (bool): Normalize the training data.
            use_gpu (bool): use GPU while training the gp.
            gp_model_path (str): path to a pretrained GP model. If None, will train a new one.
            kernel (str): 'Matern' or 'RBF' kernel.
            output_dir (str): directory to store model and results.
            prob (float): desired probabilistic safety level.
            initial_rollout_std (float): the initial std (across all states) for the mean_eq rollout.
            prior_info (dict): Dictionary specifiy the algorithms prior model parameters.
            prior_param_coeff (float): constant multiplying factor to adjust the prior model intertial properties.
            input_mask (list): list of which input dimensions to use in GP model. If None, all are used.
            target_mask (list): list of which output dimensions to use in the GP model. If None, all are used.
            gp_approx (str): 'mean_eq' used mean equivalence rollout for the GP dynamics. Only one that works currently.
            sparse_gp (bool): True to use sparse GP approximations, otherwise no spare approximation is used.
            n_ind_points (int): Number of inducing points to use got the FTIC gp approximation.
            inducing_point_selection_method (str): kmeans for kmeans clustering, 'random' for random.
            recalc_inducing_points_at_every_step (bool): True to recompute the gp approx at every time step.
            online_learning (bool): if true, GP kernel values will be updated using past trajectory values.
            additional_constraints (list): list of Constraint objects defining additional constraints to be used.
        '''
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
            use_linear_prior=use_linear_prior,
            plot_trained_gp=plot_trained_gp,
            **kwargs)

        # Initialize the method using linear MPC.
        if self.use_linear_prior:
            self.prior_ctrl = LinearMPC(
                self.prior_env_func,
                horizon=horizon,
                q_mpc=q_mpc,
                r_mpc=r_mpc,
                warmstart=warmstart,
                soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
                terminate_run_on_done=terminate_run_on_done,
                prior_info=prior_info,
                # runner args
                # shared/base args
                output_dir=output_dir,
                additional_constraints=additional_constraints,
            )
            self.prior_ctrl.reset()
        else:
            self.prior_ctrl = MPC(
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
                compute_ipopt_initial_guess=True,
                prior_info=prior_info,
            )
            self.prior_ctrl.reset()

        if self.use_linear_prior:
            self.prior_dynamics_func = self.prior_ctrl.linear_dynamics_func
        else:
            self.prior_dynamics_func = self.prior_ctrl.dynamics_func
            self.prior_dynamcis_func_c = self.prior_ctrl.model.fc_func

        self.X_EQ = self.prior_ctrl.X_EQ
        self.U_EQ = self.prior_ctrl.U_EQ

    def select_action_with_gp(self,
                              obs
                              ):
        '''Solves nonlinear MPC problem to get next action.

         Args:
             obs (np.array): current state/observation.

         Returns:
             np.array: input/action to the task/env.
         '''
        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']
        u_var = opti_dict['u_var']
        x_init = opti_dict['x_init']
        x_ref = opti_dict['x_ref']
        state_constraint_set = opti_dict['state_constraint_set']
        input_constraint_set = opti_dict['input_constraint_set']
        mean_post_factor = opti_dict['mean_post_factor']
        z_ind = opti_dict['z_ind']
        n_ind_points = opti_dict['n_ind_points']
        # Assign the initial state.
        opti.set_value(x_init, obs)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        if self.mode == 'tracking':
            self.traj_step += 1
        # Set the probabilistic state and input constraint set limits.
        state_constraint_set_prev, input_constraint_set_prev = self.precompute_probabilistic_limits()

        for si in range(len(self.constraints.state_constraints)):
            opti.set_value(state_constraint_set[si], state_constraint_set_prev[si])
        for ui in range(len(self.constraints.input_constraints)):
            opti.set_value(input_constraint_set[ui], input_constraint_set_prev[ui])
        if self.recalc_inducing_points_at_every_step:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.results_dict['inducing_points'].append(z_ind_val)
        else:
            mean_post_factor_val = self.mean_post_factor_val
            z_ind_val = self.z_ind_val
            self.results_dict['inducing_points'] = [z_ind_val]

        opti.set_value(mean_post_factor, mean_post_factor_val)
        opti.set_value(z_ind, z_ind_val)
        # Initial guess for the optimization problem.
        if self.warmstart and self.x_prev is None and self.u_prev is None:
            if self.gaussian_process is None:
                if self.use_linear_prior:
                    x_guess, u_guess \
                        = self.prior_ctrl.compute_initial_guess(obs, goal_states, self.X_EQ, self.U_EQ)
                else:
                    x_guess, u_guess = self.compute_initial_guess(obs, goal_states)
            else:
                x_guess, u_guess = self.compute_initial_guess(obs, goal_states)
                # set the solver back
                self.setup_gp_optimizer(n_ind_points=n_ind_points,
                                        solver=self.solver)
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)  # Initial guess for optimization problem.
        elif self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            x_guess = deepcopy(self.x_prev)
            u_guess = deepcopy(self.u_prev)
            x_guess[:, :-1] = x_guess[:, 1:]
            u_guess[:-1] = u_guess[1:]
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
            self.u_prev = u_val
            self.x_prev = x_val
        except RuntimeError:
            # sol = opti.solve()
            if self.solver == 'ipopt':
                x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)
            else:
                return_status = opti.return_status()
                print(f'Optimization failed with status: {return_status}')
                if return_status == 'unknown':
                    # self.terminate_loop = True
                    u_val = self.u_prev
                    x_val = self.x_prev
                    if u_val is None:
                        print('[WARN]: MPC Infeasible first step.')
                        u_val = u_guess
                        x_val = x_guess
                elif return_status == 'Maximum_Iterations_Exceeded':
                    self.terminate_loop = True
                    u_val = opti.debug.value(u_var)
                    x_val = opti.debug.value(x_var)
                elif return_status == 'Search_Direction_Becomes_Too_Small':
                    self.terminate_loop = True
                    u_val = opti.debug.value(u_var)
                    x_val = opti.debug.value(x_var)

        u_val = np.atleast_2d(u_val)
        self.x_prev = x_val
        self.u_prev = u_val
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        # self.results_dict['t_wall'].append(opti.stats()['t_wall_total'])
        zi = np.hstack((x_val[:, 0], u_val[:, 0]))
        zi = zi[self.input_mask]
        gp_contribution = np.sum(self.K_z_zind_func(z1=zi, z2=z_ind_val)['K'].toarray() * mean_post_factor_val, axis=1)
        print(f'GP Mean eq Contribution: {gp_contribution}')
        zi = np.hstack((x_val[:, 0], u_val[:, 0]))
        pred, _, _ = self.gaussian_process.predict(zi[None, :])
        print(f'True GP value: {pred.numpy()}')
        # lin_pred = self.prior_dynamics_func(x0=x_val[:, 0] - self.prior_ctrl.X_EQ,
        #                                     p=u_val[:, 0] - self.prior_ctrl.U_EQ)['xf'].toarray() + \
        #     self.prior_ctrl.X_EQ[:, None]
        # self.results_dict['linear_pred'].append(lin_pred)
        self.results_dict['gp_mean_eq_pred'].append(gp_contribution)
        self.results_dict['gp_pred'].append(pred.numpy())
        # Take the first one from solved action sequence.
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        self.prev_action = action,
        # use the ancillary gain
        if hasattr(self, 'K'):
            action += self.K @ (x_val[:, 0] - obs)
        return action

    def select_action(self,
                      obs,
                      info=None,
                      ):
        '''Select the action based on the given observation.

        Args:
            obs (ndarray): Current observed state.
            info (dict): Current info.

        Returns:
            action (ndarray): Desired policy action.
        '''
        # try:
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            # if (self.last_obs is not None and self.last_action is not None and self.online_learning):
            #     print('[ERROR]: Not yet supported.')
            #     exit()
            t1 = time.perf_counter()
            action = self.select_action_with_gp(obs)
            t2 = time.perf_counter()
            self.results_dict['runtime'].append(t2 - t1)
            self.last_obs = obs
            self.last_action = action
        return action

    def reset(self):
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
        if self.gaussian_process is not None:
            # if self.kernel in ['RBF', 'RBF_single']:
            #     self.compute_terminal_cost_and_ancillary_gain()
            if self.sparse_gp and self.train_data['train_targets'].shape[0] <= self.n_ind_points:
                n_ind_points = self.train_data['train_targets'].shape[0]
            elif self.sparse_gp:
                n_ind_points = self.n_ind_points
            else:
                n_ind_points = self.train_data['train_targets'].shape[0]

            self.set_gp_dynamics_func(n_ind_points)
            self.setup_gp_optimizer(n_ind_points)
        self.prior_ctrl.reset()
        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None
