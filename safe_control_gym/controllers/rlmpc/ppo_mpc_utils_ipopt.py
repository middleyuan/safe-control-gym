'''PPO utilities.'''

from collections import defaultdict, deque
from copy import deepcopy

import casadi as cs
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.controllers.mpc.mpc_utils import reset_constraints
from safe_control_gym.controllers.rlmpc.rlmpc_utils import AdamOptimizer, euler_discrete
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.math_and_models.distributions import Normal
from safe_control_gym.math_and_models.neural_networks import MLP


class PPO_MPC_Agent:
    '''A PPO class that encapsulates models, optimizers and update functions.'''

    def __init__(self,
                 env_fun,
                 obs_space,
                 act_space,
                 gamma,
                 model,
                 hidden_dim=64,
                 activation='tanh',
                 actor_config=None,
                 use_clipped_value=False,
                 clip_param=0.2,
                 target_kl=0.02,
                 entropy_coef=0.002,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 opt_epochs=10,
                 mini_batch_size=64,
                 **kwargs
                 ):

        # Parameters.
        self.env = env_fun
        self.obs_space = obs_space
        self.act_space = act_space
        self.use_clipped_value = use_clipped_value
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        self.activation = activation

        # Model.
        self.ac = MLPActorCritic(
            self.env,
            obs_space,
            act_space,
            gamma,
            model,
            hidden_dims=[hidden_dim] * 2,
            activation=self.activation,
            actor_config=actor_config
        )

        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.actor_mpc_opt = AdamOptimizer(actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), critic_lr)

    def to(self, device):
        '''Puts agent to device.'''
        self.ac.to(device)

    def train(self):
        '''Sets training mode.'''
        self.ac.train()

    def eval(self):
        '''Sets evaluation mode.'''
        self.ac.eval()

    def reset(self):
        '''Reset function, especially needed for resetting MPC actor'''
        self.ac.reset()

    def state_dict(self):
        '''Snapshots agent state.'''
        return {
            'ac': self.ac.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        '''Restores agent state.'''
        self.ac.load_state_dict(state_dict['ac'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])

    def compute_policy_loss(self, batch, batch_th):
        '''Returns policy loss(es) given batch of data.'''
        obs, act, logp_old, adv = batch_th['obs'], batch_th['act'], batch_th['logp'], batch_th['adv']
        info = batch['info']
        (action_th, dist, logp, nabla_pi_ref,
         nabla_pi_cost, nabla_pi_model, optimal) = self.ac.actor.forward_train(obs, act, info)

        # Policy.
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv)
        # mask = ratio*adv > clip_adv
        # policy_loss[mask] = 0.0
        policy_loss = torch.where(optimal > 0.9, policy_loss, torch.nan).nanmean()
        # Entropy.
        entropy_loss = torch.where(optimal > 0.9, -dist.entropy(), torch.nan).nanmean()
        # KL/trust region.
        approx_kl = torch.where(optimal > 0.9, (logp_old - logp), torch.nan).nanmean()
        return policy_loss, entropy_loss, approx_kl, action_th, nabla_pi_ref, nabla_pi_cost, nabla_pi_model

    def compute_value_loss(self, batch_th):
        '''Returns value loss(es) given batch of data.'''
        obs, ret, v_old = batch_th['obs'], batch_th['ret'], batch_th['v']
        v_cur = self.ac.critic(obs)
        if self.use_clipped_value:
            v_old_clipped = v_old + (v_cur - v_old).clamp(-self.clip_param, self.clip_param)
            v_loss = (v_cur - ret).pow(2)
            v_loss_clipped = (v_old_clipped - ret).pow(2)
            value_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def update(self, rollouts, device='cpu'):
        '''Updates model parameters based on current training batch.'''
        results = defaultdict(list)
        num_mini_batch = rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        # assert if num_mini_batch is 0
        assert num_mini_batch != 0, 'num_mini_batch is 0'
        iter_count = 0
        for _ in range(self.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            cost_loss_epoch, model_loss_epoch = 0, 0
            for batch, batch_th in rollouts.sampler(self.mini_batch_size, device):
                # Actor update.
                if iter_count % 3 == 0:
                    (policy_loss, entropy_loss, approx_kl, action_th,
                     nabla_pi_ref, nabla_pi_cost, nabla_pi_model) = self.compute_policy_loss(batch, batch_th)
                    # Update only when no KL constraint or constraint is satisfied.
                    if (self.target_kl <= 0) or (self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl):
                        self.actor_opt.zero_grad()
                        theta = self.ac.actor.linear_param.repeat(batch_th['obs'].shape[0], 1)
                        (policy_loss + self.entropy_coef * entropy_loss).backward()

                        temp1 = action_th.grad.unsqueeze(1) @ torch.FloatTensor(np.array(nabla_pi_cost))
                        temp1 @= (theta[:, :self.ac.actor.mpc.param_dict['l'].shape[0]].unsqueeze(2))
                        temp2 = action_th.grad.unsqueeze(1) @ torch.FloatTensor(np.array(nabla_pi_model))
                        temp2 @= (theta[:, self.ac.actor.mpc.param_dict['l'].shape[0]:].unsqueeze(2))
                        # temp3 = action_th.grad.unsqueeze(1) @ torch.FloatTensor(np.array(nabla_pi_ref))
                        # temp3 @= (theta[:, self.ac.actor.mpc.param_dict['l'].shape[0]:].unsqueeze(2))
                        (temp1.mean() + temp2.mean()).backward()
                        self.actor_opt.step()

                        p_loss_epoch += policy_loss.item()
                        e_loss_epoch += entropy_loss.item()
                        kl_epoch += approx_kl.item()
                        cost_loss_epoch += temp1.mean().item()
                        model_loss_epoch += temp2.mean().item()
                # Critic update.
                value_loss = self.compute_value_loss(batch_th)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
                v_loss_epoch += value_loss.item()
                iter_count += 1
            results['policy_loss'].append(p_loss_epoch / num_mini_batch)
            results['value_loss'].append(v_loss_epoch / num_mini_batch)
            results['entropy_loss'].append(e_loss_epoch / num_mini_batch)
            results['approx_kl'].append(kl_epoch / num_mini_batch)
            results['cost_loss'].append(cost_loss_epoch / num_mini_batch)
            results['model_loss'].append(model_loss_epoch / num_mini_batch)
        # print(self.ac.actor.pi_net.param_dict)
        print(self.ac.actor.linear_param.detach().numpy())
        results = {k: sum(v) / len(v) for k, v in results.items()}
        return results


# -----------------------------------------------------------------------------------
#                   Models
# -----------------------------------------------------------------------------------


class MLPActorCritic(nn.Module):
    '''Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        critic (MLPCritic): value network.
    '''

    def __init__(self,
                 env,
                 obs_space,
                 act_space,
                 gamma,
                 model,
                 hidden_dims=(64, 64),
                 activation='tanh',
                 actor_config=None,
                 ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            raise Exception('PPO-MPC is currently only implemented for continuous action spaces')
        # Policy.
        self.actor = MPCActor(env, obs_dim, act_dim, hidden_dims, activation, gamma, model, actor_config)
        # Value function.
        self.critic = MLPCritic(obs_dim, hidden_dims, activation)

    def step(self, obs):
        dist, _, info, results_dict, optimal_flag = self.actor(obs)
        a = dist.sample()
        logp_a = dist.log_prob(a)
        v = self.critic(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), info, results_dict, optimal_flag

    def act(self, obs):
        dist, _, _, _, _ = self.actor(obs)
        a = dist.mode()
        return a.cpu().numpy().astype(np.float32)

    def reset(self):
        self.actor.reset()


class MLPCritic(nn.Module):
    '''Critic MLP model.'''

    def __init__(self,
                 obs_dim,
                 hidden_dims,
                 activation
                 ):
        super().__init__()
        self.v_net = MLP(obs_dim, 1, hidden_dims, activation)

    def forward(self, obs):
        return self.v_net(obs)


class MPCActor(nn.Module):
    '''Actor MPC model.'''

    def __init__(self, env, obs_dim, act_dim, hidden_dims, activation, gamma, model, actor_config):
        super().__init__()
        # mpc actor
        self.mpc = MPCPolicyFunction(env, gamma, model, **actor_config['mpc_config'])

        # Parameters
        self.q_mpc = actor_config['q_mpc']
        self.r_mpc = actor_config['r_mpc']
        self.qt_mpc = actor_config['qt_mpc']
        self.model_param = actor_config['model_param']
        self._init_param_val()
        self.n_learnable_param = 0
        for k in self.param_dict.keys():
            self.n_learnable_param += self.param_dict[k].shape[0]
        temp = np.concatenate((self.q_mpc, self.r_mpc, self.qt_mpc, self.model_param))
        self.mpc_param = nn.Parameter(torch.FloatTensor(temp))
        self.param_net = MLP(obs_dim, self.n_learnable_param, hidden_dims, activation)
        self.traj_param = nn.Parameter(torch.FloatTensor(self.mpc.traj))

        # Construct output action distribution.
        self.logstd = nn.Parameter(-2.0 * torch.ones(act_dim))
        self.dist_fn = lambda x: Normal(x, self.logstd.exp())

    def _init_param_val(self):
        self.param_dict = {'l': np.concatenate((self.q_mpc, self.r_mpc, self.qt_mpc)),
                           'f': np.array(self.model_param)}

    def forward(self, obs, act=None):
        theta = self.get_theta_param(obs)
        if obs.ndim > 1:
            action, info, results_dict, optimal_flag = self.mpc.select_action_batch(obs, theta.numpy(),
                                                                                    self.traj_param.numpy())
        else:
            action, info, results_dict, optimal_flag = self.mpc.select_action(obs, theta.numpy(),
                                                                              self.traj_param.numpy())
        action = torch.FloatTensor(np.array(action))
        optimal_flag = torch.FloatTensor(np.array(optimal_flag))
        dist = self.dist_fn(action)
        logp_a = None
        if act is not None:
            logp_a = dist.log_prob(act)
        return dist, logp_a, info, results_dict, optimal_flag

    def forward_train(self, obs, act, info):
        theta = self.get_theta_param(obs)
        action, nabla_pi_ref, nabla_pi_theta, optimal_flag = self.mpc.select_action_batch_train(
            obs, theta.detach().numpy(), self.traj_param.detach().numpy(), info
        )
        action_th = action
        action_th.requires_grad_()
        dist = self.dist_fn(action_th)
        logp_a = dist.log_prob(act)
        return action_th, dist, logp_a, nabla_pi_ref, nabla_pi_theta, optimal_flag

    def reset(self):
        self.mpc.reset()

    def get_theta_param(self, obs):
        if obs.ndim > 1:
            theta = self.mpc_param.repeat(obs.shape[0], 1) + 0.1 * self.param_net.forward(torch.FloatTensor(obs))
        else:
            theta = self.mpc_param + 0.1 * self.param_net.forward(torch.FloatTensor(obs))
        return theta


class MPCPolicyFunction:
    def __init__(self,
                 env_fun,
                 gamma,
                 model,
                 horizon: int = 5,
                 warmstart: bool = True,
                 soft_constraints: bool = True,
                 constraint_tol: float = 1e-6,
                 additional_constraints: list = None,
                 n_parallel_solver: int = 1,
                 n_train_solver: int = 1,
                 ):
        self.env = env_fun
        self.model = model
        self.dt = self.model.dt
        self.T = horizon
        self.gamma = gamma
        self.update_step_count = 0
        self.soft_constraints = soft_constraints
        self.constraint_tol = constraint_tol
        self.warmstart = warmstart
        self.n_parallel_solver = n_parallel_solver
        self.n_train_solver = n_train_solver

        # Constraint list
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            (self.constraints, self.state_constraints_sym,
             self.input_constraints_sym) = reset_constraints(self.env.constraints.constraints
                                                             + self.additional_constraints)
        else:
            (self.constraints, self.state_constraints_sym,
             self.input_constraints_sym) = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []

        # Additional entries
        self.u_prev = None
        self.x_prev = None
        self.sigma_prev = None
        self.x_goal = None
        self.mode = None
        self.traj = None
        self.traj_step = 0
        self.infos = None
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

        # Setup optimizer
        self.solver_dict, self.qfn_dict = None, None
        self.dynamics_func = None
        self.set_dynamics_func()
        self.setup_optimizer()

    def reset(self):
        # Previously solved states & inputs, useful for warm start.
        self.u_prev = None
        self.x_prev = None
        self.sigma_prev = None
        self.x_goal = None
        self.traj = None
        self.traj_step = 0
        self.infos = None
        # self.X_EQ = self.env.X_EQ
        # self.U_EQ = self.env.U_EQ

        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

    def add_constraints(self, constraints):
        '''Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        '''
        (self.constraints, self.state_constraints_sym,
         self.input_constraints_sym) = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self, constraints):
        '''Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        '''
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
            old_constraints_list)

    def set_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        # self.dynamics_func = rk_discrete(self.model.fc_func,
        #                                  self.model.nx,
        #                                  self.model.nu,
        #                                  self.dt)
        self.dynamics_func = euler_discrete(self.model.param_fc_func,
                                            self.model.nx,
                                            self.model.nu,
                                            self.model.npl,
                                            self.dt)

    def setup_optimizer(self):
        '''Sets up nonlinear optimization problem.'''
        nx, nu, npl = self.model.nx, self.model.nu, self.model.npl
        T = self.T
        etau = 1e-4

        # Define optimizer and variables.
        # States.
        x_var = cs.MX.sym('x_var', nx, T + 1)
        # Inputs.
        u_var = cs.MX.sym('u_var', nu, T)
        # Add slack variables
        state_slack = cs.MX.sym('sigma_var', nx, T + 1)
        opt_vars = cs.vertcat(cs.reshape(u_var, -1, 1),
                              cs.reshape(x_var, -1, 1),
                              cs.reshape(state_slack, -1, 1))
        opt_vars_fn = cs.Function('opt_vars_fun', [opt_vars], [x_var, u_var, state_slack])

        # Parameters
        # Fixed parameters
        # Initial state.
        x_init = cs.MX.sym('x_init', nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = cs.MX.sym('x_ref', nx, T + 1)
        fixed_param = cs.vertcat(x_init)
        ref_param = cs.reshape(x_ref, -1, 1)

        # Learnable parameters
        # Cost
        Q, th_q, nq = _create_semi_definite_matrix(nx)
        R, th_r, nr = _create_semi_definite_matrix(nu)
        Qt, th_qt, nqt = _create_semi_definite_matrix(nx)
        # theta_param = cs.MX.sym('theta_var', nq + nr)
        cost_param = cs.vertcat(th_q, th_r, th_qt)
        # Model
        model_param = cs.MX.sym('f_param', npl)

        # cost (cumulative)
        cost = 0
        w = 1e3 * np.ones((1, nx))
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += self.gamma ** i * cost_func(x=x_var[:, i],
                                                u=u_var[:, i],
                                                Xr=x_ref[:, i],
                                                Ur=np.zeros((nu, 1)),
                                                Q=Q,
                                                R=R)['l']
        # Terminal cost.
        cost += self.gamma ** T * cost_func(x=x_var[:, -1],
                                            u=np.zeros((nu, 1)),
                                            Xr=x_ref[:, -1],
                                            Ur=np.zeros((nu, 1)),
                                            Q=Qt,
                                            R=R)['l']
        # Constraints
        g, hu, hx, hs = [], [], [], []
        # initial condition constraints
        g.append(x_var[:, 0] - x_init)
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], u=u_var[:, i], p=model_param)['xf']
            g.append(x_var[:, i + 1] - next_state)
            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                cost += w @ state_slack[:, i]
                hx.append(state_constraint(x_var[:, i])[:nx] - state_slack[:, i])
                hx.append(state_constraint(x_var[:, i])[nx:] - state_slack[:, i])
                hs.append(-state_slack[:, i])
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                hu.append(input_constraint(u_var[:, i]) + self.constraint_tol)
        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            cost += w @ state_slack[:, -1]
            hx.append(state_constraint(x_var[:, -1])[:nx] - state_slack[:, -1])
            hx.append(state_constraint(x_var[:, -1])[nx:] - state_slack[:, -1])
            hs.append(-state_slack[:, -1])
        # Setting casadi constraints and bounds
        G = cs.vertcat(*g)
        Hu = cs.vertcat(*hu)
        Hx = cs.vertcat(*hx)
        Hs = cs.vertcat(*hs)
        constraint_exp = cs.vertcat(*g, *hu, *hx, *hs)
        lbg = [0] * G.shape[0] + [-np.inf] * (Hu.shape[0] + Hx.shape[0] + Hs.shape[0])
        ubg = [0] * G.shape[0] + [0] * (Hu.shape[0] + Hx.shape[0] + Hs.shape[0])
        lbg = cs.vertcat(*lbg)
        ubg = cs.vertcat(*ubg)
        con_eq = [True] * G.shape[0] + [False] * (Hu.shape[0] + Hx.shape[0] + Hs.shape[0])

        # Create solver (IPOPT solver in this version)
        opts_setting = {
            'ipopt.max_iter': 200,
            'ipopt.print_level': 0,
            'print_time': 0,
            'expand': True,
            'equality': con_eq,
            'record_time': True,
            # 'ipopt.mu_target': etau,
            'ipopt.mu_init': etau,
            'ipopt.acceptable_tol': 1e-5,
            'ipopt.acceptable_obj_change_tol': 1e-5,
        }
        vnlp_prob = {
            'f': cost,
            'x': opt_vars,
            'p': cs.vertcat(fixed_param, ref_param, cost_param, model_param),
            'g': constraint_exp,
        }
        vsolver = cs.nlpsol('vsolver', 'ipopt', vnlp_prob, opts_setting)

        # Sensitivity
        # Multipliers
        lamb = cs.MX.sym('lambda', G.shape[0])
        mu_u = cs.MX.sym('muu', Hu.shape[0])
        mu_x = cs.MX.sym('mux', Hx.shape[0])
        mu_s = cs.MX.sym('mus', Hs.shape[0])

        # Build Lagrangian
        lagrangian = (
            cost
            + cs.transpose(lamb) @ G
            + cs.transpose(mu_u) @ Hu
            + cs.transpose(mu_x) @ Hx
            + cs.transpose(mu_s) @ Hs
        )
        dlag_dw = cs.jacobian(lagrangian, opt_vars)

        # Build KKT matrix
        R_kkt = cs.vertcat(
            cs.transpose(dlag_dw),
            G,
            mu_u * Hu + etau,
            mu_x * Hx + etau,
            mu_s * Hs + etau,
        )

        # z contains all variables of the lagrangian
        z = cs.vertcat(opt_vars, lamb, mu_u, mu_x, mu_s)

        # Generate sensitivity of the KKT matrix
        Rfun = cs.Function('Rfun', [z, fixed_param, ref_param, cost_param, model_param], [R_kkt])
        dR_sensfunc = Rfun.factory(
            'dR', ['i0', 'i1', 'i2', 'i3', 'i4'], ['jac:o0:i0', 'jac:o0:i2', 'jac:o0:i3', 'jac:o0:i4']
        )
        [dRdz, dRdP_ref, dRdP_cost, dRdP_model] = dR_sensfunc(z, fixed_param, ref_param, cost_param, model_param)
        dRdP = cs.horzcat(dRdP_ref, dRdP_cost, dRdP_model)

        # Generate sensitivity of the optimal solution
        dzdP = -cs.inv(dRdz) @ dRdP
        dPi = cs.Function('dPi', [z, fixed_param, ref_param, cost_param, model_param], [dzdP[: nu, :]])
        dzdP_ref = -cs.inv(dRdz) @ dRdP_ref
        dPi_ref = cs.Function('dPi_ref',
                              [z, fixed_param, ref_param, cost_param, model_param],
                              [dzdP_ref[: nu, :]])
        dzdP_cost = -cs.inv(dRdz) @ dRdP_cost
        dPi_cost = cs.Function('dPi_cost',
                               [z, fixed_param, ref_param, cost_param, model_param],
                               [dzdP_cost[: nu, :]])
        dzdP_model = -cs.inv(dRdz) @ dRdP_model
        dPi_model = cs.Function('dPi_model',
                                [z, fixed_param, ref_param, cost_param, model_param],
                                [dzdP_model[: nu, :]])

        self.solver_dict = {
            'solver': vsolver,
            'x_var': x_var,
            'u_var': u_var,
            'state_slack': state_slack,
            'opt_vars': opt_vars,
            'opt_vars_fn': opt_vars_fn,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost_param': cost_param,
            'cost': cost,
            'lower_bound': lbg,
            'upper_bound': ubg,
            'dpi_fn': dPi,
            'dpi_ref_fn': dPi_ref,
            'dpi_cost_fn': dPi_cost,
            'dpi_model_fn': dPi_model
        }

    def get_references(self, traj_step=None, traj_ref=None):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            if traj_step is None:
                traj_step = self.traj_step
            if traj_ref is None:
                traj_ref = self.traj
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(traj_step, self.traj.shape[-1])
            end = min(traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            # goal_states = np.concatenate([
            #     self.traj[:, start:end] + 1.0 * traj_change[:, start:end],
            #     np.tile(self.traj[:, -1:] + 1.0 * traj_change[:, -1:], (1, remain))
            # ], -1)
            goal_states = np.concatenate([
                traj_ref[:, start:end],
                np.tile(traj_ref[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def select_action(self, obs, theta, traj_change, info=None, mode='eval'):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            theta (ndarray): Learnable param based on current state
            traj_change (ndarray): Learnable trajectory
            info (dict): Current info
            mode (string): Current mode of evaluation (eval vs train)

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        solver_dict = self.solver_dict
        solver = solver_dict['solver']

        # Collect the fixed param
        # Assign reference trajectory within horizon.
        goal_states = self.get_references(self.traj_step, traj_change)
        fixed_param = obs[:self.model.nx, None]
        ref_param = goal_states.T.reshape(-1, 1)
        # # Collect learnable parameters
        # cost_param = (self.param_dict['l'] + 1.0 * theta[0, :self.param_dict['l'].shape[0]])[:, None]
        # model_param = (self.param_dict['f'] + 1.0 * theta[0, self.param_dict['l'].shape[0]:])[:, None]
        if self.mode == 'tracking':
            self.traj_step += 1

        opt_vars_init = np.zeros((solver_dict['opt_vars'].shape[0], solver_dict['opt_vars'].shape[1]))
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            opt_vars_init = update_initial_guess(self.x_prev, self.u_prev, self.sigma_prev)

        # Solve the optimization problem.
        soln = solver(
            x0=opt_vars_init,
            p=np.concatenate((fixed_param, ref_param, theta[:, None]))[:, 0],
            lbg=solver_dict['lower_bound'],
            ubg=solver_dict['upper_bound'],
        )
        optimal = solver.stats()['success']

        # Post-processing the solution
        opt_vars = soln['x'].full()
        x_val, u_val, sigma_val = solver_dict['opt_vars_fn'](opt_vars)
        self.x_prev = x_val.full()
        self.u_prev = u_val.full()
        self.sigma_prev = sigma_val.full()
        results_dict = {'horizon_states': deepcopy(self.x_prev), 'horizon_inputs': deepcopy(self.u_prev),
                        'goal_states': deepcopy(ref_param), 't_wall': solver.stats()['t_wall_total']}

        # Take the first action from the solved action sequence.
        if self.u_prev.ndim > 1:
            action = self.u_prev[:, 0]
        else:
            action = np.array([self.u_prev[0]])

        # additional info
        info = {
            'success': optimal,
            'soln': deepcopy(soln),
            'fixed_param': deepcopy(fixed_param),
            'ref_param': deepcopy(ref_param),
            # 'cost_param': deepcopy(cost_param),  # Not defined - commented out
            # 'model_param': deepcopy(model_param),  # Not defined - commented out
            'traj_step': deepcopy(self.traj_step) - 1
        }
        return action, info, results_dict, optimal

    def select_action_batch(self, obs_batch, theta, traj_change):
        solver_dict = self.solver_dict
        solver = solver_dict['solver']
        lbg = solver_dict['lower_bound']
        ubg = solver_dict['upper_bound']
        traj_step = self.traj_step
        goal_states = self.get_references(traj_step, traj_change)
        if self.mode == 'tracking':
            self.traj_step += 1

        eval_data_batch = []
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        for i, obs in enumerate(obs_batch):
            fixed_param = obs[:self.model.nx, None]
            ref_param = goal_states.T.reshape(-1, 1)
            cost_param = (self.param_dict['l'] + 1.0 * theta[i, :self.param_dict['l'].shape[0]])[:, None]
            model_param = (self.param_dict['f'] + 1.0 * theta[i, self.param_dict['l'].shape[0]:])[:, None]

            opt_vars_init = np.zeros((solver_dict['opt_vars'].shape[0], solver_dict['opt_vars'].shape[1]))
            if self.infos is not None:
                info = self.infos[i]
                soln = info['soln']
                # shift previous solutions by 1 step
                opt_vars_init = soln['x'].full()
                x_prev, u_prev, sigma_prev = solver_dict['opt_vars_fn'](opt_vars_init)
                x_prev, u_prev, sigma_prev = x_prev.full(), u_prev.full(), sigma_prev.full()
                opt_vars_init = update_initial_guess(x_prev, u_prev, sigma_prev)
            temp = [solver, opt_vars_init, fixed_param, ref_param, cost_param, model_param, lbg, ubg]
            eval_data_batch.append(temp)
        data_batch = self.multi_pool.map(_select_action, eval_data_batch)

        action_batch, results_dict_batch, info_batch, optimal_batch = [], [], [], []
        for data in data_batch:
            goal_states, soln, optimal = data

            # Post-processing the solution
            opt_vars = soln['x'].full()
            x_val, u_val, sigma_val = solver_dict['opt_vars_fn'](opt_vars)
            self.x_prev = x_val.full()
            self.u_prev = u_val.full()
            self.sigma_prev = sigma_val.full()
            results_dict = {'horizon_states': deepcopy(self.x_prev),
                            'horizon_inputs': deepcopy(self.u_prev),
                            'horizon_slacks': deepcopy(self.sigma_prev),
                            'goal_states': deepcopy(goal_states)}
            # results_dict['t_wall'].append(opti.stats()['t_wall_total'])

            # Take the first action from the solved action sequence.
            if self.u_prev.ndim > 1:
                action = self.u_prev[:, 0]
            else:
                action = np.array([self.u_prev[0]])

            # additional info
            info = {
                'success': optimal,
                'soln': deepcopy(soln),
                'fixed_param': deepcopy(fixed_param),
                'ref_param': deepcopy(ref_param),
                'cost_param': deepcopy(cost_param),
                'model_param': deepcopy(model_param),
                'traj_step': deepcopy(self.traj_step) - 1
            }

            # results batch
            action_batch.append(action)
            results_dict_batch.append(results_dict)
            info_batch.append(info)
            optimal_batch.append(optimal)
        action_batch = torch.FloatTensor(np.array(action_batch))
        optimal_batch = torch.FloatTensor(np.array(optimal_batch))
        self.infos = deepcopy(info_batch)
        return action_batch, info_batch, results_dict_batch, optimal_batch

    def select_action_batch_train(self, obs_batch, theta, traj_change, info_batch):
        solver_dict = self.solver_dict
        solver = solver_dict['solver']
        lbg = solver_dict['lower_bound']
        ubg = solver_dict['upper_bound']
        dpi = solver_dict['dpi_fn']
        dpi_cost = solver_dict['dpi_cost_fn']
        dpi_model = solver_dict['dpi_model_fn']
        opt_vars_fn = solver_dict['opt_vars_fn']

        eval_data_batch = []
        for i, obs in enumerate(obs_batch):
            info = info_batch[i]
            traj_step, soln = info['traj_step'], info['soln']
            opt_vars_init = soln['x'].full()
            goal_states = self.get_references(traj_step, traj_change)
            fixed_param = np.array(obs)[:self.model.nx, None]
            ref_param = goal_states.T.reshape(-1, 1)
            cost_param = (self.param_dict['l'] + 1.0 * theta[i, :self.param_dict['l'].shape[0]])[:, None]
            model_param = (self.param_dict['f'] + 1.0 * theta[i, self.param_dict['l'].shape[0]:])[:, None]
            temp = [solver, opt_vars_init, fixed_param, ref_param, cost_param, model_param,
                    lbg, ubg, opt_vars_fn, dpi, dpi_cost, dpi_model]
            eval_data_batch.append(temp)
        data_batch = self.multi_pool.map(_select_action_train, eval_data_batch)

        action_batch = []
        nabla_pi_ref_batch = []
        nabla_pi_cost_batch = []
        nabla_pi_model_batch = []
        optimal_batch = []
        for data in data_batch:
            action, _, nabla_pi_ref, nabla_pi_cost, nabla_pi_model, optimal = data
            action_batch.append(action)
            nabla_pi_ref_batch.append(nabla_pi_ref)
            nabla_pi_cost_batch.append(nabla_pi_cost)
            nabla_pi_model_batch.append(nabla_pi_model)
            optimal_batch.append(optimal)
        action_batch = torch.FloatTensor(np.array(action_batch))
        optimal_batch = torch.FloatTensor(np.array(optimal_batch))
        return action_batch, nabla_pi_ref_batch, nabla_pi_cost_batch, nabla_pi_model_batch, optimal_batch


class PPOBuffer(object):
    '''Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 max_length,
                 batch_size
                 ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        T, N = max_length, batch_size
        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.scheme = {
            'obs': {
                'vshape': (T, N, *obs_dim)
            },
            'act': {
                'vshape': (T, N, act_dim)
            },
            'rew': {
                'vshape': (T, N, 1)
            },
            'mask': {
                'vshape': (T, N, 1),
                'init': np.ones
            },
            'v': {
                'vshape': (T, N, 1)
            },
            'logp': {
                'vshape': (T, N, 1)
            },
            'ret': {
                'vshape': (T, N, 1)
            },
            'adv': {
                'vshape': (T, N, 1)
            },
            'terminal_v': {
                'vshape': (T, N, 1)
            },
            'info': {
                'vshape': T * N
            },
            'results_dict': {
                'vshape': T * N
            },
            'optimal': {
                'vshape': (T, N, 1)
            },
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocates space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            if k in ['info', 'results_dict']:
                self.__dict__[k] = deque([], maxlen=info['vshape'])
            else:
                vshape = info['vshape']
                dtype = info.get('dtype', np.float32)
                init = info.get('init', np.zeros)
                self.__dict__[k] = init(vshape, dtype=dtype)
        self.t = 0

    def push(self, batch):
        '''Inserts transition step data (as dict) to storage.'''
        for k, v in batch.items():
            assert k in self.keys
            if k in ['info', 'results_dict']:
                self.__dict__[k].extend(v)
            else:
                shape = self.scheme[k]['vshape'][1:]
                dtype = self.scheme[k].get('dtype', np.float32)
                v_ = np.asarray(deepcopy(v), dtype=dtype).reshape(shape)
                self.__dict__[k][self.t] = v_
        self.t = (self.t + 1) % self.max_length

    def get(self, device='cpu'):
        '''Returns all data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = torch.as_tensor(data, device=device)
        return batch

    def sample(self, indices):
        '''Returns partial data.'''
        batch = {}
        for k, info in self.scheme.items():
            if k in ['info', 'results_dict']:
                batch[k] = [self.__dict__[k][i] for i in indices]
            else:
                shape = info['vshape'][2:]
                batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self,
                mini_batch_size,
                device='cpu',
                drop_last=True
                ):
        '''Makes sampler to loop through all data.'''
        total_steps = self.max_length * self.batch_size
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            # batch = {
            #     k: torch.as_tensor(v, device=device) for k, v in batch.items()
            # }
            batch_th = {}
            for k, v in batch.items():
                if k not in ['info', 'results_dict']:
                    batch_th[k] = torch.as_tensor(v, device=device)
            yield batch, batch_th


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------


def update_initial_guess(x_prev, u_prev, sigma_prev):
    # shift previous solutions by 1 step
    u_guess = deepcopy(u_prev)
    x_guess = deepcopy(x_prev)
    sigma_guess = deepcopy(sigma_prev)
    u_guess[:, :-1] = u_guess[:, 1:]
    x_guess[:, :-1] = x_guess[:, 1:]
    sigma_guess[:, :-1] = sigma_guess[:, 1:]
    opt_vars_init = np.concatenate((u_guess.T.reshape(-1, 1),
                                    x_guess.T.reshape(-1, 1),
                                    sigma_guess.T.reshape(-1, 1)))
    return opt_vars_init


def _select_action(eval_data):
    solver, opt_vars_init, fixed_param, ref_param, cost_param, model_param, lbg, ubg = eval_data
    # Solve the optimization problem.
    soln = solver(
        x0=opt_vars_init,
        p=np.concatenate((fixed_param, ref_param, cost_param, model_param))[:, 0],
        lbg=lbg,
        ubg=ubg,
    )
    optimal = solver.stats()['success']
    return ref_param, soln, optimal


def _select_action_train(eval_data):
    (solver, opt_vars_init, fixed_param, ref_param, cost_param, model_param,
     lbg, ubg, opt_vars_fn, dpi, dpi_cost, dpi_model) = eval_data

    # Solve the optimization problem.
    soln = solver(
        x0=opt_vars_init,
        p=np.concatenate((fixed_param, ref_param, cost_param, model_param))[:, 0],
        lbg=lbg,
        ubg=ubg,
    )
    opt_vars = soln['x'].full()
    optimal = solver.stats()['success']
    _, u_val, _ = opt_vars_fn(opt_vars)
    u_val = u_val.full()
    if u_val.ndim > 1:
        action = u_val[:, 0]
    else:
        action = np.array([u_val[0]])

    # Sensitivity computation
    if optimal:
        mult = soln['lam_g'].full()
        z = np.concatenate((opt_vars, mult), axis=0)
        nabla_pi = dpi(z, fixed_param, ref_param, cost_param, model_param).full()
        nabla_pi_ref = nabla_pi[:, :ref_param.shape[0]]
        nabla_pi_cost = nabla_pi[:, ref_param.shape[0]:ref_param.shape[0] + cost_param.shape[0]]
        nabla_pi_model = nabla_pi[:, ref_param.shape[0] + cost_param.shape[0]:]
        # nabla_pi_cost = dpi_cost(z, fixed_param, cost_param, model_param).full()
        # nabla_pi_model = dpi_model(z, fixed_param, cost_param, model_param).full()
    else:
        nabla_pi_ref = np.zeros((u_val.shape[0], ref_param.shape[0]))
        nabla_pi_cost = np.zeros((u_val.shape[0], cost_param.shape[0]))
        nabla_pi_model = np.zeros((u_val.shape[0], model_param.shape[0]))
    return action, soln, nabla_pi_ref, nabla_pi_cost, nabla_pi_model, optimal


def _create_semi_definite_matrix(n):
    # U = cs.SX.sym('U', cs.Sparsity.lower(n))
    # u = cs.vertcat(*U.nonzeros())
    # W_upper = cs.Function('Lower_tri_W', [u], [U])
    # np = int(n * (n + 1) / 2)
    # p = cs.MX.sym('p', np)
    # W = W_upper(p)
    # WW = W.T @ W

    np = n
    P = cs.MX.sym('P', n)
    W = cs.diag(P)
    WW = cs.sqrt(W.T @ W)
    return WW, P, np


def soft_update(source, target, tau):
    '''Synchronizes target networks with exponential moving average.'''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(source, target):
    '''Synchronizes target networks by copying over parameters directly.'''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def random_sample(indices,
                  batch_size,
                  drop_last=True
                  ):
    '''Returns index batches to iterate over.'''
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(
        -1, batch_size)
    for batch in batches:
        yield batch
    if not drop_last:
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]


def compute_returns_and_advantages(rews,
                                   vals,
                                   masks,
                                   terminal_vals=0,
                                   last_val=0,
                                   gamma=0.99,
                                   use_gae=False,
                                   gae_lambda=0.95
                                   ):
    '''Useful for policy-gradient algorithms.'''
    T, N = rews.shape[:2]
    rets, advs = np.zeros((T, N, 1)), np.zeros((T, N, 1))
    ret, adv = last_val, np.zeros((N, 1))
    vals = np.concatenate([vals, last_val[np.newaxis, ...]], 0)
    # Compensate for time truncation.
    rews += gamma * terminal_vals
    # Cumulative discounted sums.
    for i in reversed(range(T)):
        ret = rews[i] + gamma * masks[i] * ret
        if not use_gae:
            adv = ret - vals[i]
        else:
            td_error = rews[i] + gamma * masks[i] * vals[i + 1] - vals[i]
            adv = adv * gae_lambda * gamma * masks[i] + td_error
        rets[i] = deepcopy(ret)
        advs[i] = deepcopy(adv)
    return rets, advs
