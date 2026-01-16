'''PPO utilities.'''

from collections import defaultdict, deque
from copy import deepcopy
import time

import casadi as cs
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.controllers.mpc.mpc_utils import reset_constraints
from safe_control_gym.controllers.rlmpc.rlmpc_utils import AdamOptimizer, rk_discrete
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.math_and_models.distributions import Normal
from safe_control_gym.math_and_models.neural_networks import MLP


class APPO_MPC_Agent:
    '''A PPO class that encapsulates models, optimizers and update functions.'''

    def __init__(
            self,
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
            exploration_init=-1.0,
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
        self.exploration_init = exploration_init
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
            exploration_init=self.exploration_init,
            activation=self.activation,
            actor_config=actor_config,
        )

        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
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

    def reset(self, idx=None):
        '''Reset function, especially needed for resetting MPC actor'''
        self.ac.reset(idx)

    def state_dict(self):
        '''Snapshots agent state.'''
        return {
            'ac': self.ac.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict()
        }

    def load_state_dict(self, state_dict, strict=True):
        '''Restores agent state.'''
        self.ac.load_state_dict(state_dict['ac'], strict=strict)
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])

    def compute_policy_loss(self, batch, theta_old):
        '''Returns policy loss(es) given batch of data.'''
        obs, act, logp_old, adv = batch['obs'], batch['act'], batch['logp'], batch['adv']
        mpc_act, nabla_pi_theta, optimal = batch['mpc_act'], batch['nabla_pi_theta'], batch['optimal']
        action, dist, logp = self.ac.actor.forward_train(obs, act, theta_old, mpc_act, nabla_pi_theta)

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
        return policy_loss, entropy_loss, approx_kl

    def compute_value_loss(self, batch):
        '''Returns value loss(es) given batch of data.'''
        obs, ret, v_old = batch['obs'], batch['ret'], batch['v']
        v_cur = self.ac.critic(obs)
        value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def update(self, rollouts, device='cpu'):
        '''Updates model parameters based on current training batch.'''
        results = defaultdict(list)
        num_mini_batch = rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        # assert if num_mini_batch is 0
        assert num_mini_batch != 0, 'num_mini_batch is 0'
        theta_old = self.ac.actor.mpc_param.detach().clone()
        for _ in range(self.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            theta_loss_epoch, ref_loss_epoch = 0, 0
            for batch in rollouts.sampler(self.mini_batch_size, device):
                # Actor update.
                policy_loss, entropy_loss, approx_kl = self.compute_policy_loss(batch, theta_old)
                # Update only when no KL constraint or constraint is satisfied.
                if (self.target_kl <= 0) or (self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl):
                    self.actor_opt.zero_grad()
                    (policy_loss + self.entropy_coef * entropy_loss).backward()
                    self.actor_opt.step()
                    with torch.no_grad():
                        self.ac.actor.mpc_param.clamp_(1e-5, 100.0)

                    p_loss_epoch += policy_loss.item()
                    e_loss_epoch += entropy_loss.item()
                    kl_epoch += approx_kl.item()

                # Critic update.
                value_loss = self.compute_value_loss(batch)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
                v_loss_epoch += value_loss.item()
            results['policy_loss'].append(p_loss_epoch / num_mini_batch)
            results['value_loss'].append(v_loss_epoch / num_mini_batch)
            results['entropy_loss'].append(e_loss_epoch / num_mini_batch)
            results['approx_kl'].append(kl_epoch / num_mini_batch)
            results['theta_loss'].append(theta_loss_epoch / num_mini_batch)
            results['ref_loss'].append(ref_loss_epoch / num_mini_batch)
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

    def __init__(
            self,
            env,
            obs_space,
            act_space,
            gamma,
            model,
            hidden_dims=(64, 64),
            exploration_init=-1.0,
            activation='tanh',
            actor_config=None
    ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            raise Exception('PPO-MPC is currently only implemented for continuous action spaces')
        # Policy.
        self.actor = MPCActor(
            env, obs_dim, act_dim, hidden_dims, activation, gamma, model, exploration_init, actor_config
        )
        # Value function.
        self.critic = MLPCritic(obs_dim, hidden_dims, activation)

    def step(self, obs, info=None):
        dist, _, mpc_act, nabla_pi_theta, optimal_flag = self.actor(obs, actor_info=info)
        mpc_act = np.array(mpc_act)
        nabla_pi_theta = np.array(nabla_pi_theta)
        a = dist.sample()
        logp_a = dist.log_prob(a)
        v = self.critic(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), mpc_act, nabla_pi_theta, optimal_flag

    def act(self, obs, info=None):
        dist, _, _, _, _ = self.actor(obs, actor_info=info)
        a = dist.mode()
        return a.cpu().numpy()

    def reset(self, idx):
        self.actor.reset(idx)


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

    def __init__(self, env, obs_dim, act_dim, hidden_dims, activation, gamma, model, exploration_init, actor_config):
        super().__init__()
        # mpc actor
        self.mpc = MPCPolicyFunction(env, gamma, model, **actor_config['mpc_config'])

        # Parameters
        self.q_mpc = actor_config['q_mpc']
        self.r_mpc = actor_config['r_mpc']
        self.qt_mpc = actor_config['qt_mpc']
        self.back_off = actor_config['back_off']
        self.model_param = actor_config['model_param']
        self._init_param_val()
        self.n_learnable_param = 0
        for k in self.param_dict.keys():
            self.n_learnable_param += self.param_dict[k].shape[0]
        temp = np.concatenate((self.q_mpc, self.r_mpc, self.qt_mpc, self.back_off, self.model_param))
        self.mpc_param = nn.Parameter(torch.FloatTensor(temp))
        self.param_net = MLP(obs_dim, self.n_learnable_param, hidden_dims, activation)
        # self.traj_param = nn.Parameter(torch.FloatTensor(self.mpc.traj))
        self.traj_param = torch.FloatTensor(self.mpc.traj)

        # Construct output action distribution.
        self.logstd = nn.Parameter(exploration_init * torch.ones(act_dim))
        self.dist_fn = lambda x: Normal(x, self.logstd.exp())

    def _init_param_val(self):
        self.param_dict = {'l': np.concatenate((self.q_mpc, self.r_mpc, self.qt_mpc)),
                           'b': np.array(self.back_off),
                           'f': np.array(self.model_param)}

    def forward(self, obs, act=None, actor_info=None):
        theta = self.get_theta_param(obs)
        traj_param = self.get_references(actor_info)
        if obs.ndim > 1:
            mpc_act, nabla_pi_theta, optimal_flag = self.mpc.select_action_batch(
                obs.cpu().numpy(), theta.cpu().numpy(), traj_param, actor_info
            )
        else:
            mpc_act, optimal_flag = self.mpc.select_action(
                obs.cpu().numpy(), theta.cpu().numpy(), traj_param
            )
            nabla_pi_theta = None
        action = torch.FloatTensor(np.array(mpc_act)).to(theta.device)
        dist = self.dist_fn(action)
        logp_a = None
        if act is not None:
            logp_a = dist.log_prob(act)
        return dist, logp_a, mpc_act, nabla_pi_theta, optimal_flag

    def forward_train(self, obs, act, theta_old, mpc_act, nabla_pi_theta):
        theta = self.get_theta_param(obs)
        action = mpc_act.unsqueeze(2) + nabla_pi_theta @ (theta - theta_old.repeat(obs.shape[0], 1)).unsqueeze(2)
        action = action.squeeze(2)
        dist = self.dist_fn(action)
        logp_a = dist.log_prob(act)
        return action, dist, logp_a

    def reset(self, idx):
        self.mpc.reset(idx)

    def get_theta_param(self, obs):
        # obs = torch.FloatTensor(obs).to(self.mpc_param.device)
        if obs.ndim > 1:
            theta = self.mpc_param.repeat(obs.shape[0], 1) + 0.0 * self.param_net.forward(obs)
        else:
            theta = self.mpc_param + 0.0 * self.param_net.forward(obs)
        theta += torch.rand_like(theta) * 1e-6
        return theta

    def get_references(self, info_batch):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        goal_states_batch = []
        for info in info_batch:
            traj_step = info['current_step']
            traj_ref = info['x_ref'].T
            if self.mpc.env.TASK == Task.STABILIZATION:
                # Repeat goal state for horizon steps.
                goal_states = np.tile(self.mpc.env.X_GOAL.reshape(-1, 1), (1, self.mpc.T + 1))
            elif self.mpc.env.TASK == Task.TRAJ_TRACKING:
                # Slice trajectory for horizon steps, if not long enough, repeat last state.
                start = min(traj_step, traj_ref.shape[-1])
                end = min(traj_step + self.mpc.T + 1, traj_ref.shape[-1])
                remain = max(0, self.mpc.T + 1 - (end - start))
                goal_states = np.concatenate([
                    traj_ref[:, start:end],
                    np.tile(traj_ref[:, -1:], (1, remain))
                ], -1)
            else:
                raise Exception('Reference for this mode is not implemented.')
            goal_states_batch.append(goal_states)
        return goal_states_batch  # list of (nx, T+1).

    def get_ref_param(self, info_batch):
        goal_states_batch = torch.FloatTensor()
        if self.mpc.env.TASK == Task.TRAJ_TRACKING:
            for info in info_batch:
                traj_step = info['traj_step']
                # Slice trajectory for horizon steps, if not long enough, repeat last state.
                start = min(traj_step, self.mpc.traj.shape[-1])
                end = min(traj_step + self.mpc.T + 1, self.mpc.traj.shape[-1])
                remain = max(0, self.mpc.T + 1 - (end - start))
                goal_states = torch.cat((
                    self.traj_param[:, start:end],
                    torch.tile(self.traj_param[:, -1:], (1, remain))
                ), -1).T.reshape(-1, 1).T
                goal_states_batch = torch.cat((goal_states_batch, goal_states), 0)
        else:
            raise Exception('Reference update for this mode is not implemented.')
        return goal_states_batch  # (nx, T+1).


class MPCPolicyFunction:
    def __init__(
            self,
            env_fun,
            gamma,
            model,
            horizon: int = 5,
            warmstart: bool = True,
            soft_constraints: bool = True,
            constraint_tol: float = 1e-6,
            additional_constraints: list = None,
            n_parallel_solver: int = 1,
            jit: bool = False,
            jit_options: dict = None,
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
        self.jit = jit
        self.jit_options = jit_options if jit_options is not None else {}

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
        self.infos = [None] * self.n_parallel_solver
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
        self.solver_dict = None
        self.dynamics_func = None
        self.set_dynamics_func()
        self.setup_optimizer()

        # Parallel solvers
        self.pi_solvers, _, _, self.all_solvers = self.get_parallel_solver(self.n_parallel_solver)

    def reset(self, idx=None):
        # Previously solved states & inputs, useful for warm start.
        self.u_prev = None
        self.x_prev = None
        self.sigma_prev = None
        self.x_goal = None
        self.traj = None
        self.traj_step = 0
        if idx is not None:
            self.infos[idx] = None
        else:
            self.infos = [None] * self.n_parallel_solver

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
        """Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        (self.constraints, self.state_constraints_sym,
         self.input_constraints_sym) = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self, constraints):
        """Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        """
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
            old_constraints_list)

    def set_dynamics_func(self):
        """Updates symbolic dynamics with actual control frequency."""
        self.dynamics_func = rk_discrete(self.model.param_fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.model.npl,
                                         self.dt)
        # self.dynamics_func = euler_discrete(self.model.param_fc_func,
        #                                     self.model.nx,
        #                                     self.model.nu,
        #                                     self.model.npl,
        #                                     self.dt)

    def setup_optimizer(self):
        """Sets up nonlinear optimization problem."""
        nx, nu, npl = self.model.nx, self.model.nu, self.model.npl
        T = self.T
        etau = 1e-5  # barrier parameter for interior point method
        start_time = time.time()

        # Optimization variable: [x0, u0, sigma0, x1, u1, ...]
        opt_vars = []
        x_var, u_var, sigma_var = [], [], []
        for i in range(T):
            x = cs.MX.sym('x', nx)
            u = cs.MX.sym('u', nu)
            sigma = cs.MX.sym('sigma', nx)
            # append the casadi var to opt_Vars
            opt_vars.append(x)
            opt_vars.append(u)
            opt_vars.append(sigma)
            # append the casadi var to individual vars for ease of use
            x_var.append(x)
            u_var.append(u)
            sigma_var.append(sigma)
        x = cs.MX.sym('x', nx)  # final state
        sigma = cs.MX.sym('sigma', nx)
        opt_vars.append(x)
        opt_vars.append(sigma)
        x_var.append(x)
        sigma_var.append(sigma)
        # compile the variable vectors/matrices
        opt_vars = cs.vcat(opt_vars)
        x_var, u_var, sigma_var = cs.hcat(x_var), cs.hcat(u_var), cs.hcat(sigma_var)
        # function definitions for conversion
        opt_vars_fn = cs.Function('opt_vars_fun', [x_var, u_var, sigma_var], [opt_vars])
        xus_fn = cs.Function('xus_fun', [opt_vars], [x_var, u_var, sigma_var])
        opt_act_fn = cs.Function('opt_act_fun', [opt_vars], [u_var[:, 0]])

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
        Q, th_q, _ = _create_semi_definite_matrix(nx)
        R, th_r, _ = _create_semi_definite_matrix(nu)
        Qt, th_qt, _ = _create_semi_definite_matrix(nx)
        # theta_param = cs.MX.sym("theta_var", nq + nr)
        cost_param = cs.vertcat(th_q, th_r, th_qt)
        back_off_param = cs.MX.sym('back_off_param', nx)
        # Model
        model_param = cs.MX.sym('f_param', npl)

        # cost (cumulative)
        cost = 0
        w = 1e3 * np.ones((1, nx))
        cost_func = self.model.loss
        for i in range(T):
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
                                            R=np.zeros((nu, nu)))['l']
        # Constraints
        con_list, con_lbg, con_ubg, con_eq = [], [], [], []
        H_eq, H_ieq = [], []
        mult, lamb, mu = [], [], []
        # initial condition constraints
        con_list.append(x_var[:, 0] - x_init)
        con_lbg.append(cs.DM.zeros(nx, 1))
        con_ubg.append(cs.DM.zeros(nx, 1))
        con_eq += [True] * nx

        H_eq.append(x_var[:, 0] - x_init)
        lm = cs.MX.sym('lm', nx)
        mult.append(lm)
        lamb.append(lm)
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], u=u_var[:, i], p=model_param)['xf']
            con_list.append(x_var[:, i + 1] - next_state)
            con_lbg.append(cs.DM.zeros(nx, 1))
            con_ubg.append(cs.DM.zeros(nx, 1))
            con_eq += [True] * nx

            H_eq.append(x_var[:, i + 1] - next_state)
            lm = cs.MX.sym('lm', nx)
            mult.append(lm)
            lamb.append(lm)

            # State bounds
            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                cost += w @ sigma_var[:, i]
                con_list.append(state_constraint(x_var[:, i])[:nx] - sigma_var[:, i] + back_off_param)
                con_list.append(state_constraint(x_var[:, i])[nx:] - sigma_var[:, i] + back_off_param)
                con_list.append(-sigma_var[:, i])
                con_lbg.append(-cs.DM.inf(3 * nx, 1))
                con_ubg.append(cs.DM.zeros(3 * nx, 1))
                con_eq += [False] * 3 * nx

                H_ieq.append(state_constraint(x_var[:, i])[:nx] - sigma_var[:, i] + back_off_param)
                H_ieq.append(state_constraint(x_var[:, i])[nx:] - sigma_var[:, i] + back_off_param)
                H_ieq.append(-sigma_var[:, i])
                lm = cs.MX.sym('lm', 3 * nx)
                mult.append(lm)
                mu.append(lm)

            # Action bounds
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                con_list.append(input_constraint(u_var[:, i]) + self.constraint_tol)
                con_lbg.append(-cs.DM.inf(2 * nu, 1))
                con_ubg.append(cs.DM.zeros(2 * nu, 1))
                con_eq += [False] * 2 * nu

                H_ieq.append(input_constraint(u_var[:, i]) + self.constraint_tol)
                lm = cs.MX.sym('lm', 2 * nu)
                mult.append(lm)
                mu.append(lm)

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            cost += w @ sigma_var[:, -1]
            con_list.append(state_constraint(x_var[:, -1])[:nx] - sigma_var[:, -1] + back_off_param)
            con_list.append(state_constraint(x_var[:, -1])[nx:] - sigma_var[:, -1] + back_off_param)
            con_list.append(-sigma_var[:, -1])
            con_lbg.append(-cs.DM.inf(3 * nx, 1))
            con_ubg.append(cs.DM.zeros(3 * nx, 1))
            con_eq += [False] * 3 * nx

            H_ieq.append(state_constraint(x_var[:, -1])[:nx] - sigma_var[:, -1] + back_off_param)
            H_ieq.append(state_constraint(x_var[:, -1])[nx:] - sigma_var[:, -1] + back_off_param)
            H_ieq.append(-sigma_var[:, -1])
            lm = cs.MX.sym('lm', 3 * nx)
            mult.append(lm)
            mu.append(lm)
        # concatenating all the lists
        con_list, H_eq, H_ieq = cs.vcat(con_list), cs.vcat(H_eq), cs.vcat(H_ieq)
        mult, lamb, mu = cs.vcat(mult), cs.vcat(lamb), cs.vcat(mu)
        con_lbg, con_ubg = cs.vcat(con_lbg), cs.vcat(con_ubg)
        lang_mult_fn = cs.Function('lang_mult_fn', [mult], [lamb, mu])

        # Create solver (IPOPT solver in this version)
        opts_setting = {
            'print_time': 0,
            'record_time': True,
            'expand': True,
            'equality': con_eq,
            'structure_detection': 'auto',
            'debug': False,
            "jit": self.jit,
            "jit_cleanup": False,
            "jit_temp_suffix": False,
            "jit_options": self.jit_options,
            'fatrop.mu_init': etau,
            'fatrop.max_iter': 500,
            'fatrop.print_level': 0,
            'fatrop.acceptable_tol': 1e-5,
        }
        vnlp_prob = {
            'f': cost,
            'x': opt_vars,
            'p': cs.vertcat(fixed_param, ref_param, cost_param, back_off_param, model_param),
            'g': con_list,
        }
        pisolver = cs.nlpsol('pisolver', 'fatrop', vnlp_prob, opts_setting)
        print(f'[MPC Setup] NLP problem setup time: {time.time() - start_time:.3f} seconds.')
        start_time = time.time()

        # Build Lagrangian
        lagrangian = (
            cost
            + cs.transpose(lamb) @ H_eq
            + cs.transpose(mu) @ H_ieq
        )
        dlag_dw = cs.jacobian(lagrangian, opt_vars)
        # Build KKT matrix
        R_kkt = cs.vertcat(
            cs.transpose(dlag_dw),
            H_eq,
            mu * H_ieq + etau,
        )
        # z contains all variables of the lagrangian
        z = cs.vertcat(opt_vars, mult)
        theta = cs.vertcat(cost_param, back_off_param, model_param)

        # Generate sensitivity of the KKT matrix
        rkkt_fn = cs.Function('rkkt_fn', [z, fixed_param, ref_param, theta], [R_kkt])
        rkkt_norm_fn = cs.Function('rkkt_norm_fn', [z, fixed_param, ref_param, theta], [cs.norm_2(R_kkt)])
        dR_sensfunc = rkkt_fn.factory('dR', ['i0', 'i1', 'i2', 'i3'], ['jac:o0:i0', 'jac:o0:i2', 'jac:o0:i3'])
        [dRdz, dRdP_ref, dRdP_theta] = dR_sensfunc(z, fixed_param, ref_param, theta)
        # dRdP = cs.horzcat(dRdP_ref, dRdP_theta)
        dRdP = cs.horzcat(dRdP_theta)
        print(f'[MPC Setup] KKT matrix setup time: {time.time() - start_time:.3f} seconds.')
        start_time = time.time()

        # Generate sensitivity of the optimal solution
        # dzdP = -cs.inv(dRdz) @ dRdP
        # dzdP = -cs.solve(dRdz, dRdP)
        # dPi = dzdP[nx: nx + nu, :].T
        S = cs.DM.zeros(nu, dRdz.shape[0])
        for i in range(nu):
            S[i, nx + i] = 1.0
        dPi_prime = cs.solve(dRdz.T, S.T).T
        dPi = -(dPi_prime @ dRdP).T
        dPi_zeros = cs.MX.zeros(dPi.shape)
        f_true = cs.Function('f_true', [z, fixed_param, ref_param, theta], [dPi])
        f_false = cs.Function('f_false', [z, fixed_param, ref_param, theta], [dPi_zeros])
        dPi_fn = cs.Function.if_else('dPi_fn', f_true, f_false)
        print(f'[MPC Setup] Sensitivity setup time: {time.time() - start_time:.3f} seconds.')
        start_time = time.time()

        # R_kkt function with jit
        jit_opts = {
            "jit": self.jit,
            "jit_cleanup": False,
            "jit_temp_suffix": False,
            "jit_options": self.jit_options
        }
        all_fn = cs.Function("all_fn", [z, fixed_param, ref_param, theta], [cs.norm_2(R_kkt), dPi], jit_opts)
        # all_fn.save("all_fn.casadi")
        print(f'[MPC Setup] JIT compilation time: {time.time() - start_time:.3f} seconds.')

        self.solver_dict = {
            'x_var': x_var,
            'u_var': u_var,
            'state_slack': sigma_var,
            'opt_vars': opt_vars,
            'opt_vars_fn': opt_vars_fn,
            'xus_fn': xus_fn,
            'opt_act_fn': opt_act_fn,
            'cost': cost,
            'lower_bound': con_lbg,
            'upper_bound': con_ubg,
            'lang_mult_fn': lang_mult_fn,
            'solver': pisolver,
            'rkkt_fn': rkkt_fn,
            'rkkt_norm_fn': rkkt_norm_fn,
            'dpi_fn': dPi_fn,
            'all_fn': all_fn
        }

    def get_references(self, traj_step=None, traj_ref=None):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
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
            goal_states = np.concatenate([
                traj_ref[:, start:end],
                np.tile(traj_ref[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def select_action(self, obs, theta, traj_ref, info=None, mode='eval'):
        """Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            theta (ndarray): Learnable param based on current state
            traj_ref (ndarray): Learnable trajectory
            info (dict): Current info
            mode (string): Current mode of evaluation (eval vs train)

        Returns:
            action (ndarray): Input/action to the task/env.
        """
        solver_dict = self.solver_dict
        solver = solver_dict['solver']
        opt_vars_fn = solver_dict['opt_vars_fn']
        xus_fn = solver_dict['xus_fn']

        # Collect the fixed param
        # Assign reference trajectory within horizon.
        fixed_param = obs[:self.model.nx, None]
        # goal_states = self.get_references(self.traj_step, traj_ref)
        goal_states = traj_ref[0].copy()
        ref_param = goal_states.T.reshape(-1, 1)
        # Collect learnable parameters
        p_param = np.concatenate((fixed_param, ref_param, theta[:, None]))[:, 0]
        if self.mode == 'tracking':
            self.traj_step += 1

        opt_vars_init = np.zeros((solver_dict['opt_vars'].shape[0], solver_dict['opt_vars'].shape[1]))
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            opt_vars_init = update_initial_guess(self.x_prev, self.u_prev, self.sigma_prev, opt_vars_fn)

        # Solve the optimization problem.
        soln = solver(
            x0=opt_vars_init,
            p=p_param,
            lbg=solver_dict['lower_bound'],
            ubg=solver_dict['upper_bound'],
        )
        optimal = solver.stats()['success']

        # Post-processing the solution
        opt_vars = soln['x'].full()
        x_val, u_val, sigma_val = xus_fn(opt_vars)
        self.x_prev = x_val.full()
        self.u_prev = u_val.full()
        self.sigma_prev = sigma_val.full()
        results_dict = {
            'horizon_states': deepcopy(self.x_prev),
            'horizon_inputs': deepcopy(self.u_prev),
            'goal_states': deepcopy(ref_param),
            't_wall': solver.stats()['t_wall_total']
        }

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
            'theta_param': deepcopy(theta),
            'traj_step': deepcopy(self.traj_step) - 1
        }
        return action, optimal

    def select_action_batch(self, obs_batch, theta, traj_ref, actor_info):
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        con_lbg = self.solver_dict['lower_bound']
        con_ubg = self.solver_dict['upper_bound']
        opt_vars_fn = self.solver_dict['opt_vars_fn']
        xus_fn = self.solver_dict['xus_fn']
        traj_step = self.traj_step
        # goal_states = self.get_references(traj_step, traj_ref)
        if self.mode == 'tracking':
            self.traj_step += 1

        # eval_data_batch = []
        x0, fixed_p, ref_p = [], [], []
        # ref_param = goal_states.T.reshape(-1, 1).repeat(obs_batch.shape[0], 1)
        lbg = con_lbg.full().repeat(obs_batch.shape[0], 1)
        ubg = con_ubg.full().repeat(obs_batch.shape[0], 1)
        for i, obs in enumerate(obs_batch):
            fixed_param = obs[:self.model.nx]
            ref_param = traj_ref[i].T.reshape(-1, 1)[:, 0]
            opt_vars_init = np.zeros(self.solver_dict['opt_vars'].shape)
            if self.infos[i] is not None:  # shift previous solutions by 1 step based on last soln
                opt_vars_init = self.infos[i]['opt_var']
                x_prev, u_prev, sigma_prev = xus_fn(opt_vars_init)
                x_prev, u_prev, sigma_prev = x_prev.full(), u_prev.full(), sigma_prev.full()
                opt_vars_init = update_initial_guess(x_prev, u_prev, sigma_prev, opt_vars_fn)
                if actor_info[i]['current_step'] == 0:
                    opt_vars_init = np.zeros_like(opt_vars_init)

            x0.append(opt_vars_init[:, 0])
            fixed_p.append(fixed_param)
            ref_p.append(ref_param)
        x0, fixed_p, ref_p = np.array(x0).T, np.array(fixed_p).T, np.array(ref_p).T
        p = np.concatenate((fixed_p, ref_p, theta.T), axis=0)

        # Forward pass through solver
        soln_batch = self.pi_solvers(x0=x0, p=p, lbg=lbg, ubg=ubg)
        z = cs.vertcat(soln_batch['x'], soln_batch['lam_g'])
        rkkt_norm_batch, dpidp_batch = self.all_solvers(z, fixed_p, ref_p, theta.T)
        optimal_batch = rkkt_norm_batch.full() < 1e-3

        # Post-processing the solution
        action_batch, nabla_pi_theta_batch, info_batch = [], [], []
        for i, obs in enumerate(obs_batch):
            opt_vars = soln_batch['x'].full()[:, i]
            x_val, u_val, sigma_val = xus_fn(opt_vars)
            x_prev = x_val.full()
            u_prev = u_val.full()
            sigma_prev = sigma_val.full()
            results_dict = {
                'horizon_states': deepcopy(x_prev),
                'horizon_inputs': deepcopy(u_prev),
                'horizon_slacks': deepcopy(sigma_prev),
                'goal_states': deepcopy(ref_p[:, i])
            }
            # results_dict['t_wall'].append(opti.stats()['t_wall_total'])

            # Take the first action from the solved action sequence.
            if u_prev.ndim > 1:
                action = u_prev[:, 0]
            else:
                action = np.array([u_prev[0]])

            # additional info
            info = {
                'success': optimal_batch[0, i],
                'opt_var': opt_vars,
                'action': deepcopy(action),
                'fixed_param': deepcopy(fixed_p[:, i]),
                'ref_param': deepcopy(ref_p[:, i]),
                'theta_param': deepcopy(theta[i, :]),
                'traj_step': deepcopy(actor_info[i]['current_step']),
                'x_ref': deepcopy(actor_info[i]['x_ref']),
                'dpidp': deepcopy(int(optimal_batch[0, i])*dpidp_batch[:, self.model.nu * i: self.model.nu * (i + 1)].T)
            }

            action_batch.append(action)
            nabla_pi_theta_batch.append(int(optimal_batch[0, i])*dpidp_batch[:, self.model.nu * i: self.model.nu * (i + 1)].T)
            info_batch.append(info)
        self.infos = deepcopy(info_batch)
        return action_batch, nabla_pi_theta_batch, optimal_batch.T

    def get_parallel_solver(self, n_solvers):
        pi_solvers = self.solver_dict['solver'].map(n_solvers, 'thread')
        rkkt_norm_solvers = self.solver_dict['rkkt_norm_fn'].map(n_solvers, 'thread')
        # rkkt_fn = cs.Function.load('rkkt_fn.casadi')
        # rkkt_solvers = rkkt_fn.map(n_solvers, 'thread')
        dpi_solvers = self.solver_dict['dpi_fn'].map(n_solvers, 'thread')
        # dPi_fn = cs.Function.load('dPi_fn.casadi')
        # dpi_solvers = dPi_fn.map(n_solvers, 'thread')
        all_solvers = self.solver_dict['all_fn'].map(n_solvers, 'thread')
        return pi_solvers, rkkt_norm_solvers, dpi_solvers, all_solvers


class APPOBuffer(object):
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
                 theta_dim,
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
            'obs': {'vshape': (T, N, *obs_dim)},
            'act': {'vshape': (T, N, act_dim)},
            'rew': {'vshape': (T, N, 1)},
            'mask': {'vshape': (T, N, 1), 'init': np.ones},
            'v': {'vshape': (T, N, 1)},
            'logp': {'vshape': (T, N, 1)},
            'ret': {'vshape': (T, N, 1)},
            'adv': {'vshape': (T, N, 1)},
            'terminal_v': {'vshape': (T, N, 1)},
            'mpc_act': {'vshape': (T, N, act_dim)},
            'nabla_pi_theta': {'vshape': (T, N, act_dim, theta_dim)},
            'optimal': {'vshape': (T, N, 1)},
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocates space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.t = 0

    def push(self, batch):
        '''Inserts transition step data (as dict) to storage.'''
        for k, v in batch.items():
            assert k in self.keys
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
            shape = info['vshape'][2:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self, mini_batch_size, device='cpu', drop_last=True):
        '''Makes sampler to loop through all data.'''
        total_steps = self.max_length * self.batch_size
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {
                k: torch.as_tensor(v, device=device) for k, v in batch.items()
            }
            yield batch


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------


def update_initial_guess(x_prev, u_prev, sigma_prev, opt_vars_fn):
    # shift previous solutions by 1 step
    u_guess = deepcopy(u_prev)
    x_guess = deepcopy(x_prev)
    sigma_guess = deepcopy(sigma_prev)
    u_guess[:, :-1] = u_guess[:, 1:]
    x_guess[:, :-1] = x_guess[:, 1:]
    sigma_guess[:, :-1] = sigma_guess[:, 1:]
    opt_vars_init = opt_vars_fn(x_guess, u_guess, sigma_guess).full()
    return opt_vars_init


def _create_semi_definite_matrix(n):
    # U = cs.SX.sym("U", cs.Sparsity.lower(n))
    # u = cs.vertcat(*U.nonzeros())
    # W_upper = cs.Function("Lower_tri_W", [u], [U])
    # np = int(n * (n + 1) / 2)
    # p = cs.MX.sym("p", np)
    # W = W_upper(p)
    # WW = W.T @ W

    n_param = n
    P = cs.MX.sym('P', n)
    W = cs.diag(P)
    # WW = cs.sqrt(W.T @ W)
    return W, P, n_param


def random_sample(indices, batch_size, drop_last=True):
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


def compute_returns_and_advantages(
        rews,
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
