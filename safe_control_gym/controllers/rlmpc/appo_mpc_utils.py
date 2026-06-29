"""PPO utilities."""

from collections import defaultdict
import time

import casadi as cs
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.controllers.rlmpc.rlmpc_utils import (
    update_initial_guess,
    MPCFunction,
)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.math_and_models.distributions import Normal
from safe_control_gym.math_and_models.neural_networks import MLP


class APPO_MPC_Agent:
    """A PPO class that encapsulates models, optimizers and update functions."""

    def __init__(
        self,
        env_fun,
        obs_space,
        act_space,
        gamma,
        model,
        hidden_dim=64,
        activation="tanh",
        actor_config=None,
        use_clipped_value=False,
        clip_param=0.2,
        target_kl=0.02,
        entropy_coef=0.002,
        exploration_init=-1.0,
        actor_lr=0.001,
        critic_lr=0.001,
        opt_epochs=10,
        rollout_batch_size=10,
        mini_batch_size=64,
        **kwargs,
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
        self.rollout_batch_size = rollout_batch_size

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
            rollout_batch_size=self.rollout_batch_size,
            mini_batch_size=self.mini_batch_size,
        )

        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), critic_lr)

    def to(self, device):
        """Puts agent to device."""
        self.ac.to(device)

    def train(self):
        """Sets training mode."""
        self.ac.train()

    def eval(self):
        """Sets evaluation mode."""
        self.ac.eval()

    def reset(self, idx=None):
        """Reset function, especially needed for resetting MPC actor"""
        self.ac.reset(idx)

    def state_dict(self):
        """Snapshots agent state."""
        return {
            "ac": self.ac.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        """Restores agent state."""
        self.ac.load_state_dict(state_dict["ac"], strict=strict)
        self.actor_opt.load_state_dict(state_dict["actor_opt"])
        self.critic_opt.load_state_dict(state_dict["critic_opt"])

    def compute_policy_loss(self, batch, theta_old):
        """Returns policy loss(es) given batch of data."""
        obs, act, logp_old, adv = (
            batch["obs"],
            batch["act"],
            batch["logp"],
            batch["adv"],
        )
        mpc_act, nabla_pi_theta, optimal = (
            batch["mpc_act"],
            batch["nabla_pi_theta"],
            batch["optimal"],
        )
        action, dist, logp = self.ac.actor.forward_train(
            obs, act, theta_old, mpc_act, nabla_pi_theta
        )

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
        """Returns value loss(es) given batch of data."""
        obs, ret, v_old = batch["obs"], batch["ret"], batch["v"]
        v_cur = self.ac.critic(obs)
        if self.use_clipped_value:
            v_old_clipped = v_old + (v_cur - v_old).clamp(
                -self.clip_param, self.clip_param
            )
            v_loss = (v_cur - ret).pow(2)
            v_loss_clipped = (v_old_clipped - ret).pow(2)
            value_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def update(self, rollouts, device="cpu"):
        """Updates model parameters based on current training batch."""
        results = defaultdict(list)
        num_mini_batch = (
            rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        )
        # assert if num_mini_batch is 0
        assert num_mini_batch != 0, "num_mini_batch is 0"
        theta_old = self.ac.actor._build_mpc_param().detach().clone()
        for _ in range(self.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            theta_loss_epoch, n_updates = 0, 0
            for batch in rollouts.sampler(self.mini_batch_size, device):
                # Actor update.
                policy_loss, entropy_loss, approx_kl = self.compute_policy_loss(
                    batch, theta_old
                )
                # Update only when no KL constraint or constraint is satisfied.
                if (self.target_kl <= 0) or (
                    self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl
                ):
                    self.actor_opt.zero_grad()
                    (policy_loss + self.entropy_coef * entropy_loss).backward()
                    self.actor_opt.step()
                    with torch.no_grad():
                        self.ac.actor.q_param.clamp_(1e-5, 100.0)
                        self.ac.actor.r_param.clamp_(1e-5, 100.0)
                        self.ac.actor.qt_param.clamp_(1e-5, 100.0)
                        self.ac.actor.model_param.clamp_(1e-5, 100.0)
                        self.ac.actor.log_std.clamp_(
                            self.ac.actor.log_std_min, self.ac.actor.log_std_max
                        )

                    p_loss_epoch += policy_loss.item()
                    e_loss_epoch += entropy_loss.item()
                    kl_epoch += approx_kl.item()
                    n_updates += 1
                else:
                    break

                # Critic update.
                value_loss = self.compute_value_loss(batch)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
                v_loss_epoch += value_loss.item()
            results["policy_loss"].append(p_loss_epoch / max(n_updates, 1))
            results["value_loss"].append(v_loss_epoch / max(n_updates, 1))
            results["entropy_loss"].append(e_loss_epoch / max(n_updates, 1))
            results["approx_kl"].append(kl_epoch / max(n_updates, 1))
            results["theta_loss"].append(theta_loss_epoch / max(n_updates, 1))
        results = {k: sum(v) / len(v) for k, v in results.items()}
        return results


# -----------------------------------------------------------------------------------
#                   Models
# -----------------------------------------------------------------------------------


class MLPActorCritic(nn.Module):
    """Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        critic (MLPCritic): value network.
    """

    def __init__(
        self,
        env,
        obs_space,
        act_space,
        gamma,
        model,
        hidden_dims=(64, 64),
        exploration_init=-1.0,
        activation="tanh",
        actor_config=None,
        rollout_batch_size=10,
        mini_batch_size=64,
    ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            raise Exception(
                "PPO-MPC is currently only implemented for continuous action spaces"
            )
        # Policy.
        self.actor = MPCActor(
            env,
            obs_dim,
            act_dim,
            hidden_dims,
            activation,
            gamma,
            model,
            exploration_init,
            actor_config,
            rollout_batch_size,
            mini_batch_size,
        )
        # Value function.
        self.critic = MLPCritic(obs_dim, hidden_dims, activation)

    def step(self, obs, info=None):
        dist, _, mpc_act, nabla_pi_theta, soln_info, results_dict, optimal_flag = self.actor(
            obs, actor_info=info
        )
        mpc_act = np.array(mpc_act)
        nabla_pi_theta = np.array(nabla_pi_theta)
        a = dist.sample()
        logp_a = dist.log_prob(a)
        v = self.critic(obs)
        return (
            a.cpu().numpy(),
            v.cpu().numpy(),
            logp_a.cpu().numpy(),
            soln_info,
            results_dict,
            mpc_act,
            nabla_pi_theta,
            optimal_flag,
        )

    def act(self, obs, info=None):
        dist, _, _, _, _, _, _ = self.actor(obs, actor_info=info)
        a = dist.mode()
        return a.cpu().numpy()

    def reset(self, idx):
        self.actor.reset(idx)


class MLPCritic(nn.Module):
    """Critic MLP model."""

    def __init__(self, obs_dim, hidden_dims, activation):
        super().__init__()
        self.v_net = MLP(obs_dim, 1, hidden_dims, activation)

    def forward(self, obs):
        return self.v_net(obs)


class MPCActor(nn.Module):
    """Actor MPC model."""

    def __init__(
        self,
        env,
        obs_dim,
        act_dim,
        hidden_dims,
        activation,
        gamma,
        model,
        exploration_init,
        actor_config,
        rollout_batch_size=10,
        mini_batch_size=64,
    ):
        super().__init__()
        # mpc actor
        self.mpc = MPCPolicyFunction(
            env,
            gamma,
            model,
            **actor_config["mpc_config"],
            n_rollout_solver=rollout_batch_size,
            n_train_solver=mini_batch_size,
        )

        # Parameters
        self.q_init = actor_config["q_mpc"]
        self.r_init = actor_config["r_mpc"]
        self.qt_init = actor_config["qt_mpc"]
        self.back_off_init = actor_config["back_off"]
        self.model_init = actor_config["model_param"]
        self.q_param = nn.Parameter(torch.tensor(self.q_init, dtype=torch.float32))
        self.r_param = nn.Parameter(torch.tensor(self.r_init, dtype=torch.float32))
        self.qt_param = nn.Parameter(torch.tensor(self.qt_init, dtype=torch.float32))
        self.model_param = nn.Parameter(
            torch.tensor(self.model_init, dtype=torch.float32)
        )
        # self.back_off_param = nn.Parameter(
        #     torch.tensor(self.back_off_init, dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "back_off_fixed",
        #     torch.tensor(self.back_off_init, dtype=torch.float32)
        # )

        # Construct output action distribution.
        # self.net = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        # self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std = nn.Parameter(exploration_init * torch.ones(act_dim))
        self.dist_fn = lambda x, log_std: Normal(x, log_std.exp())
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs, act=None, actor_info=None):
        theta = self.get_theta_param(obs)
        traj_param = self.get_references(actor_info)
        obs_np = obs.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        if obs.ndim > 1:
            (
                mpc_act,
                nabla_pi_theta,
                soln_info,
                results_dict,
                optimal_flag,
            ) = self.mpc.select_action_batch(
                obs_np, theta_np, traj_param, actor_info
            )
        else:
            mpc_act, soln_info, results_dict, optimal_flag = self.mpc.select_action(
                obs_np, theta_np, traj_param
            )
            nabla_pi_theta = None
        action = torch.as_tensor(
            np.asarray(mpc_act), dtype=theta.dtype, device=theta.device
        )
        # net_out = self.net(obs)
        # log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        dist = self.dist_fn(action, log_std)
        logp_a = None
        if act is not None:
            logp_a = dist.log_prob(act)
        return dist, logp_a, mpc_act, nabla_pi_theta, soln_info, results_dict, optimal_flag

    def forward_train(self, obs, act, theta_old, mpc_act, nabla_pi_theta):
        theta = self.get_theta_param(obs)
        action = mpc_act.unsqueeze(2) + nabla_pi_theta @ (
            theta - theta_old.repeat(obs.shape[0], 1)
        ).unsqueeze(2)
        action = action.squeeze(2)
        # net_out = self.net(obs)
        # log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        dist = self.dist_fn(action, log_std)
        logp_a = dist.log_prob(act)
        return action, dist, logp_a

    def reset(self, idx):
        self.mpc.reset(idx)

    def _build_mpc_param(self):
        return torch.cat(
            [
                self.q_param,
                self.r_param,
                self.qt_param,
                # self.back_off_fixed,
                self.model_param,
            ],
            dim=0,
        )

    def get_theta_param(self, obs):
        theta = self._build_mpc_param()
        if obs.ndim > 1:
            return theta.unsqueeze(0).repeat(obs.shape[0], 1)
        return theta

    def get_references(self, info_batch):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        goal_states_batch = []
        for info in info_batch:
            traj_step = info["current_step"]
            traj_ref = info["x_ref"].T
            if self.mpc.env.TASK == Task.STABILIZATION:
                # Repeat goal state for horizon steps.
                goal_states = np.tile(
                    self.mpc.env.X_GOAL.reshape(-1, 1), (1, self.mpc.T + 1)
                )
            elif self.mpc.env.TASK == Task.TRAJ_TRACKING:
                # Slice trajectory for horizon steps, if not long enough, repeat last state.
                start = min(traj_step, traj_ref.shape[-1])
                end = min(traj_step + self.mpc.T + 1, traj_ref.shape[-1])
                remain = max(0, self.mpc.T + 1 - (end - start))
                goal_states = np.concatenate(
                    [traj_ref[:, start:end], np.tile(traj_ref[:, -1:], (1, remain))], -1
                )
            else:
                raise Exception("Reference for this mode is not implemented.")
            goal_states_batch.append(goal_states)
        return goal_states_batch  # list of (nx, T+1).


class MPCPolicyFunction(MPCFunction):
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
        cs_workers: int = 1,
        jit: bool = False,
        jit_options: dict = None,
        n_rollout_solver: int = 1,
        n_train_solver: int = 1,
    ):
        super().__init__(
            env_fun,
            gamma,
            model,
            horizon=horizon,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            constraint_tol=constraint_tol,
            additional_constraints=additional_constraints,
            jit=jit,
            jit_options=jit_options,
        )
        self.cs_workers = cs_workers
        self.n_parallel_solver = n_rollout_solver
        self.n_train_solver = n_train_solver
        self.infos = [None] * self.n_parallel_solver

        # Parallel solvers
        self.pi_solvers, self.rkkt_norm_fns, self.all_solvers = self.get_parallel_solver(
            self.n_parallel_solver, self.cs_workers
        )

    def reset(self, idx=None):
        super().reset()
        if idx is not None:
            self.infos[idx] = None
        else:
            self.infos = [None] * self.n_parallel_solver

    def select_action_batch(self, obs_batch, theta, traj_ref, actor_info):
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        con_lbg = self.solver_dict["lower_bound"]
        con_ubg = self.solver_dict["upper_bound"]
        opt_vars_fn = self.solver_dict["opt_vars_fn"]
        xus_fn = self.solver_dict["xus_fn"]

        # eval_data_batch = []
        x0, fixed_p, ref_p = [], [], []
        # ref_param = goal_states.T.reshape(-1, 1).repeat(obs_batch.shape[0], 1)
        lbg = con_lbg.full().repeat(obs_batch.shape[0], 1)
        ubg = con_ubg.full().repeat(obs_batch.shape[0], 1)
        for i, obs in enumerate(obs_batch):
            fixed_param = np.zeros((self.model.nx + self.model.nu))
            fixed_param[: self.model.nx] = obs[: self.model.nx]
            ref_param = traj_ref[i].T.reshape(-1, 1)[:, 0]
            opt_vars_init = np.zeros(self.solver_dict["opt_vars"].shape)
            info = actor_info[i].get("soln_info")
            if info is not None:
                opt_vars_init = info["opt_var"]
            elif self.infos[i] is not None:
                opt_vars_init = self.infos[i]["opt_var"]
            else:
                opt_vars_init = np.zeros_like(opt_vars_init)

            if opt_vars_init is not None:
                x_prev, u_prev, sigma_prev, sigma_u0_prev = xus_fn(opt_vars_init)
                x_prev, u_prev, sigma_prev, sigma_u0_prev = (
                    x_prev.full(),
                    u_prev.full(),
                    sigma_prev.full(),
                    sigma_u0_prev.full(),
                )
                opt_vars_init = update_initial_guess(
                    x_prev, u_prev, sigma_prev, sigma_u0_prev, opt_vars_fn
                )

            x0.append(opt_vars_init[:, 0])
            fixed_p.append(fixed_param)
            ref_p.append(ref_param)
        x0, fixed_p, ref_p = np.array(x0).T, np.array(fixed_p).T, np.array(ref_p).T
        p = np.concatenate((fixed_p, ref_p, theta.T), axis=0)

        # Forward pass through solver
        soln_batch = self.pi_solvers(x0=x0, p=p, lbg=lbg, ubg=ubg)
        z = cs.vertcat(soln_batch["x"], soln_batch["lam_g"])
        rkkt_norm_batch, dpidp_batch = self.all_solvers(z, fixed_p, ref_p, theta.T)
        optimal_batch = rkkt_norm_batch.full() < 1e-3
        dpidp_batch = dpidp_batch.full()
        soln_x = soln_batch["x"].full()

        # Post-processing the solution
        action_batch, nabla_pi_theta_batch, info_batch, results_dict_batch = [], [], [], []
        for i, obs in enumerate(obs_batch):
            opt_vars = soln_x[:, i]
            x_val, u_val, sigma_val, sigma_u0_val = xus_fn(opt_vars)
            x_prev = x_val.full()
            u_prev = u_val.full()
            sigma_prev = sigma_val.full()
            sigma_u0_prev = sigma_u0_val.full()
            results_dict = {
                "horizon_states": x_prev.copy(),
                "horizon_inputs": u_prev.copy(),
                "horizon_slacks": sigma_prev.copy(),
                "horizon_u0_slacks": sigma_u0_prev.copy(),
                "goal_states": ref_p[:, i].copy(),
            }
            # results_dict['t_wall'].append(opti.stats()['t_wall_total'])

            # Take the first action from the solved action sequence.
            if u_prev.ndim > 1:
                action = u_prev[:, 0]
            else:
                action = np.array([u_prev[0]])

            # additional info
            info = {
                "success": optimal_batch[0, i],
                "opt_var": opt_vars,
                "action": action.copy(),
                "fixed_param": fixed_p[:, i].copy(),
                "ref_param": ref_p[:, i].copy(),
                "theta_param": theta[i, :].copy(),
                "traj_step": actor_info[i]["current_step"],
                "x_ref": actor_info[i]["x_ref"].copy(),
            }

            action_batch.append(action)
            results_dict_batch.append(results_dict)
            nabla_pi_theta_batch.append(
                int(optimal_batch[0, i])
                * dpidp_batch[
                    :,
                    self.solver_dict["theta_param"].shape[0]
                    * i : self.solver_dict["theta_param"].shape[0]
                    * (i + 1),
                ]
            )
            info_batch.append(info)
        self.infos = [info.copy() for info in info_batch]
        return (
            action_batch,
            nabla_pi_theta_batch,
            info_batch,
            results_dict_batch,
            optimal_batch.T,
        )

    def get_parallel_solver(self, n_solvers, cs_workers):
        requested = min(n_solvers, cs_workers)
        n_workers = requested
        while n_solvers % n_workers != 0:
            n_workers -= 1

        if n_workers != requested:
            print(
                f"[MPC solver] Adjusting CasADi workers from {requested} to {n_workers} "
                f"for {n_solvers} mapped solvers so batches split evenly."
            )

        pi_solvers = self.solver_dict["solver"].map(n_solvers, "thread", n_workers)
        rkkt_norm_solvers = self.pi_sensitivity_dict["rkkt_norm_fn"].map(
            n_solvers, "thread", n_workers
        )
        all_solvers = self.pi_sensitivity_dict["all_fn"].map(
            n_solvers, "thread", n_workers
        )
        return pi_solvers, rkkt_norm_solvers, all_solvers


class APPOBuffer(object):
    """Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    """

    def __init__(self, obs_space, act_space, theta_dim, max_length, batch_size):
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
            "obs": {"vshape": (T, N, *obs_dim)},
            "act": {"vshape": (T, N, act_dim)},
            "rew": {"vshape": (T, N, 1)},
            "mask": {"vshape": (T, N, 1), "init": np.ones},
            "v": {"vshape": (T, N, 1)},
            "logp": {"vshape": (T, N, 1)},
            "ret": {"vshape": (T, N, 1)},
            "adv": {"vshape": (T, N, 1)},
            "terminal_v": {"vshape": (T, N, 1)},
            "mpc_act": {"vshape": (T, N, act_dim)},
            "nabla_pi_theta": {"vshape": (T, N, act_dim, theta_dim)},
            "optimal": {"vshape": (T, N, 1)},
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        """Allocates space for containers."""
        for k, info in self.scheme.items():
            assert "vshape" in info, f"Scheme must define vshape for {k}"
            vshape = info["vshape"]
            dtype = info.get("dtype", np.float32)
            init = info.get("init", np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.t = 0

    def push(self, batch):
        """Inserts transition step data (as dict) to storage."""
        for k, v in batch.items():
            assert k in self.keys
            shape = self.scheme[k]["vshape"][1:]
            dtype = self.scheme[k].get("dtype", np.float32)
            v_ = np.asarray(v, dtype=dtype).reshape(shape)
            self.__dict__[k][self.t] = v_
        self.t = (self.t + 1) % self.max_length

    def get(self, device="cpu"):
        """Returns all data."""
        batch = {}
        for k, info in self.scheme.items():
            shape = info["vshape"][2:]
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = torch.as_tensor(data, device=device)
        return batch

    def sample(self, indices):
        """Returns partial data."""
        batch = {}
        for k, info in self.scheme.items():
            shape = info["vshape"][2:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self, mini_batch_size, device="cpu", drop_last=True):
        """Makes sampler to loop through all data."""
        total_steps = self.max_length * self.batch_size
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
            yield batch


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------


def random_sample(indices, batch_size, drop_last=True):
    """Returns index batches to iterate over."""
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[: len(indices) // batch_size * batch_size].reshape(-1, batch_size)
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
    gae_lambda=0.95,
):
    """Useful for policy-gradient algorithms."""
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
        rets[i] = ret.copy()
        advs[i] = adv.copy()
    return rets, advs
