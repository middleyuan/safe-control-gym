'''Soft Actor Critic (SAC) with MPC'''

import os
import time
from collections import defaultdict

import numpy as np
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.rlmpc.sac_mpc_utils import SAC_MPC_Agent, SACBuffer
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (RecordEpisodeStatistics,
                                                                          VecRecordEpisodeStatistics)
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_obs, _unflatten_obs
from safe_control_gym.math_and_models.normalization import (BaseNormalizer, MeanStdNormalizer,
                                                            RewardStdNormalizer)
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, is_wrapped, set_random_state


class SAC_MPC(BaseController):
    '''Soft Actor Critic with MPC'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)

        # Task.
        self.env = env_func()
        if self.training:
            # Training and testing.
            self.venv = make_vec_envs(env_func, None, self.rollout_batch_size, self.num_workers, seed)
            self.venv = VecRecordEpisodeStatistics(self.venv, self.deque_size)
            self.eval_venv = env_func(seed=seed * 111)
            self.eval_venv = RecordEpisodeStatistics(self.eval_venv, self.deque_size)
            # self.eval_venv = make_vec_envs(env_func, None, self.eval_batch_size, self.num_workers, seed * 111)
            # self.eval_venv = VecRecordEpisodeStatistics(self.eval_venv, self.deque_size)
        else:
            # Testing only.
            self.env = RecordEpisodeStatistics(self.env)

        # Agent.
        model = self.get_prior(self.env)
        self.agent = SAC_MPC_Agent(
            self.env,
            self.env.observation_space,
            self.env.action_space,
            self.gamma,
            model,
            hidden_dim=self.hidden_dim,
            actor_config=self.actor_config,
            tau=self.tau,
            init_temperature=self.init_temperature,
            use_entropy_tuning=self.use_entropy_tuning,
            target_entropy=self.target_entropy,
            exploration_init=self.exploration_init,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            entropy_lr=self.entropy_lr,
            activation=self.activation,
        )
        self.agent.to(self.device)

        # Pre-/post-processing.
        self.obs_normalizer = BaseNormalizer()
        if self.norm_obs:
            self.obs_normalizer = MeanStdNormalizer(shape=self.venv.observation_space.shape, clip=self.clip_obs,
                                                    epsilon=1e-8)
        self.reward_normalizer = BaseNormalizer()
        if self.norm_reward:
            self.reward_normalizer = RewardStdNormalizer(gamma=self.gamma, clip=self.clip_reward, epsilon=1e-8)

        # Logging.
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # Disable logging to file and tfboard for evaluation.
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def reset(self):
        '''Do initializations for training or evaluation.'''
        self.agent.reset()
        if self.training:
            # set up stats tracking
            self.venv.add_tracker('constraint_violation', 0)
            self.venv.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_venv.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_venv.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, _ = self.venv.reset()
            self.obs = self.obs_normalizer(obs)
            self.agent_info = []
            for env in self.venv.envs:
                self.agent_info.append({'current_step': 0, 'x_ref': env.X_GOAL})
            self.buffer = SACBuffer(self.env.observation_space, self.env.action_space, self.max_buffer_size,
                                    self.train_batch_size)
        else:
            # Add episodic stats to be tracked.
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')

    def reset_before_run(self, obs, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.reset()

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()
        if self.training:
            self.venv.close()
            self.eval_venv.close()
        self.logger.close()

    def save(self, path, save_buffer=False):
        '''Saves model params and experiment state to checkpoint path.'''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        state_dict = {
            'agent': self.agent.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
        }
        if self.training:
            exp_state = {
                'total_steps': self.total_steps,
                'obs': self.obs,
                'random_state': get_random_state(),
                'env_random_state': self.venv.get_env_random_state()
            }
            if save_buffer:
                exp_state['buffer'] = self.buffer.state_dict()
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self, path):
        '''Restores model and experiment given checkpoint path.'''
        state = torch.load(path, weights_only=False)
        # Restore policy.
        self.agent.load_state_dict(state['agent'], strict=False)
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])
        # Restore experiment state.
        if self.training:
            self.total_steps = state['total_steps']
            self.obs = state['obs']
            set_random_state(state['random_state'])
            self.venv.set_env_random_state(state['env_random_state'])
            if 'buffer' in state:
                self.buffer = SACBuffer(self.env.observation_space, self.env.action_space, self.max_buffer_size,
                                        self.train_batch_size)
                self.buffer.load_state_dict(state['buffer'])

            self.logger.load(self.total_steps)

    def learn(self, env=None, **kwargs):
        '''Performs learning (pre-training, training, fine-tuning, etc.).'''
        # Initial Evaluation.
        if self.eval_interval:
            results = defaultdict(list)
            eval_results = self.run(env=self.eval_venv, n_episodes=self.eval_batch_size)
            self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(
                eval_results['ep_lengths'].mean(),
                eval_results['ep_lengths'].std(),
                eval_results['ep_returns'].mean(),
                eval_results['ep_returns'].std()))
            results.update({'step': self.total_steps, 'elapsed_time': 0})
            results['eval'] = eval_results
            self.log_step(results)
        if self.num_checkpoints > 0:
            step_interval = np.linspace(0, self.max_env_steps, self.num_checkpoints)
            interval_save = np.zeros_like(step_interval, dtype=bool)

        while self.total_steps < self.max_env_steps:
            results = self.train_step()

            # Checkpoint.
            if (self.total_steps >= self.max_env_steps
                    or (self.save_interval and self.total_steps % self.save_interval == 0)):
                # Latest/final checkpoint.
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
                path = os.path.join(self.output_dir, 'checkpoints', 'model_{}.pt'.format(self.total_steps))
                self.save(path)
            if self.num_checkpoints > 0:
                interval_id = np.argmin(np.abs(np.array(step_interval) - self.total_steps))
                if interval_save[interval_id] is False:
                    # Intermediate checkpoint.
                    path = os.path.join(self.output_dir, 'checkpoints', f'model_{self.total_steps}.pt')
                    self.save(path)
                    interval_save[interval_id] = True

            # Evaluation.
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_venv, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(
                    eval_results['ep_lengths'].mean(),
                    eval_results['ep_lengths'].std(),
                    eval_results['ep_returns'].mean(),
                    eval_results['ep_returns'].std()))
                # Save best model.
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.inf)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))

            # Logging.
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        with torch.inference_mode():
            action = self.agent.ac.act(obs, info=info)
        return action

    def train_step(self):
        '''Performs a training/fine-tuning step.'''
        self.agent.reset()
        self.agent.train()
        self.obs_normalizer.unset_read_only()

        obs = self.obs
        start = time.time()
        with torch.inference_mode():
            action, _, soln_info, results_dict, optimal = self.agent.ac.step(
                torch.FloatTensor(obs).to(self.device), info=self.agent_info
            )
        next_obs, rew, done, info = self.venv.step(action)
        next_obs = self.obs_normalizer(next_obs)
        rew = self.reward_normalizer(rew, done)
        mask = 1 - np.asarray(done)

        # time truncation is not true termination
        terminal_idx, terminal_obs = [], []
        for idx, inf in enumerate(info['n']):
            self.agent_info[idx] = {'current_step': inf['current_step'], 'x_ref': self.venv.envs[idx].X_GOAL}
            if 'terminal_info' not in inf:
                continue
            inff = inf['terminal_info']
            if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
                terminal_idx.append(idx)
                terminal_obs.append(inf['terminal_observation'])
            elif inff['out_of_bounds']:
                terminal_idx.append(idx)
                terminal_obs.append(inf['terminal_observation'])
        if len(terminal_obs) > 0:
            terminal_obs = _unflatten_obs(self.obs_normalizer(_flatten_obs(terminal_obs)))

        # collect the true next states and masks (accounting for time truncation and out of bounds)
        true_next_obs = _unflatten_obs(next_obs)
        true_mask = mask.copy()
        for idx, term_ob in zip(terminal_idx, terminal_obs):
            true_next_obs[idx] = term_ob
            true_mask[idx] = 1.0
        true_next_obs = _flatten_obs(true_next_obs)

        self.buffer.push({
            'obs': obs, 'act': action, 'rew': rew, 'next_obs': true_next_obs,
            'mask': true_mask, 'info': soln_info, 'results_dict': results_dict, 'optimal': optimal
        })
        obs = next_obs

        self.obs = obs
        self.total_steps += self.rollout_batch_size

        # learn
        results = defaultdict(list)
        if self.total_steps > self.warm_up_steps and not self.total_steps % self.train_interval:
            # Regardless of how long you wait between updates,
            # the ratio of env steps to gradient steps is locked to 1.
            # alternatively, can update once each step
            for _ in range(self.train_interval):
                batch, batch_th = self.buffer.sample(self.train_batch_size, self.device)
                res = self.agent.update(batch, batch_th)
                for k, v in res.items():
                    results[k].append(v)
        results = {k: sum(v) / len(v) for k, v in results.items()}
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def run(self, env=None, render=False, n_episodes=1, verbose=False):
        '''Runs evaluation with current policy.'''
        self.agent.reset()
        self.agent.eval()
        self.obs_normalizer.set_read_only()
        if env is None:
            env = self.venv
        else:
            if not is_wrapped(env, RecordEpisodeStatistics) or is_wrapped(env, VecRecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add episodic stats to be tracked.
                env.add_tracker('constraint_violation', 0, mode='queue')
                env.add_tracker('constraint_values', 0, mode='queue')
                env.add_tracker('mse', 0, mode='queue')

        obs, env_info = env.reset()
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        agent_info = [{'current_step': 0, 'x_ref': env.X_GOAL}]
        mse, ep_rmse = [], []
        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=agent_info)
            obs, _, done, info = env.step(action)
            mse.append(info['mse'])
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            if verbose:
                print(f'obs {obs} | act {action}')
            if done:
                assert 'episode' in info
                ep_rmse.append(np.array(mse).mean() ** 0.5)
                mse = []
                ep_returns.append(info['episode']['r'])
                ep_lengths.append(info['episode']['l'])
                obs, env_info = env.reset()
                info['current_step'] = 0
                self.agent.reset()
            obs = self.obs_normalizer(obs)
            agent_info[0] = {'current_step': info['current_step'], 'x_ref': env.X_GOAL}
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths,
                        'rmse': np.array(ep_rmse).mean(),
                        'rmse_std': np.array(ep_rmse).std()}
        if len(frames) > 0:
            eval_results['frames'] = frames
        # Other episodic stats from evaluation env.
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def log_step(self, results):
        '''Does logging after a training step.'''
        step = results['step']
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'step_time': results['elapsed_time'],
                'progress': step / self.max_env_steps
            },
            step,
            prefix='time')
        # Learning stats.
        if 'train' in results:
            self.logger.add_scalars(
                {
                    k: results['train'][k]
                    for k in ['policy_loss', 'value_loss', 'entropy_loss', 'approx_kl', 'theta_loss', 'ref_loss']
                },
                step,
                prefix='loss')
            # Performance stats.
            ep_lengths = np.asarray(self.venv.length_queue)
            ep_returns = np.asarray(self.venv.return_queue)
            ep_constraint_violation = np.asarray(self.venv.queued_stats['constraint_violation'])
            self.logger.add_scalars(
                {
                    'ep_length': ep_lengths.mean(),
                    'ep_return': ep_returns.mean(),
                    'ep_return_std': ep_returns.std(),
                    'ep_reward': (ep_returns / ep_lengths).mean(),
                    'ep_constraint_violation': ep_constraint_violation.mean()
                },
                step,
                prefix='stat')
            # Total constraint violation during learning.
            total_violations = self.venv.accumulated_stats['constraint_violation']
            self.logger.add_scalars({'constraint_violation': total_violations}, step, prefix='stat')
        if 'eval' in results:
            eval_ep_lengths = results['eval']['ep_lengths']
            eval_ep_returns = results['eval']['ep_returns']
            eval_constraint_violation = results['eval']['constraint_violation']
            eval_rmse = results['eval']['rmse']
            eval_rmse_std = results['eval']['rmse_std']
            self.logger.add_scalars(
                {
                    'ep_length': eval_ep_lengths.mean(),
                    'ep_return': eval_ep_returns.mean(),
                    'ep_return_std': eval_ep_returns.std(),
                    'ep_reward': (eval_ep_returns / eval_ep_lengths).mean(),
                    'constraint_violation': eval_constraint_violation.mean(),
                    'rmse': eval_rmse,
                    'rmse_std': eval_rmse_std
                },
                step,
                prefix='stat_eval')
        # Print summary table
        self.logger.dump_scalars()
        print(self.agent.ac.actor.mpc_param.detach().numpy())
        print(self.agent.ac.actor.logstd.detach().numpy())
