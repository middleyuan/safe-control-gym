'''Register controllers.'''

from safe_control_gym.utils.registration import register

register(idx='lqr',
         entry_point='safe_control_gym.controllers.lqr.lqr:LQR',
         config_entry_point='safe_control_gym.controllers.lqr:lqr.yaml')

register(idx='ilqr',
         entry_point='safe_control_gym.controllers.lqr.ilqr:iLQR',
         config_entry_point='safe_control_gym.controllers.lqr:ilqr.yaml')

register(idx='mpc',
         entry_point='safe_control_gym.controllers.mpc.mpc:MPC',
         config_entry_point='safe_control_gym.controllers.mpc:mpc.yaml')

register(idx='linear_mpc',
         entry_point='safe_control_gym.controllers.mpc.linear_mpc:LinearMPC',
         config_entry_point='safe_control_gym.controllers.mpc:linear_mpc.yaml')

register(idx='gp_mpc',
         entry_point='safe_control_gym.controllers.mpc.gp_mpc:GPMPC',
         config_entry_point='safe_control_gym.controllers.mpc:gp_mpc.yaml')

register(idx='pid',
         entry_point='safe_control_gym.controllers.pid.pid:PID',
         config_entry_point='safe_control_gym.controllers.pid:pid.yaml')

register(idx='ppo',
         entry_point='safe_control_gym.controllers.ppo.ppo:PPO',
         config_entry_point='safe_control_gym.controllers.ppo:ppo.yaml')

register(idx='sac',
         entry_point='safe_control_gym.controllers.sac.sac:SAC',
         config_entry_point='safe_control_gym.controllers.sac:sac.yaml')

register(idx='td3',
         entry_point='safe_control_gym.controllers.td3.td3:TD3',
         config_entry_point='safe_control_gym.controllers.td3:td3.yaml')

register(idx='ddpg',
         entry_point='safe_control_gym.controllers.ddpg.ddpg:DDPG',
         config_entry_point='safe_control_gym.controllers.ddpg:ddpg.yaml')

register(idx='safe_explorer_ppo',
         entry_point='safe_control_gym.controllers.safe_explorer.safe_ppo:SafeExplorerPPO',
         config_entry_point='safe_control_gym.controllers.safe_explorer:safe_ppo.yaml')

register(idx='dppo',
         entry_point='safe_control_gym.controllers.dppo.dppo:DPPO',
         config_entry_point='safe_control_gym.controllers.dppo:dppo.yaml')

register(idx='rarl',
         entry_point='safe_control_gym.controllers.rarl.rarl:RARL',
         config_entry_point='safe_control_gym.controllers.rarl:rarl.yaml')

register(idx='rap',
         entry_point='safe_control_gym.controllers.rarl.rap:RAP',
         config_entry_point='safe_control_gym.controllers.rarl:rap.yaml')

register(idx='sqp_mpc',
         entry_point='safe_control_gym.controllers.mpc.sqp_mpc:SQPMPC',
         config_entry_point='safe_control_gym.controllers.mpc:sqp_mpc.yaml')

register(idx='sqp_gp_mpc',
            entry_point='safe_control_gym.controllers.mpc.sqp_gp_mpc:SQPGPMPC',
            config_entry_point='safe_control_gym.controllers.mpc:sqp_gp_mpc.yaml')

register(idx='mpc_acados',
            entry_point='safe_control_gym.controllers.mpc.mpc_acados:MPC_ACADOS',
            config_entry_point='safe_control_gym.controllers.mpc:mpc_acados.yaml')

register(idx='gpmpc_acados',
            entry_point='safe_control_gym.controllers.mpc.gpmpc_acados:GPMPC_ACADOS',
            config_entry_point='safe_control_gym.controllers.mpc:gpmpc_acados.yaml')

register(idx='lqr_c',
            entry_point='safe_control_gym.controllers.lqr.lqr_c:LQR_C',
            config_entry_point='safe_control_gym.controllers.lqr:lqr_c.yaml')

register(idx='gpmpc_casadi',
         entry_point='safe_control_gym.controllers.mpc.gpmpc_casadi:GPMPC_CASADI',
         config_entry_point='safe_control_gym.controllers.mpc:gpmpc_casadi.yaml')

register(idx='gpmpc_acados_TP',
            entry_point='safe_control_gym.controllers.mpc.gpmpc_acados_TP:GPMPC_ACADOS_TP',
            config_entry_point='safe_control_gym.controllers.mpc:gpmpc_acados_TP.yaml')

register(idx='linear_mpc_acados',
            entry_point='safe_control_gym.controllers.mpc.linear_mpc_acados:LinearMPC_ACADOS',
            config_entry_point='safe_control_gym.controllers.mpc:linear_mpc_acados.yaml')

register(idx='gpmpc_acados_TRPY',
            entry_point='safe_control_gym.controllers.mpc.gpmpc_acados_TRPY:GPMPC_ACADOS_TRPY',
            config_entry_point='safe_control_gym.controllers.mpc:gpmpc_acados_TRPY.yaml')

register(idx='fmpc',
         entry_point='safe_control_gym.controllers.mpc.fmpc:FlatMPC',
         config_entry_point='safe_control_gym.controllers.mpc:fmpc.yaml')

register(idx='ppo_mpc',
         entry_point='safe_control_gym.controllers.rlmpc.ppo_mpc:PPO_MPC',
         config_entry_point='safe_control_gym.controllers.rlmpc:ppo_mpc.yaml')

register(idx='appo_mpc',
         entry_point='safe_control_gym.controllers.rlmpc.appo_mpc:APPO_MPC',
         config_entry_point='safe_control_gym.controllers.rlmpc:appo_mpc.yaml')

register(idx='sac_mpc',
         entry_point='safe_control_gym.controllers.rlmpc.sac_mpc:SAC_MPC',
         config_entry_point='safe_control_gym.controllers.rlmpc:sac_mpc.yaml')
