'''PID control class for Crazyflies.

Based on work conducted at UTIAS' DSL by SiQi Zhou and James Xu.
'''

import math
import os
import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.benchmark_env import Environment, Task
from line_profiler import profile

def cross_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''Computes the cross product of two 3D vectors.
    Args:
        a (ndarray): The first 3D vector.
        b (ndarray): The second 3D vector.
    Returns:
        ndarray: The cross product of the two vectors.
    '''
    assert a.shape == b.shape, "Input arrays must have the same shape"
    assert a.shape == (3,), "Input arrays must be 1D arrays of length 3"
    skew_symmetric = np.array([[0, -a[2], a[1]],
                                [a[2], 0, -a[0]],
                                [-a[1], a[0], 0]])
    return np.dot(skew_symmetric, b)

def wrap2pi_vec(angle_vec: np.ndarray) -> np.ndarray:
    '''Wraps a vector of angles between (-pi, pi].

    Args:
        angle_vec (ndarray): A vector of angles.
    '''
    for k, angle in enumerate(angle_vec):
        while angle > np.pi:
            angle -= 2*np.pi
        while angle <= -np.pi:
            angle += 2*np.pi
        angle_vec[k] = angle
    return angle_vec

@profile
def rot2eul(R: np.ndarray) -> np.ndarray:
    '''Convert rotation matrix to euler angles.
    Args:
        R (ndarray): The rotation matrix.
    Returns:
        ndarray: The euler angles.
    
    The rotation matrix is assumed to be follow the intrinsic rotation in the order of X -> Y -> Z
    Gimbal lock is handled by the setting the third angle to zero.
    '''
    EPSILON = 1e-7
    # assert np.linalg.norm(np.dot(R, R.transpose()) - np.eye(3)) < 1e-5, "Input must be a valid rotation matrix"
    beta = np.arctan2(R[0, 2], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    safe1 = np.abs(beta) >= EPSILON
    safe2 = np.abs(beta - np.pi) >= EPSILON
    safe = safe1 and safe2
    if safe:
        gamma = np.arctan2(-R[0, 1]/np.cos(beta), R[0, 0]/np.cos(beta))
        alpha = np.arctan2(-R[1, 2]/np.cos(beta), R[2, 2]/np.cos(beta))
    else:
        gamma = 0
        alpha = np.arctan2(R[2, 1], R[1, 1])
        print('Warning: Gimbal lock detected. Setting gamma to 0.')

    return np.array((alpha, beta, gamma))



class PID(BaseController):
    '''PID Class.'''

    def __init__(self,
                 env_func=None,
                 g: float = 9.8,
                 kf: float = 3.16e-10,
                 km: float = 7.94e-12,
                 p_coeff_for=np.array([.4, .4, 1.25]),
                 i_coeff_for=np.array([.05, .05, .05]),
                 d_coeff_for=np.array([.2, .2, .5]),
                 p_coeff_tor=np.array([70000., 70000., 60000.]),
                 i_coeff_tor=np.array([.0, .0, 500.]),
                 d_coeff_tor=np.array([20000., 20000., 12000.]),
                 pwm2rpm_scale: float = 0.2685,
                 pwm2rpm_const: float = 4070.3,
                 min_pwm: float = 20000,
                 max_pwm: float = 65535,
                 **kwargs
                 ):
        '''Common control classes __init__ method.

        Args:
            g (float, optional): The gravitational acceleration in m/s^2.
            kf (float, optional): Thrust coefficient.
            km (float, optional): Torque coefficient.
            p_coeff_for (ndarray, optional): Position proportional coefficients.
            i_coeff_for (ndarray, optional): Position integral coefficients.
            d_coeff_for (ndarray, optional): Position derivative coefficients.
            p_coeff_tor (ndarray, optional): Attitude proportional coefficients.
            i_coeff_tor (ndarray, optional): Attitude integral coefficients.
            d_coeff_tor (ndarray, optional): Attitude derivative coefficients.
            pwm2rpm_scale (float, optional): PWM-to-RPM scale factor.
            pwm2rpm_const (float, optional): PWM-to-RPM constant factor.
            min_pwm (float, optional): Minimum PWM.
            max_pwm (float, optional): Maximum PWM.
        '''

        super().__init__(env_func, **kwargs)

        self.env = env_func()

        if self.env.NAME != Environment.QUADROTOR:
            raise NotImplementedError('[ERROR] PID not implemented for any system other than Quadrotor (2D and 3D).')

        self.env.reset()
        self.g = g
        self.KF = kf
        self.KM = km
        self.P_COEFF_FOR = np.array(p_coeff_for)
        self.I_COEFF_FOR = np.array(i_coeff_for)
        self.D_COEFF_FOR = np.array(d_coeff_for)
        self.P_COEFF_TOR = np.array(p_coeff_tor)
        self.I_COEFF_TOR = np.array(i_coeff_tor)
        self.D_COEFF_TOR = np.array(d_coeff_tor)
        self.PWM2RPM_SCALE = np.array(pwm2rpm_scale)
        self.PWM2RPM_CONST = np.array(pwm2rpm_const)
        self.MIN_PWM = np.array(min_pwm)
        self.MAX_PWM = np.array(max_pwm)
        self.MIXER_MATRIX = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])

        self.control_timestep = self.env.CTRL_TIMESTEP
        self.reference = self.env.X_GOAL
        self.last_action = None

        self.reset()

    @profile
    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        step = self.extract_step(info)

        # Step the environment and print all returned information.
        if self.env.QUAD_TYPE in [2, 4]:
            cur_pos = np.array([obs[0], 0, obs[2]])
            cur_quat = np.array(p.getQuaternionFromEuler([0, obs[4], 0]))
            cur_vel = np.array([obs[1], 0, obs[3]])
        elif self.env.QUAD_TYPE in [3, 6, 8]:
            cur_pos = np.array([obs[0], obs[2], obs[4]])
            cur_quat = np.array(p.getQuaternionFromEuler([obs[6], obs[7], obs[8]]))
            cur_vel = np.array([obs[1], obs[3], obs[5]])

        if self.env.QUAD_TYPE in [2, 4]:
            if self.env.TASK == Task.TRAJ_TRACKING:
                target_pos = np.array([self.reference[step, 0],
                                       0,
                                       self.reference[step, 2]])
                target_vel = np.array([self.reference[step, 1],
                                       0,
                                       self.reference[step, 3]])
            elif self.env.TASK == Task.STABILIZATION:
                target_pos = np.array([self.reference[0], 0, self.reference[2]])
                target_vel = np.array([0, 0, 0])
        elif self.env.QUAD_TYPE in [3, 6, 8]:
            if self.env.TASK == Task.TRAJ_TRACKING:
                target_pos = np.array([self.reference[step, 0],
                                       self.reference[step, 2],
                                       self.reference[step, 4]])
                target_vel = np.array([self.reference[step, 1],
                                       self.reference[step, 3],
                                       self.reference[step, 5]])
            elif self.env.TASK == Task.STABILIZATION:
                target_pos = np.array([self.reference[0], self.reference[2], self.reference[4]])
                target_vel = np.array([0, 0, 0])

        target_rpy = np.zeros(3)
        target_rpy_rates = np.zeros(3)

        # Compute the next action.
        time_before = time.perf_counter()
        try:
            thrust, computed_target_rpy, _ = self._dslPIDPositionControl(cur_pos,
                                                                        cur_quat,
                                                                        cur_vel,
                                                                        target_pos,
                                                                        target_rpy,
                                                                        target_vel
                                                                        )
            if self.env.QUAD_TYPE in [4, 6, 8]:
                if self.env.QUAD_TYPE == 4:  # 2D quadrotor with attitude control
                    action = np.array([self.env.attitude_control.pwm2thrust(thrust/3)*4, computed_target_rpy[1]])
                
                elif self.env.QUAD_TYPE == 6:  # 3D quadrotor with attitude control
                    action = np.array([self.env.attitude_control.pwm2thrust(thrust/3)*4,
                                    computed_target_rpy[0],
                                    computed_target_rpy[1],
                                    computed_target_rpy[2]])
                elif self.env.QUAD_TYPE == 8:  # 3D quadrotor with attitude control
                    action = np.array([self.env.attitude_control.pwm2thrust(thrust/3)*4,
                                    computed_target_rpy[0],
                                    computed_target_rpy[1],])
                self.last_action = action
                time_after = time.perf_counter()
                self.results_dict['inference_time'].append(time_after - time_before)
                return action
            
            rpm = self._dslPIDAttitudeControl(thrust,
                                            cur_quat,
                                            computed_target_rpy,
                                            target_rpy_rates
                                            )
        except ValueError as e:
            print(e)
            print('Error in Control._dslPIDPositionControl() or Control._dslPIDAttitudeControl()')
            print('Returning last action')
            rpm = self.last_action
        time_after = time.perf_counter()
        self.results_dict['inference_time'].append(time_after - time_before)
        action = rpm
        action = self.KF * action**2
        if self.env.QUAD_TYPE == 2:
            action = np.array([action[0] + action[3], action[1] + action[2]])
        self.last_action = action
        return action

    @profile
    def _dslPIDPositionControl(self,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        '''DSL's CF2.x PID position control.

        Args:
            cur_pos (ndarray): (3,1)-shaped array of floats containing the current position.
            cur_quat (ndarray): (4,1)-shaped array of floats containing the current orientation as a quaternion.
            cur_vel (ndarray): (3,1)-shaped array of floats containing the current velocity.
            target_pos (ndarray): (3,1)-shaped array of floats containing the desired position.
            target_rpy (ndarray): (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
            target_vel (ndarray): (3,1)-shaped array of floats containing the desired velocity.

        Returns:
            thrust (float): The target thrust along the drone z-axis.
            target_euler (ndarray): (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
            pos_e (float): The current position error.
        '''

        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e * self.control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)

        # PID target thrust.
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
            + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
            + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        # deno = np.linalg.norm(np.cross(target_z_ax, target_x_c)) 
        # target_y_ax = np.cross(target_z_ax, target_x_c) / deno
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_y_ax_cross = cross_3d(target_z_ax, target_x_ax) / np.linalg.norm(cross_3d(target_z_ax, target_x_ax))
        target_x_ax_cross = cross_3d(target_y_ax, target_z_ax)
        print('target_z_ax:', target_z_ax)
        print('target_y_ax:', target_y_ax)
        print('target_y_ax_cross:', target_y_ax_cross)
        print('target_x_ax:', target_x_ax)
        print('target_x_ax_cross:', target_x_ax_cross)
        print('')
        assert np.isclose(target_y_ax_cross, target_y_ax).all(), \
            'target_y_ax_cross and target_y_ax are not the same'
        assert np.isclose(target_x_ax_cross, target_x_ax).all(), \
            'target_x_ax_cross and target_x_ax are not the same'

        # NOTE: a proper rotation matrix by definition
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()

        # Target rotation.
        # NOTE: intrinsic rotation (around the body frame)
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        eul = rot2eul(target_rotation)

        # wrap angles to (-pi, pi]
        eul = wrap2pi_vec(eul)
        target_euler = wrap2pi_vec(target_euler)
        print('target euler angles:', target_euler)
        print('euler angles:', eul)
        print('')

        assert np.isclose(eul, target_euler).all(), \
            'target rotation and target euler angles are not the same'
        target_euler = eul

        if np.any(np.abs(target_euler) > math.pi):
            raise ValueError('\n[ERROR] ctrl it', self.control_counter, 'in Control._dslPIDPositionControl(), values outside range [-pi,pi]')

        return thrust, target_euler, pos_e

    def _dslPIDAttitudeControl(self,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        '''DSL's CF2.x PID attitude control.

        Args:
            thrust (float): The target thrust along the drone z-axis.
            cur_quat (ndarray): (4,1)-shaped array of floats containing the current orientation as a quaternion.
            target_euler (ndarray): (3,1)-shaped array of floats containing the computed target Euler angles.
            target_rpy_rates (ndarray): (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns:
            rpm (ndarray): (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        '''

        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(), target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / self.control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * self.control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)

        # PID target torques.
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
            + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
            + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    def reset(self):
        '''Resets the control classes. The previous step's and integral
           errors for both position and attitude are set to zero.
        '''
        self.model = self.get_prior(self.env, self.prior_info)
        self.GRAVITY = self.g * self.model.quad_mass  # The gravitational force (g*M) acting on each drone.
        self.env.reset()
        self.reset_before_run()

    def reset_before_run(self, obs=None, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        # Clear PID control variables.
        self.integral_pos_e = np.zeros(3)
        self.last_rpy = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.setup_results_dict()

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()

    def save(self, path):
        '''Saves integral errors to checkpoint path.

        Args:
            path (str): The path where to the saved integral errors.
        '''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        np.save(path, (self.integral_pos_e, self.last_rpy, self.integral_rpy_e))

    def load(self, path):
        '''Restores integral errors given checkpoint path.

        Args:
            path (str): The path where the integral errors are saved.
        '''
        self.integral_pos_e, self.last_rpy, self.integral_rpy_e = np.load(path)

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {
            'inference_time': []
            }