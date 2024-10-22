'''SQP MPC utility functions.'''

import casadi as cs
import numpy as np
import scipy


def get_cost(r, Q, n_lookahead):
    '''Get the cost function for the SQP MPC.

    Args:
        r (np.array): Actuation cost
        Q (np.array): State cost
        n_lookahead (int): Number of steps to look ahead

    Returns:
        S (np.array): The cost matrix
        Q (np.array): The state cost matrix
        R (np.array): The actuation cost matrix
    
    '''

    I_r = np.eye(r.shape[0])
    I_Q = np.eye(Q.shape[0])

    cost_action = np.kron(I_r, np.eye(n_lookahead))
    cost_state = np.kron(I_Q, np.eye(n_lookahead + 1))
    S = scipy.linalg.block_diag(cost_state, cost_action)

    nx = Q.shape[0]
    nu = r.shape[0]

    assert S.shape[0] == nx * (n_lookahead + 1) + nu * n_lookahead
    assert cost_action.shape[0] == nu * n_lookahead
    assert cost_state.shape[0] == nx * (n_lookahead + 1)
    
    return S, cost_state, cost_action

