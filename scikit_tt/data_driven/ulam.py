# -*- coding: utf-8 -*-


from __future__ import division
import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT


def ulam_2d(transitions, states, simulations):
    """
    TT approximation of the Perron-Frobenius operator in 2D.

    Given transitions of particles in a 2-dimensional potential, compute the approximation of the corresponding Perron-
    Frobenius operator in TT format. See [1]_ for details.

    Parameters
    ----------
    transitions : np.ndarray
        matrix containing the transitions, each row is of the form [x_1, x_2, y_1, y_2] representing a transition from
        state (x_1, x_2) to (y_1, y_2)
    states : list[int]
        number of states in x- and y-direction
    simulations : int
        number of simulations per state

    Returns
    -------
    TT
        TT approximation of the Perron-Frobenius operator

     References
    ----------
    .. [1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
           Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
    """

    # find unique indices for transitions in the first dimension
    [ind_unique, ind_inv] = np.unique(transitions[[0, 2], :], axis=1, return_inverse=True)
    rank = ind_unique.shape[1]

    # construct core for the first dimension
    cores = [np.zeros([1, states[0], states[0], rank])]
    for i in range(rank):
        cores[0][0, ind_unique[0, i] - 1, ind_unique[1, i] - 1, i] = 1

    # construct core for the second dimension
    cores.append(np.zeros([rank, states[1], states[1], 1]))
    for i in range(transitions.shape[1]):
        cores[1][ind_inv[i], transitions[1, i] - 1, transitions[3, i] - 1, 0] += 1

    # transpose and normalize operator
    operator = (1 / simulations) * TT(cores).transpose()

    return operator


def ulam_3d(transitions, states, simulations):
    """
    TT approximation of the Perron-Frobenius operator in 3D.

    Given transitions of particles in a 3-dimensional potential, compute the approximation of the corresponding Perron-
    Frobenius operator in TT format. See [1]_ for details.

    Parameters
    ----------
    transitions : np.ndarray
        matrix containing the transitions, each row is of the form [x_1, x_2, x_3, y_1, y_2, y_3] representing a
        transition from state (x_1, x_2, x_3) to (y_1, y_2, y_3)
    states : list[int]
        number of states in x-, y-, and z-direction
    simulations : int
        number of simulations per state

    Returns
    -------
    TT
        TT approximation of the Perron-Frobenius operator

     References
    ----------
    .. [1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
           Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
    """

    # find unique indices for transitions in the first dimension
    [ind_1_unique, ind_1_inv] = np.unique(transitions[[0, 3], :], axis=1, return_inverse=True)
    rank_1 = ind_1_unique.shape[1]

    # construct core for the first dimension
    cores = [np.zeros([1, states[0], states[0], rank_1])]
    for i in range(rank_1):
        cores[0][0, ind_1_unique[0, i] - 1, ind_1_unique[1, i] - 1, i] = 1

    # find unique indices for transitions in the third dimension
    [ind_2_unique, ind_2_inv] = np.unique(transitions[[2, 5], :], axis=1, return_inverse=True)
    rank_2 = ind_2_unique.shape[1]

    # list entry for the second core
    cores.append(None)

    # construct core for the third dimension
    cores.append(np.zeros([rank_2, states[2], states[2], 1]))
    for i in range(rank_2):
        cores[2][i, ind_2_unique[0, i] - 1, ind_2_unique[1, i] - 1, 0] = 1

    # construct core for the second dimension
    cores[1] = np.zeros([rank_1, states[1], states[1], rank_2])
    for i in range(transitions.shape[1]):
        cores[1][ind_1_inv[i], transitions[1, i] - 1, transitions[4, i] - 1, ind_2_inv[i]] += 1

    # transpose and normalize operator
    operator = (1 / simulations) * tt.TT(cores).transpose()

    return operator
