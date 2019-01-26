#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT


def perron_frobenius_2d(transitions, states, simulations):
    """TT approximation of the Perron-Frobenius operator in 2D

    Given transitions of particles in a 2-dimensional potential, compute the approximation of the corresponding Perron-
    Frobenius operator in TT format. See [1]_ for details.

    Parameters
    ----------
    transitions: ndarray
        matrix containing the transitions, each row is of the form [x_1, x_2, y_1, y_2] representing a transition from
        state (x_1, x_2) to (y_1, y_2)
    states: list of ints
        number of states in x- and y-direction
    simulations: int
        number of simulations per state

    Returns
    -------
    operator: instance of TT class
        TT approximation of the Perron-Frobenius operator

     References
    ----------
    .. [1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
           Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
    """

    # find unique indices for transitions in the first dimension
    flat_indices = np.ravel_multi_index(transitions[[0, 2], :] - 1, (states[0], states[0]))
    [ind_unique, ind_inv] = np.unique(flat_indices, return_inverse=True)
    ind_unique = np.array(np.unravel_index(ind_unique, (states[0], states[0])))
    rank = ind_unique.shape[1]

    # construct core for the first dimension
    cores = [np.zeros([1, states[0], states[0], rank])]
    for i in range(rank):
        cores[0][0, ind_unique[0, i], ind_unique[1, i], i] = 1

    # construct core for the second dimension
    cores.append(np.zeros([rank, states[1], states[1], 1]))
    for i in range(transitions.shape[1]):
        cores[1][ind_inv[i], transitions[1, i] - 1, transitions[3, i] - 1, 0] += 1

    # transpose and normalize operator
    operator = (np.true_divide(1, simulations)) * TT(cores).transpose()

    return operator


def perron_frobenius_3d(transitions, states, simulations):
    """TT approximation of the Perron-Frobenius operator in 3D

    Given transitions of particles in a 3-dimensional potential, compute the approximation of the corresponding Perron-
    Frobenius operator in TT format. See [1]_ for details.

    Parameters
    ----------
    transitions: ndarray
        matrix containing the transitions, each row is of the form [x_1, x_2, x_3, y_1, y_2, y_3] representing a
        transition from state (x_1, x_2, x_3) to (y_1, y_2, y_3)
    states: list of ints
        number of states in x-, y-, and z-direction
    simulations: int
        number of simulations per state

    Returns
    -------
    operator: instance of TT class
        TT approximation of the Perron-Frobenius operator

     References
    ----------
    .. [1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
           Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
    """

    # find unique indices for transitions in the first dimension
    flat_indices = np.ravel_multi_index(transitions[[0, 3], :] - 1, (states[0], states[0]))
    [ind_1_unique, ind_1_inv] = np.unique(flat_indices, return_inverse=True)
    ind_1_unique = np.array(np.unravel_index(ind_1_unique, (states[0], states[0])))
    rank_1 = ind_1_unique.shape[1]

    # construct core for the first dimension
    cores = [np.zeros([1, states[0], states[0], rank_1])]
    for i in range(rank_1):
        cores[0][0, ind_1_unique[0, i], ind_1_unique[1, i], i] = 1

    # find unique indices for transitions in the third dimension
    flat_indices = np.ravel_multi_index(transitions[[2, 5], :] - 1, (states[2], states[2]))
    [ind_2_unique, ind_2_inv] = np.unique(flat_indices, return_inverse=True)
    ind_2_unique = np.array(np.unravel_index(ind_2_unique, (states[2], states[2])))
    rank_2 = ind_2_unique.shape[1]

    # list entry for the second core
    cores.append(None)

    # construct core for the third dimension
    cores.append(np.zeros([rank_2, states[2], states[2], 1]))
    for i in range(rank_2):
        cores[2][i, ind_2_unique[0, i], ind_2_unique[1, i], 0] = 1

    # construct core for the second dimension
    cores[1] = np.zeros([rank_1, states[1], states[1], rank_2])
    for i in range(transitions.shape[1]):
        cores[1][ind_1_inv[i], transitions[1, i] - 1, transitions[4, i] - 1, ind_2_inv[i]] += 1

    # transpose and normalize operator
    operator = (np.true_divide(1, simulations)) * tt.TT(cores).transpose()

    return operator
