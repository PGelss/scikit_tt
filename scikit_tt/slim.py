#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scikit_tt.tensor_train import TT
import scikit_tt.subfunctions as sf


def slim_mme(state_space, single_cell_reactions, two_cell_reactions, threshold=10 ** -14):
    """SLIM decomposition for Markov generators

    Construct a tensor-train decomposition of a Markov generator representing a nearest-neighbor interaction system
    (NNIS) described by a list of single-cell and two-cell reactions. The output is a TT operator of the form

                                       -                     -         -                         -   -       -
             -                     -   | I[1]  0    0    0   |         | I[-2]   0     0     0   |   | I[-1] |
             | S[0] L[0] I[0] M[0] | x | M[1]  0    0    0   | x ... x | M[-2]   0     0     0   | x | M[-1] | ,
             -                     -   | S[1] L[1] I[1]  0   |         | S[-2] L[-2] I[-2]   0   |   | S[-1] |
                                       |  0    0    0   J[1] |         |   0     0     0   J[-2] |   | L[-1] |
                                       -                     -         -                         -   -       -

    where S[i] represents the singel-cell reactions on cell i and L[i] and M[i+1] the two-cell reactions between cell i
    and cell i+1. I[i] denotes an identity matrix.

    Note that the implementation of the SLIM algorithm differs from that described in [1]_ and [2]_ in order to simplify
    the parameter passing. However, the resulting operator corresponds to the described decompositions in [1]_ and [2]_.

    Parameters
    ----------
    state_space: list of ints
        number of states of each cell in the NNIS
    single_cell_reactions: list of lists of lists of ints and floats
        list of considered single-cell reactions, i.e. single_cell_reactions[i][j] is a list of the form
        [reactant_state, product_state, reactions_rate] describing the jth reaction on the ith cell.
    two_cell_reactions: list of lists of lists of ints and floats
        list of considered two-cell reactions, i.e. two_cell_reactions[i][j] is a list of the form
        [reactant_state_i, product_state_i, reactant_state_i+1, product_state_i+1, reactions_rate] describing the jth
        reaction between the ith and the (i+1)th cell.
    threshold: float
        threshold for the singular value decomposition of the two-cell reaction cores

    Returns
    -------
    operator: instance of TT class
        TT representation of the Markov generator of the NNIS

    References
    ----------
    .. [1] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-neighbor interaction systems in the tensor-train format",
           Journal of Computational Physics, 2017
    .. [2] P. Gelß, "The Tensor-Train Format and Its Applications", dissertation, FU Berlin, 2017

    """

    S = [None] * len(state_space)
    L = [None] * len(state_space)
    I = [None] * len(state_space)
    M = [None] * len(state_space)
    beta = [0] * len(state_space)

    # identity matrices

    for i in range(len(state_space)):
        I[i] = np.eye(state_space[i])

    # single-cell reactions

    for i in range(len(state_space)):
        S[i] = np.zeros([state_space[i]] * 2)
        for j in range(len(single_cell_reactions[i])):
            dimension = state_space[i]
            reactant_state = single_cell_reactions[i][j][0]
            product_state = single_cell_reactions[i][j][1]
            net_change = product_state - reactant_state
            reaction_rate = single_cell_reactions[i][j][2]
            S[i] = S[i] + reaction_rate * (np.eye(dimension, k=-net_change) - np.eye(dimension)) \
                   @ np.diag(sf.unit_vector(dimension, reactant_state))

    # two-cell reactions

    for i in range(len(state_space) - 1):
        LM = np.zeros([state_space[i]] * 2 + [state_space[i + 1]] * 2)
        for j in range(len(two_cell_reactions[i])):
            dimension_1 = state_space[i]
            dimension_2 = state_space[i + 1]
            reactant_state_1 = two_cell_reactions[i][j][0]
            reactant_state_2 = two_cell_reactions[i][j][2]
            product_state_1 = two_cell_reactions[i][j][1]
            product_state_2 = two_cell_reactions[i][j][3]
            net_change_1 = product_state_1 - reactant_state_1
            net_change_2 = product_state_2 - reactant_state_2
            reaction_rate = two_cell_reactions[i][j][4]
            LM = LM + reaction_rate * (np.tensordot(
                np.eye(dimension_1, k=-net_change_1) @ np.diag(sf.unit_vector(dimension_1, reactant_state_1)),
                np.eye(dimension_2, k=-net_change_2) @ np.diag(sf.unit_vector(dimension_2, reactant_state_2)), axes=0)
                                       - np.tensordot(np.diag(sf.unit_vector(dimension_1, reactant_state_1)),
                                                      np.diag(sf.unit_vector(dimension_2, reactant_state_2)), axes=0))
        L[i], M[i + 1], beta[i] = slim_tcr_decomposition(LM, threshold=threshold)

    # for cyclic NNIS

    if len(two_cell_reactions) == len(state_space):
        LM = np.zeros([state_space[-1]] * 2 + [state_space[0]] * 2)
        for j in range(len(two_cell_reactions[-1])):
            dimension_1 = state_space[-1]
            dimension_2 = state_space[0]
            reactant_state_1 = two_cell_reactions[-1][j][0]
            reactant_state_2 = two_cell_reactions[-1][j][2]
            product_state_1 = two_cell_reactions[-1][j][1]
            product_state_2 = two_cell_reactions[-1][j][3]
            net_change_1 = product_state_1 - reactant_state_1
            net_change_2 = product_state_2 - reactant_state_2
            reaction_rate = two_cell_reactions[-1][j][4]
            LM = LM + reaction_rate * (np.tensordot(
                np.eye(dimension_1, k=-net_change_1) @ np.diag(sf.unit_vector(dimension_1, reactant_state_1)),
                np.eye(dimension_2, k=-net_change_2) @ np.diag(sf.unit_vector(dimension_2, reactant_state_2)), axes=0)
                                       - np.tensordot(np.diag(sf.unit_vector(dimension_1, reactant_state_1)),
                                                      np.diag(sf.unit_vector(dimension_2, reactant_state_2)), axes=0))
        L[-1], M[0], beta[-1] = slim_tcr_decomposition(LM, threshold=threshold)
        L[-1] = L[-1].transpose(2, 0, 1)
        M[0] = M[0].transpose(1, 2, 0)

    # construct tensor train

    cores = [None] * len(state_space)
    cores[0] = np.zeros([1, state_space[0], state_space[0], 2 + beta[0] + beta[-1]])
    cores[0][0, :, :, 0] = S[0]
    cores[0][0, :, :, 1:1 + beta[0]] = L[0]
    cores[0][0, :, :, 1 + beta[0]] = I[0]
    cores[0][0, :, :, 2 + beta[0]:2 + beta[0] + beta[-1]] = M[0]

    for i in range(1, len(state_space) - 1):
        cores[i] = np.zeros([2 + beta[i - 1] + beta[-1], state_space[i], state_space[i], 2 + beta[i] + beta[-1]])
        cores[i][0, :, :, 0] = I[i]
        cores[i][1:1 + beta[i - 1], :, :, 0] = M[i]
        cores[i][1 + beta[i - 1], :, :, 0] = S[i]
        cores[i][1 + beta[i - 1], :, :, 1:1 + beta[i]] = L[i]
        cores[i][1 + beta[i - 1], :, :, 1 + beta[i]] = I[i]
        for j in range(beta[-1]):
            cores[i][2 + beta[i] + j, :, :, 2 + beta[i] + j] = I[i]

    cores[-1] = np.zeros([2 + beta[-2] + beta[-1], state_space[-1], state_space[-1], 1])
    cores[-1][0, :, :, 0] = I[-1]
    cores[-1][1:1 + beta[-2], :, :, 0] = M[-1]
    cores[-1][1 + beta[-2], :, :, 0] = S[-1]
    cores[-1][2 + beta[-2]:2 + beta[-2] + beta[-1], :, :, 0] = L[-1]

    operator = TT(cores)

    return operator


def slim_tcr_decomposition(LM, threshold):
    dimension_1 = LM.shape[0]
    dimension_2 = LM.shape[2]
    [U, S, V] = sp.linalg.svd(LM.reshape(dimension_1 ** 2, dimension_2 ** 2),
                              overwrite_a=True, full_matrices=False, lapack_driver='gesvd')
    if threshold != 0:
        indices = np.where(S / S[0] > threshold)[0]
        U = U[:, indices]
        S = S[indices]
        V = V[indices, :]
    rank = U.shape[1]
    L = (U @ np.diag(S)).reshape(dimension_1, dimension_1, rank)
    M = V.reshape(rank, dimension_2, dimension_2)
    return L, M, rank
