#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scikit_tt.tensor_train import TT


def slim_mme(state_space, single_cell_reactions, two_cell_reactions, threshold=0):
    """
    SLIM decomposition for Markov generators.

    Construct a tensor-train decomposition of a Markov generator representing a nearest-neighbor interaction system
    (NNIS) described by a list of single-cell and two-cell reactions. Note that the implementation of the SLIM algorithm
    differs from that described in [1]_ and [2]_ in order to simplify the parameter passing. However, the resulting
    operator corresponds to the described decompositions in [1]_ and [2]_.

    Parameters
    ----------
    state_space : list[int]
        number of states of each cell in the NNIS
    single_cell_reactions :  list[list[list[int or float]]]
        list of considered single-cell reactions, i.e. single_cell_reactions[i][j] is a list of the form
        [reactant_state, product_state, reactions_rate] describing the jth reaction on the ith cell.
    two_cell_reactions : list[list[list[int or float]]]
        list of considered two-cell reactions, i.e. two_cell_reactions[i][j] is a list of the form
        [reactant_state_i, product_state_i, reactant_state_i+1, product_state_i+1, reactions_rate] describing the jth
        reaction between the ith and the (i+1)th cell.
    threshold : float, optional
        threshold for the singular value decomposition of the two-cell reaction cores, default is 1e-14

    Returns
    -------
    TT
        TT representation of the Markov generator of the NNIS

    References
    ----------
    .. [1] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-neighbor interaction systems in the tensor-train format",
           Journal of Computational Physics, 2017
    .. [2] P. Gelß, "The Tensor-Train Format and Its Applications", dissertation, FU Berlin, 2017
    """

    # define core elements
    # --------------------

    s_mat = []
    l_mat = []
    i_mat = []
    m_mat = [None]
    ranks = []

    # core elements equal to identity matrices
    # ----------------------------------------

    for i in range(len(state_space)):
        i_mat.append(np.eye(state_space[i]))

    # core elements for single-cell reactions
    # ---------------------------------------

    for i in range(len(state_space)):

        # append matrix of all zeros
        s_mat.append(np.zeros([state_space[i]] * 2))

        # sum over all single-cell reactions
        for j in range(len(single_cell_reactions[i])):
            dimension = state_space[i]
            reactant_state = single_cell_reactions[i][j][0]
            product_state = single_cell_reactions[i][j][1]
            net_change = product_state - reactant_state
            reaction_rate = single_cell_reactions[i][j][2]
            s_mat[i] = s_mat[i] + reaction_rate * (np.eye(dimension, k=-net_change) - np.eye(dimension)).dot(np.diag(
                np.eye(dimension)[:, reactant_state]))

    # core elements for two-cell reactions
    # ------------------------------------

    for i in range(len(state_space) - 1):

        # define super-core as a 4-dimensional tensor of all zeros
        super_core = np.zeros([state_space[i]] * 2 + [state_space[i + 1]] * 2)

        # sum over all two-cell reactions
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
            super_core = super_core + reaction_rate * (np.tensordot(
                np.eye(dimension_1, k=-net_change_1).dot(np.diag(np.eye(dimension_1)[:, reactant_state_1])),
                np.eye(dimension_2, k=-net_change_2).dot(np.diag(np.eye(dimension_2)[:, reactant_state_2])), axes=0)
                                                       - np.tensordot(
                        np.diag(np.eye(dimension_1)[:, reactant_state_1]),
                        np.diag(np.eye(dimension_2)[:, reactant_state_2]), axes=0))

        # split super-core append quantities
        core_left, core_right, rank = __slim_tcr_decomposition(super_core, threshold=threshold)
        l_mat.append(core_left)
        m_mat.append(core_right)
        ranks.append(rank)

    # for cyclic nearest-neighbor interaction systems
    # -----------------------------------------------

    if len(two_cell_reactions) == len(state_space):

        # define super-core as a 4-dimensional tensor of all zeros
        super_core = np.zeros([state_space[-1]] * 2 + [state_space[0]] * 2)

        # sum over all two-cell reactions between first and last cell
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
            super_core = super_core + reaction_rate * (np.tensordot(
                np.eye(dimension_1, k=-net_change_1).dot(np.diag(np.eye(dimension_1)[:, reactant_state_1])),
                np.eye(dimension_2, k=-net_change_2).dot(np.diag(np.eye(dimension_2)[:, reactant_state_2])), axes=0)
                                                       - np.tensordot(
                        np.diag(np.eye(dimension_1)[:, reactant_state_1]),
                        np.diag(np.eye(dimension_2)[:, reactant_state_2]), axes=0))

        # split super-core append quantities
        core_left, core_right, rank = __slim_tcr_decomposition(super_core, threshold=threshold)
        l_mat.append(core_left.transpose(2, 0, 1))
        m_mat[0] = core_right.transpose(1, 2, 0)
        ranks.append(rank)
    else:
        l_mat.append(0)
        m_mat[0] = 0
        ranks.append(0)

    # construct tensor train operator
    # -------------------------------

    cores = [np.zeros([1, state_space[0], state_space[0], 2 + ranks[0] + ranks[-1]])]
    cores[0][0, :, :, 0] = s_mat[0]
    cores[0][0, :, :, 1:1 + ranks[0]] = l_mat[0]
    cores[0][0, :, :, 1 + ranks[0]] = i_mat[0]
    cores[0][0, :, :, 2 + ranks[0]:2 + ranks[0] + ranks[-1]] = m_mat[0]

    for i in range(1, len(state_space) - 1):
        cores.append(np.zeros([2 + ranks[i - 1] + ranks[-1], state_space[i], state_space[i], 2 + ranks[i] + ranks[-1]]))
        cores[i][0, :, :, 0] = i_mat[i]
        cores[i][1:1 + ranks[i - 1], :, :, 0] = m_mat[i]
        cores[i][1 + ranks[i - 1], :, :, 0] = s_mat[i]
        cores[i][1 + ranks[i - 1], :, :, 1:1 + ranks[i]] = l_mat[i]
        cores[i][1 + ranks[i - 1], :, :, 1 + ranks[i]] = i_mat[i]
        for j in range(ranks[-1]):
            cores[i][2 + ranks[i] + j, :, :, 2 + ranks[i] + j] = i_mat[i]

    cores.append(np.zeros([2 + ranks[-2] + ranks[-1], state_space[-1], state_space[-1], 1]))
    cores[-1][0, :, :, 0] = i_mat[-1]
    cores[-1][1:1 + ranks[-2], :, :, 0] = m_mat[-1]
    cores[-1][1 + ranks[-2], :, :, 0] = s_mat[-1]
    cores[-1][2 + ranks[-2]:2 + ranks[-2] + ranks[-1], :, :, 0] = l_mat[-1]

    operator = TT(cores)

    return operator


def slim_mme_hom(state_space, single_cell_reactions, two_cell_reactions, cyclic=True, threshold=0):
    """
    Homogeneous SLIM decomposition for Markov generators.

    Construct a tensor-train decomposition of a Markov generator representing a homogeneous nearest-neighbor interaction
    system. See [1]_ and [2]_ for details.

    Parameters
    ----------
    state_space : list[int]
        number of states of each cell in the NNIS
    single_cell_reactions : list[list[int or float]]
        list of considered single-cell reactions, i.e. single_cell_reactions[i] is a list of the form
        [reactant_state, product_state, reactions_rate] describing the ith reaction on each cell.
    two_cell_reactions : list[list[int or float]]
        list of considered two-cell reactions, i.e. two_cell_reactions[i] is a list of the form
        [reactant_state_i, product_state_i, reactant_state_i+1, product_state_i+1, reactions_rate] describing the ith
        reaction between neighboring cells
    cyclic : bool, optional
        whether the system is cyclic or not, default is True
    threshold : float, optional
        threshold for the singular value decomposition of the two-cell reaction core, default is 1e-14

    Returns
    -------
    TT
        TT representation of the Markov generator of the NNIS

    References
    ----------
    .. [1] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-neighbor interaction systems in the tensor-train format",
           Journal of Computational Physics, 2017
    .. [2] P. Gelß, "The Tensor-Train Format and Its Applications", dissertation, FU Berlin, 2017
    """

    # adapt parameters
    order = len(state_space)
    single_cell_reactions = [single_cell_reactions for _ in range(order)]
    if cyclic is True:
        two_cell_reactions = [two_cell_reactions for _ in range(order)]
    else:
        two_cell_reactions = [two_cell_reactions for _ in range(order - 1)]

    # construct TT operator by using slim_mme
    operator = slim_mme(state_space, single_cell_reactions, two_cell_reactions, threshold=threshold)

    return operator


def __slim_tcr_decomposition(super_core, threshold):
    """
    Two-cell reaction decomposition.

    Decompose a super-core representing the interactions between two cells.

    Parameters
    ----------
    super_core : np.ndarray
        tensor with order 4
    threshold : float
            threshold for reduced SVD decompositions

    Returns
    -------
    core_left : np.ndarray
        TT core for first cell
    core_right : np.ndarry
        TT core for second cell
    rank : int
        TT rank
    """

    # number of states
    dimension_1 = super_core.shape[0]
    dimension_2 = super_core.shape[2]

    # apply SVD in order to split the super-core
    [u, s, v] = linalg.svd(super_core.reshape(dimension_1 ** 2, dimension_2 ** 2),
                           full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

    # rank reduction
    if threshold != 0:
        indices = np.where(s / s[0] > threshold)[0]
        u = u[:, indices]
        s = s[indices]
        v = v[indices, :]

    # set quantities for decomposition
    rank = u.shape[1]
    core_left = (u.dot(np.diag(s))).reshape(dimension_1, dimension_1, rank)
    core_right = v.reshape(rank, dimension_2, dimension_2)
    return core_left, core_right, rank
