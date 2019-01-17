#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate as spint
from scikit_tt.tensor_train import TT
import scikit_tt.slim as slim


def co_oxidation(order, k_ad_co, cyclic=True):
    """"CO oxidation on RuO2

    Model for the CO oxidation on a RuO2 surface. For a detailed description of the process and the construction of the
    corresponding TT operator, we refer to [1]_,[2]_, and [3]_.

    Arguments
    ---------
    order: int
        number of reaction sites (= order of the operator)
    k_ad_co: float
        reaction rate constant for the adsorption of CO
    cyclic: bool, optional
        whether model should be cyclic or not, default=True

    Returns
    -------
    operator: instance of TT class
        TT operator of the process


    References
    ----------
    .. [1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
           Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
    .. [2] P. Gelß, S. Matera, C. Schütte, "Solving the master equation without kinetic Monte Carlo: Tensor train
           approximations for a CO oxidation model", Journal of Computational Physics 314 (2016) 489–502
    .. [3] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-neighbor interaction systems in the tensor-train format",
           Journal of Computational Physics 341 (2017) 140-162
    """

    # define reaction rate constants
    k_ad_o2 = 9.7e7
    k_de_co = 9.2e6
    k_de_o2 = 2.8e1
    k_diff_co = 6.6e-2
    k_diff_o = 5.0e-1
    k_de_co2 = 1.7e5

    # define state space
    state_space = [3] * order

    # define operator using automatic construction of SLIM decomposition
    # ------------------------------------------------------------------

    # define list of reactions
    single_cell_reactions = [[0, 2, k_ad_co], [2, 0, k_de_co]]
    two_cell_reactions = [[0, 1, 0, 1, k_ad_o2], [1, 0, 1, 0, k_de_o2], [2, 0, 1, 0, k_de_co2],
                          [1, 0, 2, 0, k_de_co2], [1, 0, 0, 1, k_diff_o], [0, 1, 1, 0, k_diff_o],
                          [0, 2, 2, 0, k_diff_co], [2, 0, 0, 2, k_diff_co]]

    # define operator
    operator = slim.slim_mme_hom(state_space, single_cell_reactions, two_cell_reactions, cyclic=cyclic)

    return operator


def fermi_pasta_ulam(number_of_oscillators, number_of_snapshots):
    """Fermi–Pasta–Ulam problem.

    Generate data for the Fermi–Pasta–Ulam problem represented by the differential equation

        d^2/dt^2 x_i = (x_i+1 - 2x_i + x_i-1) + 0.7((x_i+1 - x_i)^3 - (x_i-x_i-1)^3).

    See [1]_ for details.

    Parameters
    ----------
    number_of_oscillators: int
        number of oscillators
    number_of_snapshots: int
        number of snapshots

    Returns
    -------
    snapshots: ndarray(number_of_oscillators, number_of_snapshots)
        snapshot matrix containing random displacements of the oscillators in [-0.1,0.1]
    derivatives: ndarray(number_of_oscillators, number_of_snapshots)
        matrix containing the corresponding derivatives

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # define random snapshot matrix
    snapshots = 0.2 * np.random.rand(number_of_oscillators, number_of_snapshots) - 0.1

    # compute derivatives
    derivatives = np.zeros((number_of_oscillators, number_of_snapshots))
    for j in range(number_of_snapshots):
        derivatives[0, j] = snapshots[1, j] - 2 * snapshots[0, j] + 0.7 * (
                (snapshots[1, j] - snapshots[0, j]) ** 3 - snapshots[0, j] ** 3)
        for i in range(1, number_of_oscillators - 1):
            derivatives[i, j] = snapshots[i + 1, j] - 2 * snapshots[i, j] + snapshots[i - 1, j] + 0.7 * (
                    (snapshots[i + 1, j] - snapshots[i, j]) ** 3 - (snapshots[i, j] - snapshots[i - 1, j]) ** 3)
        derivatives[-1, j] = - 2 * snapshots[-1, j] + snapshots[-2, j] + 0.7 * (
                -snapshots[-1, j] ** 3 - (snapshots[-1, j] - snapshots[-2, j]) ** 3)

    return snapshots, derivatives


def fpu_coefficients(d):
    """Construction of the exact coefficient tensor for the application of MANDy to the Fermi-Pasta-Ulam problem using
    the basis set {1, x, x^2, x^3}. See [1]_ for details.

    Parameters
    ----------
    d: int
        number of oscillators

    Returns
    -------
    coefficient_tensor: instance of TT class
        exact coefficient tensor

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # define core types
    core_type_1 = np.zeros([1, 4, 1, 1])  # define core types
    core_type_1[0, 0, 0, 0] = 1
    core_type_2 = np.eye(4).reshape([1, 4, 1, 4])
    core_type_3 = np.zeros([4, 4, 1, 4])
    core_type_3[0, 1, 0, 0] = -2
    core_type_3[0, 3, 0, 0] = -1.4
    core_type_3[0, 0, 0, 1] = 1
    core_type_3[0, 2, 0, 1] = 2.1
    core_type_3[0, 1, 0, 2] = -2.1
    core_type_3[0, 0, 0, 3] = 0.7
    core_type_3[1, 0, 0, 0] = 1
    core_type_3[1, 2, 0, 0] = 2.1
    core_type_3[2, 1, 0, 0] = -2.1
    core_type_3[3, 0, 0, 0] = 0.7
    core_type_4 = np.eye(4).reshape([4, 4, 1, 1])

    # construct cores
    cores = [np.zeros([1, 4, 1, 4])]
    cores[0][0, :, :, :] = core_type_3[0, :, :, :]
    cores.append(core_type_4)
    for _ in range(2, d):
        cores.append(core_type_1)
    cores.append(np.zeros([1, d, 1, 1]))
    cores[d][0, 0, 0, 0] = 1
    coefficient_tensor = TT(cores)
    for q in range(1, d - 1):
        cores = []
        for _ in range(q - 1):
            cores.append(core_type_1)
        cores.append(core_type_2)
        cores.append(core_type_3)
        cores.append(core_type_4)
        for _ in range(q + 2, d):
            cores.append(core_type_1)
        cores.append(np.zeros([1, d, 1, 1]))
        cores[d][0, q, 0, 0] = 1
        coefficient_tensor = coefficient_tensor + TT(cores)
    cores = []
    for _ in range(d - 2):
        cores.append(core_type_1)
    cores.append(core_type_2)
    cores.append(np.zeros([4, 4, 1, 1]))
    cores[d - 1][:, :, :, 0] = core_type_3[:, :, :, 0]
    cores.append(np.zeros([1, d, 1, 1]))
    cores[d][0, d - 1, 0, 0] = 1
    coefficient_tensor = coefficient_tensor + TT(cores)

    return coefficient_tensor


def kuramoto(theta_init, frequencies, time, number_of_snapshots):
    """Kuramoto model

    Generate data for the Kuramoto model represented by the differential equation

        d/dt x_i = w_i + (2/d) * (sin(x_1 - x_i) + ... + sin(x_d - x_i)) + 0.2 * sin(x_i).

    See [1]_ and [2]_ for details.

    Parameters
    ----------
    theta_init: ndarray
        initial distribution of the oscillators
    frequencies: ndarray
        natural frequencies of the oscillators
    time: float
        integration time for BDF method
    number_of_snapshots: int
        number of snapshots

    Returns
    -------
    snapshots: ndarray(number_of_oscillators, number_of_snapshots)
        snapshot matrix containing random displacements of the oscillators in [-0.1,0.1]
    derivatives: ndarray(number_of_oscillators, number_of_snapshots)
        matrix containing the corresponding derivatives

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    .. [2] J. A. Acebrón, L. L. Bonilla, C. J. Pérez Vicente, F. Ritort, R. Spigler, "The Kuramoto model: A simple
           paradigm for synchronization phenomena", Rev. Mod. Phys. 77, pp. 137-185 , 2005
    """

    number_of_oscillators = len(theta_init)

    def kuramoto_ode(_, theta):
        [theta_i, theta_j] = np.meshgrid(theta, theta)
        return frequencies + 2 / number_of_oscillators * np.sin(theta_j - theta_i).sum(0) + 0.2 * np.sin(theta)

    sol = spint.solve_ivp(kuramoto_ode, [0, time], theta_init, method='BDF',
                          t_eval=np.linspace(0, time, number_of_snapshots))
    snapshots = sol.y
    derivatives = np.zeros([number_of_oscillators, number_of_snapshots])
    for i in range(number_of_snapshots):
        derivatives[:, i] = kuramoto_ode(0, snapshots[:, i])
    return snapshots, derivatives


def kuramoto_coefficients(d, w):
    """Construction of the exact coefficient tensor for the application of MANDy to the Kuramoto model using the basis
    set {1, x, x^2, x^3}. See [1]_ for details.

    Parameters
    ----------
    d: int
        number of oscillators
    w: ndarray
        natural frequencies

    Returns
    -------
    coefficient_tensor: instance of TT class
        exact coefficient tensor

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    cores = [np.zeros([1, d + 1, 1, 2 * d + 1]), np.zeros([2 * d + 1, d + 1, 1, 2 * d + 1]),
             np.zeros([2 * d + 1, d, 1, 1])]
    cores[0][0, 0, 0, 0] = 1
    cores[1][0, 0, 0, 0] = 1
    cores[2][0, :, 0, 0] = w
    for q in range(d):
        cores[0][0, 1:, 0, 2 * q + 1] = (2 / d) * np.ones([d])
        cores[0][0, q + 1, 0, 2 * q + 1] = 0
        cores[1][2 * q + 1, q + 1, 0, 2 * q + 1] = 1
        cores[2][2 * q + 1, q, 0, 0] = 1
        cores[0][0, q + 1, 0, 2 * q + 2] = 1
        cores[1][2 * q + 2, 0, 0, 2 * q + 2] = 0.2
        cores[1][2 * q + 2, 1:, 0, 2 * q + 2] = -(2 / d) * np.ones([d])
        cores[1][2 * q + 2, q + 1, 0, 2 * q + 2] = 0
        cores[2][2 * q + 2, q, 0, 0] = 1
    coefficient_tensor = TT(cores)
    return coefficient_tensor


def signaling_cascade(d):
    """Signaling cascade

    Model for a cascading process on a genetic network consisting of genes of species S_1 , ..., S_d. For a detailed
    description of the process and the construction of the corresponding TT operator, we refer to [1]_.

    Arguments
    ---------
    d: int
        number of species (= order of the operator)

    Returns
    -------
    operator: instance of TT class
        TT operator of the model


    References
    ----------
    .. [1] P. Gelß, "The Tensor-Train Format and Its Applications", dissertation, FU Berlin, 2017
    """

    # define core elements
    s_mat_0 = 0.7 * (np.eye(64, k=-1) - np.eye(64)) + 0.07 * (np.eye(64, k=1) - np.eye(64)) @ np.diag(np.arange(64))
    s_mat = 0.07 * (np.eye(64, k=1) - np.eye(64)) @ np.diag(np.arange(64))
    l_mat = np.diag(np.arange(64)) @ np.diag(np.reciprocal(np.arange(5.0, 69.0)))
    i_mat = np.eye(64)
    m_mat = np.eye(64, k=-1) - np.eye(64)

    # define TT cores
    cores = [np.zeros([1, 64, 64, 3])]
    cores[0][0, :, :, 0] = s_mat_0
    cores[0][0, :, :, 1] = l_mat
    cores[0][0, :, :, 2] = i_mat
    for k in range(1, d - 1):
        cores.append(np.zeros([3, 64, 64, 3]))
        cores[k][0, :, :, 0] = i_mat
        cores[k][1, :, :, 0] = m_mat
        cores[k][2, :, :, 0] = s_mat
        cores[k][2, :, :, 1] = l_mat
        cores[k][2, :, :, 2] = i_mat
    cores.append(np.zeros([3, 64, 64, 1]))
    cores[d - 1][0, :, :, 0] = i_mat
    cores[d - 1][1, :, :, 0] = m_mat
    cores[d - 1][2, :, :, 0] = s_mat

    # define TT operator
    operator = TT(cores)

    return operator


def two_step_destruction(k_1, k_2, m):
    """"Two-step destruction

    Model for a two-step mechanism for the destruction of molecules. For a detailed description of the process and the
    construction of the corresponding TT operator, we refer to [1]_.

    Arguments
    ---------
    k_1: float
        rate constant for the first reaction
    k_2: float
        rate constant for the second reaction
    m: int
        exponent determining the maximum number of molecules

    Returns
    -------
    operator: instance of TT class
        TT operator of the process


    References
    ----------
    .. [1] P. Gelß, "The Tensor-Train Format and Its Applications", dissertation, FU Berlin, 2017
    """

    # define dimensions and ranks
    n = [2 ** m, 2 ** (m + 1), 2 ** m, 2 ** m]
    r = [1, 3, 5, 3, 1]

    # define TT cores
    cores = [np.zeros([r[i], n[i], n[i], r[i + 1]]) for i in range(4)]
    cores[0][0, :, :, 0] = np.eye(n[0])
    cores[0][0, :, :, 1] = k_1 * np.eye(n[0], k=1) @ np.diag(np.arange(n[0]))
    cores[0][0, :, :, 2] = -k_1 * np.diag(np.arange(n[0]))
    cores[1][0, :, :, 0] = np.eye(n[1])
    cores[1][0, :, :, 1] = k_2 * np.eye(n[1], k=1) @ np.diag(np.arange(n[1]))
    cores[1][0, :, :, 2] = -k_2 * np.diag(np.arange(n[1]))
    cores[1][1, :, :, 3] = np.eye(n[1], k=1) @ np.diag(np.arange(n[1]))
    cores[1][2, :, :, 4] = np.diag(np.arange(n[1]))
    cores[2][0, :, :, 0] = np.eye(n[2])
    cores[2][1, :, :, 1] = np.eye(n[2], k=1) @ np.diag(np.arange(n[2]))
    cores[2][2, :, :, 2] = np.diag(np.arange(n[2]))
    cores[2][3, :, :, 2] = np.eye(n[2], k=-1)
    cores[2][4, :, :, 2] = np.eye(n[2])
    cores[3][0, :, :, 0] = (np.eye(n[3], k=1) - np.eye(n[3])) @ np.diag(np.arange(n[3]))
    cores[3][1, :, :, 0] = np.eye(n[3], k=-1)
    cores[3][2, :, :, 0] = np.eye(n[3])

    # define operator
    operator = TT(cores)

    return operator
