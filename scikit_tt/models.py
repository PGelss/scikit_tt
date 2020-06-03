# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from scikit_tt.tensor_train import TT
import scikit_tt.slim as slim


def cantor_dust(dimension, level):
    """
    Construction of a (multidimensional) Cantor dust.

    Generate a binary tensor representing a Cantor dust, see [1]_, by exploiting the
    tensor-train format and Kronecker products.

    Parameters
    ----------
    dimension : int
        dimension of the Cantor dust
    level : int
        level of the fractal construction to generate

    Returns
    -------
    np.ndarray
        tensor representing the Cantor dust

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    # construct generating tensor
    cores = []
    for _ in range(dimension):
        cores.append(np.zeros([1, 3, 1, 1]))
        cores[-1][0, :, 0, 0] = [1, 0, 1]
    generator = TT(cores)
    generator = generator.full().reshape(generator.row_dims)

    # construct fractal in the form of a binary tensor
    fractal = generator
    for i in range(2, level + 1):
        fractal = np.kron(fractal, generator)
    fractal = fractal.astype(int)

    return fractal


def co_oxidation(order, k_ad_co, cyclic=True):
    """
    CO oxidation on RuO2.

    Model for the CO oxidation on a RuO2 surface. For a detailed description of the process and the construction of the
    corresponding TT operator, we refer to [1]_,[2]_, and [3]_.

    Arguments
    ---------
    order : int
        number of reaction sites (= order of the operator)
    k_ad_co : float
        reaction rate constant for the adsorption of CO
    cyclic : bool, optional
        whether model should be cyclic or not, default=True

    Returns
    -------
    TT
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


def fpu_coefficients(d):
    """
    Construction of the exact coefficient tensor for the application of MANDy to the Fermi-Pasta-Ulam problem using
    the basis set {1, x, x^2, x^3}. See [1]_ for details.

    Parameters
    ----------
    d : int
        number of oscillators

    Returns
    -------
    TT
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


def kuramoto_coefficients(d, w):
    """
    Construction of the exact coefficient tensor for the application of MANDy to the Kuramoto model using the basis
    set {1, x, x^2, x^3}. See [1]_ for details.

    Parameters
    ----------
    d : int
        number of oscillators
    w : np.ndarray
        natural frequencies

    Returns
    -------
    TT
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


def multisponge(dimension, level):
    """
    Construction of a multisponge.

    Generate a binary tensor representing a multisponge fractal (e.g., Sierpinski carpet,
    Menger sponge, etc.), see [1]_, by exploiting the tensor-train format and Kronecker
    products.

    Parameters
    ----------
    dimension : int
        dimension (>1) of the multisponge
    level : int
        level of the fractal construction to generate

    Returns
    -------
    np.ndarray
        tensor representing the multisponge fractal

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    if dimension > 1:

        # construct generating tensor
        cores = [np.zeros([1, 3, 1, 2])]
        cores[0][0, :, 0, 0] = [1, 1, 1]
        cores[0][0, :, 0, 1] = [1, 0, 1]
        for _ in range(1, dimension - 1):
            cores.append(np.zeros([2, 3, 1, 2]))
            cores[-1][0, :, 0, 0] = [1, 0, 1]
            cores[-1][1, :, 0, 0] = [0, 1, 0]
            cores[-1][1, :, 0, 1] = [1, 0, 1]
        cores.append(np.zeros([2, 3, 1, 1]))
        cores[-1][0, :, 0, 0] = [1, 0, 1]
        cores[-1][1, :, 0, 0] = [0, 1, 0]
        generator = TT(cores)
        generator = generator.full().reshape(generator.row_dims)

        # construct fractal in the form of a binary tensor
        fractal = generator
        for i in range(2, level + 1):
            fractal = np.kron(fractal, generator)
        fractal = fractal.astype(int)

    else:

        raise ValueError('dimension must be larger than 1')

    return fractal


def rgb_fractal(matrix_r, matrix_g, matrix_b, level):
    """
    Construction of an RGB fractal.

    Generate a 3-dimensional tensor representing an RGB fractal, see [1]_, by exploiting
    the tensor-train format.

    Parameters
    ----------
    matrix_r : np.ndarray
        matrix representing red primaries
    matrix_g : np.ndarray
        matrix representing green primaries
    matrix_b : np.ndarray
        matrix representing blue primaries
    level : int
        level of the fractal construction to generate

    Returns
    -------
    np.ndarray
        tensor representing the RGB fractal

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    # dimension of RGB matrices
    n = matrix_r.shape[0]

    # construct RGB fractal
    cores = [np.zeros([1, n, n, 3])]
    cores[0][0, :, :, 0] = matrix_r
    cores[0][0, :, :, 1] = matrix_g
    cores[0][0, :, :, 2] = matrix_b
    for _ in range(1, level):
        cores.append(np.zeros([3, n, n, 3]))
        cores[-1][0, :, :, 0] = matrix_r
        cores[-1][1, :, :, 1] = matrix_g
        cores[-1][2, :, :, 2] = matrix_b
    cores.append(np.zeros([3, 3, 1, 1]))
    cores[-1][0, :, 0, 0] = [1, 0, 0]
    cores[-1][1, :, 0, 0] = [0, 1, 0]
    cores[-1][2, :, 0, 0] = [0, 0, 1]
    fractal = TT(cores).full().reshape([n ** level, 3, n ** level]).transpose([0, 2, 1])

    return fractal


def signaling_cascade(d):
    """
    Signaling cascade.

    Model for a cascading process on a genetic network consisting of genes of species S_1 , ..., S_d. For a detailed
    description of the process and the construction of the corresponding TT operator, we refer to [1]_.

    Arguments
    ---------
    d : int
        number of species (= order of the operator)

    Returns
    -------
    TT
        TT operator of the model

    References
    ----------
    .. [1] P. Gelß, "The Tensor-Train Format and Its Applications", dissertation, FU Berlin, 2017
    """

    # define core elements
    s_mat_0 = 0.7 * (np.eye(64, k=-1) - np.eye(64)) + 0.07 * (np.eye(64, k=1) - np.eye(64)).dot(np.diag(np.arange(64)))
    s_mat = 0.07 * (np.eye(64, k=1) - np.eye(64)).dot(np.diag(np.arange(64)))
    l_mat = np.diag(np.arange(64)).dot(np.diag(np.reciprocal(np.arange(5.0, 69.0))))
    i_mat = np.eye(64)
    m_mat = np.eye(64, k=-1) - np.eye(64)

    # make operator stochastic
    s_mat_0[-1, -1] = -0.07 * 63
    m_mat[-1, -1] = 0

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


def toll_station(number_of_lanes, number_of_cars):
    """"
    Toll station.

    Model for a quasi-realistic traffic problem.

    Arguments
    ---------
    number_of_lanes : int
    number_of_cars : int

    Returns
    -------
    TT
        TT operator of the process

    References
    ----------
    .. [1] P. Gelß, "The Tensor-Train Format and Its Applications", dissertation, FU Berlin, 2017
    """

    theta_in = np.sqrt(2.5)
    theta_out_left = 1
    theta_out_right = np.sqrt(0.5)
    nu_out_left = -1.5
    nu_out_right = 1.5
    change_rate = 5

    def f_in(t):
        return (1 / np.sqrt(2 * np.pi * theta_in ** 2)) * np.exp(-0.5 * t ** 2 / theta_in ** 2) + 0.05

    def f_out(t):
        return (1 / np.sqrt(2 * np.pi * theta_out_left ** 2)) * np.exp(
            -0.5 * (t - nu_out_left) ** 2 / theta_out_left ** 2) + (
                           1 / np.sqrt(2 * np.pi * theta_out_right ** 2)) * np.exp(
            -0.5 * (t - nu_out_right) ** 2 / theta_out_right ** 2)

    # define state space
    state_space = [number_of_cars + 1] * number_of_lanes

    # define single-cell reactions
    single_cell_reactions = []
    for i in range(number_of_lanes):
        scr_lane = []
        for j in range(number_of_cars):
            position = -2 + 4/(number_of_lanes-1) * i
            scr_lane.append([j, j + 1, f_in(position)])
            scr_lane.append([j + 1, j, f_out(position)])
        single_cell_reactions.append(scr_lane)

    two_cell_reactions = []
    for i in range(number_of_lanes - 1):
        tcr_lane = []
        for j in range(number_of_cars):
            for k in range(j + 1):
                tcr_lane.append([j + 1, j, k, k + 1, change_rate])
                tcr_lane.append([k, k + 1, j + 1, j, change_rate])
        two_cell_reactions.append(tcr_lane)

    operator = slim.slim_mme(state_space, single_cell_reactions, two_cell_reactions, threshold=1e-14)

    return operator


def two_step_destruction(k_1, k_2, k_3, m):
    """"
    Two-step destruction.

    Model for a two-step mechanism for the destruction of molecules. For a detailed description of the process and the
    construction of the corresponding TT operator, we refer to [1]_.

    Arguments
    ---------
    k_1 : float
        rate constant for the first reaction
    k_2 : float
        rate constant for the second reaction
    k_3 : float
        rate constant for the third reaction
    m : int
        exponent determining the maximum number of molecules

    Returns
    -------
    TT
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
    cores[0][0, :, :, 1] = -k_1 * np.diag(np.arange(n[0]))
    cores[0][0, :, :, 2] = k_1 * np.eye(n[0], k=1).dot(np.diag(np.arange(n[0])))
    cores[1][0, :, :, 0] = k_2 * np.eye(n[1], k=1).dot(np.diag(np.arange(n[1])))
    cores[1][0, :, :, 1] = np.eye(n[1])
    cores[1][0, :, :, 2] = -k_2 * np.diag(np.arange(n[1]))
    cores[1][1, :, :, 3] = np.diag(np.arange(n[1]))
    cores[1][2, :, :, 4] = np.eye(n[1], k=1).dot(np.diag(np.arange(n[1])))
    cores[2][0, :, :, 0] = np.eye(n[2], k=1).dot(np.diag(np.arange(n[2])))
    cores[2][1, :, :, 1] = np.eye(n[2])
    cores[2][2, :, :, 2] = np.diag(np.arange(n[2]))
    cores[2][3, :, :, 2] = np.eye(n[2])
    cores[2][4, :, :, 2] = np.eye(n[2], k=-1)
    cores[3][0, :, :, 0] = np.eye(n[3], k=-1)
    cores[3][1, :, :, 0] = k_3 * np.eye(n[3], k=1).dot(np.diag(np.arange(n[3]))) - k_3 * np.diag(np.arange(n[3]))  
    cores[3][2, :, :, 0] = np.eye(n[3])

    # make operator a stochastic tensor
    cores[2][4, -1, -1, 2] = 1
    cores[3][0, -1, -1, 0] = 1

    # define operator
    operator = TT(cores)

    return operator


def vicsek_fractal(dimension, level):
    """
    Construction of a Vicsek fractal.

    Generate a binary tensor representing a Vicsek fractal, see [1]_, by exploiting the
    tensor-train format and Kronecker products.

    Parameters
    ----------
    dimension : int
        dimension (>1) of the Vicsek fractal
    level : int
        level of the fractal construction to generate

    Returns
    -------
    np.ndarray
        tensor representing the Vicsek fractal

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    if dimension > 1:

        # construct generating tensor
        cores = [np.zeros([1, 3, 1, 2])]
        cores[0][0, :, 0, 0] = [1, 1, 1]
        cores[0][0, :, 0, 1] = [0, 1, 0]
        for _ in range(1, dimension - 1):
            cores.append(np.zeros([2, 3, 1, 2]))
            cores[-1][0, :, 0, 0] = [0, 1, 0]
            cores[-1][1, :, 0, 0] = [1, 0, 1]
            cores[-1][1, :, 0, 1] = [0, 1, 0]
        cores.append(np.zeros([2, 3, 1, 1]))
        cores[-1][0, :, 0, 0] = [0, 1, 0]
        cores[-1][1, :, 0, 0] = [1, 0, 1]
        generator = TT(cores)
        generator = generator.full().reshape(generator.row_dims)

        # construct fractal in the form of a binary tensor
        fractal = generator
        for i in range(2, level + 1):
            fractal = np.kron(fractal, generator)
        fractal = fractal.astype(int)

    else:

        raise ValueError('dimension must be larger than 1')

    return fractal
