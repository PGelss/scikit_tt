#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as splin

from typing import List, Union, Tuple

import scikit_tt.tensor_train as tt
import scikit_tt.solvers.sle as sle
from scikit_tt.tensor_train import TT

class Object(object):
    pass

def als(operator: 'TT',
        initial_guess: 'TT',
        previous: List['TT']=[],
        shift: float=0,
        operator_gevp: 'TT'=None,
        number_ev: int=1,
        repeats: int=1,
        conv_eps: float=1e-10, 
        solver: str='eig',
        sigma: float=1,
        real: bool=True) -> Tuple[Union[float, List[float]], Union['TT', List['TT']], int]:
    """
    Alternating linear scheme.

    Approximates eigenvalues and corresponding eigentensors of an (generalized) eigenvalue problem in the TT format.
    For details, see [1]_.

    Parameters
    ----------
    operator : TT
        TT operator, left-hand side

    initial_guess : TT
        initial guess for the solution

    previous : list of TT, optional
        list of known eigentensors whose eigenvalues should be shifted

    shift : float, optional
        shift parameter for known eigenpairs

    operator_gevp : TT, optional
        TT operator, right-hand side (for generalized eigenvalue problems), default is None

    number_ev : int, optional
        number of eigenvalues and corresponding eigentensor to compute, default is 1

    repeats : int, optional
        number of repeats of the ALS, default is 1

    conv_eps : float, optional
        threshold for convergence of the eigenvalue, default is 0

    solver : string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'eig', 'eigs' or 'eigh', default is 'eig'

    sigma : float, optional
        find eigenvalues near sigma, default is 1

    real : bool, optional
        whether to compute only real eigenvalues and eigentensors or not, default is True

    Returns
    -------
    eigenvalues: float or list[float]
        approximated eigenvalues, if number_ev>1 eigenvalues is a list[float]

    eigentensors: TT or list[TT]
        approximated eigentensors, if number_ev>1 eigentensors is a list of tensor trains

    iterations: int
    	number of ALS iterations, if conv_eps<=0 iterations is equal to repeats

    References
    ----------
    ..[1] S. Holtz, T. Rohwedder, R. Schneider, "The Alternating Linear Scheme for Tensor Optimization in the Tensor
          Train Format", SIAM Journal on Scientific Computing 34 (2), 2012
    """

    # define tensor trains
    trains = Object()
    trains.operator      = operator
    trains.operator_gevp = operator_gevp
    trains.solution      = initial_guess.copy()
    trains.previous      = previous

    # define stacks
    stacks = Object()
    stacks.op_left         = [None] * operator.order
    stacks.op_right        = [None] * operator.order
    stacks.op_gevp_left    = [None] * operator.order
    stacks.op_gevp_right   = [None] * operator.order
    stacks.previous_left   = [[None] * operator.order for _ in range(len(previous))]
    stacks.previous_right  = [[None] * operator.order for _ in range(len(previous))]

    # construct right stacks for the left-hand side
    for i in range(operator.order - 1, -1, -1):
        __construct_right_stacks(i, trains, stacks)

    # define iteration number
    current_iteration = 1

    # initialize variables for convergence detection
    eigenvalues_pre = np.array([np.inf]*number_ev)[None,:]
    conv_tf = False

    # initialize variables for optimal eigenpair (number_ev=1)
    eigenvalue_opt  = np.inf
    eigentensor_opt = None

    # begin ALS
    while current_iteration <= repeats and not conv_tf:

        # first half sweep
        for i in range(operator.order):

            # update left stack for the left-hand side
            __construct_left_stacks(i, trains, stacks)

            if i < operator.order - 1:

                # construct micro system
                micro_op, micro_op_gevp = __construct_micro_matrices(i, trains, stacks, shift)

                # update solution
                eigenvalues = __update_core(i, micro_op, micro_op_gevp, number_ev, trains.solution, solver, sigma, real,
                                                'forward')

        # second half sweep
        for i in range(operator.order - 1, -1, -1):
            # update right stack for the left-hand side
            __construct_right_stacks(i, trains, stacks)

            # construct micro system
            micro_op, micro_op_gevp = __construct_micro_matrices(i, trains, stacks, shift)

            # update solution
            eigenvalues = __update_core(i, micro_op, micro_op_gevp, number_ev, trains.solution, solver, sigma, real,
                                            'backward')

        # increase iteration number
        current_iteration += 1

        # save optimal eigenpair
        if number_ev == 1:

            if np.abs(eigenvalues[0]-sigma)<np.abs(eigenvalue_opt-sigma):
                eigenvalue_opt = eigenvalues[0].copy()
                eigentensor_opt = TT([trains.solution.cores[0][:, :, :, :, 0]] + trains.solution.cores[1:])

        # check for convergence
        last = eigenvalues_pre[-np.amin([3, eigenvalues_pre.shape[0]]):, :]

        # last_rel_diff = np.abs(np.dot(last - eigenvalues, np.diag(np.reciprocal(eigenvalues))))
        last_diff = np.abs(last - eigenvalues)

        if np.amax(last_diff)<conv_eps:
            conv_tf = True
        eigenvalues_pre = np.vstack((eigenvalues_pre, eigenvalues))

    # define form of the final solution depending on the number of eigenvalues to compute
    if number_ev == 1:
        eigentensors = eigentensor_opt
        eigenvalues  = eigenvalue_opt
    else:
        eigentensors = []

        for i in range(number_ev):
            eigentensors.append(TT([trains.solution.cores[0][:, :, :, :, i]] + trains.solution.cores[1:]))

    iterations = current_iteration - 1

    return eigenvalues, eigentensors, iterations


def power_method(operator: 'TT', initial_guess: 'TT', operator_gevp: 'TT'=None, repeats: int=10, sigma: float=0.999) -> Tuple[float, 'TT']:
    """
    Inverse power iteration method.

    Approximates eigenvalues and corresponding eigentensors of an (generalized) eigenvalue problem in the TT format.
    For details, see [1]_.

    Parameters
    ----------
    operator : TT
        TT operator, left-hand side

    initial_guess : TT
        initial guess for the solution
        
    operator_gevp : TT, optional
        TT operator, right-hand side (for generalized eigenvalue problems), default is None

    repeats : int, optional
        number of iterations, default is 10

    sigma : float, optional
        find eigenvalues near sigma, default is 1

    Returns
    -------
    eigenvalue : float
        approximated eigenvalue

    eigentensor : TT
        approximated eigentensors

    References
    ----------
    ..[1] S. Klus, C. Schütte, "Towards tensor-based methods for the numerical approximation of the Perron-Frobenius
          and Koopman operator", Journal of Computational Dynamics 3 (2), 2016
    """

    # define shift operator
    if operator_gevp is None:
        operator_shift = operator - sigma * tt.eye(operator.row_dims)

    else:
        operator_shift = operator - sigma * operator_gevp

    # define eigenvalue and eigentensor
    eigenvalue = 0
    eigentensor = initial_guess

    # start iteration
    for i in range(repeats):

        # solve system of linear equations in the TT format
        if operator_gevp is None:
            eigentensor = sle.als(operator_shift, eigentensor, eigentensor)

        else:
            eigentensor = sle.als(operator_shift, eigentensor, operator_gevp.dot(eigentensor))

        # normalize eigentensor
        eigentensor *= (1 / eigentensor.norm())

        # compute eigenvalue
        eigenvalue = (eigentensor.transpose().dot(operator).dot(eigentensor))

        if operator_gevp is not None:
            eigenvalue *= 1 / (eigentensor.transpose().dot(operator_gevp).dot(eigentensor))

    return eigenvalue, eigentensor


def __construct_left_stacks(i: int, trains, stacks):
    """
    Construct left stack for left-hand side.

    Parameters
    ----------
    i : int
        core index
    trains : Object
        collection of tensor trains
    stacks: Object
        collection of stacks
    """

    if i == 0:

        # first stack element is 1
        stacks.op_left[i] = np.array([1], ndmin=3)

        if trains.operator_gevp is not None:
            stacks.op_gevp_left[i] = np.array([1], ndmin=3)

        for j in range(len(trains.previous)):
            stacks.previous_left[j][i] = np.array([1], ndmin=2)

    else:

        # contract previous stack element with solution and operator cores
        stacks.op_left[i] = np.tensordot(stacks.op_left[i - 1], np.conjugate(trains.solution.cores[i - 1][:, :, 0, :]), axes=(0, 0))
        stacks.op_left[i] = np.tensordot(stacks.op_left[i], trains.operator.cores[i - 1], axes=([0, 2], [0, 2]))
        stacks.op_left[i] = np.tensordot(stacks.op_left[i], trains.solution.cores[i - 1][:, :, 0, :], axes=([0, 2], [0, 1]))

        if trains.operator_gevp is not None:
            stacks.op_gevp_left[i] = np.tensordot(stacks.op_gevp_left[i - 1], np.conjugate(trains.solution.cores[i - 1][:, :, 0, :]), axes=(0, 0))
            stacks.op_gevp_left[i] = np.tensordot(stacks.op_gevp_left[i], trains.operator_gevp.cores[i - 1], axes=([0, 2], [0, 2]))
            stacks.op_gevp_left[i] = np.tensordot(stacks.op_gevp_left[i], trains.solution.cores[i - 1][:, :, 0, :], axes=([0, 2], [0, 1]))

        for j in range(len(trains.previous)):
            stacks.previous_left[j][i] = np.tensordot(stacks.previous_left[j][i - 1], trains.previous[j].cores[i - 1][:, :, 0, :], axes=(0, 0))
            stacks.previous_left[j][i] = np.tensordot(stacks.previous_left[j][i], np.conjugate(trains.solution.cores[i - 1][:, :, 0, :]), axes=([0, 1], [0, 1]))


def __construct_right_stacks(i: int, trains, stacks):
    """
    Construct right stacks.

    Parameters
    ----------
    i : int
        core index
    trains : Object
        collection of tensor trains
    stacks: Object
        collection of stacks
    """

    if i == trains.operator.order - 1:

        # last stack element is 1
        stacks.op_right[i] = np.array([1], ndmin=3)

        if trains.operator_gevp is not None:
            stacks.op_gevp_right[i] = np.array([1], ndmin=3)

        for j in range(len(trains.previous)):
            stacks.previous_right[j][i] = np.array([1], ndmin=2)

    else:

        # contract previous stack element with solution and operator cores
        stacks.op_right[i] = np.tensordot(np.conjugate(trains.solution.cores[i + 1][:, :, 0, :]), stacks.op_right[i + 1], axes=(2, 2))
        stacks.op_right[i] = np.tensordot(trains.operator.cores[i + 1], stacks.op_right[i], axes=([1, 3], [1, 3]))
        stacks.op_right[i] = np.tensordot(trains.solution.cores[i + 1][:, :, 0, :], stacks.op_right[i], axes=([1, 2], [1, 3]))

        if trains.operator_gevp is not None:
            stacks.op_gevp_right[i] = np.tensordot(np.conjugate(trains.solution.cores[i + 1][:, :, 0, :]), stacks.op_gevp_right[i + 1], axes=(2, 2))
            stacks.op_gevp_right[i] = np.tensordot(trains.operator_gevp.cores[i + 1], stacks.op_gevp_right[i], axes=([1, 3], [1, 3]))
            stacks.op_gevp_right[i] = np.tensordot(trains.solution.cores[i + 1][:, :, 0, :], stacks.op_gevp_right[i], axes=([1, 2], [1, 3]))

        for j in range(len(trains.previous)):
            stacks.previous_right[j][i] = np.tensordot(np.conjugate(trains.solution.cores[i + 1][:, :, 0, :]), stacks.previous_right[j][i + 1], axes=(2, 1))   
            stacks.previous_right[j][i] = np.tensordot(trains.previous[j].cores[i + 1][:, :, 0, :], stacks.previous_right[j][i], axes=([1, 2], [1, 2]))


def __construct_micro_matrices(i: int, trains, stacks, shift: float) -> np.ndarray:
    """
    Construct micro matrix for ALS.

    Parameters
    ----------
    i : int
        core index
    trains : Object
        collection of tensor trains
    stacks: Object
        collection of stacks
    shift : float
        shift parameter for known eigenvalues

    Returns
    -------
    np.ndarray
        ith micro matrix
    """

    # contract stack elements and operator core
    micro_op = np.tensordot(stacks.op_left[i], trains.operator.cores[i], axes=(1, 0))
    micro_op = np.tensordot(micro_op, stacks.op_right[i], axes=(4, 1))

    micro_op = micro_op.transpose([1, 2, 5, 0, 3, 4]).reshape(

        trains.solution.ranks[i] * trains.operator.row_dims[i] * trains.solution.ranks[i + 1],
        trains.solution.ranks[i] * trains.operator.col_dims[i] * trains.solution.ranks[i + 1])

    if trains.operator_gevp is not None:
        micro_op_gevp = np.tensordot(stacks.op_gevp_left[i], trains.operator_gevp.cores[i], axes=(1, 0))
        micro_op_gevp = np.tensordot(micro_op_gevp, stacks.op_gevp_right[i], axes=(4, 1))
        micro_op_gevp = micro_op_gevp.transpose([1, 2, 5, 0, 3, 4]).reshape(
            trains.solution.ranks[i] * trains.operator_gevp.row_dims[i] * trains.solution.ranks[i + 1],
            trains.solution.ranks[i] * trains.operator_gevp.col_dims[i] * trains.solution.ranks[i + 1])
    else:
        micro_op_gevp = None

    for j in range(len(trains.previous)):
        tmp = np.tensordot(stacks.previous_left[j][i], trains.previous[j].cores[i][:, :, 0, :], axes=(0, 0))
        tmp = np.tensordot(tmp, stacks.previous_right[j][i], axes=(2, 0))
        tmp = tmp.reshape(trains.solution.ranks[i] * trains.previous[j].row_dims[i] * trains.solution.ranks[i + 1], 1)

        micro_op += shift*tmp.dot(np.conjugate(tmp.T))

    return micro_op, micro_op_gevp


def __update_core(i: int, micro_op: np.ndarray, 
                  micro_op_gevp: 'TT', 
                  number_ev, solution: 'TT', 
                  solver: str, sigma: float, real: bool, direction: str):
    """
    Update TT core for ALS.

    Parameters
    ----------
    i : int
        core index
    micro_op : np.ndarray
        micro matrix for ith TT core
    solution : TT
        approximated solution of the eigenvalue problem
    solver : string
        algorithm for obtaining the solutions of the micro systems
    sigma : float
        find eigenvalues near sigma
    real : bool
        whether to compute only real eigenvalues and eigentensors or not
    direction : string
        'forward' if first half sweep, 'backward' if second half sweep
    """

    # solve the micro system for the ith TT core
    # ------------------------------------------

    eigenvalues  = None
    eigenvectors = None

    if solver == 'eigs':
        v0 = np.ones(micro_op.shape[0])
        eigenvalues, eigenvectors = splin.eigs(micro_op, M=micro_op_gevp, sigma=sigma, k=number_ev, v0=v0)
        idx = np.abs(eigenvalues-sigma).argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    if solver == 'eig':
        # noinspection PyTupleAssignmentBalance
        eigenvalues, eigenvectors = lin.eig(micro_op, b=micro_op_gevp, overwrite_a=True, overwrite_b=True,
                                            check_finite=False)

        idx = (np.abs(eigenvalues - sigma)).argsort()

        eigenvalues  = eigenvalues[idx[:number_ev]]
        eigenvectors = eigenvectors[:, idx[:number_ev]]

    if solver == 'eigh':
        eigenvalues, eigenvectors = lin.eigh(micro_op, b=micro_op_gevp, overwrite_a=True, overwrite_b=True,
                                             check_finite=False, 
                                             subset_by_index=(micro_op.shape[0] - number_ev, micro_op.shape[0] - 1))
        eigenvalues  = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

    if real is True:
        eigenvalues  = np.real(eigenvalues)
        #eigenvectors = np.real(eigenvectors)

    # reshape solution and orthonormalization
    # ---------------------------------------

    # first half sweep
    if direction == 'forward':

        # decompose solution
        [u, _, _] = lin.svd(
            eigenvectors.reshape(solution.ranks[i] * solution.row_dims[i], solution.ranks[i + 1] * number_ev),
            overwrite_a=True, check_finite=False, lapack_driver='gesvd')

        # rank reduction
        r = np.minimum(solution.ranks[i + 1], u.shape[1])
        u = u[:, :r]

        # set new rank
        solution.ranks[i + 1] = r

        # save orthonormal part
        solution.cores[i] = u.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])

    # second half sweep
    if direction == 'backward':

        if i > 0:

            # transpose
            eigenvectors = eigenvectors.transpose()

            # decompose solution
            [_, _, v] = lin.svd(
                eigenvectors.reshape([number_ev * solution.ranks[i], solution.row_dims[i] * solution.ranks[i + 1]]),
                overwrite_a=True, check_finite=False, lapack_driver='gesvd')

            # rank reduction
            r = np.minimum(solution.ranks[i], v.shape[0])
            v = v[:r, :]

            # set new rank
            solution.ranks[i] = r

            # save orthonormal part
            solution.cores[i] = v.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])

        else:

            # last iteration step
            solution.cores[i] = eigenvectors.reshape(solution.ranks[i], solution.row_dims[i], 1,
                                                     solution.ranks[i + 1], number_ev)

    return eigenvalues
