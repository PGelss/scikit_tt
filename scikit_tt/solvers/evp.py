#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as splin


def als(operator, initial_guess, number_ev=1, repeats=1, solver='eigs', sigma=1, real=True):
    """Alternating linear scheme

    Approximates the leading eigenvalues and corresponding eigentensors of an eigenvalue problem in the TT format. For
    details, see [1]_.

    Parameters
    ----------
    operator: instance of TT class
        TT operator
    initial_guess: instance of TT class
        initial guess for the solution
    number_ev: int, optional
        number of eigenvalues and corresponding eigentensor to compute, default is 1
    repeats: int, optional
        number of repeats of the ALS, default is 1
    solver: string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'eigs' or 'eig', default is 'eigs'
    sigma: float, optional
        find eigenvalues near sigma, default is 1
    real: bool, optional
        whether to compute only real eigenvalues and eigentensors or not, default is True

    Returns
    -------
    solution: instance of TT class or list of instances of TT class
        approximated solution of the eigenvalue problem, if number_ev>1 solution is a list of tensor trains

    References
    ----------
    ..[1] S. Holtz, T. Rohwedder, R. Schneider, "The Alternating Linear Scheme for Tensor Optimization in the Tensor
          Train Format", SIAM Journal on Scientific Computing 34 (2), 2012
    """

    # define solution tensor and eigenvalues
    solution = initial_guess.copy()
    eigenvalues = None

    # define stacks
    stack_left_op = [None] * operator.order
    stack_right_op = [None] * operator.order

    # construct right stacks for the left- and right-hand side
    for i in range(operator.order - 1, -1, -1):
        __construct_stack_right_op(i, stack_right_op, operator, solution)

    # define iteration number
    current_iteration = 1

    # begin ALS
    while current_iteration <= repeats:

        # first half sweep
        for i in range(operator.order):

            # update left stack for the left-hand side
            __construct_stack_left_op(i, stack_left_op, operator, solution)

            if i < operator.order - 1:
                # construct micro system
                micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution)

                # update solution
                eigenvalues = __update_core_als(i, micro_op, number_ev, solution, solver, sigma, real, 'forward')

        # second half sweep
        for i in range(operator.order - 1, -1, -1):
            # update right stack for the left--hand side
            __construct_stack_right_op(i, stack_right_op, operator, solution)

            # construct micro system
            micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution)

            # update solution
            eigenvalues = __update_core_als(i, micro_op, number_ev, solution, solver, sigma, real, 'backward')

        # increase iteration number
        current_iteration += 1

    # define form of the final solution depending on the number of eigenvalues to compute
    if number_ev == 1:
        solution_final = solution.copy()
        solution_final.cores[0] = solution.cores[0][:, :, :, :, 0]
    else:
        solution_final = [None] * number_ev
        for i in range(number_ev):
            solution_final[i] = solution.copy()
            solution_final[i].cores[0] = solution.cores[0][:, :, :, :, i]

    return eigenvalues, solution_final


def __construct_stack_left_op(i, stack_left_op, operator, solution):
    """Construct left stack for left-hand side

    Parameters
    ----------
    i: int
        core index
    stack_left_op: list of ndarrays
        left stack for left-hand side
    operator: instance of TT class
        TT operator of the system of linear equations
    solution: instance of TT class
        approximated solution of the system of linear equations
    """

    if i == 0:

        # first stack element is 1
        stack_left_op[i] = np.array([1], ndmin=3)

    else:

        # contract previous stack element with solution and operator cores
        stack_left_op[i] = np.tensordot(stack_left_op[i - 1], solution.cores[i - 1][:, :, 0, :], axes=(0, 0))
        stack_left_op[i] = np.tensordot(stack_left_op[i], operator.cores[i - 1], axes=([0, 2], [0, 2]))
        stack_left_op[i] = np.tensordot(stack_left_op[i], solution.cores[i - 1][:, :, 0, :], axes=([0, 2], [0, 1]))


def __construct_stack_right_op(i, stack_right_op, operator, solution):
    """Construct right stack for left-hand side

    Parameters
    ----------
    i: int
        core index
    stack_right_op: list of ndarrays
        right stack for left-hand side
    operator: instance of TT class
        TT operator side of the system of linear equations
    solution: instance of TT class
        approximated solution of the system of linear equations
    """

    if i == operator.order - 1:

        # last stack element is 1
        stack_right_op[i] = np.array([1], ndmin=3)

    else:

        # contract previous stack element with solution and operator cores
        stack_right_op[i] = np.tensordot(solution.cores[i + 1][:, :, 0, :], stack_right_op[i + 1], axes=(2, 2))
        stack_right_op[i] = np.tensordot(operator.cores[i + 1], stack_right_op[i], axes=([1, 3], [1, 3]))
        stack_right_op[i] = np.tensordot(solution.cores[i + 1][:, :, 0, :], stack_right_op[i], axes=([1, 2], [1, 3]))


def __construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution):
    """Construct micro matrix for ALS

    Parameters
    ----------
    i: int
        core index
    stack_left_op: list of ndarrays
        left stack for left-hand side
    stack_right_op: list of ndarrays
        right stack for left-hand side
    operator: instance of TT class
        TT operator of the system of linear equations
    solution: instance of TT class
        approximated solution of the system of linear equations

    Returns
    -------
    micro_op: ndarray
        ith micro matrix
    """

    # contract stack elements and operator core
    micro_op = np.tensordot(stack_left_op[i], operator.cores[i], axes=(1, 0))
    micro_op = np.tensordot(micro_op, stack_right_op[i], axes=(4, 1))

    # transpose and reshape micro matrix
    micro_op = micro_op.transpose([1, 2, 5, 0, 3, 4]).reshape(
        solution.ranks[i] * operator.row_dims[i] * solution.ranks[i + 1],
        solution.ranks[i] * operator.col_dims[i] * solution.ranks[i + 1])
    return micro_op


def __update_core_als(i, micro_op, number_ev, solution, solver, sigma, real, direction):
    """Update TT core for ALS

    Parameters
    ----------
    i: int
        core index
    micro_op: ndarray
        micro matrix for ith TT core
    solution: instance of TT class
        approximated solution of the eigenvalue problem
    solver: string
        algorithm for obtaining the solutions of the micro systems
    sigma: float
        find eigenvalues near sigma
    real: bool
        whether to compute only real eigenvalues and eigentensors or not
    direction: string
        'forward' if first half sweep, 'backward' if second half sweep
    """

    # solve the micro system for the ith TT core
    # ------------------------------------------

    eigenvalues = None
    eigenvectors = None
    if solver == 'eigs':
        eigenvalues, eigenvectors = splin.eigs(micro_op, sigma=sigma, k=number_ev)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    if solver == 'eig':
        eigenvalues, eigenvectors = np.linalg.eig(micro_op)
        idx = (np.abs(eigenvalues - sigma)).argsort()
        eigenvalues = eigenvalues[idx[:number_ev]]
        eigenvectors = eigenvectors[:, idx[:number_ev]]
    if real is True:
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

    # reshape solution and orthonormalization
    # ---------------------------------------

    # first half sweep
    if direction == 'forward':
        # decompose solution
        [u, _, _] = lin.svd(
            eigenvectors.reshape(solution.ranks[i] * solution.row_dims[i], solution.ranks[i + 1] * number_ev),
            overwrite_a=True, check_finite=False)

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
                overwrite_a=True, check_finite=False)

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
