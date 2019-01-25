#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as splin
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.sle as sle


def als(operator, initial_guess, operator_gevp=None, number_ev=1, repeats=1, solver='eig', sigma=1, real=True):
    """Alternating linear scheme

    Approximates eigenvalues and corresponding eigentensors of an (generalized) eigenvalue problem in the TT format.
    For details, see [1]_.

    Parameters
    ----------
    operator: instance of TT class
        TT operator, left-hand side
    initial_guess: instance of TT class
        initial guess for the solution
    operator_gevp: instance of TT class, optional
        TT operator, right-hand side (for generalized eigenvalue problems), default is None
    number_ev: int, optional
        number of eigenvalues and corresponding eigentensor to compute, default is 1
    repeats: int, optional
        number of repeats of the ALS, default is 1
    solver: string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'eig', 'eigs' or 'eigh', default is 'eig'
    sigma: float, optional
        find eigenvalues near sigma, default is 1
    real: bool, optional
        whether to compute only real eigenvalues and eigentensors or not, default is True

    Returns
    -------
    eigenvalues: float or list of floats
        approximated eigenvalues, if number_ev>1 eigenvalues is a list of floats
    eigentensors: instance of TT class or list of instances of TT class
        approximated eigentensors, if number_ev>1 eigentensors is a list of tensor trains

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
    stack_left_op_gevp = [None] * operator.order
    stack_right_op_gevp = [None] * operator.order

    # define micro operator for generalized eigenvalue problems
    micro_op_gevp = None

    # construct right stacks for the left-hand side
    for i in range(operator.order - 1, -1, -1):
        __construct_stack_right_op(i, stack_right_op, operator, solution)

    # construct right stacks for the right-hand side
    if operator_gevp is not None:
        for i in range(operator.order - 1, -1, -1):
            __construct_stack_right_op(i, stack_right_op_gevp, operator_gevp, solution)

    # define iteration number
    current_iteration = 1

    # begin ALS
    while current_iteration <= repeats:

        # first half sweep
        for i in range(operator.order):

            # update left stack for the left-hand side
            __construct_stack_left_op(i, stack_left_op, operator, solution)

            # update left stack for the right-hand side
            if operator_gevp is not None:
                __construct_stack_left_op(i, stack_left_op_gevp, operator_gevp, solution)

            if i < operator.order - 1:

                # construct micro system
                micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution)
                if operator_gevp is not None:
                    micro_op_gevp = __construct_micro_matrix_als(i, stack_left_op_gevp, stack_right_op_gevp,
                                                                 operator_gevp, solution)

                # update solution
                eigenvalues = __update_core_als(i, micro_op, micro_op_gevp, number_ev, solution, solver, sigma, real,
                                                'forward')

        # second half sweep
        for i in range(operator.order - 1, -1, -1):
            # update right stack for the left-hand side
            __construct_stack_right_op(i, stack_right_op, operator, solution)

            # update right stack for the right-hand side
            if operator_gevp is not None:
                __construct_stack_right_op(i, stack_right_op_gevp, operator_gevp, solution)

            # construct micro system
            micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution)
            if operator_gevp is not None:
                micro_op_gevp = __construct_micro_matrix_als(i, stack_left_op_gevp, stack_right_op_gevp, operator_gevp,
                                                             solution)

            # update solution
            eigenvalues = __update_core_als(i, micro_op, micro_op_gevp, number_ev, solution, solver, sigma, real,
                                            'backward')

        # increase iteration number
        current_iteration += 1

    # define form of the final solution depending on the number of eigenvalues to compute
    if number_ev == 1:
        eigentensors = solution.copy()
        eigentensors.cores[0] = solution.cores[0][:, :, :, :, 0]
        eigenvalues = eigenvalues[0]
    else:
        eigentensors = [None] * number_ev
        for i in range(number_ev):
            eigentensors[i] = solution.copy()
            eigentensors[i].cores[0] = solution.cores[0][:, :, :, :, i]

    return eigenvalues, eigentensors


def power_method(operator, initial_guess, operator_gevp=None, repeats=10, sigma=0.999):
    """Inverse power iteration method

    Approximates eigenvalues and corresponding eigentensors of an (generalized) eigenvalue problem in the TT format.
    For details, see [1]_.

    Parameters
    ----------
    operator: instance of TT class
        TT operator, left-hand side
    initial_guess: instance of TT class
        initial guess for the solution
    operator_gevp: instance of TT class, optional
        TT operator, right-hand side (for generalized eigenvalue problems), default is None
    repeats: int, optional
        number of iterations, default is 10
    sigma: float, optional
        find eigenvalues near sigma, default is 1

    Returns
    -------
    eigenvalue: float
        approximated eigenvalue
    eigentensor: instance of TT class
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


def __update_core_als(i, micro_op, micro_op_gevp, number_ev, solution, solver, sigma, real, direction):
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
        v0 = np.ones(micro_op.shape[0])
        eigenvalues, eigenvectors = splin.eigs(micro_op, M=micro_op_gevp, sigma=sigma, k=number_ev, v0=v0)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    if solver == 'eig':
        # noinspection PyTupleAssignmentBalance
        eigenvalues, eigenvectors = lin.eig(micro_op, b=micro_op_gevp, overwrite_a=True, overwrite_b=True,
                                            check_finite=False)
        idx = (np.abs(eigenvalues - sigma)).argsort()
        eigenvalues = eigenvalues[idx[:number_ev]]
        eigenvectors = eigenvectors[:, idx[:number_ev]]
    if solver == 'eigh':
        eigenvalues, eigenvectors = lin.eigh(micro_op, b=micro_op_gevp, overwrite_a=True, overwrite_b=True,
                                             check_finite=False,
                                             eigvals=(micro_op.shape[0] - number_ev, micro_op.shape[0] - 1))
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
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
