#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.linalg as lin
import scikit_tt.data_driven.transform as tdt
from scikit_tt.data_driven.transform import Function
from scikit_tt.tensor_train import TT
import scikit_tt.utils as utl
import time as _time


def arr(x_data, y_data, basis_list, initial_guess, repeats=1, rcond=10**-2, string='ARR', progress=True):
    """
    Alternating ridge regression on transformed data tensors.

    Approximates the solution of a ridge regression in the TT format. For details, see [1]_.

    Parameters
    ----------
    x_data : np.ndarray
        snapshot matrix which is transformed
    y_data : np.ndarray
        snapshot matrix for the right-hand side
    basis_list : list[list[Function]]
        list of basis functions in every mode
    initial_guess : TT
        initial guess for the solution of operator @ x = right_hand_side
    repeats : int, optional
        number of repeats of the ALS, default is 1
    rcond : float, optional
        cut-off ratio for singular values of the subproblems, parameter for NumPy's lstsq, default is 1e-2
    string : string, optional
        string to show above progress bar
    progress : boolean, optional
        whether to show progress bar, default is True

    Returns
    -------
    TT
        approximated solution of the regression problem

    References
    ----------
    ..[1] S. Klus, P. Gelß, "Tensor-Based Algorithms for Image Classification", Algorithms, 2019
    """

    # progress bar
    start_time = utl.progress(string, 0, show=progress)

    # define order of the transformed data tensor
    order = len(basis_list)

    # number of core optimizations and counter
    number_of_optimizations = y_data.shape[0]*repeats*(2*order-1)
    counter = 0


    if isinstance(initial_guess, list):
        solution = initial_guess
    else:
        # define list of solutions
        solution = [initial_guess.copy() for _ in range(y_data.shape[0])]

    x_data_org = x_data
    y_data_org = y_data
    m = x_data_org.shape[1]

    counter =0

    for k in range(y_data.shape[0]):

        # define right-hand side and left stack
        rhs = y_data[k, :]
        stack_left = [None] * order
        stack_right = [None] * order

        # copy initial right stack and solution
        # construct right stacks for the left- and right-hand side
        for i in range(order - 1, -1, -1):
            __arr_construct_stack_right(i, stack_right, x_data, basis_list, solution[k])

        # define iteration number
        current_iteration = 1

        # begin ARR
        while current_iteration <= repeats:

            # first half sweep
            for i in range(order):

                # update left stack
                __arr_construct_stack_left(i, stack_left, x_data, basis_list, solution[k])

                if i < order - 1:
                    # construct micro system
                    micro_matrix = __arr_construct_micro_matrix(i, stack_left, stack_right, x_data, basis_list, solution[k])

                    # update solution
                    __arr_update_core(i, micro_matrix, rhs, solution[k], rcond, 'forward')

                    # progress bar
                    counter += 1
                    
            utl.progress(string, 100 * counter/number_of_optimizations, cpu_time=_time.time() - start_time, show=progress)
 
            # second half sweep
            for i in range(order - 1, -1, -1):
                # update right stack
                __arr_construct_stack_right(i, stack_right, x_data, basis_list, solution[k])

                # construct micro system
                micro_matrix = __arr_construct_micro_matrix(i, stack_left, stack_right, x_data, basis_list, solution[k])

                # update solution
                __arr_update_core(i, micro_matrix, rhs, solution[k], rcond, 'backward')

                # progress bar
                counter += 1
                
            utl.progress(string, 100 * counter/number_of_optimizations, cpu_time=_time.time() - start_time, show=progress)

            # increase iteration number
            current_iteration += 1

    return solution


def mandy_cm(x, y, phi, threshold=0):
    """
    Multidimensional Approximation of Nonlinear Dynamics (MANDy).

    Coordinate-major approach for construction of the tensor train xi. See [1]_ for details.

    Parameters
    ----------
    x : np.ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    y : np.ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    phi : list[Function]
        list of basis functions
    threshold : float, optional
        threshold for SVDs, default is 0

    Returns
    -------
    TT
        tensor train of coefficients for chosen basis functions

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           Journal of Computational Nonlinear Dynamics, 2019
    """

    # parameters
    d = x.shape[0]
    m = x.shape[1]

    # construct transformed data tensor
    psi = tdt.coordinate_major(x, phi)

    # define xi as pseudoinverse of psi
    xi = psi.pinv(d, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[d] = (xi.cores[d].reshape([xi.ranks[d], m]).dot(y.transpose())).reshape(xi.ranks[d], d, 1, 1)

    # set new row dimension
    xi.row_dims[d] = d

    return xi


def mandy_fm(x, y, phi, threshold=0, add_one=True):
    """
    Multidimensional Approximation of Nonlinear Dynamics (MANDy).

    Function-major approach for construction of the tensor train xi. See [1]_ for details.

    Parameters
    ----------
    x : np.ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    y : np.ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    phi : list[Function]
        list of basis functions
    threshold : float, optional
        threshold for SVDs, default is 0
    add_one : bool, optional
        whether to add the basis function 1 to the cores or not, default is True

    Returns
    -------
    TT
        tensor train of coefficients for chosen basis functions

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           Journal of Computational Nonlinear Dynamics, 2019
    """

    # parameters
    d = x.shape[0]
    m = x.shape[1]
    p = len(phi)

    # construct transformed data tensor
    psi = tdt.function_major(x, phi, add_one=add_one)

    # define xi as pseudoinverse of psi
    xi = psi.pinv(p, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[p] = (xi.cores[p].reshape([xi.ranks[p], m]).dot(y.transpose())).reshape(xi.ranks[p], d, 1, 1)

    # set new row dimension
    xi.row_dims[p] = d

    return xi


def mandy_kb(x, y, basis_list):
    """
    Kernel-based MANDy.

    Kernel-based version of MANDy for solving regression problems on transformed data tensors. See [1]_ and _[2] 
    for details.

    Parameters
    ----------
    x : np.ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    y : np.ndarray
        corresponding snapshot matrix of size d' x m (e.g., derivatives)
    basis_list : list[list[Function]]
        list of basis functions in every mode

    Returns
    -------
    np.ndarray
        matrix such that the solution of the regression problem can be expressed as z@psi_1^T

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           Journal of Computational Nonlinear Dynamics, 2019
    .. [2] S. Klus, P. Gelß, "Tensor-Based Algorithms for Image Classification", Algorithms, 2019
    """

    # compute Gram matrix
    gram = tdt.gram(x, x, basis_list)

    # solve system of linear equations
    if np.linalg.cond(gram) < 1 / sys.float_info.epsilon:
        z = lin.solve(gram, y.T, assume_a='pos', check_finite=False).T
    else:
        z, _, _, _ = lin.lstsq(gram, y.T, lapack_driver='gelss')
        z = z.T

    return z

# private functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def __arr_construct_stack_left(i, stack_left, x_data, basis_list, solution):
    """
    Construct left stack for ARR

    Parameters
    ----------
    i : int
        core index
    stack_left : list[np.ndarray]
        left stack
    x_data : np.ndarray
        snapshot matrix which is transformed
    basis_list : list[list[Function]]
        list of basis functions in every mode
    solution : TT
        approximated solution of the system of linear equations
    """

    if i == 0:

        # last stack element is 1
        stack_left[i] = np.array([1], ndmin=2)

    else:

        # number of basis functions and number of snapshots
        n = len(basis_list[i - 1])
        m = x_data.shape[1]

        # contract previous stack element with solution and TDT cores
        stack_left[i] = np.array([[basis_list[i-1][k](x_data[:, j]) for j in range(m)] for k in range(n)])
        stack_left[i] = np.einsum('ij, kj, ikl -> lj', stack_left[i - 1], stack_left[i], solution.cores[i - 1][:,:,0,:])


def __arr_construct_stack_right(i, stack_right, x_data, basis_list, solution):
    """
    Construct right stack for ARR.

    Parameters
    ----------
    i : int
        core index
    stack_right : list[np.ndarray]
        right stack
    x_data : np.ndarray
        snapshot matrix which is transformed
    basis_list : list[list[Function]]
        list of basis functions in every mode
    solution : TT
        approximated solution of the system of linear equations
    """

    if i == len(basis_list) - 1:

        # last stack element is 1
        stack_right[i] = np.array([1], ndmin=2)

    else:

        # number of basis functions and number of snapshots
        n = len(basis_list[i+1])
        m = x_data.shape[1]

        # contract previous stack element with solution and TDT cores
        stack_right[i] = np.array([[basis_list[i+1][k](x_data[:, j]) for j in range(m)] for k in range(n)])
        stack_right[i] = np.einsum('ikl, kj, lj -> ij', solution.cores[i + 1][:,:,0,:], stack_right[i], stack_right[i + 1])


def __arr_construct_micro_matrix(i, stack_left, stack_right, x_data, basis_list, solution):
    """
    Construct micro matrix for ARR.

    Parameters
    ----------
    i : int
        core index
    stack_left : list[np.ndarray]
        left stack
    stack_right : list[np.ndarray]
        right stack
    x_data : np.ndarray
        snapshot matrix which is transformed
    basis_list : list[list[Function]]
        list of basis functions in every mode
    solution : TT
        approximated solution of the system of linear equations

    Returns
    -------
    np.ndarray
        ith micro matrix
    """

    # number of basis functions and number of snapshots
    n = len(basis_list[i])
    m = x_data.shape[1]

    # contract stack elements and TDT core
    micro_matrix = np.array([[basis_list[i][k](x_data[:, j]) for j in range(m)] for k in range(n)])
    micro_matrix = np.einsum('ij,kj,lj->iklj', stack_left[i], micro_matrix, stack_right[i])
    micro_matrix = micro_matrix.reshape([solution.ranks[i] * solution.row_dims[i] * solution.ranks[i + 1], m])

    return micro_matrix


def __arr_update_core(i, micro_matrix, rhs, solution, rcond, direction):
    """
    Update TT core for ARR.

    Parameters
    ----------
    i : int
        core index
    micro_op : np.ndarray
        micro matrix for ith TT core
    rhs : np.ndarray
        right-hand side for ith TT core
    solution : TT
        approximated solution of the system of linear equations
    rcond : float
        cut-off ratio for singular values of the subproblems, parameter for NumPy's lstsq
    direction : string
        'forward' if first half sweep, 'backward' if second half sweep
    """

    # solve the micro system for the ith TT core
    solution.cores[i], _, _, _ = lin.lstsq(micro_matrix.T, rhs, cond=rcond, lapack_driver='gelss')

    # reshape solution and orthonormalization
    # ---------------------------------------

    # first half sweep
    if direction == 'forward':
        # decompose solution
        [q, _] = lin.qr(
            solution.cores[i].reshape(solution.ranks[i] * solution.row_dims[i], solution.ranks[i + 1]),
            overwrite_a=True, mode='economic', check_finite=False)

        # set new rank
        solution.ranks[i + 1] = q.shape[1]

        # save orthonormal part
        solution.cores[i] = q.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])

    # second half sweep
    if direction == 'backward':

        if i > 0:

            # decompose solution
            [_, q] = lin.rq(
                solution.cores[i].reshape(solution.ranks[i], solution.row_dims[i] * solution.ranks[i + 1]),
                overwrite_a=True, mode='economic', check_finite=False)

            # set new rank
            solution.ranks[i] = q.shape[0]

            # save orthonormal part
            solution.cores[i] = q.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])

        else:

            # last iteration step
            solution.cores[i] = solution.cores[i].reshape(solution.ranks[i], solution.row_dims[i], 1,
                                                          solution.ranks[i + 1])
