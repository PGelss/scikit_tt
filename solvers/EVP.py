#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as splin

def als(operator, initial_guess, number_ev=1, repeats = 1):
    """Alternating linear scheme for solving eigenvalue problems in the TT format.

    """
    solution = initial_guess.copy()

    # # block TT format
    # block_core = np.zeros(list(solution.cores[0].shape)+[number_ev])
    # for i in range(number_ev):
    #     block_core[:,:,:,:,i] = solution.cores[0]
    #
    # solution.cores[0] = block_core

    stack_left_op = [None] * operator.order
    stack_right_op = [None] * operator.order

    for i in range(operator.order - 1, -1, -1):
        stack_right_op[i] = construct_stack_right_op(i, stack_right_op, operator, solution)

    current_iteration = 1

    while current_iteration <= repeats:
        for i in range(operator.order):
            stack_left_op[i] = construct_stack_left_op(i, stack_left_op, operator, solution)
            if i < operator.order-1:
                micro_op = construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution)
                solution, eigenvalues = update_core_als(i, micro_op, number_ev, solution, 'forward')
        for i in range(operator.order - 1, -1, -1):
            stack_right_op[i] = construct_stack_right_op(i, stack_right_op, operator, solution)
            micro_op = construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution)
            solution, eigenvalues = update_core_als(i, micro_op, number_ev, solution, 'backward')
        current_iteration += 1
    if number_ev == 1:
        solution_final = solution.copy()
        solution_final.cores[0] = solution.cores[0][:,:,:,:,0]
    else:
        solution_final = [None]*number_ev
        for i in range(number_ev):
            solution_final[i] = solution.copy()
            solution_final[i].cores[0] = solution.cores[0][:,:,:,:,i]
    return solution_final, eigenvalues


def construct_stack_left_op(i, stack_left_op, operator, solution):
    if i == 0:
        stack_left_op[i] = np.array([1], ndmin=3)
    else:
        stack_left_op[i] = np.tensordot(stack_left_op[i - 1], solution.cores[i - 1][:, :, 0, :], axes=(0, 0))
        stack_left_op[i] = np.tensordot(stack_left_op[i], operator.cores[i - 1], axes=([0, 2], [0, 2]))
        stack_left_op[i] = np.tensordot(stack_left_op[i], solution.cores[i - 1][:, :, 0, :], axes=([0, 2], [0, 1]))
    return stack_left_op[i]


def construct_stack_right_op(i, stack_right_op, operator, solution):
    if i == operator.order - 1:
        stack_right_op[i] = np.array([1], ndmin=3)
    else:
        stack_right_op[i] = np.tensordot(solution.cores[i + 1][:, :, 0, :], stack_right_op[i + 1], axes=(2, 2))
        stack_right_op[i] = np.tensordot(operator.cores[i + 1], stack_right_op[i], axes=([1, 3], [1, 3]))
        stack_right_op[i] = np.tensordot(solution.cores[i + 1][:, :, 0, :], stack_right_op[i], axes=([1, 2], [1, 3]))
    return stack_right_op[i]


def construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution):
    micro_op = np.tensordot(stack_left_op[i], operator.cores[i], axes=(1, 0))
    micro_op = np.tensordot(micro_op, stack_right_op[i], axes=(4, 1))
    micro_op = micro_op.transpose([1, 2, 5, 0, 3, 4]).reshape(
            solution.ranks[i] * operator.row_dims[i] * solution.ranks[i + 1],
            solution.ranks[i] * operator.col_dims[i] * solution.ranks[i + 1])
    return micro_op


def update_core_als(i, micro_op, number_ev, solution, direction):

    eigenValues, eigenVectors = splin.eigs(micro_op, k = number_ev)
    eig_val = np.real(eigenValues)
    eig_vec = np.real(eigenVectors)
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]

    # eigenValues, eigenVectors = np.linalg.eig(micro_op)
    # idx = (np.abs(eigenValues - 1)).argsort()
    # eig_val = np.real(eigenValues[idx[:number_ev]])
    # eig_vec = np.real(eigenVectors[:,idx[:number_ev]])

    if direction == 'forward':
        u, s, v = sp.linalg.svd(eig_vec.reshape([solution.ranks[i]*solution.row_dims[i], solution.ranks[i+1]*number_ev]), full_matrices=False, lapack_driver='gesvd')
        r = np.minimum(solution.ranks[i+1], u.shape[1])
        u = u[:,:r]
        s = s[:r]
        v = v[:r,:]
        v = v.reshape(r, solution.ranks[i+1],number_ev)[:,:,0]
        solution.ranks[i + 1] = r
        solution.cores[i] = u.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])
        solution.cores[i+1] = np.tensordot(np.diag(s)@v,solution.cores[i+1], axes = (1,0))
    elif direction == 'backward' and i > 0:
        eig_vec=eig_vec.reshape(list(solution.cores[i].shape)+[number_ev]).transpose(4,0,1,2,3)
        u, s, v = sp.linalg.svd(eig_vec.reshape([number_ev*solution.ranks[i], solution.row_dims[i]*solution.ranks[i+1]]), full_matrices=False, lapack_driver='gesvd')
        r = np.minimum(solution.ranks[i], v.shape[0])
        u = u[:, :r]
        u = u.reshape(number_ev, solution.ranks[i], r)[0,:,:]
        s = s[:r]
        v = v[:r, :]
        solution.ranks[i] = r
        solution.cores[i] = v.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])
        solution.cores[i-1] = np.tensordot(solution.cores[i-1],u@np.diag(s), axes=(3,0))
    else:
        solution.cores[i] = eig_vec.reshape(list(solution.cores[0].shape)+[number_ev])
    return solution, eig_val