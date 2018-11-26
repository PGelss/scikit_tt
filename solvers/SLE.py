#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la

def als(operator, initial_guess, right_hand_side, repeats = 1, solver = 'solve'):
    """Alternating linear scheme for solving systems of linear equations in the TT format.

    """
    solution = initial_guess.copy()
    stack_left_op = [None] * operator.order
    stack_left_rhs = [None] * operator.order
    stack_right_op = [None] * operator.order
    stack_right_rhs = [None] * operator.order

    for i in range(operator.order - 1, -1, -1):
        stack_right_op[i] = construct_stack_right_op(i, stack_right_op, operator, solution)
        stack_right_rhs[i] = construct_stack_right_rhs(i, stack_right_rhs, right_hand_side, solution)

    current_iteration = 1

    while current_iteration <= repeats:
        for i in range(operator.order):
            stack_left_op[i] = construct_stack_left_op(i, stack_left_op, operator, solution)
            stack_left_rhs[i] = construct_stack_left_rhs(i, stack_left_rhs, right_hand_side, solution)
            micro_op = construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution, solver)
            micro_rhs = construct_micro_rhs_als(i, stack_left_rhs, stack_right_rhs, right_hand_side, solution, solver)
            solution = update_core_als(i, micro_op, micro_rhs, solution, solver, 'forward')
        for i in range(operator.order - 1, -1, -1):
            stack_right_op[i] = construct_stack_right_op(i, stack_right_op, operator, solution)
            stack_right_rhs[i] = construct_stack_right_rhs(i, stack_right_rhs, right_hand_side, solution)
            micro_op = construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution, solver)
            micro_rhs = construct_micro_rhs_als(i, stack_left_rhs, stack_right_rhs, right_hand_side, solution, solver)
            solution = update_core_als(i, micro_op, micro_rhs, solution, solver, 'backward')
        current_iteration += 1
    return solution

def mals(operator, solution, right_hand_side, repeats = 1, threshold = 1e-10, solver = 'solve', max_rank = 0):
    """Alternating linear scheme for solving systems of linear equations in the TT format.

    """

    stack_left_op = [None] * operator.order
    stack_left_rhs = [None] * operator.order
    stack_right_op = [None] * operator.order
    stack_right_rhs = [None] * operator.order

    for i in range(operator.order - 1, 0, -1):
        stack_right_op[i] = construct_stack_right_op(i, stack_right_op, operator, solution)
        stack_right_rhs[i] = construct_stack_right_rhs(i, stack_right_rhs, right_hand_side, solution)

    current_iteration = 1

    while current_iteration <= repeats:
        for i in range(operator.order-1):
            stack_left_op[i] = construct_stack_left_op(i, stack_left_op, operator, solution)
            stack_left_rhs[i] = construct_stack_left_rhs(i, stack_left_rhs, right_hand_side, solution)
            if i < operator.order-2:
                micro_op = construct_micro_matrix_mals(i, stack_left_op, stack_right_op, operator, solution, solver)
                micro_rhs = construct_micro_rhs_mals(i, stack_left_rhs, stack_right_rhs, right_hand_side, solution, solver)
                solution = update_core_mals(i, micro_op, micro_rhs, solution, solver, threshold, max_rank, 'forward')
        for i in range(operator.order - 2, -1, -1):
            stack_right_op[i] = construct_stack_right_op(i+1, stack_right_op, operator, solution)
            stack_right_rhs[i] = construct_stack_right_rhs(i+1, stack_right_rhs, right_hand_side, solution)
            micro_op = construct_micro_matrix_mals(i, stack_left_op, stack_right_op, operator, solution, solver)
            micro_rhs = construct_micro_rhs_mals(i, stack_left_rhs, stack_right_rhs, right_hand_side, solution, solver)
            solution = update_core_mals(i, micro_op, micro_rhs, solution, solver, threshold, max_rank, 'backward')
        current_iteration += 1
    return solution


def construct_stack_left_op(i, stack_left_op, operator, solution):
    if i == 0:
        stack_left_op[i] = np.array([1], ndmin=3)
    else:
        stack_left_op[i] = np.tensordot(stack_left_op[i - 1], solution.cores[i - 1][:, :, 0, :], axes=(0, 0))
        stack_left_op[i] = np.tensordot(stack_left_op[i], operator.cores[i - 1], axes=([0, 2], [0, 2]))
        stack_left_op[i] = np.tensordot(stack_left_op[i], solution.cores[i - 1][:, :, 0, :], axes=([0, 2], [0, 1]))
    return stack_left_op[i]


def construct_stack_left_rhs(i, stack_left_rhs, right_hand_side, solution):
    if i == 0:
        stack_left_rhs[i] = np.array([1], ndmin=2)
    else:
        stack_left_rhs[i] = np.tensordot(stack_left_rhs[i - 1], right_hand_side.cores[i - 1][:, :, 0, :], axes=(0, 0))
        stack_left_rhs[i] = np.tensordot(stack_left_rhs[i], solution.cores[i - 1][:, :, 0, :], axes=([0, 1], [0, 1]))
    return stack_left_rhs[i]


def construct_stack_right_op(i, stack_right_op, operator, solution):
    if i == operator.order - 1:
        stack_right_op[i] = np.array([1], ndmin=3)
    else:
        stack_right_op[i] = np.tensordot(solution.cores[i + 1][:, :, 0, :], stack_right_op[i + 1], axes=(2, 2))
        stack_right_op[i] = np.tensordot(operator.cores[i + 1], stack_right_op[i], axes=([1, 3], [1, 3]))
        stack_right_op[i] = np.tensordot(solution.cores[i + 1][:, :, 0, :], stack_right_op[i], axes=([1, 2], [1, 3]))
    return stack_right_op[i]


def construct_stack_right_rhs(i, stack_right_rhs, right_hand_side, solution):
    if i == right_hand_side.order - 1:
        stack_right_rhs[i] = np.array([1], ndmin=2)
    else:
        stack_right_rhs[i] = np.tensordot(solution.cores[i + 1][:, :, 0, :], stack_right_rhs[i + 1], axes=(2, 1))
        stack_right_rhs[i] = np.tensordot(right_hand_side.cores[i + 1][:, :, 0, :], stack_right_rhs[i],
                                          axes=([1, 2], [1, 2]))
    return stack_right_rhs[i]


def construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, solution, solver):
    micro_op = np.tensordot(stack_left_op[i], operator.cores[i], axes=(1, 0))
    micro_op = np.tensordot(micro_op, stack_right_op[i], axes=(4, 1))
    if solver != 'tensorsolve':
        micro_op = micro_op.transpose([1, 2, 5, 0, 3, 4]).reshape(
            solution.ranks[i] * operator.row_dims[i] * solution.ranks[i + 1],
            solution.ranks[i] * operator.col_dims[i] * solution.ranks[i + 1])
    return micro_op


def construct_micro_matrix_mals(i, stack_left_op, stack_right_op, operator, solution, solver):
    micro_op = np.tensordot(stack_left_op[i], operator.cores[i], axes=(1, 0))
    micro_op = np.tensordot(micro_op, operator.cores[i+1], axes=(4, 0))
    micro_op = np.tensordot(micro_op, stack_right_op[i+1], axes=(6, 1))
    if solver != 'tensorsolve':
        micro_op = micro_op.transpose([1, 2, 4, 7, 0, 3, 5, 6]).reshape(
            solution.ranks[i] * operator.row_dims[i] * operator.row_dims[i+1] * solution.ranks[i + 2],
            solution.ranks[i] * operator.col_dims[i] * operator.col_dims[i + 1] * solution.ranks[i + 2])
    return micro_op


def construct_micro_rhs_als(i, stack_left_rhs, stack_right_rhs, right_hand_side, solution, solver):
    micro_rhs = np.tensordot(stack_left_rhs[i], right_hand_side.cores[i][:, :, 0, :], axes=(0, 0))
    micro_rhs = np.tensordot(micro_rhs, stack_right_rhs[i], axes=(2, 0))
    if solver != 'tensorsolve':
        micro_rhs = micro_rhs.reshape(solution.ranks[i] * right_hand_side.row_dims[i] * solution.ranks[i + 1], 1)
    return micro_rhs

def construct_micro_rhs_mals(i, stack_left_rhs, stack_right_rhs, right_hand_side, solution, solver):
    #print(right_hand_side.cores[i+1].shape)
    micro_rhs = np.tensordot(stack_left_rhs[i], right_hand_side.cores[i][:, :, 0, :], axes=(0, 0))
    micro_rhs = np.tensordot(micro_rhs, right_hand_side.cores[i+1][:, :, 0, :], axes=(2, 0))
    micro_rhs = np.tensordot(micro_rhs, stack_right_rhs[i+1], axes=(3, 0))
    if solver != 'tensorsolve':
        micro_rhs = micro_rhs.reshape(solution.ranks[i] * right_hand_side.row_dims[i] * right_hand_side.row_dims[i+1] * solution.ranks[i + 2], 1)
    return micro_rhs


def update_core_als(i, micro_op, micro_rhs, solution, solver, direction):

    if solver == 'solve':
        solution.cores[i] = np.linalg.solve(micro_op, micro_rhs)
    if solver == 'lu':
        lu = la.lu_factor(micro_op, overwrite_a=True, check_finite=False)
        solution.cores[i] = la.lu_solve(lu, micro_rhs, trans=0, overwrite_b=True, check_finite=False)
    if solver == 'tensorsolve':
        solution.cores[i] = np.linalg.tensorsolve(micro_op, micro_rhs, axes=(0, 3, 4))

    if direction == 'forward' and i < solution.order - 1:
        [q, r] = la.qr(solution.cores[i].reshape(solution.ranks[i] * solution.row_dims[i], solution.ranks[i + 1]),
                       overwrite_a=True, mode='economic', check_finite=False)
        solution.ranks[i + 1] = q.shape[1]
        solution.cores[i] = q.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])
        solution.cores[i + 1] = np.tensordot(r, solution.cores[i + 1], axes=(1, 0))

# > 0 ??????

    elif direction == 'backward' and i > 1:
        [r, q] = la.rq(solution.cores[i].reshape(solution.ranks[i], solution.row_dims[i] * solution.ranks[i + 1]),
                       overwrite_a=True, mode='economic', check_finite=False)
        solution.ranks[i] = q.shape[0]
        solution.cores[i] = q.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])
        solution.cores[i - 1] = np.tensordot(solution.cores[i - 1], r, axes=(3, 0))
    else:
        solution.cores[i] = solution.cores[i].reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])
    return solution

def update_core_mals(i, micro_op, micro_rhs, solution, solver, threshold, max_rank, direction):
    if solver == 'solve':
        solution.cores[i] = la.solve(micro_op, micro_rhs, overwrite_a=True, overwrite_b=True, check_finite=False)
    if solver == 'lu':
        lu = la.lu_factor(micro_op, overwrite_a=True, check_finite=False)
        solution.cores[i] = la.lu_solve(lu, micro_rhs, trans=0, overwrite_b=True, check_finite=False)
    if solver == 'tensorsolve':
        solution.cores[i] = np.linalg.tensorsolve(micro_op, micro_rhs, axes=(0, 3, 4))

    if direction == 'forward':
        [u, s, v] = la.svd(solution.cores[i].reshape(solution.ranks[i]*solution.row_dims[i], solution.row_dims[i+1]*solution.ranks[i + 2]), full_matrices=False, lapack_driver='gesvd')
        if threshold != 0:
            indices = np.where(s / s[0] > threshold)[0]
            u = u[:, indices]
            s = s[indices]
            v = v[indices, :]
        if max_rank != 0:
            u = u[:, :np.minimum(u.shape[1], max_rank)]
            s = s[:np.minimum(s.shape[0], max_rank)]
            v = v[:np.minimum(v.shape[0], max_rank),:]
        solution.ranks[i+1] = s.shape[0]
        solution.cores[i] = u.reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])
        solution.cores[i + 1] = (np.diag(s)@v).reshape(solution.ranks[i+1], solution.row_dims[i+1], 1, solution.ranks[i + 2])
    if direction == 'backward':
        [u, s, v] = la.svd(solution.cores[i].reshape(solution.ranks[i]*solution.row_dims[i], solution.row_dims[i+1]*solution.ranks[i + 2]), full_matrices=False, lapack_driver='gesvd')
        if threshold != 0:
            indices = np.where(s / s[0] > threshold)[0]
            u = u[:, indices]
            s = s[indices]
            v = v[indices, :]
        if max_rank != 0:
            u = u[:, :np.minimum(u.shape[1], max_rank)]
            s = s[:np.minimum(s.shape[0], max_rank)]
            v = v[:np.minimum(v.shape[0], max_rank),:]
        solution.ranks[i + 1] = s.shape[0]
        solution.cores[i + 1] = v.reshape(solution.ranks[i+1], solution.row_dims[i+1], 1, solution.ranks[i + 2])
        if i == 0:
            solution.cores[i] = (u @ np.diag(s)).reshape(solution.ranks[i], solution.row_dims[i], 1, solution.ranks[i + 1])
    return solution