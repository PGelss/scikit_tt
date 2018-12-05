#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scikit_tt.tensor_train as tt
import scikit_tt.utils as utl
from scikit_tt.solvers import sle
import numpy as np


def implicit_euler(operator, initial_value, initial_guess, step_sizes, repeats=1, tt_solver='als', threshold=1e-12,
                   max_rank=np.infty, micro_solver='solve', progress=True):
    """Implicit Euler method for linear differential equations in the TT format

    Parameters
    ----------
    operator: instance of TT class
        TT operator of the differential equation
    initial_value: instance of TT class
        initial value of the differential equation
    initial_guess: instance of TT class
        initial guess for the first step
    step_sizes: list of floats
        step sizes for the application of the implicit Euler method
    repeats: int, optional
        number of repeats of the ALS in each iteration step, default is 1
    tt_solver: string, optional
        algorithm for solving the systems of linear equations in the TT format, default is 'als'
    threshold: float, optional
        threshold for reduced SVD decompositions, default is 1e-12
    max_rank: int, optional
        maximum rank of the solution, default is infinity
    micro_solver: string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'solve' or 'lu', default is 'solve'
    progress: bool, optional
        whether to show the progress of the algorithm or not, default is True

    Returns
    -------
    solution: list of instances of the TT class
        numerical solution of the differential equation
    """

    # define solution
    solution = [initial_value]

    # define temporary tensor train
    tt_tmp = initial_guess

    # begin implicit Euler method
    # ---------------------------

    for i in range(len(step_sizes)):

        # solve system of linear equations for current time step
        if tt_solver == 'als':
            tt_tmp = sle.als(tt.eye(operator.row_dims) - step_sizes[i] * operator, tt_tmp, solution[i],
                             solver=micro_solver, repeats=repeats)
        if tt_solver == 'mals':
            tt_tmp = sle.mals(tt.eye(operator.row_dims) - step_sizes[i] * operator, tt_tmp, solution[i],
                              solver=micro_solver, threshold=threshold, repeats=repeats, max_rank=max_rank)

        # normalize solution
        tt_tmp = (1 / tt_tmp.norm(p=1)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        if progress is True:
            utl.progress('Running implicit Euler method', 100 * i / (len(step_sizes) - 1))

    return solution


def trapezoidal_rule(operator, initial_value, initial_guess, step_sizes, repeats=1, tt_solver='als', threshold=1e-12,
                     max_rank=np.infty, micro_solver='solve', progress=True):
    """Trapezoidal rule for linear differential equations in the TT format

    Parameters
    ----------
    operator: instance of TT class
        TT operator of the differential equation
    initial_value: instance of TT class
        initial value of the differential equation
    initial_guess: instance of TT class
        initial guess for the first step
    step_sizes: list of floats
        step sizes for the application of the trapezoidal rule
    repeats: int, optional
        number of repeats of the ALS in each iteration step, default is 1
    tt_solver: string, optional
        algorithm for solving the systems of linear equations in the TT format, default is 'als'
    threshold: float, optional
        threshold for reduced SVD decompositions, default is 1e-12
    max_rank: int, optional
        maximum rank of the solution, default is infinity
    micro_solver: string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'solve' or 'lu', default is 'solve'
    progress: bool, optional
        whether to show the progress of the algorithm or not, default is True

    Returns
    -------
    solution: list of instances of the TT class
        numerical solution of the differential equation
    """

    # define solution
    solution = [initial_value]

    # define temporary tensor train
    tt_tmp = initial_guess

    # begin trapezoidal rule
    # ----------------------

    for i in range(len(step_sizes)):

        # solve system of linear equations for current time step
        if tt_solver == 'als':
            tt_tmp = sle.als(tt.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator, tt_tmp,
                             (tt.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[i],
                             solver=micro_solver, repeats=repeats)
        if tt_solver == 'mals':
            tt_tmp = sle.mals(tt.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator, tt_tmp,
                              (tt.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[i],
                              solver=micro_solver, repeats=repeats, threshold=threshold, max_rank=max_rank)

        # normalize solution
        tt_tmp = (1 / tt_tmp.norm(p=1)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        if progress is True:
            utl.progress('Running trapezoidal rule', 100 * i / (len(step_sizes) - 1))

    return solution

# TODO:
#
# def adaptive_als(operator, initial_value, initial_guess, first_step, T_end, repeats=1, loc_tol=10 ** -3, tau_max=10,
#                  max_factor=1.25, safety_factor=0.9, solver='lu', compute_errors=False):
#     solution = [None]
#     solution[0] = initial_value
#     X = initial_guess
#     errors = []
#     time = 0
#     tau = first_step
#     i = 0
#     while time < T_end:
#
#         # S1 = SLE.als(TT.eye(operator.row_dims) - 0.5 * tau * operator, X,
#         #            (TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
#         #            repeats=repeats)
#
#         S1 = SLE.als(TT.eye(operator.row_dims) - tau * operator, X, solution[i], solver=solver, repeats=repeats)
#         S1 = (1 / S1.norm(p=1)) * S1
#
#         # S2 = SLE.als(TT.eye(operator.row_dims) - 0.5 * tau * operator, X,
#         #            (TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
#         #            repeats=repeats)
#         # S2 = SLE.als(TT.eye(operator.row_dims) - 0.5 * tau * operator, S2,
#         #             (TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
#         #             repeats=repeats)
#
#         S2 = SLE.als(TT.eye(operator.row_dims) - 0.5 * tau * operator, X, solution[i], solver=solver,
#                      repeats=repeats)
#         S2 = SLE.als(TT.eye(operator.row_dims) - 0.5 * tau * operator, S2, solution[i], solver=solver,
#                      repeats=repeats)
#         S2 = (1 / S2.norm(p=1)) * S2
#
#         loc_err = (S1 - S2).norm()
#         closeness = (operator @ S1).norm()
#         factor = (loc_tol) / loc_err
#
#         tau_new = np.amin([max_factor * tau, safety_factor * factor * tau])
#         if factor > 1:
#             time = time + tau
#             tau = np.amin([tau_new, T_end - time, tau_max])
#             solution.append(S1.copy())
#             # print((operator@solution[-1]).norm())
#             X = S1
#             print('tau: ' + str("%.2e" % tau) + ', closeness: ' + str("%.2e" % closeness))
#         else:
#             tau = tau_new
#     return solution
