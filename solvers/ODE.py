#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import solvers.SLE as SLE
import scikit_tt.tt as tt
import numpy as np
import tools.tools as tls


def adaptive_als(operator, initial_value, initial_guess, first_step, T_end, repeats=1, loc_tol=10 ** -3, tau_max=10,
                 max_factor=1.25, safety_factor=0.9, solver='lu', compute_errors=False):
    solution = [None]
    solution[0] = initial_value
    X = initial_guess
    errors = []
    time = 0
    tau = first_step
    i = 0
    while time < T_end:

        # S1 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, X,
        #            (tt.TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
        #            repeats=repeats)

        S1 = SLE.als(tt.TT.eye(operator.row_dims) - tau * operator, X, solution[i], solver=solver, repeats=repeats)
        S1 = (1 / S1.norm(p=1)) * S1

        # S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, X,
        #            (tt.TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
        #            repeats=repeats)
        # S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, S2,
        #             (tt.TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
        #             repeats=repeats)

        S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, X, solution[i], solver=solver,
                     repeats=repeats)
        S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, S2, solution[i], solver=solver,
                     repeats=repeats)
        S2 = (1 / S2.norm(p=1)) * S2

        loc_err = (S1 - S2).norm()
        closeness = (operator @ S1).norm()
        factor = (loc_tol) / loc_err

        tau_new = np.amin([max_factor * tau, safety_factor * factor * tau])
        if factor > 1:
            time = time + tau
            tau = np.amin([tau_new, T_end - time, tau_max])
            solution.append(S1.copy())
            # print((operator@solution[-1]).norm())
            X = S1
            print('tau: ' + str("%.2e" % tau) + ', closeness: ' + str("%.2e" % closeness))
        else:
            tau = tau_new
    return solution


def implicit_euler_als(operator, initial_value, initial_guess, step_sizes, repeats=1, solver='solve',
                       compute_errors=False):
    solution = [None]
    solution[0] = initial_value
    X = initial_guess
    errors = []

    for i in range(len(step_sizes)):

        X = SLE.als(tt.TT.eye(operator.row_dims) - step_sizes[i] * operator, X, solution[i], solver=solver,
                    repeats=repeats)
        X = (1 / X.norm(p=1)) * X
        solution.append(X.copy())
        sys.stdout.write(
            '\r' + 'Running implicit_euler_mals... ' + str(int(1000 * (i + 1) / len(step_sizes)) / 10) + '%')
        if compute_errors:
            errors.append(
                ((tt.TT.eye(operator.row_dims) - step_sizes[i] * operator) @ solution[i + 1] - solution[i]).norm() /
                solution[i].norm())

    if compute_errors:
        return solution, errors
    else:
        return solution


def implicit_euler_mals(operator, initial_value, initial_guess, step_sizes, repeats=1, solver='solve',
                        threshold=1e-12, max_rank=np.infty, compute_errors=False):
    """Implicit Euler method for linear differential equations in the TT format using MALS

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
        number of repeats of the ALS in each iteration step
    solver: string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'solve' or 'lu', default is 'solve'
    threshold: float, optional
        threshold for reduced SVD decompositions, default is 1e-12
    max_rank: int, optional
        maximum rank of the solution, default is infinity
    compute_errors: bool, optional
        whether to compute the relative errors of the systems of linear equations in each step, default is False

    Returns
    -------
    solution: list of instances of the TT class
        numerical solution of the differential equation
    errors: list of floats, optional
        relative errors of the systems of linear equations
    """

    # define solution
    solution = [initial_value]

    # define temporary tensor train
    tt_tmp = initial_guess

    # define errors
    errors = []

    # begin implicit Euler method
    # ---------------------------

    for i in range(len(step_sizes)):

        # solve system of linear equations for current time step
        tt_tmp = SLE.mals(tt.TT.eye(operator.row_dims) - step_sizes[i] * operator, tt_tmp, solution[i],
                          solver=solver, threshold=threshold, repeats=repeats, max_rank=max_rank)

        # normalize solution
        tt_tmp = (1 / tt_tmp.norm(p=1)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        tls.progress('Running implicit_euler_mals', 100 * i / (len(step_sizes) - 1))

        # compute error and append
        if compute_errors:
            errors.append(
                ((tt.TT.eye(operator.row_dims) - step_sizes[i] * operator) @ solution[i + 1] - solution[i]).norm() /
                solution[i].norm())

    if compute_errors:
        return solution, errors
    else:
        return solution


def trapezoidal_rule_als(operator, initial_value, initial_guess, step_sizes, repeats=1, solver='solve',
                         compute_errors=False):
    """Trapezoidal rule for linear differential equations in the TT format using ALS

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
        number of repeats of the ALS in each iteration step
    solver: string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'solve' or 'lu', default is 'solve'
    compute_errors: bool, optional
        whether to compute the relative errors of the systems of linear equations in each step, default is False

    Returns
    -------
    solution: list of instances of the TT class
        numerical solution of the differential equation
    errors: list of floats, optional
        relative errors of the systems of linear equations
    """

    # define solution
    solution = [initial_value]

    # define temporary tensor train
    tt_tmp = initial_guess

    # define errors
    errors = []

    # begin trapezoidal rule
    # ----------------------

    for i in range(len(step_sizes)):

        # solve system of linear equations for current time step
        tt_tmp = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator, tt_tmp,
                         (tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[i], solver=solver,
                         repeats=repeats)

        # normalize solution
        tt_tmp = (1 / tt_tmp.norm(p=1)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        tls.progress('Running trapezoidal_rule_als', 100 * i / (len(step_sizes) - 1))

        # compute error and append
        if compute_errors:
            errors.append(((tt.TT.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator) @ solution[i + 1] - (
                    tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[i]).norm() / (
                                  (tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[
                              i]).norm())

    if compute_errors:
        return solution, errors
    else:
        return solution


def trapezoidal_rule_mals(operator, initial_value, initial_guess, step_sizes, repeats=1, solver='solve',
                          threshold=10 ** -14, max_rank=30, compute_errors=False):
    solution = [None]
    solution[0] = initial_value
    X = initial_guess
    errors = []

    for i in range(len(step_sizes)):

        X = SLE.mals(tt.TT.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator, X,
                     (tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[i], solver=solver,
                     threshold=threshold, repeats=repeats, max_rank=max_rank)
        X = (1 / X.norm(p=1)) * X
        solution.append(X.copy())
        sys.stdout.write(
            '\r' + 'Running trapezoidal_rule_mals... ' + str(int(1000 * (i + 1) / len(step_sizes)) / 10) + '%')
        if compute_errors:
            errors.append(((tt.TT.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator) @ solution[i + 1] - (
                    tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[i]).norm() / (
                                  (tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator) @ solution[
                              i]).norm())

    if compute_errors:
        return solution, errors
    else:
        return solution
