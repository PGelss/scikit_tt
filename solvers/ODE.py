#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import solvers.SLE as SLE
import scikit_tt.tt as tt
import numpy as np

def adaptive_als(operator, initial_value, initial_guess, first_step, T_end, repeats=1, loc_tol = 10**-3, tau_max = 10, max_factor = 1.25, safety_factor = 0.9, solver='lu', compute_errors=False):
    solution = [None]
    solution[0] = initial_value
    X = initial_guess
    errors = []
    time = 0
    tau = first_step
    i = 0
    while time < T_end:

        #S1 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, X,
        #            (tt.TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
        #            repeats=repeats)

        S1 = SLE.als(tt.TT.eye(operator.row_dims) - tau * operator, X, solution[i], solver=solver, repeats=repeats)
        S1 = (1 / S1.norm(p=1)) * S1

        #S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, X,
        #            (tt.TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
        #            repeats=repeats)
        #S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5 * tau * operator, S2,
        #             (tt.TT.eye(operator.row_dims) + 0.5 * tau * operator) @ solution[i], solver=solver,
        #             repeats=repeats)

        S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5*tau * operator, X, solution[i], solver=solver, repeats=repeats)
        S2 = SLE.als(tt.TT.eye(operator.row_dims) - 0.5*tau * operator, S2, solution[i], solver=solver, repeats=repeats)
        S2 = (1 / S2.norm(p=1)) * S2




        loc_err = (S1-S2).norm()
        closeness = (operator@S1).norm()
        factor = (loc_tol)/loc_err


        tau_new = np.amin([max_factor*tau, safety_factor*factor*tau])
        if factor > 1:
            time = time + tau
            tau = np.amin([tau_new, T_end-time, tau_max])
            solution.append(S1.copy())
            #print((operator@solution[-1]).norm())
            X = S1
            print('tau: '+str("%.2e" % tau)+', closeness: '+str("%.2e" % closeness))
        else:
            tau = tau_new
    return solution

def implicit_euler_als(operator, initial_value, initial_guess, step_sizes, repeats = 1, solver='solve', compute_errors=False):

    solution=[None]
    solution[0] = initial_value
    X = initial_guess
    errors = []

    for i in range(len(step_sizes)):

        X = SLE.als(tt.TT.eye(operator.row_dims) - step_sizes[i] * operator, X, solution[i], solver=solver, repeats=repeats)
        X = (1 / X.norm(p=1)) * X
        solution.append(X.copy())
        sys.stdout.write('\r' + 'Running implicit_euler_mals... ' + str(int(1000 * (i + 1) / len(step_sizes)) / 10) + '%')
        if compute_errors:
            errors.append(((tt.TT.eye(operator.row_dims) - step_sizes[i] * operator)@solution[i+1]-solution[i]).norm()/solution[i].norm())

    if compute_errors:
        return solution, errors
    else:
        return solution


def implicit_euler_mals(operator, initial_value, initial_guess, step_sizes, repeats = 1, solver='solve', threshold = 10**-14, max_rank =30, compute_errors=False):

    solution=[None]
    solution[0] = initial_value
    X = initial_guess
    errors = []

    for i in range(len(step_sizes)):

        X = SLE.mals(tt.TT.eye(operator.row_dims) - step_sizes[i] * operator, X, solution[i], solver=solver, threshold=threshold, repeats=repeats, max_rank=max_rank)
        X = (1 / X.norm(p=1)) * X
        solution.append(X.copy())
        sys.stdout.write('\r' + 'Running implicit_euler_mals... ' + str(int(1000 * (i + 1) / len(step_sizes)) / 10) + '%')
        if compute_errors:
            errors.append(((tt.TT.eye(operator.row_dims) - step_sizes[i] * operator)@solution[i+1]-solution[i]).norm()/solution[i].norm())

    if compute_errors:
        return solution, errors
    else:
        return solution

def trapezoidal_rule_als(operator, initial_value, initial_guess, step_sizes, repeats = 1, solver='solve', compute_errors=False):

    solution=[None]
    solution[0] = initial_value
    X = initial_guess
    errors = []

    for i in range(len(step_sizes)):

        X = SLE.als(tt.TT.eye(operator.row_dims) - 0.5*step_sizes[i] * operator, X, (tt.TT.eye(operator.row_dims) + 0.5*step_sizes[i] * operator) @ solution[i], solver=solver, repeats=repeats)
        X = (1 / X.norm(p=1)) * X
        solution.append(X.copy())
        sys.stdout.write('\r' + 'Running trapezoidal_rule_als... ' + str(int(1000 * (i + 1) / len(step_sizes)) / 10) + '%')
        if compute_errors:
            errors.append(((tt.TT.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator)@solution[i+1]-(tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator)@solution[i]).norm()/((tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator)@solution[i]).norm())

    if compute_errors:
        return solution, errors
    else:
        return solution

def trapezoidal_rule_mals(operator, initial_value, initial_guess, step_sizes, repeats = 1, solver='solve', threshold = 10**-14, max_rank =30, compute_errors=False):

    solution=[None]
    solution[0] = initial_value
    X = initial_guess
    errors = []

    for i in range(len(step_sizes)):

        X = SLE.mals(tt.TT.eye(operator.row_dims) - 0.5*step_sizes[i] * operator, X, (tt.TT.eye(operator.row_dims) + 0.5*step_sizes[i] * operator) @ solution[i], solver=solver, threshold=threshold, repeats=repeats, max_rank=max_rank)
        X = (1 / X.norm(p=1)) * X
        solution.append(X.copy())
        sys.stdout.write('\r' + 'Running trapezoidal_rule_mals... ' + str(int(1000 * (i + 1) / len(step_sizes)) / 10) + '%')
        if compute_errors:
            errors.append(((tt.TT.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator)@solution[i+1]-(tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator)@solution[i]).norm()/((tt.TT.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator)@solution[i]).norm())

    if compute_errors:
        return solution, errors
    else:
        return solution