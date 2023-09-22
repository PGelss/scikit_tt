import numpy as np
import scipy as sp
import scipy.linalg as lin

from typing import List, Union

import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.utils as utl
from scikit_tt.solvers import sle
import time as _time
from scikit_tt.solvers.sle import __construct_stack_right_op, __construct_stack_left_op, __construct_micro_matrix_als
from scipy.sparse.linalg import expm_multiply

def explicit_euler(operator: 'TT', 
                   initial_value: 'TT',
                   step_sizes: List[float], 
                   threshold: float=1e-12,
                   max_rank: int=50,
                   normalize: int=1,
                   progress: bool=True) -> List['TT']:
    """
    Explicit Euler method for linear differential equations in the TT format.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation

    initial_value : TT
        initial value of the differential equation

    step_sizes : list[float]
        step sizes for the application of the implicit Euler method

    threshold : float, optional
        threshold for reduced SVD decompositions, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step

    progress : bool, optional
        whether to show the progress of the algorithm or not, default is True

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # return current time
    start_time = utl.progress('Running explicit Euler method', 0, show=progress)

    # define solution
    solution = [initial_value]

    # begin explicit Euler method
    # ---------------------------

    for i in range(len(step_sizes)):
        # compute next time step
        tt_tmp = (tt.eye(operator.row_dims) + step_sizes[i] * operator).dot(solution[i])

        # truncate ranks of the solution
        tt_tmp = tt_tmp.ortho(threshold=threshold, max_rank=max_rank)

        # normalize solution
        if normalize > 0:
            tt_tmp = (1 / tt_tmp.norm(p=normalize)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        utl.progress('Running explicit Euler method', 100 * (i + 1) / len(step_sizes), show=progress,
                     cpu_time=_time.time() - start_time)

    return solution


def errors_expl_euler(operator: 'TT', solution: List['TT'], step_sizes: List[float]) -> List[float]:
    """
    Compute approximation errors of the explicit Euler method.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation

    solution : list[TT]
        approximate solution of the linear differential equation

    step_sizes : list[float]
        step sizes for the application of the implicit Euler method

    Returns
    -------
    list[float]
        approximation errors
    """

    # define errors
    errors = []

    # compute relative approximation errors
    for i in range(len(solution) - 1):
        errors.append(
            (solution[i + 1] - (tt.eye(operator.row_dims) + step_sizes[i] * operator).dot(solution[i])).norm() /
            solution[i].norm())

    return errors

def hod(operator: 'TT', 
        initial_value: 'TT',
        step_size: float, 
        number_of_steps: int,
        order: int=2,
        previous_value: 'TT'=None,
        op_hod: 'TT'=None,
        threshold: float=1e-12,
        max_rank: int=50,
        normalize: int=1,
        progress: bool=True) -> List['TT']:
    """
    Higher-order differencing for linear differential equations in the TT format.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation (assuming operator = -iH with Hamiltonian H)

    initial_value : TT
        initial value of the differential equation

    step_size : float
        step size

    number_of_steps : int
        number of time steps

    order : int, optional
        order of the differncing scheme, must be even, default is 2

    previous_value: TT, optional, default is None
        previous step; if not given one explicit Euler half-step and afterwards one HOD half-step are computed backwards in time

    op_hod : TT, optional, default is None
        TT operator for the HOD scheme

    threshold : float, optional
        threshold for reduced SVD decompositions, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step

    progress : bool, optional
        whether to show the progress of the algorithm or not, default is True

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # check if order is even, otherwise add 1
    if (order % 2) != 0:
        order = order + 1

    # return current time
    start_time = utl.progress('Running higher-order differencing method', 0, show=progress)

    # construct TT operator and orthonormalize
    if op_hod == None:
        op_hod = 2*step_size*operator.copy()
        op_tmp = operator.copy()

        for k in range(2,order//2+1):
            op_tmp = op_tmp.dot(operator).dot(operator)
            op_hod = op_hod + 2/np.math.factorial(2*k-1) * step_size**(2*k-1) * op_tmp

        op_hod = op_hod.ortho(threshold=threshold)

    # initialize solution
    solution = [initial_value]

    # begin loop over time steps
    # --------------------------

    for i in range(number_of_steps):

        if i == 0: # initialize: one expl. Euler and HOD half step backwards in time if previous step is not given

            if previous_value==None:
                op_first = step_size*operator.copy()
                op_tmp = operator.copy()

                for k in range(2,order//2+1):
                    op_tmp = op_tmp.dot(operator).dot(operator)
                    op_first = op_first + 2/np.math.factorial(2*k-1) * (step_size/2)**(2*k-1) * op_tmp

                op_first = op_first.ortho(threshold=threshold)

                # explicit Euler half step
                solution_prev = (tt.eye(operator.row_dims) - 0.5*step_size*operator).dot(solution[0])

                # HOD half step
                solution_prev = solution[i] - op_first.dot(solution_prev)

            else:

                solution_prev = previous_value

            # normalize
            if normalize > 0:
                solution_prev = (1 / solution_prev.norm(p=normalize)) * solution_prev

            solution_prev = solution_prev.ortho(threshold=threshold, max_rank=max_rank)

        else:
            solution_prev = solution[i-1].copy()

        # compute next time step from current and previous time step
        tt_tmp = solution_prev + op_hod.dot(solution[i])

        # truncate ranks of the solution
        tt_tmp = tt_tmp.ortho(threshold=threshold, max_rank=max_rank)

        # normalize solution
        if normalize > 0:
            tt_tmp = (1 / tt_tmp.norm(p=normalize)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        utl.progress('Running higher-order differencing method', 100 * (i + 1) / number_of_steps, show=progress,
                     cpu_time=_time.time() - start_time)

    return solution


def implicit_euler(operator: 'TT', initial_value: 'TT', initial_guess: 'TT', 
                   step_sizes: List[float], repeats: int=1, 
                   tt_solver: str='als', threshold: float=1e-12,
                   max_rank=np.infty, micro_solver='solve', normalize=1, progress=True) -> List['TT']:
    """
    Implicit Euler method for linear differential equations in the TT format.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation

    initial_value : TT
        initial value of the differential equation
        
    initial_guess : TT
        initial guess for the first step
        
    step_sizes : list[float]
        step sizes for the application of the implicit Euler method
        
    repeats : int, optional
        number of repeats of the (M)ALS in each iteration step, default is 1
        
    tt_solver : string, optional
        algorithm for solving the systems of linear equations in the TT format, default is 'als'
        
    threshold : float, optional
        threshold for reduced SVD decompositions, default is 1e-12
        
    max_rank : int, optional
        maximum rank of the solution, default is infinity
        
    micro_solver : string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'solve' or 'lu', default is 'solve'
        
    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step
        
    progress : bool, optional
        whether to show the progress of the algorithm or not, default is True

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # return current time
    start_time = utl.progress('Running implicit Euler method', 0, show=progress)

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
        if normalize > 0:
            tt_tmp = (1 / tt_tmp.norm(p=normalize)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        utl.progress('Running implicit Euler method', 100 * (i + 1) / len(step_sizes), show=progress,
                     cpu_time=_time.time() - start_time)

    return solution


def errors_impl_euler(operator: 'TT', solution: List['TT'], step_sizes: List[float]):
    """
    Compute approximation errors of the implicit Euler method.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation

    solution : list[TT]
        approximate solution of the linear differential equation

    step_sizes : list[float]
        step sizes for the application of the implicit Euler method

    Returns
    -------
    list[float]
        approximation errors
    """

    # define errors
    errors = []

    # compute relative approximation errors
    for i in range(len(solution) - 1):
        errors.append(
            ((tt.eye(operator.row_dims) - step_sizes[i] * operator).dot(solution[i + 1]) - solution[i]).norm() /
            solution[i].norm())

    return errors


def trapezoidal_rule(operator: 'TT', initial_value: 'TT', initial_guess: 'TT',
                     step_sizes: List[float], repeats: int=1, 
                     tt_solver: str='als', threshold=1e-12,
                     max_rank: int=np.infty, micro_solver: str='solve',
                     normalize: int=1, progress: bool=True) -> List['TT']:
    """
    Trapezoidal rule for linear differential equations in the TT format.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation

    initial_value : TT
        initial value of the differential equation
        
    initial_guess : TT
        initial guess for the first step

    step_sizes : list[float]
        step sizes for the application of the trapezoidal rule
        
    repeats : int, optional
        number of repeats of the (M)ALS in each iteration step, default is 1
        
    tt_solver : string, optional
        algorithm for solving the systems of linear equations in the TT format, default is 'als'
        
    threshold : float, optional
        threshold for reduced SVD decompositions, default is 1e-12
        
    max_rank : int, optional
        maximum rank of the solution, default is infinity
        
    micro_solver : string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'solve' or 'lu', default is 'solve'
        
    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step
        
    progress : bool, optional
        whether to show the progress of the algorithm or not, default is True

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # return current time
    start_time = utl.progress('Running trapezoidal rule', 0, show=progress)

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
                             (tt.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator).dot(solution[i]),
                             solver=micro_solver, repeats=repeats)
        if tt_solver == 'mals':
            tt_tmp = sle.mals(tt.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator, tt_tmp,
                              (tt.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator).dot(solution[i]),
                              solver=micro_solver, repeats=repeats, threshold=threshold, max_rank=max_rank)

        # normalize solution
        if normalize > 0:
            tt_tmp = (1 / tt_tmp.norm(p=normalize)) * tt_tmp

        # append solution
        solution.append(tt_tmp.copy())

        # print progress
        utl.progress('Running trapezoidal rule', 100 * (i + 1) / len(step_sizes), show=progress,
                     cpu_time=_time.time() - start_time)

    return solution


def errors_trapezoidal(operator: 'TT', solution: List['TT'], 
                       step_sizes: List[float]) -> List[float]:
    """
    Compute approximation errors of the trapezoidal rule.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation

    solution : list[TT]
        approximate solution of the linear differential equation

    step_sizes : list[float]
        step sizes for the application of the implicit Euler method

    Returns
    -------
    list[float]
        approximation errors
    """

    # define errors
    errors = []

    # compute relative approximation errors
    for i in range(len(solution) - 1):
        errors.append(((tt.eye(operator.row_dims) - 0.5 * step_sizes[i] * operator).dot(solution[i + 1]) -
                       (tt.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator).dot(solution[i])).norm() /
                      ((tt.eye(operator.row_dims) + 0.5 * step_sizes[i] * operator).dot(solution[i])).norm())

    return errors


def adaptive_step_size(operator: 'TT', initial_value: 'TT', initial_guess: 'TT', 
                       time_end: float, step_size_first: float=1e-10, 
                       repeats: int=1, solver: str='solve',
                       error_tol: float=1e-1, closeness_tol: float=0.5,
                       step_size_min: float=1e-14, step_size_max: float=10, 
                       closeness_min: float=1e-3, factor_max: float=2, 
                       factor_safe: float=0.9, second_method:str ='two_step_Euler', 
                       normalize: int=1, progress: bool=True) -> List['TT']:
    """
    Adaptive step size method.

    Parameters
    ----------
    operator : TT
        TT operator of the differential equation

    initial_value : TT
        initial value of the differential equation

    initial_guess : TT
        initial guess for the first step

    time_end : float
        time point to which the ODE should be integrated

    step_size_first : float, optional
        first time step, default is 1e-10

    repeats : int, optional
        number of repeats of the ALS in each iteration step, default is 1

    solver : string, optional
        algorithm for obtaining the solutions of the micro systems, can be 'solve' or 'lu', default is 'solve'

    error_tol : float, optional
        tolerance for relative local error, default is 1e-1

    closeness_tol : float, optional
        tolerance for relative change in the closeness to the stationary distribution, default is 0.5

    step_size_min : float, optional
        minimum step size, default is 1e-14

    step_size_max : float, optional
        maximum step size, default is 10

    closeness_min : float, optional
        minimum closeness value, default is 1e-3

    factor_max : float, optional
        maximum factor for step size adaption, default is 2

    factor_safe : float, optional
        safety factor for step size adaption, default is 0.9

    second_method : {'two_step_Euler', 'trapezoidal_rule'}, optional
        which higher-order method should be used, can be 'two_step_Euler' or 'trapezoidal_rule', default is
        'two_step_Euler'

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step

    progress : bool, optional
        whether to show the progress of the algorithm or not, default is True

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # return current time
    start_time = utl.progress('Running adaptive step size method', 0, show=progress)

    # define solution
    solution = [initial_value]

    # define variable for integration
    t_2 = []

    # set closeness variables
    closeness_pre = (operator.dot(initial_value)).norm()

    # define tensor train for solving the systems of linear equations
    t_tmp = initial_guess

    # set time and step size
    time = 0
    time_steps = [0]
    step_size = step_size_first

    # begin integration
    # -----------------

    while (time < time_end) and (closeness_pre > closeness_min) and (step_size > step_size_min):

        # first method
        t_1 = sle.als(tt.eye(operator.row_dims) - step_size * operator, t_tmp.copy(), solution[-1], solver=solver,
                      repeats=repeats)
        t_1 = (1 / t_1.norm(p=1)) * t_1

        # second method
        if second_method == 'two_step_Euler':
            t_2 = sle.als(tt.eye(operator.row_dims) - 0.5 * step_size * operator, t_tmp.copy(), solution[-1],
                          solver=solver,
                          repeats=repeats)

            t_2 = sle.als(tt.eye(operator.row_dims) - 0.5 * step_size * operator, t_2.copy(), solution[-1],
                          solver=solver,
                          repeats=repeats)

        if second_method == 'trapezoidal_rule':
            t_2 = sle.als(tt.eye(operator.row_dims) - 0.5 * step_size * operator, t_tmp.copy(),
                          (tt.eye(operator.row_dims) + 0.5 * step_size * operator).dot(solution[-1]), solver=solver,
                          repeats=repeats)

        # normalize solution
        if normalize > 0:
            t_2 = (1 / t_2.norm(p=normalize)) * t_2

        # compute closeness to staionary distribution
        closeness = (operator.dot(t_1)).norm()

        # compute relative local error and closeness change
        local_error = (t_1 - t_2).norm() / t_1.norm()
        closeness_difference = (closeness - closeness_pre) / closeness_pre

        # compute factors for step size adaption
        factor_local     = error_tol / local_error
        factor_closeness = closeness_tol / np.abs(closeness_difference)

        # compute new step size
        step_size_new = np.amin([factor_max, factor_safe * factor_local, factor_safe * factor_closeness]) * step_size

        # accept or reject step
        if (factor_local > 1) and (factor_closeness > 1):
            time = np.min([time + step_size, time_end])
            step_size = np.amin([step_size_new, time_end - time, step_size_max])
            solution.append(t_2.copy())
            time_steps.append(time)
            t_tmp = t_1

            utl.progress('Running adaptive step size method', 100 * time / time_end, show=progress,
                         cpu_time=_time.time() - start_time)
            closeness_pre = closeness

        else:
            step_size = step_size_new

    return solution, time_steps

def lie_splitting(S : Union[np.ndarray, List[np.ndarray]],
                  L : Union[np.ndarray, List[np.ndarray]],
                  I : Union[np.ndarray, List[np.ndarray]],
                  M : Union[np.ndarray, List[np.ndarray]],
                  initial_value: 'TT', step_size: float, number_of_steps: int, 
                  threshold: float=1e-12, max_rank: int=50,
                  normalize: int=1, K: List[np.ndarray]=None) -> List['TT']:
    """
    Lie splitting for ODEs with non-periodic SLIM operators.

    Parameters
    ----------
    S : ndarray or list[ndarrays]
        single-site components of SLIM decomposition

    L : ndarray or list[ndarrays]
        left two-site components of SLIM decomposition

    I : ndarray or list[ndarrays]
        identity components of SLIM decomposition

    M : ndarray or list[ndarrays]
        right two-site components of SLIM decomposition

    initial_value : TT
        initial value of the differential equation

    step_size : float
        step size

    number_of_steps : int
        number of time steps

    threshold : float, optional
        threshold for reduced SVDs, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step

    K : list[ndarrays], optional
        list of propagators

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # chain length
    order = initial_value.order

    # define solution list
    solution = []
    solution.append(initial_value)

    if K is None:
        K = __splitting_propagators(S, L, I, M, order, step_size, [1, 1])

    for i in range(number_of_steps):

        # copy previous solution for next step
        tmp = solution[i].copy()

        # Strang splitting
        tmp = __splitting_stage(K, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K, np.arange(1,order,2), tmp, threshold, 2*max_rank)
        tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)

        # normalize solution
        if normalize > 0:
            tmp = (1 / tmp.norm(p=normalize)) * tmp

        # append solution
        solution.append(tmp.copy())

    return solution

def strang_splitting(S : Union[np.ndarray, List[np.ndarray]],                     
                     L : Union[np.ndarray, List[np.ndarray]],
                     I : Union[np.ndarray, List[np.ndarray]],
                     M : Union[np.ndarray, List[np.ndarray]],
                     initial_value: 'TT', step_size: float, number_of_steps: int, 
                     threshold: float=1e-12, max_rank: int=50,
                     normalize: int=0, K: List[np.ndarray]=None) -> List['TT']:

    """
    Strang splitting for ODEs with non-periodic SLIM operators.

    Parameters
    ----------
    S : ndarray or list[ndarrays]
        single-site components of SLIM decomposition

    L : ndarray or list[ndarrays]
        left two-site components of SLIM decomposition

    I : ndarray or list[ndarrays]
        identity components of SLIM decomposition

    M : ndarray or list[ndarrays]
        right two-site components of SLIM decomposition

    initial_value : TT
        initial value of the differential equation

    step_size : float
        step size

    number_of_steps : int
        number of time steps

    threshold : float, optional
        threshold for reduced SVDs, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step

    K : list[ndarrays], optional
        list of propagators

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # chain length
    order = initial_value.order

    # define solution list
    solution = []
    solution.append(initial_value)

    if K is None:
        K = __splitting_propagators(S, L, I, M, order, step_size, [0.5, 1])

    for i in range(number_of_steps):

        # copy previous solution for next step
        tmp = solution[i].copy()

        # Strang splitting
        tmp = __splitting_stage(K, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K, np.arange(1,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)

        # normalize solution
        if normalize > 0:
            tmp = (1 / tmp.norm(p=normalize)) * tmp

        # append solution
        solution.append(tmp.copy())

    return solution

def yoshida_splitting(S : Union[np.ndarray, List[np.ndarray]],
                      L : Union[np.ndarray, List[np.ndarray]],
                      I : Union[np.ndarray, List[np.ndarray]],
                      M : Union[np.ndarray, List[np.ndarray]],
                      initial_value: 'TT', step_size: float, number_of_steps: int, 
                      threshold: float=1e-12, max_rank: int=50,
                      normalize: int=0) -> List['TT']:
    """
    Yoshida splitting for ODEs with non-periodic SLIM operators.

    Parameters
    ----------
    S : ndarray or list[ndarrays]
        single-site components of SLIM decomposition

    L : ndarray or list[ndarrays]
        left two-site components of SLIM decomposition

    I : ndarray or list[ndarrays]
        identity components of SLIM decomposition

    M : ndarray or list[ndarrays]
        right two-site components of SLIM decomposition

    initial_value : TT
        initial value of the differential equation

    step_size : float
        step size

    number_of_steps : int
        number of time steps

    threshold : float, optional
        threshold for reduced SVDs, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # chain length
    order = initial_value.order

    # define solution list
    solution = []
    solution.append(initial_value)

    K1 = __splitting_propagators(S, L, I, M, order, step_size, [0.5*(1/(2-2**(1/3))), 1/(2-2**(1/3))])
    K2 = __splitting_propagators(S, L, I, M, order, step_size, [-0.5*(2**(1/3)/(2-2**(1/3))), -2**(1/3)/(2-2**(1/3))])

    for i in range(number_of_steps):

        # copy previous solution for next step
        tmp = solution[i].copy()

        # splitting
        tmp = __splitting_stage(K1, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K1, np.arange(1,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K1, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        #tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)
        tmp = __splitting_stage(K2, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K2, np.arange(1,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K2, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        #tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)
        tmp = __splitting_stage(K1, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K1, np.arange(1,order,2), tmp, threshold, 2*max_rank)
        tmp = __splitting_stage(K1, np.arange(0,order,2), tmp, threshold, 2*max_rank)
        tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)

        # normalize solution
        if normalize > 0:
            tmp = (1 / tmp.norm(p=normalize)) * tmp

        # append solution
        solution.append(tmp.copy())

    return solution

def kahan_li_splitting(S : Union[np.ndarray, List[np.ndarray]],
                       L : Union[np.ndarray, List[np.ndarray]],
                       I : Union[np.ndarray, List[np.ndarray]],
                       M : Union[np.ndarray, List[np.ndarray]],
                       initial_value: 'TT', step_size: float, number_of_steps: int, 
                       threshold: float=1e-12, max_rank: int=50,
                       normalize: int=0) -> List['TT']:
    """
    Kahan-Li splitting for ODEs with non-periodic SLIM operators.

    Parameters
    ----------
    S : ndarray or list[ndarrays]
        single-site components of SLIM decomposition

    L : ndarray or list[ndarrays]
        left two-site components of SLIM decomposition

    I : ndarray or list[ndarrays]
        identity components of SLIM decomposition

    M : ndarray or list[ndarrays]
        right two-site components of SLIM decomposition

    initial_value : TT
        initial value of the differential equation

    step_size : float
        step size

    number_of_steps : int
        number of time steps

    threshold : float, optional
        threshold for reduced SVDs, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step

    Returns
    -------
    list[TT]
        numerical solution of the differential equation
    """

    # chain length
    order = initial_value.order

    # define solution list
    solution = []
    solution.append(initial_value)

    K=[None]*9

    K[0] = __splitting_propagators(S, L, I, M, order, step_size, [ 0.5*0.13020248308889008087881763,  0.13020248308889008087881763])
    K[1] = __splitting_propagators(S, L, I, M, order, step_size, [ 0.5*0.56116298177510838456196441,  0.56116298177510838456196441])
    K[2] = __splitting_propagators(S, L, I, M, order, step_size, [-0.5*0.38947496264484728640807860, -0.38947496264484728640807860])
    K[3] = __splitting_propagators(S, L, I, M, order, step_size, [ 0.5*0.15884190655515560089621075,  0.15884190655515560089621075])
    K[4] = __splitting_propagators(S, L, I, M, order, step_size, [-0.5*0.39590389413323757733623154, -0.39590389413323757733623154])
    K[5] = __splitting_propagators(S, L, I, M, order, step_size, [ 0.5*0.18453964097831570709183254,  0.18453964097831570709183254])
    K[6] = __splitting_propagators(S, L, I, M, order, step_size, [ 0.5*0.25837438768632204729397911,  0.25837438768632204729397911])
    K[7] = __splitting_propagators(S, L, I, M, order, step_size, [ 0.5*0.29501172360931029887096624,  0.29501172360931029887096624])
    K[8] = __splitting_propagators(S, L, I, M, order, step_size, [-0.5*0.60550853383003451169892108, -0.60550853383003451169892108])

    for i in range(number_of_steps):

        # copy previous solution for next step
        tmp = solution[i].copy()

        # splitting
        for j in range(9):
            tmp = __splitting_stage(K[j], np.arange(0,order,2), tmp, threshold, 2*max_rank)
            tmp = __splitting_stage(K[j], np.arange(1,order,2), tmp, threshold, 2*max_rank)
            tmp = __splitting_stage(K[j], np.arange(0,order,2), tmp, threshold, 2*max_rank)
            #tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)
        for j in range(7,-1,-1):
            tmp = __splitting_stage(K[j], np.arange(0,order,2), tmp, threshold, 2*max_rank)
            tmp = __splitting_stage(K[j], np.arange(1,order,2), tmp, threshold, 2*max_rank)
            tmp = __splitting_stage(K[j], np.arange(0,order,2), tmp, threshold, 2*max_rank)
            #tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)
        tmp = tmp.ortho(threshold=threshold, max_rank=max_rank)

        # normalize solution
        if normalize > 0:
            tmp = (1 / tmp.norm(p=normalize)) * tmp

        # append solution
        solution.append(tmp.copy())

    return solution

def __splitting_propagators(S: Union[np.ndarray, List[np.ndarray]],
                            L: Union[np.ndarray, List[np.ndarray]],
                            I: Union[np.ndarray, List[np.ndarray]],
                            M: Union[np.ndarray, List[np.ndarray]],
                            order: int, step_size: float, coefficients: List[float]):

    if isinstance(S, list):

        # inhomogeneous case

        d = S[0].shape[0]
        K = [None] * order

        for i in range(order-1):

            if len(L[i].shape) == 2:
                L[i] = L[i][:,:,None]
                M[i+1] = M[i+1][None, :, :]

            K[i] = np.kron(S[i], I[i+1]) + np.einsum('ijk, klm -> iljm', L[i], M[i+1]).reshape([L[i].shape[0]*M[i+1].shape[1], L[i].shape[1]*M[i+1].shape[2]])
            

            if np.mod(i, 2) == 0:
                K[i] = sp.linalg.expm(K[i]*coefficients[0]*step_size)
            else:
                K[i] = sp.linalg.expm(K[i]*coefficients[1]*step_size)

        if np.mod(order-1, 2) == 0:
            K[-1] = sp.linalg.expm(S[-1]*coefficients[0]*step_size)
        else:
            K[-1] = sp.linalg.expm(S[-1]*coefficients[1]*step_size)

    else: 

        # homogeneous case

        d = S.shape[0]
        K = [None] * order

        if len(L.shape) == 2:
            L = L[:,:,None]
            M = M[None, :, :]

        K_hom = np.kron(S, I) + np.einsum('ijk, klm -> iljm', L, M).reshape([L.shape[0]*M.shape[1], L.shape[1]*M.shape[2]])

        for i in range(order-1):

            K[i] = K_hom.copy()

            if np.mod(i, 2) == 0:
                K[i] = sp.linalg.expm(K[i]*coefficients[0]*step_size)
            else:
                K[i] = sp.linalg.expm(K[i]*coefficients[1]*step_size)

        if np.mod(order-1, 2) == 0:
            K[-1] = sp.linalg.expm(S*coefficients[0]*step_size)
        else:
            K[-1] = sp.linalg.expm(S*coefficients[1]*step_size)

    return K


def __splitting_stage(K: Union[np.ndarray, List[np.ndarray]],
                      indices: np.ndarray, tmp, 
                      threshold: float, max_rank: int):

    for i in indices:

        if i<tmp.order-1:

            # contract cores
            tmp_vec = np.einsum('ijkl,lmno -> ijkmno', tmp.cores[i], tmp.cores[i+1]).reshape([tmp.ranks[i], tmp.row_dims[i]*tmp.row_dims[i+1], tmp.ranks[i+2]])
            tmp_vec = np.einsum('ijk, lj -> ilk', tmp_vec, K[i]).reshape([tmp.ranks[i]*tmp.row_dims[i], tmp.row_dims[i+1]*tmp.ranks[i+2]])

            # apply SVD in order to isolate modes
            u, s, v = utl.truncated_svd(tmp_vec, threshold=threshold, max_rank=max_rank)

            # update cores
            tmp.cores[i] = u.reshape([tmp.ranks[i], tmp.row_dims[i], 1, u.shape[1]])
            tmp.cores[i+1] = (np.dot(np.diag(s),v)).reshape([u.shape[1], tmp.row_dims[i+1], 1, tmp.ranks[i+2]])
            tmp.ranks[i+1] = u.shape[1]

        else:

            tmp.cores[-1] = np.einsum('ijkl, mj -> imkl', tmp.cores[-1], K[-1])

    return tmp


def tdvp(operator: 'TT', initial_value: 'TT', step_size: float, number_of_steps: int, normalize: int=0) -> 'TT':
    """
    Time-dependent variational principle (1TDVP), see [1]_.

    Parameters
    ----------
    operator : TT
        TT operator

    initial_guess : TT
        initial guess for the solution
        
    step_size: float
        step size

    number_of_steps: int
        number of time steps
        
    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step


    Returns
    -------
    TT
        approximated solution of the Schrödinger equation

    References
    ----------
    ..[1] S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck, 
          C. Hubig, "Time-evolution methods for matrix-product states". 
          Annals of Physics, 411, 167998, 2019
    """
    
    # define solution list
    solution = []
    solution.append(initial_value)
    
    # copy previous solution for next step
    tmp = solution[0].copy()

    # define stacks
    stack_left_op   = [None] * operator.order
    stack_right_op  = [None] * operator.order

    # construct right stacks for the left- and right-hand side
    for i in range(operator.order - 1, -1, -1):
        __construct_stack_right_op(i, stack_right_op, operator, tmp)

    # define iteration number
    current_iteration = 1

    # begin TDVP
    while current_iteration <= number_of_steps:

        # first half sweep
        for i in range(operator.order):

            # update left stacks for the left- and right-hand side
            __construct_stack_left_op(i, stack_left_op, operator, tmp)
            
            # construct micro system
            micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, tmp)

            # update solution
            __update_core_tdvp(i, micro_op, tmp, step_size, 'forward')

        # second half sweep
        for i in range(operator.order - 1, -1, -1):
            
            # update right stacks for the left- and right-hand side
            __construct_stack_right_op(i, stack_right_op, operator, tmp)
            
            # construct micro system
            micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, operator, tmp)

            # update solution
            __update_core_tdvp(i, micro_op, tmp, step_size, 'backward')

        # increase iteration number
        current_iteration += 1
        
        # normalize solution
        if normalize > 0:
            tmp = (1 / tmp.norm(p=normalize)) * tmp
            
        # append solution
        solution.append(tmp.copy())

    return solution




def __update_core_tdvp(i: int, micro_op: np.ndarray, solution: 'TT', step_size: float, direction: str):
    """
    Update TT core for TDVP.

    Parameters
    ----------
    i : int
        core index

    micro_op : np.ndarray
        micro matrix for ith TT core

    solution : TT
        approximated solution of the system of linear equations

    step_size: int
        step size

    direction : string
        'forward' if first half sweep, 'backward' if second half sweep
    """

    # time step
    # ------------------------------------------

    r1 = solution.ranks[i]
    n = solution.row_dims[i]
    r2 = solution.ranks[i+1]
        
    # first half sweep
    if direction == 'forward':
            
        if i < solution.order-1:
            
            # time step
            solution.cores[i] = expm_multiply(-1j*step_size*0.5*micro_op, solution.cores[i].flatten())
        
            # decompose solution
            [q, r] = lin.qr(solution.cores[i].reshape(r1 * n, r2), overwrite_a=True, mode='economic', check_finite=False)

            # set new rank
            solution.ranks[i + 1] = q.shape[1]

            # save orthonormal part
            solution.cores[i] = q.reshape(r1, n, 1, solution.ranks[i + 1])

            # adapt micro matrix
            q = np.tensordot(q, np.eye(r2),axes=0)
            q = q.transpose([0,3,1,2]).reshape([r1*n*r2, solution.ranks[i + 1]*r2])
            micro_op = np.conj(q).T@micro_op@q
            
            # time step
            r = expm_multiply(1j*step_size*0.5*micro_op, r.flatten())
            r = r.reshape([solution.ranks[i + 1], r2])
            
            # save non-orthonormal part
            solution.cores[i+1] = np.tensordot(r, solution.cores[i+1], axes=(1,0))
        
        else:
            
            # time step
            solution.cores[i] = expm_multiply(-1j*step_size*micro_op, solution.cores[i].flatten())
            solution.cores[i] = solution.cores[i].reshape(r1, n, 1, r2)
            
    # second half sweep
    if direction == 'backward':

        if i > 0:
            
            if i < solution.order-1:
                
                # time step
                solution.cores[i] = expm_multiply(-1j*step_size*0.5*micro_op, solution.cores[i].flatten())
 
            # decompose solution
            [r, q] = lin.rq(solution.cores[i].reshape(r1, n * r2), overwrite_a=True, mode='economic', check_finite=False)

            # set new rank
            solution.ranks[i] = q.shape[0]

            # save orthonormal part
            solution.cores[i] = q.reshape(r1, n, 1, r2)
            
            # adapt micro matrix
            q = np.tensordot(np.eye(r1),q,axes=0)
            q = q.transpose([0,3,1,2]).reshape([r1*n*r2, r1*solution.ranks[i]])
            micro_op = np.conj(q).T@micro_op@q
            
            # time step
            r = expm_multiply(1j*step_size*0.5*micro_op, r.flatten())
            r = r.reshape([r1, solution.ranks[i]])
            
            # save non-orthonormal part
            solution.cores[i-1] = np.tensordot(solution.cores[i-1], r, axes=(3,0))
            
        else:
            
            # time step
            solution.cores[i] = expm_multiply(-1j*step_size*0.5*micro_op, solution.cores[i].flatten())
            solution.cores[i] = solution.cores[i].reshape(r1, n, 1, r2)

                
def krylov(operator: 'TT', initial_value: 'TT', dimension: int, step_size: float, threshold: float=1e-12, max_rank: int=50, normalize: int=0) -> 'TT':
    """
    Krylov method, see [1]_.

    Parameters
    ----------
    operator : TT
        TT operator

    initial_value : TT
        initial value for ODE
        
    dimension: int
        dimension of Krylov subspace, must be larger than 1
        
    step_size: float
        step size

    threshold : float, optional
        threshold for reduced SVD decompositions, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50
        
    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step


    Returns
    -------
    TT
        approximated solution of the Schrödinger equation

    References
    ----------
    ..[1] S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck, 
          C. Hubig, "Time-evolution methods for matrix-product states". 
          Annals of Physics, 411, 167998, 2019
    """
    
    # construct Krylov subspace basis
    krylov_tensors = [(1/initial_value.norm())*initial_value]
    v_tmp = operator@krylov_tensors[0]
    v_tmp = v_tmp - (v_tmp.transpose(conjugate=True)@krylov_tensors[0])*krylov_tensors[0]
    v_tmp = v_tmp.ortho(threshold=threshold, max_rank=max_rank)
    v_tmp = (1/v_tmp.norm())*v_tmp
    krylov_tensors.append(v_tmp)
    for i in range(2,dimension):
        v_tmp = operator@krylov_tensors[-1]
        v_tmp = v_tmp - (v_tmp.transpose(conjugate=True)@krylov_tensors[-2])*krylov_tensors[-2] - (v_tmp.transpose(conjugate=True)@krylov_tensors[-1])*krylov_tensors[-1]
        v_tmp = (1/v_tmp.norm())*v_tmp
        v_tmp = v_tmp.ortho(threshold=threshold, max_rank=max_rank)
        krylov_tensors.append(v_tmp)
        
    # compute effective H
    T = np.zeros([dimension, dimension], dtype=complex)
    for i in range(dimension):
        for j in range(dimension):
            T[i,j] = krylov_tensors[i].transpose(conjugate=True)@operator@krylov_tensors[j]
    matrix_exponential = sp.linalg.expm(-1j*T*step_size)
            
    # compute time-evolved state
    w_tmp = np.zeros([dimension], dtype=complex)
    for j in range(dimension):
        w_tmp[j] = krylov_tensors[j].transpose(conjugate=True)@initial_value
    w_tmp = matrix_exponential@w_tmp
    solution = w_tmp[0]*krylov_tensors[0]
    for j in range(1,dimension):
        solution = solution + w_tmp[j]*krylov_tensors[j]
    solution = solution.ortho(threshold=threshold, max_rank=max_rank)
    if normalize > 0:
        solution = (1 / solution.norm(p=normalize)) * solution

    return solution


    
    
