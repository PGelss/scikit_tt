# -*- coding: utf-8 -*-

"""
This is an example for the application of the TT format to a heterogeneous catalytic reaction system. For more details,
see [1]_, [2]_, and [3]_.

References
----------
.. [1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
       Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
.. [2] P. Gelß, S. Matera, C. Schütte, "Solving the master equation without kinetic Monte Carlo: Tensor train
       approximations for a CO oxidation model", Journal of Computational Physics 314 (2016) 489–502
.. [3] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-neighbor interaction systems in the tensor-train format",
       Journal of Computational Physics 341 (2017) 140-162
"""

import numpy as np
import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
import scikit_tt.solvers.evp as evp
import scikit_tt.solvers.ode as ode
import scikit_tt.utils as utl
import matplotlib.pyplot as plt


def two_cell_tof(t, reactant_states, reaction_rate):
    """Turn-over frequency of a reaction in a cyclic homogeneous nearest-neighbor interaction system

    Parameters
    ----------
    t: instance of TT class
        tensor train representing a probability distribution
    reactant_states: list of ints
        reactant states of the given reaction in the form of [reactant_state_1, reactant_state_2] where the list entries
        represent the reactant states on two neighboring cell
    reaction_rate: float
        reaction rate constant of the given reaction

    Returns
    -------
    turn_over_frequency: float
        turn-over frequency of the given reaction
    """

    tt_left = [None] * t.order
    tt_right = [None] * t.order
    for k in range(t.order):
        tt_left[k] = tt.ones([1] * t.order, t.row_dims)
        tt_right[k] = tt.ones([1] * t.order, t.row_dims)
        tt_left[k].cores[k] = np.zeros([1, 1, t.row_dims[k], 1])
        tt_left[k].cores[k][0, 0, reactant_states[0], 0] = 1
        tt_right[k].cores[k] = np.zeros([1, 1, t.row_dims[k], 1])
        tt_right[k].cores[k][0, 0, reactant_states[0], 0] = 1
        if k > 0:
            tt_left[k].cores[k - 1] = np.zeros([1, 1, t.row_dims[k - 1], 1])
            tt_left[k].cores[k - 1][0, 0, reactant_states[1], 0] = 1
        else:
            tt_left[k].cores[-1] = np.zeros([1, 1, t.row_dims[-1], 1])
            tt_left[k].cores[-1][0, 0, reactant_states[1], 0] = 1
        if k < t.order - 1:
            tt_right[k].cores[k + 1] = np.zeros([1, 1, t.row_dims[k + 1], 1])
            tt_right[k].cores[k + 1][0, 0, reactant_states[1], 0] = 1
        else:
            tt_right[k].cores[0] = np.zeros([1, 1, t.row_dims[0], 1])
            tt_right[k].cores[0][0, 0, reactant_states[1], 0] = 1
    turn_over_frequency = 0
    for k in range(t.order):
        turn_over_frequency = turn_over_frequency + (reaction_rate / t.order) * (tt_left[k] @ t) + \
                              (reaction_rate / t.order) * (tt_right[k] @ t)
    return turn_over_frequency


utl.header(title='CO oxidation')

# parameters
order = 20
p_CO_exp = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]

# TT ranks for approximations
#           - EVP -           |            - IEM -
R = [3, 4, 5, 5, 6, 6, 9, 12] + [12]

# TODO: FIND FURTHER RANKS

# define array for turn-over frequencies
tof = []

# compute solutions and print results
# -----------------------------------

print('----------------------------------------------------------')
print('p_CO in atm    Method    TT ranks    Closeness    CPU time')
print('----------------------------------------------------------')

# construct eigenvalue problems to find the stationary distributions
# ------------------------------------------------------------------

for i in range(8):
    # construct and solve eigenvalue problem for current CO pressure
    operator = mdl.co_oxidation(20, 10 ** (8 + p_CO_exp[i])).ortho_left().ortho_right()
    initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=R[i]).ortho_left().ortho_right()
    with utl.timer() as time:
        eigenvalues, solution, _ = evp.als(tt.eye(operator.row_dims) + operator, initial_guess, repeats=20, solver='eigs')
        solution = (1 / solution.norm(p=1)) * solution

    # compute turn-over frequency of CO2 desorption
    tof.append(two_cell_tof(solution, [2, 1], 1.7e5))

    # print results
    string_p_CO = '10^' + (p_CO_exp[i] >= 0) * '+' + str("%.1f" % p_CO_exp[i])
    string_method = ' ' * 9 + 'EVP' + ' ' * 9
    string_rank = (R[i] < 10) * ' ' + str(R[i]) + ' ' * 8
    string_closeness = str("%.2e" % (operator @ solution).norm()) + ' ' * 6
    string_time = (time.elapsed < 10) * ' ' + str("%.2f" % time.elapsed) + 's'
    print(string_p_CO + string_method + string_rank + string_closeness + string_time)

# apply adaptive step size scheme to the ODE to find the stationary distribution
# ------------------------------------------------------------------------------

for i in range(8, len(R)):
    # integrate ODE to approximate stationary distribution
    operator = mdl.co_oxidation(20, 10 ** (8 + p_CO_exp[i]))
    initial_value = tt.unit(operator.row_dims, [1] * operator.order)
    initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=R[i]).ortho_left().ortho_right()
    with utl.timer() as time:
        solution, _ = ode.adaptive_step_size(operator, initial_value, initial_guess, 100, progress=False)

    # compute turn-over frequency of CO2 desorption
    tof.append(two_cell_tof(solution[-1], [2, 1], 1.7e5))

    # print results
    string_p_CO = '10^' + (p_CO_exp[i] >= 0) * '+' + str("%.1f" % p_CO_exp[i])
    string_method = ' ' * 9 + 'IEM' + ' ' * 9
    string_rank = (R[i] < 10) * ' ' + str(R[i]) + ' ' * 8
    string_closeness = str("%.2e" % (operator @ solution[-1]).norm()) + ' ' * 6
    string_time = (time.elapsed < 10) * ' ' + str("%.2f" % time.elapsed) + 's'
    print(string_p_CO + string_method + string_rank + string_closeness + string_time)

print('----------------------------------------------------------')

# plot turn-over frequency
# ------------------------


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})
plt.plot(p_CO_exp[:len(tof)], tof)
plt.yscale('log')
plt.title(r'CPU times for $d = 10$', y=1.03)
plt.xlabel(r'$p_{\textrm{CO}}$')
plt.ylabel(r'$\textrm{TOF}$')
plt.title('Turn-over frequency', y=1.05)
plt.xticks([-4, -3, -2, -1, 0, 1, 2],
           (r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'))
plt.xlim(-4, 2)
plt.ylim(10 ** -4, 10 ** 6)
plt.show()
