# -*- coding: utf-8 -*-

"""
This is an example for the application of the TT and QTT format to Markovian master equations of chemical reaction
networks. For more details, see [1]_.

References
----------
..[1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
      Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
"""

from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
import scikit_tt.solvers.ode as ode
import scikit_tt.utils as utl
import numpy as np
import matplotlib.pyplot as plt

# parameters
# ----------

order = 20
tt_rank = 4
qtt_rank = 12
step_sizes = [1] * 300
qtt_modes = [[2] * 6] * order
threshold = 1e-14

# operator in TT format
# ---------------------

operator = mdl.signaling_cascade(order)

# initial distribution in TT format
# ---------------------------------

initial_distribution = tt.zeros(operator.col_dims, [1] * order)
for i in range(initial_distribution.order):
    initial_distribution.cores[i][0, 0, 0, 0] = 1

# initial guess in TT format
# --------------------------

initial_guess = tt.ones(operator.col_dims, [1] * order, ranks=tt_rank).ortho_right()

# solve Markovian master equation in TT format
# --------------------------------------------

print('\nTT approach')
print('-----------\n')
with utl.Timer() as time:
    solution = ode.trapezoidal_rule(operator, initial_distribution, initial_guess, step_sizes)
print('CPU time ' + '.' * 19 + ' ' + str("%.2f" % time.elapsed) + 's\n')

# operator in QTT format
# ----------------------

operator = TT.tt2qtt(operator, qtt_modes, qtt_modes, threshold=threshold)

# initial distribution in QTT format
# ----------------------------------

initial_distribution = tt.zeros(operator.col_dims, [1] * operator.order)
for i in range(initial_distribution.order):
    initial_distribution.cores[i][0, 0, 0, 0] = 1

# initial guess in QTT format
# ---------------------------

initial_guess = tt.ones(operator.col_dims, [1] * operator.order, ranks=qtt_rank).ortho_right()

# solve Markovian master equation in QTT format
# ---------------------------------------------

print('\nQTT approach')
print('------------\n')
with utl.Timer() as time:
    solution = ode.trapezoidal_rule(operator, initial_distribution, initial_guess, step_sizes)
print('CPU time ' + '.' * 19 + ' ' + str("%.2f" % time.elapsed) + 's\n')

# convert to TT and compute mean concentrations
# ---------------------------------------------

for i in range(len(solution)):
    solution[i] = TT.qtt2tt(solution[i], [len(qtt_modes[0])] * order)
mean = utl.mean_concentrations(solution)

# plot mean concentrations
# ------------------------

utl.plot_parameters()
plt.plot(np.insert(np.cumsum(step_sizes), 0, 0), mean)
plt.title('Mean concentrations', y=1.05)
plt.xlabel(r'$t$')
plt.ylabel(r'$\overline{x_i}(t)$')
plt.axis([0, len(step_sizes), 0, np.amax(mean) + 0.5])
plt.legend(['species ' + str(i) for i in range(1, np.amin([8, order]))] + ['...'], loc=4)
plt.show()
