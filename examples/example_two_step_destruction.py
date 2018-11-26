#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tt as tt
import models as mdl
import solvers.ODE as ODE
import subfunctions as sf
import tools as tls

# This is an example for the application of the QTT format to chemical reaction networks.
# Reference: P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical
# Reaction Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017

# parameters

m = 3
step_sizes = [0.001] * 100 + [0.1] * 9 + [1] * 9

# operator

operator = mdl.two_step_destruction(1, 2, m).tt2qtt([[2] * m] + [[2] * (m + 1)] + [[2] * m] + [[2] * m],
                                                    [[2] * m] + [[2] * (m + 1)] + [[2] * m] + [[2] * m],
                                                    threshold=10 ** -14)

# initial distribution

initial_distribution = tt.TT.zeros([2 ** m, 2 ** (m + 1), 2 ** m, 2 ** m], [1] * 4)
initial_distribution.cores[0][0, -1, 0, 0] = 1
initial_distribution.cores[1][0, -2, 0, 0] = 1
initial_distribution.cores[2][0, 0, 0, 0] = 1
initial_distribution.cores[3][0, 0, 0, 0] = 1
initial_distribution = tt.TT.tt2qtt(initial_distribution, [[2] * m] + [[2] * (m + 1)] + [[2] * m] + [[2] * m],
                                    [[1] * m] + [[1] * (m + 1)] + [[1] * m] + [[1] * m], threshold=10 ** -14)

# initial guess

initial_guess = tt.TT.uniform([2] * (4 * m + 1), 10).ortho_right()

# solve ODE

with tls.Timer() as time:
    solution, errors = ODE.implicit_euler_mals(operator, initial_distribution, initial_guess, step_sizes,
                                               threshold=10 ** -7, compute_errors=True)

print('\n\n' + 'CPU time     : ' + str("%.2f" % time.elapsed) + 's')
print('Maximum error: ' + str("%.2e" % np.amax(errors)))

# convert to TT and compute mean concentrations

for i in range(len(solution)):
    solution[i] = tt.TT.qtt2tt(solution[i], [m - 1, 2 * m, 3 * m, 4 * m])

mean = sf.mean_concentrations(solution, time_steps=np.insert(np.cumsum(step_sizes), 0, 0))
