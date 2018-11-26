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

tt_rank = 4
qtt_rank = 12
step_sizes = [1] * 300

# operator in TT format

operator = mdl.signaling_cascade(20)

# initial distribution in TT format

initial_distribution = tt.TT.zeros([64] * 20, [1] * 20)
for i in range(initial_distribution.order):
    initial_distribution.cores[i][0, 0, 0, 0] = 1

# initial guess in TT format

initial_guess = tt.TT.ones([64] * 20, [1] * 20, ranks=tt_rank).ortho_right()

# solve ODE in TT format

with tls.Timer() as time:
    solution = ODE.trapezoidal_rule_als(operator, initial_distribution, initial_guess, step_sizes)

print('\n' + 'CPU time     : ' + str("%.2f" % time.elapsed) + 's\n')

# operator in QTT format

operator = tt.TT.tt2qtt(operator, [[2] * 6] * 20, [[2] * 6] * 20, threshold=10 ** -14)

# initial distribution in QTT format

initial_distribution = tt.TT.zeros([2] * 6 * 20, [1] * 6 * 20)
for i in range(initial_distribution.order):
    initial_distribution.cores[i][0, 0, 0, 0] = 1

# initial guess in QTT format

initial_guess = tt.TT.ones([2] * 6 * 20, [1] * 6 * 20, ranks=qtt_rank).ortho_right()

# solve ODE in QTT format

with tls.Timer() as time:
    solution = ODE.trapezoidal_rule_als(operator, initial_distribution, initial_guess, step_sizes)

print('\n' + 'CPU time     : ' + str("%.2f" % time.elapsed) + 's')

# convert to TT and compute mean concentrations

for i in range(len(solution)):
    solution[i] = tt.TT.qtt2tt(solution[i], [5 + 6 * j for j in range(20)])

mean = sf.mean_concentrations(solution, time_steps=np.arange(301))
