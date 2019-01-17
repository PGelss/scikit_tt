# -*- coding: utf-8 -*-

from unittest import TestCase
import scikit_tt.models as mdl
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.ode as ode
import numpy as np
import numpy.linalg as lin
import scipy.sparse.linalg as splin


class TestODE(TestCase):

    def setUp(self):
        """Use the signaling cascade and the two-step destruction models for testing"""

        # set tolerance for the error
        self.tol = 1e-3

        # set parameters
        self.d = 2
        self.k_1 = 1
        self.k_2 = 2
        self.m = 2
        self.rank = 4
        self.qtt_modes = [[2] * 6] * self.d
        self.step_sizes_signaling_cascade = [1] * 300

        # construct operators
        self.operator_signaling_cascade = mdl.signaling_cascade(self.d).tt2qtt(self.qtt_modes, self.qtt_modes)
        self.operator_two_step_destruction = mdl.two_step_destruction(self.k_1, self.k_2, self.m)

    def test_implicit_euler(self):
        """test for implicit Euler method"""

        # compute numerical solution of the ODE
        operator = self.operator_signaling_cascade
        initial_value = tt.unit(operator.row_dims, [0]*operator.order)
        initial_guess = tt.ones(operator.row_dims, [1]*operator.order, ranks=self.rank).ortho_right()
        step_sizes = self.step_sizes_signaling_cascade
        solution = ode.implicit_euler(operator, initial_value, initial_guess, step_sizes, progress=False)

        # compute norm of the derivatives at the final 10 time steps
        derivatives = []
        for i in range(10):
            derivatives.append((operator @ solution[-i-1]).norm())

        # check if implicit Euler method converged to staionary distribution
        for i in range(10):
            self.assertLess(derivatives[i], self.tol)

