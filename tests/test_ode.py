# -*- coding: utf-8 -*-

from unittest import TestCase
import scikit_tt.models as mdl
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.ode as ode


class TestODE(TestCase):

    def setUp(self):
        """Use the signaling cascade and the two-step destruction models for testing"""

        # set tolerance for the error
        self.tol = 1e-3

        # set parameters
        self.d = 3
        self.k_1 = 1
        self.k_2 = 2
        self.m = 2
        self.rank = 4
        self.qtt_modes = [[2] * 6] * self.d
        self.step_sizes_signaling_cascade = [1] * 300
        self.step_sizes_two_step_destruction = [0.001] * 100 + [0.1] * 9 + [1] * 19

        # construct operators
        self.operator_signaling_cascade = mdl.signaling_cascade(self.d).tt2qtt(self.qtt_modes, self.qtt_modes)
        self.operator_two_step_destruction = mdl.two_step_destruction(self.k_1, self.k_2, self.m)

    def test_implicit_euler(self):
        """test for implicit Euler method"""

        # compute numerical solution of the ODE
        operator = self.operator_signaling_cascade
        initial_value = tt.unit(operator.row_dims, [0] * operator.order)
        initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=self.rank).ortho_right()
        step_sizes = self.step_sizes_signaling_cascade
        solution = ode.implicit_euler(operator, initial_value, initial_guess, step_sizes, progress=False)

        # compute norm of the derivatives at the final 10 time steps
        derivatives = []
        for i in range(10):
            derivatives.append((operator @ solution[-i - 1]).norm())

        # check if implicit Euler method converged to stationary distribution
        for i in range(10):
            self.assertLess(derivatives[i], self.tol)

    def test_trapezoidal_rule(self):
        """test for trapezoidal rule"""

        # compute numerical solution of the ODE
        operator = self.operator_two_step_destruction
        initial_value = tt.zeros([2 ** self.m, 2 ** (self.m + 1), 2 ** self.m, 2 ** self.m], [1] * 4)
        initial_value.cores[0][0, -1, 0, 0] = 1
        initial_value.cores[1][0, -2, 0, 0] = 1
        initial_value.cores[2][0, 0, 0, 0] = 1
        initial_value.cores[3][0, 0, 0, 0] = 1
        initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=self.rank).ortho_right()
        step_sizes = self.step_sizes_two_step_destruction
        solution = ode.trapezoidal_rule(operator, initial_value, initial_guess, step_sizes, progress=False)

        # compute norm of the derivatives at the final 10 time steps
        derivatives = []
        for i in range(10):
            derivatives.append((operator @ solution[-i - 1]).norm())

        # check if trapezoidal rule converged to stationary distribution
        for i in range(10):
            self.assertLess(derivatives[i], self.tol)
