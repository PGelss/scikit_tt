# -*- coding: utf-8 -*-

from unittest import TestCase
import scikit_tt.models as mdl
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.ode as ode
import numpy as np


class TestODE(TestCase):

    def setUp(self):
        """Use the signaling cascade and the two-step destruction models for testing"""

        # set tolerance for the errors
        self.tol = 1e-2

        # set ranks for approximations
        self.rank = 4
        self.max_rank = 15

    def test_explicit_euler(self):
        """test for explicit Euler method"""

        # compute numerical solution of the ODE
        operator = mdl.signaling_cascade(2).tt2qtt([[2] * 6] * 2, [[2] * 6] * 2)
        initial_value = tt.unit(operator.row_dims, [0] * operator.order)
        step_sizes = [0.1] * 3000
        solution = ode.explicit_euler(operator, initial_value, step_sizes, progress=False, max_rank=self.max_rank)

        # compute norm of the derivatives at the final 10 time steps
        derivatives = []
        for i in range(10):
            derivatives.append((operator.dot(solution[-i - 1])).norm())

        # check if explicit Euler method converged to stationary distribution
        for i in range(10):
            self.assertLess(derivatives[i], self.tol)

    def test_implicit_euler(self):
        """test for implicit Euler method"""

        # compute numerical solution of the ODE
        operator = mdl.signaling_cascade(2).tt2qtt([[2] * 6] * 2, [[2] * 6] * 2)
        initial_value = tt.unit(operator.row_dims, [0] * operator.order)
        initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=self.rank).ortho_right()
        step_sizes = [1] * 300
        solution_als = ode.implicit_euler(operator, initial_value, initial_guess, step_sizes, tt_solver='als',
                                          progress=False)
        solution_mals = ode.implicit_euler(operator, initial_value, initial_guess, step_sizes, tt_solver='mals',
                                           max_rank=self.rank, progress=False)

        # compute norm of the derivatives at the final 10 time steps
        derivatives_als = []
        derivatives_mals = []
        for i in range(10):
            derivatives_als.append((operator.dot(solution_als[-i - 1])).norm())
            derivatives_mals.append((operator.dot(solution_mals[-i - 1])).norm())

        # check if implicit Euler method converged to stationary distribution
        for i in range(10):
            self.assertLess(derivatives_als[i], self.tol)
            self.assertLess(derivatives_mals[i], self.tol)

    def test_trapezoidal_rule(self):
        """test for trapezoidal rule"""

        # compute numerical solution of the ODE
        operator = mdl.two_step_destruction(1, 2, 1, 2)
        initial_value = tt.zeros([2 ** 2, 2 ** (2 + 1), 2 ** 2, 2 ** 2], [1] * 4)
        initial_value.cores[0][0, -1, 0, 0] = 1
        initial_value.cores[1][0, -2, 0, 0] = 1
        initial_value.cores[2][0, 0, 0, 0] = 1
        initial_value.cores[3][0, 0, 0, 0] = 1
        initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=self.rank).ortho_right()
        step_sizes = [0.001] * 100 + [0.1] * 9 + [1] * 19
        solution_als = ode.trapezoidal_rule(operator, initial_value, initial_guess, step_sizes, tt_solver='als',
                                            progress=False)
        solution_mals = ode.trapezoidal_rule(operator, initial_value, initial_guess, step_sizes, tt_solver='mals',
                                             max_rank=self.rank, progress=False)

        # compute norm of the derivatives at the final 10 time steps
        derivatives_als = []
        derivatives_mals = []
        for i in range(10):
            derivatives_als.append((operator.dot(solution_als[-i - 1])).norm())
            derivatives_mals.append((operator.dot(solution_mals[-i - 1])).norm())

        # check if trapezoidal rule converged to stationary distribution
        for i in range(10):
            self.assertLess(derivatives_als[i], self.tol)
            self.assertLess(derivatives_mals[i], self.tol)

    def test_errors(self):
        """test for error computations"""

        # compute numerical solution of the ODE
        operator = mdl.signaling_cascade(3).tt2qtt([[2] * 6] * 3, [[2] * 6] * 3)
        initial_value = tt.unit(operator.row_dims, [0] * operator.order)
        initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=self.rank).ortho_right()
        step_sizes = [0.1] * 10
        solution_ee = ode.explicit_euler(operator, initial_value, step_sizes, progress=False, max_rank=self.max_rank)
        solution_ie = ode.implicit_euler(operator, initial_value, initial_guess, step_sizes, progress=False)
        solution_tr = ode.trapezoidal_rule(operator, initial_value, initial_guess, step_sizes, progress=False)

        # compute errors
        errors_ee = ode.errors_expl_euler(operator, solution_ee, step_sizes)
        errors_ie = ode.errors_impl_euler(operator, solution_ie, step_sizes)
        errors_tr = ode.errors_trapezoidal(operator, solution_tr, step_sizes)

        # check if errors are smaller than tolerance
        self.assertLess(np.max(errors_ee), self.tol)
        self.assertLess(np.max(errors_ie), self.tol)
        self.assertLess(np.max(errors_tr), self.tol)

    def test_adaptive(self):
        """test for adaptive_step_size"""

        # compute numerical solution of the ODE
        operator = mdl.signaling_cascade(2).tt2qtt([[2] * 6] * 2, [[2] * 6] * 2)
        initial_value = tt.unit(operator.row_dims, [0] * operator.order)
        initial_guess = tt.ones(operator.row_dims, [1] * operator.order, ranks=self.rank).ortho_right()
        solution_ie, _ = ode.adaptive_step_size(operator, initial_value, initial_guess, 300, step_size_first=1,
                                             second_method='two_step_Euler', progress=False)
        solution_tr, _ = ode.adaptive_step_size(operator, initial_value, initial_guess, 300, step_size_first=1,
                                             second_method='trapezoidal_rule', progress=False)

        # compute norm of the derivatives at the final time step
        derivative_ie = (operator.dot(solution_ie[-1])).norm()
        derivative_tr = (operator.dot(solution_tr[-1])).norm()

        # check if converged to stationary distribution
        self.assertLess(derivative_ie, self.tol)
        self.assertLess(derivative_tr, self.tol)
