# -*- coding: utf-8 -*-

from __future__ import division

from unittest import TestCase

import numpy as np
import scipy.integrate as spint

import scikit_tt.data_driven.regression as reg
import scikit_tt.data_driven.transform as tdt
import scikit_tt.models as mdl
import scikit_tt.tensor_train as tt


class TestMANDy(TestCase):

    def setUp(self):
        """Consider the Fermi-Pasta-Ulam problem and Kuramoto model for testing the routines in sle.py"""

        # set tolerance
        self.tol = 1e-5

        # number of oscillators
        self.fpu_d = 4
        self.kuramoto_d = 10

        # parameters for the Fermi-Pasta_ulam problem
        self.fpu_m = 2000
        self.fpu_psi = [lambda t: 1, lambda t: t, lambda t: t ** 2, lambda t: t ** 3]

        # parameters for the Kuramoto model
        self.kuramoto_x_0 = 2 * np.pi * np.random.rand(self.kuramoto_d) - np.pi
        self.kuramoto_w = np.linspace(-5, 5, self.kuramoto_d)
        self.kuramoto_t = 100
        self.kuramoto_m = 1000
        self.kuramoto_psi = [lambda t: np.sin(t), lambda t: np.cos(t)]
        self.kuramoto_basis = [[tdt.ConstantFunction(0)] + [tdt.Sin(i, 1) for i in range(self.kuramoto_d)],
                               [tdt.ConstantFunction(0)] + [tdt.Cos(i, 1) for i in range(self.kuramoto_d)]]
        self.kuramoto_initial = tt.ones([11, 11], [1, 1], 11)

        # exact coefficient tensors
        self.fpu_xi_exact = mdl.fpu_coefficients(self.fpu_d)
        self.kuramoto_xi_exact = mdl.kuramoto_coefficients(self.kuramoto_d, self.kuramoto_w)

        # generate test data for FPU
        self.fpu_x = 0.2 * np.random.rand(self.fpu_d, self.fpu_m) - 0.1
        self.fpu_y = np.zeros((self.fpu_d, self.fpu_m))
        for j in range(self.fpu_m):
            self.fpu_y[0, j] = self.fpu_x[1, j] - 2 * self.fpu_x[0, j] + 0.7 * (
                    (self.fpu_x[1, j] - self.fpu_x[0, j]) ** 3 - self.fpu_x[0, j] ** 3)
            for i in range(1, self.fpu_d - 1):
                self.fpu_y[i, j] = self.fpu_x[i + 1, j] - 2 * self.fpu_x[i, j] + self.fpu_x[i - 1, j] + 0.7 * (
                        (self.fpu_x[i + 1, j] - self.fpu_x[i, j]) ** 3 - (self.fpu_x[i, j] - self.fpu_x[i - 1, j]) ** 3)
                self.fpu_y[-1, j] = - 2 * self.fpu_x[-1, j] + self.fpu_x[-2, j] + 0.7 * (
                        -self.fpu_x[-1, j] ** 3 - (self.fpu_x[-1, j] - self.fpu_x[-2, j]) ** 3)

        # generate test data for Kuramoto
        number_of_oscillators = len(self.kuramoto_x_0)

        def kuramoto_ode(_, theta):
            [theta_i, theta_j] = np.meshgrid(theta, theta)
            return self.kuramoto_w + 2 / number_of_oscillators * np.sin(theta_j - theta_i).sum(0) + 0.2 * np.sin(theta)

        sol = spint.solve_ivp(kuramoto_ode, [0, self.kuramoto_t], self.kuramoto_x_0, method='BDF',
                              t_eval=np.linspace(0, self.kuramoto_t, self.kuramoto_m))
        self.kuramoto_x = sol.y
        self.kuramoto_y = np.zeros([number_of_oscillators, self.kuramoto_m])
        for i in range(self.kuramoto_m):
            self.kuramoto_y[:, i] = kuramoto_ode(0, self.kuramoto_x[:, i])

    def test_mandy_cm(self):
        """test coordinate-major approach"""

        # apply MANDy
        xi = reg.mandy_cm(self.fpu_x, self.fpu_y, self.fpu_psi, threshold=1e-10)

        # compute relative error
        rel_err = (xi - self.fpu_xi_exact).norm() / self.fpu_xi_exact.norm()

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_mandy_fm(self):
        """test function-major approach"""

        # apply MANDy
        _ = reg.mandy_fm(self.kuramoto_x, self.kuramoto_y, self.kuramoto_psi, add_one=False)
        xi = reg.mandy_fm(self.kuramoto_x, self.kuramoto_y, self.kuramoto_psi)

        # compute relative error
        rel_err = (xi - self.kuramoto_xi_exact).norm() / self.kuramoto_xi_exact.norm()

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_mandy_kb(self):
        """test kernel-based approach"""

        # apply kernel-based MANDy
        z = reg.mandy_kb(self.kuramoto_x, self.kuramoto_y, self.kuramoto_basis)

        # construct coefficient tensor
        xi = tdt.basis_decomposition(self.kuramoto_x, self.kuramoto_basis)
        xi.cores[-1] = np.tensordot(xi.cores[-1], z.T, axes=(1, 0)).transpose([0, 3, 1, 2])
        xi.row_dims[-1] = z.shape[0]

        # compute relative error
        rel_err = (xi - self.kuramoto_xi_exact).norm() / self.kuramoto_xi_exact.norm()

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_arr(self):
        """test ARR"""

        # apply ARR
        _ = reg.arr(self.kuramoto_x, self.kuramoto_y, self.kuramoto_basis,
                    [self.kuramoto_initial.copy() for _ in range(10)], repeats=1, rcond=10 ** -12, progress=False)
        xi = reg.arr(self.kuramoto_x, self.kuramoto_y, self.kuramoto_basis, self.kuramoto_initial, repeats=10,
                     rcond=10 ** -12, progress=False)

        # merge tensor-trains
        for i in range(self.kuramoto_d):
            xi[i].order += 1
            xi[i].cores.append(np.eye(self.kuramoto_d)[:, i].reshape([1, 10, 1, 1]))
            xi[i].row_dims.append(10)
            xi[i].col_dims.append(1)
            xi[i].ranks.append(1)
        xi_comb = xi[0]
        for i in range(1, self.kuramoto_d):
            xi_comb = xi_comb + xi[i]

        # compute relative error
        rel_err = (xi_comb - self.kuramoto_xi_exact).norm() / self.kuramoto_xi_exact.norm()

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)
