# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import scikit_tt.models as mdl
import scikit_tt.data_driven.mandy as mandy


class TestMANDy(TestCase):

    def setUp(self):
        """Consider the Fermi-Pasta-Ulam problem and Kuramoto model for testing the routines in sle.py"""

        # set tolerance
        self.tol = 1e-5

        # number of oscillators
        self.fpu_d = 3
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

        # exact coefficient tensors
        self.fpu_xi_exact = mdl.fpu_coefficients(self.fpu_d)
        self.kuramoto_xi_exact = mdl.kuramoto_coefficients(self.kuramoto_d, self.kuramoto_w)

        # generate test data
        [self.fpu_x, self.fpu_y] = mdl.fermi_pasta_ulam(self.fpu_d, self.fpu_m)
        [self.kuramoto_x, self.kuramoto_y] = mdl.kuramoto(self.kuramoto_x_0, self.kuramoto_w, self.kuramoto_t,
                                                          self.kuramoto_m)

    def test_mandy_cm(self):
        """test coordinate-major approach"""

        # apply MANDy
        xi = mandy.mandy_cm(self.fpu_x, self.fpu_y, self.fpu_psi, threshold=1e-10)

        # compute relative error
        rel_err = (xi - self.fpu_xi_exact).norm() / self.fpu_xi_exact.norm()

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_mandy_fm(self):
        """test function-major approach"""

        # apply MANDy
        xi = mandy.mandy_fm(self.kuramoto_x, self.kuramoto_y, self.kuramoto_psi)

        # compute relative error
        rel_err = (xi - self.kuramoto_xi_exact).norm() / self.kuramoto_xi_exact.norm()

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)
