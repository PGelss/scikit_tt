# -*- coding: utf-8 -*-

from __future__ import division

from unittest import TestCase

import numpy as np

import scikit_tt.data_driven.transform as tdt


class TestBasisFunctions(TestCase):
    # def test_Function_initialized(self):
    #     f = tdt.Function(3)

    #     self.assertEqual(f([1, 2, 3]), 0)
    #     with self.assertRaises(ValueError):
    #         f([1, 2])
    #     with self.assertRaises(ValueError):
    #         f([1, 2, 3, 4])

    #     with self.assertRaises(ValueError):
    #         tdt.Function(0)

    # def test_Function_unitialized(self):
    #     f = tdt.Function()
    #     x = np.random.random((4,))

    #     self.assertEqual(f(x), 0)
    #     self.assertEqual(f.partial(x, 0), 0)
    #     self.assertEqual(f.partial2(x, 0, 0), 0)
    #     self.assertTrue((f.gradient(x) - np.zeros((4,)) == 0).all())
    #     self.assertTrue((f.hessian(x) - np.zeros((4, 4)) == 0).all())

    #     with self.assertRaises(ValueError):
    #         f([1, 2])
    #     with self.assertRaises(ValueError):
    #         f([1, 2, 3, 4, 5])
    #     with self.assertRaises(ValueError):
    #         f.partial(x, -1)
    #     with self.assertRaises(ValueError):
    #         f.partial2(x, 0, 7)

    # def test_OneCoord_initialized(self):
    #     f = tdt.OneCoordinateFunction(1, 3)

    #     self.assertEqual(f([1, 2, 3]), 0)
    #     with self.assertRaises(ValueError):
    #         f([1, 2])
    #     with self.assertRaises(ValueError):
    #         f([1, 2, 3, 4])

    #     with self.assertRaises(ValueError):
    #         tdt.OneCoordinateFunction(4, 3)
    #     with self.assertRaises(ValueError):
    #         tdt.OneCoordinateFunction(-1, 3)

    # def test_OneCoord_unitialized(self):
    #     f = tdt.OneCoordinateFunction(1)
    #     x = np.random.random((4,))

    #     self.assertEqual(f(x), 0)
    #     self.assertEqual(f.partial(x, 0), 0)
    #     self.assertEqual(f.partial2(x, 0, 0), 0)
    #     self.assertTrue((f.gradient(x) - np.zeros((4,)) == 0).all())
    #     self.assertTrue((f.hessian(x) - np.zeros((4, 4)) == 0).all())

    #     with self.assertRaises(ValueError):
    #         f([1, 2])
    #     with self.assertRaises(ValueError):
    #         f([1, 2, 3, 4, 5])
    #     with self.assertRaises(ValueError):
    #         f.partial(x, -1)
    #     with self.assertRaises(ValueError):
    #         f.partial2(x, 0, 7)

    def test_constant_function(self):
        f = tdt.ConstantFunction(0)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        hess = np.zeros((3, 3))

        self.assertEqual(f(x), 1)
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 0)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), 0)
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())

    def test_indicator_function(self):
        f = tdt.IndicatorFunction(1, 0.5, 1.0)

        self.assertEqual(f([0, 0.75]), 1)
        self.assertEqual(f([0, -2]), 0)
        self.assertEqual(f([0.75, 2]), 0)

        with self.assertRaises(NotImplementedError):
            f.partial([0, 0.75], 0)
        with self.assertRaises(NotImplementedError):
            f.partial2([0, 0.75], 0, 0)
        with self.assertRaises(NotImplementedError):
            f.gradient([0, 0.75])
        with self.assertRaises(NotImplementedError):
            f.hessian([0, 0.75])

    def test_identity_function(self):
        f = tdt.Identity(1)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 1
        hess = np.zeros((3, 3))

        self.assertEqual(f(x), x[1])
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 1)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), 0)
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())

    def test_monomial(self):
        f = tdt.Monomial(1, 3)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 3 * x[1] ** 2
        hess = np.zeros((3, 3))
        hess[1, 1] = 6 * x[1]

        self.assertEqual(f(x), x[1] ** 3)
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 3 * x[1] ** 2)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), 6 * x[1])
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())

    def test_legendre(self):
        f = tdt.Legendre(1, 3)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 15 * x[1] ** 2 / 2 - 3 / 2
        hess = np.zeros((3, 3))
        hess[1, 1] = 15 * x[1]

        self.assertAlmostEqual(f(x), 5 * x[1] ** 3 / 2 - 3 * x[1] / 2)
        self.assertAlmostEqual(f.partial(x, 0), 0)
        self.assertAlmostEqual(f.partial(x, 1), 15 * x[1] ** 2 / 2 - 3 / 2)
        self.assertAlmostEqual(f.partial2(x, 0, 0), 0)
        self.assertAlmostEqual(f.partial2(x, 1, 1), 15 * x[1])
        self.assertTrue((np.abs(f.gradient(x) - grad) <= 1e-10).all())
        self.assertTrue((np.abs(f.hessian(x) - hess) <= 1e-10).all())

        f = tdt.Legendre(1, 3, domain=2)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 15 * x[1] ** 2 / 16 - 3 / 4
        hess = np.zeros((3, 3))
        hess[1, 1] = 30 * x[1] / 16

        self.assertAlmostEqual(f(x), 5 * x[1] ** 3 / 16 - 3 * x[1] / 4)
        self.assertAlmostEqual(f.partial(x, 0), 0)
        self.assertAlmostEqual(f.partial(x, 1), 15 * x[1] ** 2 / 16 - 3 / 4)
        self.assertAlmostEqual(f.partial2(x, 0, 0), 0)
        self.assertAlmostEqual(f.partial2(x, 1, 1), 30 * x[1] / 16)
        self.assertTrue((np.abs(f.gradient(x) - grad) <= 1e-10).all())
        self.assertTrue((np.abs(f.hessian(x) - hess) <= 1e-10).all())

        with self.assertRaises(ValueError):
            f = tdt.Legendre(1, -4)

    def test_sin(self):
        f = tdt.Sin(1, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 0.5 * np.cos(0.5 * x[1])
        hess = np.zeros((3, 3))
        hess[1, 1] = -(0.5 ** 2) * np.sin(0.5 * x[1])

        self.assertEqual(f(x), np.sin(0.5 * x[1]))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 0.5 * np.cos(0.5 * x[1]))
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), -(0.5 ** 2) * np.sin(0.5 * x[1]))
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())

    def test_cos(self):
        f = tdt.Cos(1, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = -0.5 * np.sin(0.5 * x[1])
        hess = np.zeros((3, 3))
        hess[1, 1] = -(0.5 ** 2) * np.cos(0.5 * x[1])

        self.assertEqual(f(x), np.cos(0.5 * x[1]))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), -0.5 * np.sin(0.5 * x[1]))
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), -(0.5 ** 2) * np.cos(0.5 * x[1]))
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())

    def test_gauss_function(self):
        f = tdt.GaussFunction(1, 0.5, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = -np.exp(-(0.5 * (0.5 - x[1]) ** 2) / 0.5) * (-0.5 + x[1]) / 0.5

        self.assertEqual(f(x), np.exp(-0.5 * (x[1] - 0.5) ** 2 / 0.5))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), -np.exp(-(0.5 * (0.5 - x[1]) ** 2) / 0.5) * (-0.5 + x[1]) / 0.5)
        self.assertTrue((f.gradient(x) - grad == 0).all())

        with self.assertRaises(ValueError):
            tdt.GaussFunction(1, 0.5, 0)

    def test_periodicgauss_function(self):
        f = tdt.PeriodicGaussFunction(1, 0.5, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = (0.5 * np.exp(-(0.5 * np.sin(0.5 * 0.5 - 0.5 * x[1]) ** 2) / 0.5) *
                   np.cos(0.5 * 0.5 - 0.5 * x[1]) * np.sin(0.5 * 0.5 - 0.5 * x[1])) / 0.5

        self.assertEqual(f(x), np.exp(-0.5 * np.sin(0.5 * (x[1] - 0.5)) ** 2 / 0.5))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), (0.5 * np.exp(-(0.5 * np.sin(0.5 * 0.5 - 0.5 * x[1]) ** 2) / 0.5) *
                                           np.cos(0.5 * 0.5 - 0.5 * x[1]) * np.sin(0.5 * 0.5 - 0.5 * x[1])) / 0.5)
        self.assertTrue((f.gradient(x) - grad == 0).all())

        with self.assertRaises(ValueError):
            tdt.GaussFunction(1, 0.5, 0)


class TestMANDy(TestCase):

    def setUp(self):
        """..."""

        self.tol = 1e-10
        self.d = 10
        self.m = 5
        self.n = 20
        self.data = np.random.rand(self.d, self.m)
        self.data_2 = np.random.rand(self.d, self.n)
        self.phi_1 = [[tdt.ConstantFunction(0), tdt.Identity(i), tdt.Monomial(i, 2)] for i in range(self.d)]
        self.psi_1 = [lambda t: 1, lambda t: t, lambda t: t ** 2]
        self.phi_2 = [[tdt.ConstantFunction(0)] + [tdt.Sin(i, 1) for i in range(self.d)],
                      [tdt.ConstantFunction(0)] + [tdt.Cos(i, 1) for i in range(self.d)]]
        self.psi_2 = [lambda t: np.sin(t), lambda t: np.cos(t)]

    def test_basis_decomposition(self):
        """test construction of transformed data tensors"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1).transpose(cores=[self.d]).matricize()
        tdt_2 = np.zeros([3 ** self.d, self.m])
        for j in range(self.m):
            v = [1, self.data[0, j], self.data[0, j] ** 2]
            for i in range(1, self.d):
                v = np.kron(v, [1, self.data[i, j], self.data[i, j] ** 2])
            tdt_2[:, j] = v
        self.assertEqual(np.sum(np.abs(tdt_1 - tdt_2)), 0)

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1)
        core_0 = tdt.basis_decomposition(self.data, self.phi_1, single_core=0)
        core_1 = tdt.basis_decomposition(self.data, self.phi_1, single_core=1)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[0] - core_0)), 0)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[1] - core_1)), 0)

    def test_coordinate_major(self):
        """test coordinate-major decomposition"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1)
        tdt_2 = tdt.coordinate_major(self.data, self.psi_1)
        self.assertLess((tdt_1 - tdt_2).norm(), self.tol)

        core_0 = tdt.coordinate_major(self.data, self.psi_1, single_core=0)
        core_1 = tdt.coordinate_major(self.data, self.psi_1, single_core=1)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[0] - core_0)), 0)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[1] - core_1)), 0)

    def test_function_major(self):
        """test function-major decomposition"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_2)
        _ = tdt.function_major(self.data, self.psi_2, add_one=False)
        _ = tdt.function_major(self.data, self.psi_2, add_one=False, single_core=0)
        _ = tdt.function_major(self.data, self.psi_2, add_one=False, single_core=1)
        tdt_2 = tdt.function_major(self.data, self.psi_2)
        self.assertLess((tdt_1 - tdt_2).norm(), self.tol)

        core_0 = tdt.function_major(self.data, self.psi_2, single_core=0)
        core_1 = tdt.function_major(self.data, self.psi_2, single_core=1)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[0] - core_0)), 0)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[1] - core_1)), 0)

    def test_gram(self):
        """test construction of gram matrix"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1).transpose(cores=[self.d]).matricize()
        tdt_2 = tdt.basis_decomposition(self.data_2, self.phi_1).transpose(cores=[self.d]).matricize()
        gram = tdt.gram(self.data, self.data_2, self.phi_1)
        self.assertLess(np.sum(np.abs(tdt_1.T.dot(tdt_2) - gram)), self.tol)

    def test_hocur(self):
        """test higher-order CUR decomposition"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1).transpose(cores=[self.d]).matricize()
        tdt_2 = tdt.hocur(self.data, self.phi_1, 5, repeats=10, progress=False).transpose(cores=[self.d]).matricize()
        self.assertLess(np.sum(np.abs(tdt_1 - tdt_2)), self.tol)
