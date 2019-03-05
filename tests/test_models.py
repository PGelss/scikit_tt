# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
from unittest import TestCase


class TestModels(TestCase):

    def setUp(self):
        """Setup for testing models"""

        # define parameters
        # -----------------

        # set tolerance
        self.tol = 1e-1

        # set order
        self.order = 3

        # set CO adsorption rate for CO oxidation model
        self.k_ad_co = 1e4

        # set fequencies for kuramoto model
        self.frequencies = np.linspace(-5, 5, self.order)

        # set parameters for two-step destruction
        self.k_1 = 1
        self.k_2 = 2
        self.k_3 = 1
        self.m = 3

        # set parameters for toll station model
        self.number_of_lanes = 20
        self.number_of_cars = 10

        # generate Cantor dusts
        # ---------------------

        # generate 1-dimensional Cantor dust of level 3
        generator = np.array([1, 0, 1])
        self.cantor_dust = [np.kron(np.kron(generator, generator), generator)]

        # generate 2-dimensional Cantor dust of level 3
        generator = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        self.cantor_dust.append(np.kron(np.kron(generator, generator), generator))

        # generate 3-dimensional Cantor dust of level 3
        generator = np.array(
            [[[1, 0, 1], [0, 0, 0], [1, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]])
        self.cantor_dust.append(np.kron(np.kron(generator, generator), generator))

        # generate multisponges
        # ---------------------

        # generate 2-dimensional multisponge of level 3
        generator = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.multisponge = [np.kron(np.kron(generator, generator), generator)]

        # generate 3-dimensional multisponge of level 3
        generator = np.array(
            [[[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
        self.multisponge.append(np.kron(np.kron(generator, generator), generator))

        # generate RGB fractal
        # --------------------

        # define RGB matrices
        self.matrix_r = np.random.rand(3, 3)
        self.matrix_g = np.random.rand(3, 3)
        self.matrix_b = np.random.rand(3, 3)

        # generate RGB fractal
        self.rgb_fractal = np.zeros([27, 27, 3])
        self.rgb_fractal[:, :, 0] = np.kron(np.kron(self.matrix_r, self.matrix_r), self.matrix_r)
        self.rgb_fractal[:, :, 1] = np.kron(np.kron(self.matrix_g, self.matrix_g), self.matrix_g)
        self.rgb_fractal[:, :, 2] = np.kron(np.kron(self.matrix_b, self.matrix_b), self.matrix_b)

        # generate VICsek fractals
        # ------------------------

        # generate 2-dimensional Vicsek fractal of level 3
        generator = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.vicsek_fractal = [np.kron(np.kron(generator, generator), generator)]

        # generate 3-dimensional Vicsek fractal of level 3
        generator = np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        self.vicsek_fractal.append(np.kron(np.kron(generator, generator), generator))

    def test_cantor_dust(self):
        """test for Cantor dust"""

        # generating Cantor dusts
        cantor_dust = []
        for i in range(3):
            cantor_dust.append(mdl.cantor_dust(i + 1, 3))

        # check if construction is correct
        for i in range(3):
            self.assertEqual(np.sum(self.cantor_dust[i] - cantor_dust[i]), 0)

    def test_multisponge(self):
        """test for multisponge"""

        # generating multisponges
        multisponge = []
        for i in range(2):
            multisponge.append(mdl.multisponge(i + 2, 3))

        # check if construction is correct
        for i in range(2):
            self.assertEqual(np.sum(self.multisponge[i] - multisponge[i]), 0)

        # check if construction fails when dimension equal to 1
        with self.assertRaises(ValueError):
            mdl.multisponge(1, 1)

    def test_rgb_fractal(self):
        """test for rgb_fractal"""

        # generate RGB fractal
        rgb_fractal = mdl.rgb_fractal(self.matrix_r, self.matrix_g, self.matrix_b, 3)

        # check if construction is correct
        self.assertEqual(np.sum(self.rgb_fractal - rgb_fractal), 0)

    def test_vicsek_fractal(self):
        """test for vicsek_fractal"""

        # generating multisponges
        vicsek_fractal = []
        for i in range(2):
            vicsek_fractal.append(mdl.vicsek_fractal(i + 2, 3))

        # check if construction is correct
        for i in range(2):
            self.assertEqual(np.sum(self.vicsek_fractal[i] - vicsek_fractal[i]), 0)

        # check if construction fails when dimension equal to 1
        with self.assertRaises(ValueError):
            mdl.vicsek_fractal(1, 1)

    def test_fpu_kuramoto(self):
        """tests for Fermi-Pasta-Ulam and Kuramoto model"""

        mdl.fpu_coefficients(self.order)
        mdl.kuramoto_coefficients(self.order, self.frequencies)

    def test_co_oxidation(self):
        """tests for CO oxidation"""

        # construct operator
        op = mdl.co_oxidation(self.order, self.k_ad_co, cyclic=True)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)

    def test_signaling_cascade(self):
        """tests for signaling cascade"""

        # construct operator
        op = mdl.signaling_cascade(self.order)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)

    def test_toll_station(self):
        """tests for toll station"""

        # construct operator
        op = mdl.toll_station(self.number_of_lanes, self.number_of_cars)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)

    def test_two_step(self):
        """tests for two-setp destruction"""

        # construct operator
        op = mdl.two_step_destruction(self.k_1, self.k_2, self.k_3, self.m)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)
