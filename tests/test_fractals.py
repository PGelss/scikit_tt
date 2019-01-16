# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.fractals as frac
from unittest import TestCase

class TestFractals(TestCase):

    def setUp(self):
        """Setup for testing fractals.py"""

        # set tolerance for errors
        self.tol_eigval = 1e-10

        # generate Cantor dusts
        # ---------------------

        # generate 1-dimensional Cantor dust of level 3
        generator = np.array([1, 0, 1])
        self.cantor_dust = [np.kron(np.kron(generator, generator), generator)]

        # generate 2-dimensional Cantor dust of level 3
        generator = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        self.cantor_dust.append(np.kron(np.kron(generator, generator), generator))

        # generate 3-dimensional Cantor dust of level 3
        generator = np.array([[[1, 0, 1], [0, 0, 0], [1, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]])
        self.cantor_dust.append(np.kron(np.kron(generator, generator), generator))

        # generate multisponges
        # ---------------------

        # generate 2-dimensional multisponge of level 3
        generator = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.multisponge = [np.kron(np.kron(generator, generator), generator)]

        # generate 3-dimensional multisponge of level 3
        generator = np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
        self.multisponge.append(np.kron(np.kron(generator, generator), generator))

        # generate RGB fractal
        # --------------------

        # define RGB matrices
        self.matrix_r = np.random.rand(3, 3)
        self.matrix_g = np.random.rand(3, 3)
        self.matrix_b = np.random.rand(3, 3)

        # generate RGB fractal
        self.rgb_fractal = np.zeros([27, 27, 3])
        self.rgb_fractal[:,:,0] = np.kron(np.kron(self.matrix_r, self.matrix_r), self.matrix_r)
        self.rgb_fractal[:,:,1] = np.kron(np.kron(self.matrix_g, self.matrix_g), self.matrix_g)
        self.rgb_fractal[:,:,2] = np.kron(np.kron(self.matrix_b, self.matrix_b), self.matrix_b)

        # generate VICsek fractals
        # ------------------------

        # generate 2-dimensional Vicsek fractal of level 3
        generator = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.vicsek_fractal = [np.kron(np.kron(generator, generator), generator)]

        # generate 3-dimensional Vicsek fractal of level 3
        generator = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        self.vicsek_fractal.append(np.kron(np.kron(generator, generator), generator))

    def test_cantor_dust(self):
        """test for Cantor dust"""

        # generating Cantor dusts
        cantor_dust = []
        for i in range(3):
            cantor_dust.append(frac.cantor_dust(i+1,3))

        # check if construction is correct
        for i in range(3):
            self.assertEqual(np.sum(self.cantor_dust[i] - cantor_dust[i]),0)

    def test_multisponge(self):
        """test for multisponge"""

        # generating multisponges
        multisponge = []
        for i in range(2):
            multisponge.append(frac.multisponge(i+2,3))

        # check if construction is correct
        for i in range(2):
            self.assertEqual(np.sum(self.multisponge[i] - multisponge[i]),0)

    def test_rgb_fractal(self):
        """test for rgb_fractal"""

        # generate RGB fractal
        rgb_fractal = frac.rgb_fractal(self.matrix_r, self.matrix_g, self.matrix_b, 3)

        # check if construction is correct
        self.assertEqual(np.sum(self.rgb_fractal - rgb_fractal),0)

    def test_vicsek_fractal(self):
        """test for vicsek_fractal"""

        # generating multisponges
        vicsek_fractal = []
        for i in range(2):
            vicsek_fractal.append(frac.vicsek_fractal(i+2,3))

        # check if construction is correct
        for i in range(2):
            self.assertEqual(np.sum(self.vicsek_fractal[i] - vicsek_fractal[i]),0)