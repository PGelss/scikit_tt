# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tensor_train as tt
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

    def test_cantor_dust(self):
        """test for Cantor dust"""

        # generating Cantor dusts
        cantor_dust = []
        for i in range(3):
            cantor_dust.append(frac.cantor_dust(i,3))

        # check if construction is correct
        for i in range(3):
            self.assertEqual(self.cantor_dust, cantor_dust)
