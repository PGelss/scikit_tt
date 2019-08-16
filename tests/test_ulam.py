# -*- coding: utf-8 -*-

import unittest as ut
from unittest import TestCase
import scikit_tt.data_driven.ulam as ulam
import numpy as np
import os


class TestPF(TestCase):

    def setUp(self):
        """Consider triple- and quadruple well potentials for testing"""

        # set tolerance for errors
        self.tol = 1e-10

        # load transition lists
        directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.transitions_2d = np.load(directory + '/examples/data/triple_well_transitions.npz')['transitions']
        self.transitions_3d = np.load(directory + '/examples/data/quadruple_well_transitions.npz')['transitions']

        # coarse-grain 3d data
        self.transitions_3d = np.int64(np.ceil(np.true_divide(self.transitions_3d, 5)))

    def test_ulam_2d(self):
        """test for perron_frobenius_2d"""

        # construct transition operator in TT format
        operator = ulam.ulam_2d(self.transitions_2d, [50, 50], 500)

        # construct full operator
        operator_full = np.zeros([50, 50, 50, 50])
        for i in range(self.transitions_2d.shape[1]):
            [x_1, y_1, x_2, y_2] = self.transitions_2d[:, i] - 1
            operator_full[x_2, y_2, x_1, y_1] += 1
        operator_full *= np.true_divide(1, 500)

        # compute error
        error = np.abs(operator.full() - operator_full).sum()

        # check if error is smaller than tolerance         
        self.assertLess(error, self.tol)

    def test_ulam_3d(self):
        """test for perron_frobenius_3d"""

        # construct transition operator in TT format
        operator = ulam.ulam_3d(self.transitions_3d, [5, 5, 5], 12500)

        # construct full operator
        operator_full = np.zeros([5, 5, 5, 5, 5, 5])
        for i in range(self.transitions_3d.shape[1]):
            [x_1, y_1, z_1, x_2, y_2, z_2] = self.transitions_3d[:, i] - 1
            operator_full[x_2, y_2, z_2, x_1, y_1, z_1] += 1
        operator_full *= np.true_divide(1, 12500)

        # compute error
        error = np.abs(operator.full() - operator_full).sum()

        # check if error is smaller than tolerance         
        self.assertLess(error, self.tol)


if __name__ == '__main__':
    ut.main()
