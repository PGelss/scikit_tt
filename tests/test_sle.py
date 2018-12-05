#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unittest import TestCase
from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.sle as sle
import numpy as np
import scipy as sp


class TestSLE(TestCase):

    def setUp(self):
        """Construct a Toeplitz matrix for testing the routines in sle.py"""

        # set tolerance for the error
        self.tol = 1e-7

        # set order of the resulting TT operator
        self.order = 10

        # generate Toeplitz matrix
        self.operator_mat = sp.linalg.toeplitz(np.arange(1, 2 ** self.order + 1), np.arange(1, 2 ** self.order + 1))

        # decompose Toeplitz matrix into TT format
        self.operator_tt = TT(self.operator_mat.reshape([2] * 2 * self.order))

        # define right-hand side as vector of all ones (matrix case)
        self.rhs_mat = np.ones(self.operator_mat.shape[0])

        # define right-hand side as tensor train of all ones (tensor case)
        self.rhs_tt = tt.ones(self.operator_tt.row_dims, [1] * self.operator_tt.order)

        # define initial tensor train for solving the system of linear equations
        self.initial_tt = tt.ones(self.operator_tt.row_dims, [1] * self.operator_tt.order, ranks=5).ortho_right()

    def test_als(self):
        """test for ALS"""

        # solve system of linear equations in matrix format
        solution_mat = np.linalg.solve(self.operator_mat, self.rhs_mat)

        # solve system of linear equations in TT format
        solution_tt = sle.als(self.operator_tt, self.initial_tt, self.rhs_tt, repeats=1)

        # compute relative error
        rel_err = np.linalg.norm(solution_mat - solution_tt.matricize()) / np.linalg.norm(solution_mat)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_mals(self):
        """test for MALS"""

        # solve system of linear equations in matrix format
        solution_mat = np.linalg.solve(self.operator_mat, self.rhs_mat)

        # solve system of linear equations in TT format
        solution_tt = sle.mals(self.operator_tt, self.initial_tt, self.rhs_tt, repeats=1, threshold=1e-14, max_rank=10)

        # compute relative error
        rel_err = np.linalg.norm(solution_mat - solution_tt.matricize()) / np.linalg.norm(solution_mat)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)
