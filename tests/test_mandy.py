#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import mandy
import numpy as np

from unittest import TestCase


class TestMANDy(TestCase):

    def setUp(self):
        """load snapshot matrices"""
        self.tol = 0.05  # set tolerance for relative errors
        D = scipy.io.loadmat("../data/FPU_d2_m6000.mat")  # load data
        X = D["X"]
        Y = D["Y"]
        Xi = D["Xi"]
        m = 1000  # numer of snapshots
        self.X = X[:, :m]  # snapshot matrix X
        self.Y = Y[:, :m]  # snapshot matrix Y
        self.Xi = Xi  # exact solution
        self.psi = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]  # basis functions
        self.threshold = 10 ** -12  # threshold for SVDs

    def test_mandy_basis_major(self):
        """test MANDy using basis-major order"""
        Xi_tt = mandy.mandy_basis_major(self.X, self.Y, self.psi, threshold=self.threshold)
        Xi_tt = Xi_tt.full().reshape(np.prod(Xi_tt.row_dims[:-1]), Xi_tt.row_dims[-1], order='F')
        self.assertLess(np.linalg.norm(self.Xi - Xi_tt) / np.linalg.norm(self.Xi), self.tol)

    def test_mandy_basis_major_matrix(self):
        """test matrix-based MANDy using basis-major order"""
        Xi_mat = mandy.mandy_basis_major_matrix(self.X, self.Y, self.psi)
        self.assertLess(np.linalg.norm(self.Xi - Xi_mat.transpose()) / np.linalg.norm(self.Xi), self.tol)

