# -*- coding: utf-8 -*-

from unittest import TestCase
import scikit_tt.utils as utl
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.evp as evp
import numpy as np
import scipy.sparse.linalg as splin
import scipy.io as io
import os


class TestSLE(TestCase):

    def setUp(self):
        """Consider the triple-well model for testing the routines in sle.py"""

        # set tolerance for the error of the eigenvalues
        self.tol_eigval = 1e-3

        # set tolerance for the error of the eigenvectors
        self.tol_eigvec = 5e-2

        # set number of eigenvalues to compute
        self.number_ev = 3

        # generate operator in TT format
        directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        transitions = io.loadmat(directory + '/examples/data/TripleWell2D_500.mat')["indices"]
        self.operator_tt = utl.perron_frobenius_2d(transitions, [50, 50], 500)

        # matricize TT operator
        self.operator_mat = self.operator_tt.matricize()

        # define initial tensor train for solving the eigenvalue problem
        self.initial_tt = tt.ones(self.operator_tt.row_dims, [1] * self.operator_tt.order, ranks=11).ortho_right()

    def test_als(self):
        """test for ALS"""

        # solve eigenvalue problem in TT format
        eigenvalues_tt, eigenvectors_tt = evp.als(self.operator_tt, self.initial_tt, number_ev=self.number_ev,
                                                  repeats=10)

        # solve eigenvalue problem in matrix format
        eigenvalues_mat, eigenvectors_mat = splin.eigs(self.operator_mat, k=self.number_ev)
        eigenvalues_mat = np.real(eigenvalues_mat)
        eigenvectors_mat = np.real(eigenvectors_mat)
        idx = eigenvalues_mat.argsort()[::-1]
        eigenvalues_mat = eigenvalues_mat[idx]
        eigenvectors_mat = eigenvectors_mat[:, idx]

        # compute relative error between exact and approximate eigenvalues
        rel_err_val = []
        for i in range(self.number_ev):
            rel_err_val.append(np.abs(eigenvalues_mat[i] - eigenvalues_tt[i]) / eigenvalues_mat[i])

        # compute relative error between exact and approximate eigenvectors
        rel_err_vec = []
        for i in range(self.number_ev):
            norm_1 = np.linalg.norm(eigenvectors_mat[:, i] + eigenvectors_tt[i].matricize())
            norm_2 = np.linalg.norm(eigenvectors_mat[:, i] - eigenvectors_tt[i].matricize())
            rel_err_vec.append(np.amin([norm_1, norm_2]) / np.linalg.norm(eigenvectors_mat[:, i]))

        # check if relative errors are smaller than tolerance
        for i in range(self.number_ev):
            self.assertLess(rel_err_val[i], self.tol_eigval)
            self.assertLess(rel_err_vec[i], self.tol_eigvec)
