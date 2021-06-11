# -*- coding: utf-8 -*-

import unittest as ut
from unittest import TestCase
from scikit_tt.tensor_train import TT
import scikit_tt.data_driven.ulam as ulam
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.evp as evp
import numpy as np
import scipy.sparse.linalg as splin
import os


class TestEVP(TestCase):

    @classmethod
    def setUpClass(cls):

        super(TestEVP, cls).setUpClass()

        # set tolerance for the error of the eigenvalues
        cls.tol_eigval = 5e-3

        # set tolerance for the error of the eigenvectors
        cls.tol_eigvec = 5e-2

        # set number of eigenvalues to compute
        cls.number_ev = 3

        # load data
        directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        transitions = np.load(directory + '/examples/data/triple_well_transitions.npz')["transitions"]

        # coarse-grain data
        transitions = np.int64(np.ceil(np.true_divide(transitions, 2)))

        # generate operators in TT format
        cls.operator_tt = ulam.ulam_2d(transitions, [25, 25], 2000)
        cls.operator_gevp = tt.eye(cls.operator_tt.row_dims)

        # matricize TT operator
        cls.operator_mat = cls.operator_tt.matricize()

        # define initial tensor train for solving the eigenvalue problem
        cls.initial_tt = tt.ones(cls.operator_tt.row_dims, [1] * cls.operator_tt.order, ranks=15).ortho_right()

    def test_als_eig(self):
        """test for ALS with solver='eig'"""

        # solve eigenvalue problem in TT format (number_ev=1 and operator_gevp is defined as identity tensor)
        _, eigentensor, _ = evp.als(self.operator_tt, self.initial_tt, operator_gevp=self.operator_gevp, repeats=10)
        self.assertTrue(isinstance(eigentensor, TT))

        # solve eigenvalue problem in TT format (number_ev=self.number_ev)
        eigenvalues_tt, eigenvectors_tt, _ = evp.als(self.operator_tt, self.initial_tt, number_ev=self.number_ev,
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

    def test_als_eigs(self):
        """test for ALS with solver='eigs'"""

        # solve eigenvalue problem in TT format (number_ev=self.number_ev)
        eigenvalues_tt, eigenvectors_tt, _ = evp.als(self.operator_tt, self.initial_tt, number_ev=self.number_ev,
                                                  repeats=10, solver='eigs')

        # solve eigenvalue problem in matrix format
        eigenvalues_mat, eigenvectors_mat = splin.eigs(self.operator_mat, k=self.number_ev)
        eigenvalues_mat = np.real(eigenvalues_mat)
        eigenvectors_mat = np.real(eigenvectors_mat)
        idx = eigenvalues_mat.argsort()
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

    def test_als_eigh(self):
        """test for ALS with solver='eigh'"""

        # make problem symmetric
        self.operator_tt = 0.5 * (self.operator_tt + self.operator_tt.transpose())
        self.operator_mat = 0.5 * (self.operator_mat + self.operator_mat.transpose())

        # solve eigenvalue problem in TT format (number_ev=self.number_ev)
        eigenvalues_tt, eigenvectors_tt, _ = evp.als(self.operator_tt, self.initial_tt, number_ev=self.number_ev,
                                                  repeats=10, solver='eigh')

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

    def test_power_method(self):
        """test for inverse power iteration method"""

        # solve eigenvalue problem in TT format
        evp.power_method(self.operator_tt, self.initial_tt, operator_gevp=self.operator_gevp)
        eigenvalue_tt, eigenvector_tt = evp.power_method(self.operator_tt, self.initial_tt)

        # solve eigenvalue problem in matrix format
        eigenvalue_mat, eigenvector_mat = splin.eigs(self.operator_mat, k=1)

        # compute relative error between exact and approximate eigenvalues
        rel_err_val = np.abs(eigenvalue_mat - eigenvalue_tt) / np.abs(eigenvalue_mat)

        # compute relative error between exact and approximate eigenvectors
        norm_1 = np.linalg.norm(eigenvector_mat + eigenvector_tt.matricize()[:, None])
        norm_2 = np.linalg.norm(eigenvector_mat - eigenvector_tt.matricize()[:, None])
        rel_err_vec = np.amin([norm_1, norm_2]) / np.linalg.norm(eigenvector_mat)

        # check if relative errors are smaller than tolerance
        self.assertLess(rel_err_val, self.tol_eigval)
        self.assertLess(rel_err_vec, self.tol_eigvec)


if __name__ == '__main__':
    ut.main()
