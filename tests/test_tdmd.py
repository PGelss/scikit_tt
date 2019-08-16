# -*- coding: utf-8 -*-

from unittest import TestCase
from scikit_tt.tensor_train import TT
import scikit_tt.data_driven.tdmd as tdmd
import numpy as np
import scipy.linalg as lin
import os


class TestTDMD(TestCase):

    def setUp(self):
        """Consider the von Kármán vortex street for testing"""

        # load data
        path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data = np.load(path + "/examples/data/karman_snapshots.npz")['snapshots']
        self.x = data[:, :, 0:-1]
        self.y = data[:, :, 1:]

        # set tolerance for the errors
        self.tol = 1e-8

    # noinspection PyTupleAssignmentBalance
    def test_tdmd_exact(self):
        """test for tdmd_exact"""

        # apply exact TDMD
        # ----------------

        # convert data to TT format
        x = TT(self.x[:, :, :, None, None, None])
        y = TT(self.y[:, :, :, None, None, None])

        # compute eigenvalues and modes
        eigenvalues_tdmd, modes_tdmd = tdmd.tdmd_exact(x, y, threshold=1e-10)

        # convert modes to full format and reshape for comparison
        modes_tdmd = modes_tdmd.full()
        modes_tdmd = modes_tdmd.reshape(modes_tdmd.shape[0] * modes_tdmd.shape[1], modes_tdmd.shape[2])

        # apply classical exact DMD
        # -------------------------

        # reshape data
        x = self.x.reshape(self.x.shape[0] * self.x.shape[1], self.x.shape[2])
        y = self.y.reshape(self.y.shape[0] * self.y.shape[1], self.y.shape[2])

        # decompose x
        u, s, v = lin.svd(x, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

        # construct reduced matrix
        reduced_matrix = u.T.dot(y).dot(v.T).dot(np.diag(np.reciprocal(s)))

        # find eigenvalues
        eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)

        # sort eigenvalues
        ind = np.argsort(eigenvalues)[::-1]
        eigenvalues_dmd = eigenvalues[ind]

        # compute modes
        modes_dmd = y.dot(v.T).dot(np.diag(np.reciprocal(s))).dot(eigenvectors[:, ind]).dot(
            np.diag(np.reciprocal(eigenvalues_dmd)))

        # compute errors
        error_ev = np.linalg.norm(eigenvalues_tdmd - eigenvalues_dmd)
        error_md = []
        for i in range(modes_dmd.shape[1]):
            error_md.append(np.min([np.linalg.norm(modes_dmd[:, i] - modes_tdmd[:, i]),
                                    np.linalg.norm(modes_dmd[:, i] + modes_tdmd[:, i])]))

        # check if relative errors are smaller than tolerance
        self.assertLess(error_ev, self.tol)
        for i in range(len(error_md)):
            self.assertLess(error_md[i], self.tol)

    # noinspection PyTupleAssignmentBalance
    def test_tdmd_standard(self):
        """test for tdmd_standard"""

        # apply standard TDMD
        # ----------------

        # convert data to TT format
        x = TT(self.x[:, :, :, None, None, None])
        y = TT(self.y[:, :, :, None, None, None])

        # compute eigenvalues and modes
        eigenvalues_tdmd, modes_tdmd = tdmd.tdmd_standard(x, y, threshold=1e-10)

        # convert modes to full format and reshape for comparison
        modes_tdmd = modes_tdmd.full()
        modes_tdmd = modes_tdmd.reshape(modes_tdmd.shape[0] * modes_tdmd.shape[1], modes_tdmd.shape[2])

        # apply classical standard DMD
        # -------------------------

        # reshape data
        x = self.x.reshape(self.x.shape[0] * self.x.shape[1], self.x.shape[2])
        y = self.y.reshape(self.y.shape[0] * self.y.shape[1], self.y.shape[2])

        # decompose x
        u, s, v = lin.svd(x, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

        # construct reduced matrix
        reduced_matrix = u.T.dot(y).dot(v.T).dot(np.diag(np.reciprocal(s)))

        # find eigenvalues
        eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)

        # sort eigenvalues
        ind = np.argsort(eigenvalues)[::-1]
        eigenvalues_dmd = eigenvalues[ind]

        # compute modes
        modes_dmd = u.dot(eigenvectors[:, ind])

        # compute errors
        error_ev = np.linalg.norm(eigenvalues_tdmd - eigenvalues_dmd)
        error_md = []
        for i in range(modes_dmd.shape[1]):
            error_md.append(np.min([np.linalg.norm(modes_dmd[:, i] - modes_tdmd[:, i]),
                                    np.linalg.norm(modes_dmd[:, i] + modes_tdmd[:, i])]))

        # check if relative errors are smaller than tolerance
        self.assertLess(error_ev, self.tol)
        for i in range(len(error_md)):
            self.assertLess(error_md[i], self.tol)
