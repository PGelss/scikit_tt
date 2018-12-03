#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
from unittest import TestCase


class TestTT(TestCase):

    def setUp(self):
        """Generate random parameters for a tensor train"""

        # set tolerance for relative errors
        self.tol = 10e-10

        # generate random order in [3,5]
        self.order = np.random.randint(3, 6)

        # generate random ranks in [3,5]
        self.ranks = [1] + list(np.random.randint(3, high=6, size=self.order - 1)) + [1]

        # generate random row  and column dimensions in [3,5]
        self.row_dims = list(np.random.randint(3, high=6, size=self.order))
        self.col_dims = list(np.random.randint(3, high=6, size=self.order))

        # define cores
        self.cores = [2 * np.random.rand(self.ranks[i], self.row_dims[i], self.col_dims[i], self.ranks[i + 1]) - 1
                      for i in range(self.order)]

        # construct tensor train
        self.t = TT(self.cores)

    def test_construction_from_cores(self):
        """test tensor train class for list of cores"""

        # check if all parameters are correct
        self.assertEqual(self.t.order, self.order)
        self.assertEqual(self.t.ranks, self.ranks)
        self.assertEqual(self.t.row_dims, self.row_dims)
        self.assertEqual(self.t.col_dims, self.col_dims)
        self.assertEqual(self.t.cores, self.cores)

    def test_conversion(self):
        """test conversion to full format and element extraction"""

        # convert to full format
        t_full = self.t.full()

        # number of wrong entries
        err = 0

        # loop through all elements of the tensor
        for i in range(np.int(np.prod(self.row_dims + self.col_dims))):

            # convert flat index
            j = np.unravel_index(i, self.row_dims + self.col_dims)

            # extract elements of both representations
            v = self.t.element(list(j))
            w = t_full[j]

            # count wrong entries
            if (v - w) / v > self.tol:
                err += 1

        # check if no wrong entry exists
        self.assertEqual(err, 0)

    def test_matricize(self):
        """test matricization of tensor trains"""

        # matricize t
        t_mat = self.t.matricize()

        # convert t to full array and reshape
        t_full = self.t.full().reshape([np.prod(self.row_dims), np.prod(self.col_dims)])

        # compute relative error
        rel_err = np.linalg.norm(t_mat - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_addition(self):
        """test addition/subtraction of tensor trains"""

        # compute difference of t and itself
        t_diff = (self.t - self.t)

        # convert to full array and reshape to vector
        t_diff = t_diff.full().flatten()

        # convert t to full array and reshape to vector
        t_tmp = self.t.full().flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_diff) / np.linalg.norm(t_tmp)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_scalar_multiplication(self):
        """test scalar multiplication"""

        # random constant in [0,10]
        c = 10 * np.random.rand(1)[0]

        # multiply tensor train with scalar value, convert to full array, and reshape to vector
        t_tmp = TT.__mul__(self.t, c)
        t_tmp = t_tmp.full().flatten()

        # convert t to full array and reshape to vector
        t_full = self.t.full().flatten()

        # compute error
        err = np.linalg.norm(t_tmp) / np.linalg.norm(t_full) - c

        # check if error is smaller than tolerance
        self.assertLess(err, self.tol)

    def test_transpose(self):
        """test transpose of tensor trains"""

        # transpose in TT format, convert to full format, and reshape to vector
        t_trans = self.t.transpose().full().flatten()

        # convert to full format, transpose, and rehape to vector
        t_full = self.t.full().transpose(
            list(np.arange(self.order) + self.order) + list(np.arange(self.order))).flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_trans - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_multiplication(self):
        """test multiplication of tensor trains"""

        # multiply t with its tranpose
        t_tmp = self.t.transpose() @ self.t

        # convert to full format and reshape to vector
        t_tmp = t_tmp.full().flatten()

        # convert t to full format and matricize
        t_full = self.t.full().reshape([np.prod(self.row_dims), np.prod(self.col_dims)])

        # multiply with its transpose and flatten
        t_full = (t_full.transpose() @ t_full).flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_tmp - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_construction_from_array(self):
        """test tensor train class for arrays"""

        # convert t to full format and construct tensor train form array
        t_full = self.t.full()

        # construct tensor train
        t_tmp = TT(t_full)

        # compute difference, convert to full format, and flatten
        t_diff = (self.t - t_tmp).full().flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_diff) / np.linalg.norm(t_full.flatten())

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_operator(self):
        """test operator check"""

        # check t
        t_check = self.t.isoperator()

        # construct non-operator tensor trains
        cores = [self.cores[i][:, :, 0:1, :] for i in range(self.order)]
        u = TT(cores)
        cores = [self.cores[i][:, 0:1, :, :] for i in range(self.order)]
        v = TT(cores)

        # check u and v
        u_check = u.isoperator()
        v_check = v.isoperator()

        # check if operator checks are correct
        self.assertTrue(t_check)
        self.assertFalse(u_check)
        self.assertFalse(v_check)

    def test_left_orthonormalization(self):
        """test left-orthonormalization"""

        # construct non-operator tensor train
        cores = [self.cores[i][:, :, 0:1, :] for i in range(self.order)]
        t_col = TT(cores)

        # left-orthonormalize t
        t_left = t_col.ortho_left()

        # test if cores are left-orthonormal
        err = 0
        for i in range(self.order - 1):
            c = np.tensordot(t_left.cores[i], t_left.cores[i], axes=([0, 1], [0, 1])).squeeze()
            if np.linalg.norm(c - np.eye(t_left.ranks[i + 1])) > self.tol:
                err += 1

        # convert t_col to full format and flatten
        t_full = t_col.full().flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_left.full().flatten() - t_full) / np.linalg.norm(t_full)

        # check if t_left is left-orthonormal and equal to t_col
        self.assertEqual(err, 0)
        self.assertLess(rel_err, self.tol)

    def test_right_orthonormalization(self):
        """test right-orthonormalization"""

        # construct non-operator tensor train
        cores = [self.cores[i][:, :, 0:1, :] for i in range(self.order)]
        t_col = TT(cores)

        # right-orthonormalize t
        t_right = t_col.ortho_right()

        # test if cores are right-orthonormal
        err = 0
        for i in range(1, self.order):
            c = np.tensordot(t_right.cores[i], t_right.cores[i], axes=([1, 3], [1, 3])).squeeze()
            if np.linalg.norm(c - np.eye(t_right.ranks[i])) > self.tol:
                err += 1

        # convert t_col to full format and flatten
        t_full = t_col.full().flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_right.full().flatten() - t_full) / np.linalg.norm(t_full)

        # check if t_right is right-orthonormal and equal to t_col
        self.assertEqual(err, 0)
        self.assertLess(rel_err, self.tol)

    def test_1_norm(self):
        """test 1-norm"""

        # construct non-operator tensor train
        cores = [np.abs(self.cores[i][:, :, 0:1, :]) for i in range(self.order)]
        t_col = TT(cores)

        # convert to full array and flatten
        t_full = t_col.full().flatten()

        # compute norms
        norm_tt = t_col.norm(p=1)
        norm_full = np.linalg.norm(t_full, 1)

        # compute relative error
        rel_err = (norm_tt - norm_full) / norm_full

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_2_norm(self):
        """test 2-norm"""

        # convert to full array and flatten
        t_full = self.t.full().flatten()

        # compute norms
        norm_tt = self.t.norm()
        norm_full = np.linalg.norm(t_full)

        # compute relative error
        rel_err = (norm_tt - norm_full) / norm_full

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_qtt2tt_tt2qtt(self):
        """test qtt2tt and tt2qtt"""

        # suppose t to be in QTT format and contract first two cores of t
        t_tt = self.t.qtt2tt([2, self.order - 2])

        # convert t_tt to full array, and flatten
        t_tt_full = t_tt.full().flatten()

        # split the cores of t_tt, convert to full array, and flatten
        t_qtt = t_tt.tt2qtt([self.row_dims[0:2], self.row_dims[2:]], [self.col_dims[0:2], self.col_dims[2:]])

        # convert t_qtt to full array, and flatten
        t_qtt_full = t_qtt.full().flatten()

        # convert t and to full format and flatten
        t_full = self.t.full().flatten()

        # compute relative errors
        rel_err_tt = np.linalg.norm(t_tt_full - t_full) / np.linalg.norm(t_full)
        rel_err_qtt = np.linalg.norm(t_qtt_full - t_full) / np.linalg.norm(t_full)

        # check if relative errors are smaller than tolerance
        self.assertLess(rel_err_tt, self.tol)
        self.assertLess(rel_err_qtt, self.tol)

    def test_zeros(self):
        """test tensor train of all zeros"""

        # construct tensor train of all zeros
        t_zeros = tt.zeros(self.t.row_dims, self.t.col_dims)

        # compute norm
        t_norm = np.linalg.norm(t_zeros.full().flatten())

        # check if norm is 0
        self.assertEqual(t_norm, 0)

    def test_ones(self):
        """test tensor train of all ones"""

        # construct tensor train of all ones, convert to full format, and flatten
        t_ones = tt.ones(self.row_dims, self.col_dims).full().flatten()

        # construct full array of all ones
        t_full = np.ones(np.int(np.prod(self.row_dims)) * np.int(np.prod(self.col_dims)))

        # compute relative error
        rel_err = np.linalg.norm(t_ones - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_identity(self):
        """test identity tensor train"""

        # construct identity tensor train, convert to full format, and flatten
        t_eye = tt.eye(self.row_dims).full().flatten()

        # construct identity matrix and flatten
        t_full = np.eye(np.int(np.prod(self.row_dims))).flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_eye - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_uniform(self):
        """test uniform tensor train"""

        # construct uniform tensor train
        norm = 10 * np.random.rand()
        t_uni = tt.uniform(self.row_dims, ranks=self.ranks, norm=norm)

        # convert to full array and flatten
        t_full = t_uni.full().flatten()

        # compute norms
        norm_tt = t_uni.norm()
        norm_full = np.linalg.norm(t_full)

        # compute relative error
        rel_err = (norm_tt - norm_full) / norm_full

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)
