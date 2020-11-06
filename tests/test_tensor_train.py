# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT, rand
import unittest as ut
from unittest import TestCase


class TestTT(TestCase):

    def setUp(self):
        """Generate random parameters for a tensor train"""

        # set tolerance for relative errors
        self.tol = 1e-7

        # set threshold and maximum rank for orthonormalization
        self.threshold = 1e-14
        self.max_rank = 50

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
        self.t = TT(self.cores, threshold=self.threshold, max_rank=self.max_rank)

    def test_construction_from_cores(self):
        """test tensor train class for list of cores"""

        # check if all parameters are correct
        self.assertEqual(self.t.order, self.order)
        self.assertEqual(self.t.ranks, self.ranks)
        self.assertEqual(self.t.row_dims, self.row_dims)
        self.assertEqual(self.t.col_dims, self.col_dims)
        self.assertEqual(self.t.cores, self.cores)

        # check if construction fails if ranks are inconsistent
        with self.assertRaises(ValueError):
            TT([np.random.rand(1, 2, 3, 3), np.random.rand(4, 3, 2, 1)])

        # check if construction fails if cores are not 4-dimensional
        with self.assertRaises(ValueError):
            TT([np.random.rand(1, 2, 3), np.random.rand(3, 3, 2, 1)])

    def test_representation(self):
        """test string representation of tensor trains"""

        # get string representation
        string = self.t.__repr__()

        # check if string is not empty
        self.assertIsNotNone(string)

    def test_element(self):
        """test element extraction"""

        # indices of last entry
        indices = self.row_dims + self.col_dims

        # check if element extraction fails if indices are out of range
        with self.assertRaises(IndexError):
            indices[0] += 1
            self.t.element(indices)

        # check if element extraction fails if number of indices is not correct
        with self.assertRaises(ValueError):
            self.t.element(indices[1:])

        # check if element extraction fails if an index is not an integer
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            indices[0] = None
            self.t.element(indices)

        # check if element extraction fails if input is not a list of integers
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            self.t.element("a")

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

        with self.assertRaises(ValueError):
            rand([1, 2], [2, 3], [2, 2, 2]).full()

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

        # check if addition fails when inputs do not have the same dimensions
        with self.assertRaises(ValueError):
            cores = self.t.cores
            cores[0] = np.random.rand(self.ranks[0], self.row_dims[0] + 1, self.col_dims[0], self.ranks[1])
            self.t + TT(cores)

        # check if addition fails when input is not a tensor train
        with self.assertRaises(TypeError):
            self.t + 0

    def test_scalar_multiplication(self):
        """test scalar multiplication"""

        # random constant in [0,10]
        c = 10 * np.random.rand(1)[0]

        # multiply tensor train with scalar value, convert to full array, and reshape to vector
        t_tmp = c * self.t
        t_tmp = t_tmp.full().flatten()

        # convert t to full array and reshape to vector
        t_full = self.t.full().flatten()

        # compute error
        err = np.linalg.norm(t_tmp) / np.linalg.norm(t_full) - c

        # check if error is smaller than tolerance
        self.assertLess(err, self.tol)

        # check if multiplication fails when input is neither integer, float, nor complex
        with self.assertRaises(TypeError):
            self.t * "a"

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

    def test_rank_transpose(self):
        t_trans = self.t.rank_transpose()
        p = self.t.order
        for i in range(p):
            err = self.t.cores[i] - np.transpose(t_trans.cores[p - 1 - i], [3, 1, 2, 0])
            self.assertLess(np.linalg.norm(err), self.tol)

    def test_concatenate(self):
        # test concatenate with other TT
        p = self.t.order
        t_other = rand(row_dims=[2, 3], col_dims=[3, 2], ranks=[1, 3, 1])
        concat = self.t.concatenate(t_other)
        for i in range(concat.order):
            if i < p:
                err = concat.cores[i] - self.t.cores[i]
            else:
                err = concat.cores[i] - t_other.cores[i - p]
            self.assertLess(np.linalg.norm(err), self.tol)

        t_other = rand(row_dims=[2, 3], col_dims=[3, 2], ranks=[3, 3, 1])
        with self.assertRaises(ValueError):
            concat = self.t.concatenate(t_other)

        # test concatenate with list of cores
        t_other = []
        ranks = [1, 2, 3, 1]
        for i in range(len(ranks) - 1):
            t_other.append(np.random.random((ranks[i], np.random.randint(1, 4), np.random.randint(1, 4), ranks[i + 1])))
        concat = self.t.concatenate(t_other)
        for i in range(concat.order):
            if i < p:
                err = concat.cores[i] - self.t.cores[i]
            else:
                err = concat.cores[i] - t_other[i - p]
            self.assertLess(np.linalg.norm(err), self.tol)

        t_other.append(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            concat = self.t.concatenate(t_other)
        t_other = [np.random.random((3, 2, 2, 2))]
        with self.assertRaises(ValueError):
            concat = self.t.concatenate(t_other)

    def test_multiplication(self):
        """test multiplication of tensor trains"""

        # multiply t with its tranpose
        t_tmp = self.t.transpose().dot(self.t)

        # convert to full format and reshape to vector
        t_tmp = t_tmp.full().flatten()

        # convert t to full format and matricize
        t_full = self.t.full().reshape([np.prod(self.row_dims), np.prod(self.col_dims)])

        # multiply with its transpose and flatten
        t_full = (t_full.transpose().dot(t_full)).flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_tmp - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

        # check if multiplication fails when dimensions do not match
        with self.assertRaises(ValueError):
            t_tmp = self.t.copy()
            t_tmp.cores[0] = np.random.rand(self.ranks[0], self.row_dims[0] + 1, self.col_dims[0], self.ranks[1])
            self.t.transpose().dot(t_tmp)

        # check if multiplication fails when input is not a tensor train
        with self.assertRaises(TypeError):
            self.t.dot(0)

    def test_rank_tensordot(self):
        t = rand([2, 3, 4], [4, 2, 1], [2, 3, 4, 2])
        mat_front = np.random.random((1, 2))
        mat_back = np.random.random((2, 1))
        t2 = t.rank_tensordot(mat_back)
        t2 = t2.rank_tensordot(mat_front, mode='first')

        t.cores[-1] = np.tensordot(t.cores[-1], mat_back, axes=([3], [0]))
        t.cores[0] = np.tensordot(mat_front, t.cores[0], axes=([1], [0]))
        t.ranks = [t.cores[i].shape[0] for i in range(t.order)] + [t.cores[-1].shape[3]]
        err = t.full() - t2.full()
        self.assertLess(np.linalg.norm(err), self.tol)

        with self.assertRaises(ValueError):
            t.rank_tensordot(np.zeros((2, 3, 2)))
        with self.assertRaises(ValueError):
            t.rank_tensordot(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            t.rank_tensordot(np.zeros((3, 3)), mode='first')

    def test_construction_from_array(self):
        """test tensor train class for arrays"""

        # convert t to full format and construct tensor train form array
        t_full = self.t.full()

        # construct tensor train
        t_tmp = TT(t_full, threshold=self.threshold, max_rank=self.max_rank)

        # compute difference, convert to full format, and flatten
        t_diff = (self.t - t_tmp).full().flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_diff) / np.linalg.norm(t_full.flatten())

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

        # check if construction fails if number of dimensions is not a multiple of 2
        with self.assertRaises(ValueError):
            TT(np.random.rand(1, 2, 3))

        # check if construction fails if input is neither a list of cores nor an ndarray
        with self.assertRaises(TypeError):
            TT(None)

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
        t_left = t_col.ortho_left(threshold=1e-14)

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

        # check if orthonormalization fails if maximum rank is not positive
        with self.assertRaises(ValueError):
            t_col.ortho_left(max_rank=0)

        # check if orthonormalization fails if threshold is negative
        with self.assertRaises(ValueError):
            t_col.ortho_left(threshold=-1)

        # check if orthonormalization fails if start and end indices are not integers
        with self.assertRaises(TypeError):
            t_col.ortho_left(start_index="a")
            t_col.ortho_left(end_index="b")

    def test_right_orthonormalization(self):
        """test right-orthonormalization"""

        # construct non-operator tensor train
        cores = [self.cores[i][:, :, 0:1, :] for i in range(self.order)]
        t_col = TT(cores)

        # right-orthonormalize t
        t_right = t_col.ortho_right(threshold=1e-14)

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

        # check if orthonormalization fails if maximum rank is not positive
        with self.assertRaises(ValueError):
            t_col.ortho_right(max_rank=0)

        # check if orthonormalization fails if threshold is negative
        with self.assertRaises(ValueError):
            t_col.ortho_right(threshold=-1)

        # check if orthonormalization fails if start and end indices are not integers
        with self.assertRaises(TypeError):
            t_col.ortho_right(start_index="a")
            t_col.ortho_right(end_index="b")

    def test_orthonormalization(self):
        """test orthonormalization"""

        # construct non-operator tensor train
        cores = [self.cores[i][:, :, 0:1, :] for i in range(self.order)]
        t_col = TT(cores)

        # orthonormalize t
        t_right = t_col.ortho(threshold=1e-14)

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

        # check if orthonormalization fails if maximum rank is not positive
        with self.assertRaises(ValueError):
            t_col.ortho(max_rank=0)

        # check if orthonormalization fails if threshold is negative
        with self.assertRaises(ValueError):
            t_col.ortho(threshold=-1)

    def test_1_norm(self):
        """test 1-norm"""

        # construct tensor train without negative entries
        cores = [np.abs(self.cores[i][:, :, 0:1, :]) for i in range(self.order)]
        tt_col = TT(cores)

        # transpose
        tt_row = tt_col.transpose()

        # convert to full matrix
        tt_mat = tt_col.matricize()

        # compute norms
        norm_tt_row = tt_row.norm(p=1)
        norm_tt_col = tt_col.norm(p=1)
        norm_full = np.linalg.norm(tt_mat, 1)

        # compute relative errors
        rel_err_row = (norm_tt_row - norm_full) / norm_full
        rel_err_col = (norm_tt_col - norm_full) / norm_full

        # construct tensor-train operator without negative entries
        cores = [np.abs(self.cores[i][:, :, :, :]) for i in range(self.order)]
        tt_op = TT(cores)

        # convert to full matrix
        tt_mat = tt_op.matricize()

        # compute norms
        norm_tt = tt_op.norm(p=1)
        norm_full = np.linalg.norm(tt_mat, 1)

        # compute relative error
        rel_err = (norm_tt - norm_full) / norm_full

        # check if relative errors are smaller than tolerance
        self.assertLess(rel_err_row, self.tol)
        self.assertLess(rel_err_col, self.tol)
        self.assertLess(rel_err, self.tol)

    def test_2_norm(self):
        """test 2-norm"""

        # construct tensor train
        cores = [self.cores[i][:, :, 0:1, :] for i in range(self.order)]
        tt_col = TT(cores)

        # transpose
        tt_row = tt_col.transpose()

        # convert to full matrix
        tt_mat = tt_col.matricize()

        # compute norms
        norm_tt_row = tt_row.norm(p=2)
        norm_tt_col = tt_col.norm(p=2)
        norm_full = np.linalg.norm(tt_mat, 2)

        # compute relative errors
        rel_err_row = (norm_tt_row - norm_full) / norm_full
        rel_err_col = (norm_tt_col - norm_full) / norm_full

        # define tensor-train operator
        tt_op = self.t

        # convert to full matrix
        tt_mat = tt_op.matricize()

        # compute norms
        norm_tt = tt_op.norm(p=2)
        norm_full = np.linalg.norm(tt_mat, 'fro')

        # compute relative error
        rel_err = (norm_tt - norm_full) / norm_full

        # check if relative errors are smaller than tolerance
        self.assertLess(rel_err_row, self.tol)
        self.assertLess(rel_err_col, self.tol)
        self.assertLess(rel_err, self.tol)

    def test_p_norm(self):
        """test for p-norm, p>2"""

        with self.assertRaises(ValueError):
            self.t.norm(p=3)

    def test_qtt2tt_tt2qtt(self):
        """test qtt2tt and tt2qtt"""

        # suppose t to be in QTT format and contract first two cores of t
        t_tt = self.t.qtt2tt([2, self.order - 2])

        # convert t_tt to full array, and flatten
        t_tt_full = t_tt.full().flatten()

        # split the cores of t_tt, convert to full array, and flatten
        t_qtt = t_tt.tt2qtt([self.row_dims[0:2], self.row_dims[2:]], [self.col_dims[0:2], self.col_dims[2:]],
                            threshold=1e-14)

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

    def test_pinv(self):
        """test pinv"""

        # construct non-operator tensor train
        cores = [self.cores[i][:, :, 0:1, :] for i in range(self.order)]
        t = TT(cores)

        # compute pseudoinverse
        t_pinv = TT.pinv(t, self.order - 1)

        # matricize tensor trains
        t = t.full().reshape([np.prod(self.row_dims[:-1]), self.row_dims[-1]])
        t_pinv = t_pinv.full().reshape([np.prod(self.row_dims[:-1]), self.row_dims[-1]]).transpose()

        # compute relative errors
        rel_err_1 = np.linalg.norm(t.dot(t_pinv).dot(t) - t) / np.linalg.norm(t)
        rel_err_2 = np.linalg.norm(t_pinv.dot(t).dot(t_pinv) - t_pinv) / np.linalg.norm(t_pinv)
        rel_err_3 = np.linalg.norm((t.dot(t_pinv)).transpose() - t.dot(t_pinv)) / np.linalg.norm(t.dot(t_pinv))
        rel_err_4 = np.linalg.norm((t_pinv.dot(t)).transpose() - t_pinv.dot(t)) / np.linalg.norm(t_pinv.dot(t))

        # check if relative errors are smaller than tolerance
        self.assertLess(rel_err_1, self.tol)
        self.assertLess(rel_err_2, self.tol)
        self.assertLess(rel_err_3, self.tol)
        self.assertLess(rel_err_4, self.tol)

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

    def test_eye(self):
        """test identity tensor train"""

        # construct identity tensor train, convert to full format, and flatten
        t_eye = tt.eye(self.row_dims).full().flatten()

        # construct identity matrix and flatten
        t_full = np.eye(np.int(np.prod(self.row_dims))).flatten()

        # compute relative error
        rel_err = np.linalg.norm(t_eye - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_unit(self):
        """test unit tensor train"""

        # construct unit tensor train, convert to full format, and flatten
        t_unit = tt.unit(self.row_dims, [0] * self.order).full().flatten()

        # construct unit vector
        t_full = np.eye(np.int(np.prod(self.row_dims)), 1).T

        # compute relative error
        rel_err = np.linalg.norm(t_unit - t_full) / np.linalg.norm(t_full)

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)

    def test_random(self):
        """test random tensor train"""

        # construct random tensor train
        t_rand = tt.rand(self.row_dims, self.col_dims)

        # check if attributes are correct
        self.assertEqual(t_rand.order, self.order)
        self.assertEqual(t_rand.row_dims, self.row_dims)
        self.assertEqual(t_rand.col_dims, self.col_dims)
        self.assertEqual(t_rand.ranks, [1] * (self.order + 1))

    def test_uniform(self):
        """test uniform tensor train"""

        # construct uniform tensor train
        norm = 10 * np.random.rand()
        t_uni = tt.uniform(self.row_dims, ranks=self.ranks[1], norm=norm)

        # compute norms
        norm_tt = t_uni.norm()

        # compute relative error
        rel_err = (norm_tt - norm) / norm

        # check if relative error is smaller than tolerance
        self.assertLess(rel_err, self.tol)


if __name__ == '__main__':
    ut.main()
