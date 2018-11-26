#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import scikit_tt
import mandy

from unittest import TestCase


class TestTT(TestCase):

    def setUp(self):
        """generate random parameters"""
        self.tol = 10e-12  # set tolerance for relative errors
        self.d = np.random.randint(4, 6)  # generate random order in [3,5]
        self.r = [1] + [np.random.randint(2, 5) for i in range(self.d - 1)] + [1]  # generate random ranks in [2,4]
        self.m = [np.random.randint(2, 5) for i in range(self.d)]  # generate random row dimensions in [2,4]
        self.n = [np.random.randint(2, 5) for i in range(self.d)]  # generate random column dimensions in [2,4]
        self.cores = [np.random.rand(self.r[i], self.m[i], self.n[i], self.r[i + 1]) for i in
                      range(self.d)]  # define cores
        self.T = scikit_tt.TT(self.cores)  # construct tensor train

    def test_construction_form_cores(self):
        """test tensor train class for list of cores"""
        self.assertEqual(self.T.order, self.d)
        self.assertEqual(self.T.ranks, self.r)
        self.assertEqual(self.T.row_dims, self.m)
        self.assertEqual(self.T.col_dims, self.n)
        self.assertEqual(self.T.cores, self.cores)

    def test_conversion(self):
        """test conversion to full format"""
        U = self.T.full()  # convert to full format
        err = 0  # number of unequal entries
        for i in range(np.prod(self.T.row_dims + self.T.col_dims)):  # loop through all elements
            j = np.unravel_index(i, self.T.row_dims + self.T.col_dims)  # convert flat index
            v = self.T.element(j)  # element of tensor train
            w = U[j]  # element of full tensor
            if (v - w) / v > self.tol:  # count unequal elements
                err += 1  # increment err by 1
        self.assertEqual(err, 0)

    def test_addition(self):
        """test addition/subtraction of tensor trains"""
        U = self.T - self.T  # compute difference
        U_norm = np.linalg.norm(np.reshape(U.full(), [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute norm of U
        T_norm = np.linalg.norm(
            np.reshape(self.T.full(), [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute norm of T
        self.assertLess(U_norm / T_norm, self.tol)

    def test_scalar_multiplication(self):
        """test scalar multiplication"""
        U = 3 * self.T - 2 * self.T;  # multiply with scalar values
        U = U.full() - self.T.full()  # compute difference
        U_norm = np.linalg.norm(np.reshape(U, [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute norm of U
        T_norm = np.linalg.norm(
            np.reshape(self.T.full(), [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute norm of T
        self.assertLess(U_norm / T_norm, self.tol)

    def test_transpose(self):
        """test transpose of tensor trains"""
        U = self.T.transpose().full()  # transpose in TT format and convert to full format
        U_norm = np.linalg.norm(np.reshape(U, [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute norm
        V = self.T.full()  # convert to full format
        p = [i + self.T.order for i in range(self.T.order)] + [i for i in range(self.T.order)]  # list for permutation
        V = np.transpose(V, p)  # transpose in full format
        V = V - U  # compute difference
        V_norm = np.linalg.norm(np.reshape(V, [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute norm
        self.assertLess(V_norm / U_norm, self.tol)

    def test_multiplication(self):
        """test multiplication of tensor trains"""
        U = self.T.full()  # convert to full format
        p = [i + self.T.order for i in range(self.T.order)] + [i for i in range(self.T.order)]  # list for permutation
        U = np.transpose(U, p)  # transpose in full format
        U = np.reshape(U, [np.prod(self.T.col_dims), np.prod(self.T.row_dims)])  # reshape to matrix
        U = U @ np.reshape(self.T.full(),
                           [np.prod(self.T.row_dims), np.prod(self.T.col_dims)])  # multiply with original tensor
        U_norm = np.linalg.norm(np.reshape(U, [np.prod(self.T.col_dims + self.T.col_dims)]))  # compute norm
        V = self.T.transpose() @ self.T  # tranpose in TT format
        V = np.reshape(V.full(), [np.prod(self.T.col_dims), np.prod(self.T.col_dims)])  # reshape to matrix
        V = V - U  # compute difference
        V_norm = np.linalg.norm(np.reshape(V, [np.prod(self.T.col_dims + self.T.col_dims)]))  # compute norm
        self.assertLess(V_norm / U_norm, self.tol)

    def test_construction_from_array(self):
        """test tensor train class for arrays"""
        U = self.T.full()  # convert to full format
        U_norm = np.linalg.norm(np.reshape(U, [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute Frobenius norm
        U = scikit_tt.TT(U)  # construct tensor train
        V = self.T - U  # compute difference
        V = V.full()  # convert to full format
        V_norm = np.linalg.norm(np.reshape(V, [np.prod(self.T.row_dims + self.T.col_dims)]))  # compute norm
        self.assertLess(V_norm / U_norm, self.tol)

    def test_operator(self):
        """test operator check"""
        T_check = self.T.isOperator()  # check T
        cores = [np.random.rand(self.T.ranks[i], self.T.row_dims[i], 1, self.T.ranks[i + 1]) for i in
                 range(self.T.order)]  # define cores
        U = scikit_tt.TT(cores)  # construct tensor train
        U_check = U.isOperator()  # check U
        cores = [np.random.rand(self.T.ranks[i], 1, self.T.col_dims[i], self.T.ranks[i + 1]) for i in
                 range(self.T.order)]  # define cores
        V = scikit_tt.TT(cores)  # construct tensor train
        V_check = V.isOperator()  # check V
        self.assertTrue(T_check)
        self.assertFalse(U_check)
        self.assertFalse(V_check)

    def test_zeros(self):
        """test tensor train of all zeros"""
        U = scikit_tt.TT.zeros(self.T.row_dims, self.T.col_dims)
        U_norm = np.linalg.norm(np.reshape(U.full(), [np.prod(self.T.row_dims + self.T.col_dims)]))
        self.assertEqual(U_norm, 0)

    def test_ones(self):
        """test tensor train of all ones"""
        U = scikit_tt.TT.ones(self.T.row_dims, self.T.col_dims)
        U = np.reshape(U.full(), [np.prod(self.T.row_dims), np.prod(self.T.col_dims)]) - np.ones(
            (np.prod(self.T.row_dims), np.prod(self.T.col_dims)))
        U_norm = np.linalg.norm(np.reshape(U, [np.prod(self.T.row_dims + self.T.col_dims)]))
        self.assertLess(U_norm / np.sqrt(np.prod(self.T.row_dims + self.T.col_dims)), self.tol)

    def test_identity(self):
        """test identity tensor train"""
        U = scikit_tt.TT.eye(self.T.row_dims)
        U = np.reshape(U.full(), [np.prod(self.T.row_dims), np.prod(self.T.row_dims)]) - np.eye(
            (np.prod(self.T.row_dims)))
        U_norm = np.linalg.norm(np.reshape(U, [np.prod(self.T.row_dims + self.T.row_dims)]))
        self.assertLess(U_norm / np.sqrt(np.prod(self.T.row_dims)), self.tol)

    def test_left_orthonormalization(self):
        """test left-orthonormalization"""
        cores = [np.random.rand(self.T.ranks[i], self.T.row_dims[i], 1, self.T.ranks[i + 1]) for i in
                 range(self.T.order)]  # define cores
        U = scikit_tt.TT(cores)
        U_full = U.full().reshape(np.prod(self.T.row_dims))
        U = U.ortho_left()
        U_ortho_full = U.full().reshape(np.prod(self.T.row_dims))
        err = 0
        for i in range(self.T.order - 1):
            V = np.einsum('ijkl,ijmn->klmn', U.cores[i], U.cores[i])
            V = V.reshape(U.ranks[i + 1], U.ranks[i + 1])
            if np.linalg.norm(V - np.eye(U.ranks[i + 1])) / np.sqrt(U.ranks[i + 1]) > self.tol:
                err += 1
        self.assertEqual(err, 0)
        self.assertLess(np.linalg.norm(U_full - U_ortho_full) / np.linalg.norm(U_full), self.tol)

    def test_right_orthonormalization(self):
        """test right-orthonormalization"""
        cores = [np.random.rand(self.T.ranks[i], self.T.row_dims[i], 1, self.T.ranks[i + 1]) for i in
                 range(self.T.order)]  # define cores
        U = scikit_tt.TT(cores)
        U_full = U.full().reshape(np.prod(self.T.row_dims))

        U = U.ortho_right()
        U_ortho_full = U.full().reshape(np.prod(self.T.row_dims))
        err = 0

        for i in range(self.T.order - 1, 1, -1):
            V = np.einsum('ijkl,mjkl->im', U.cores[i], U.cores[i])
            if np.linalg.norm(V - np.eye(U.ranks[i])) / np.sqrt(U.ranks[i]) > self.tol:
                err += 1
        self.assertEqual(err, 0)
        self.assertLess(np.linalg.norm(U_full - U_ortho_full) / np.linalg.norm(U_full), self.tol)

    def test_1_norm(self):
        """test 1-norm"""
        cores = [np.random.rand(self.T.ranks[i], self.T.row_dims[i], 1, self.T.ranks[i + 1]) for i in
                 range(self.T.order)]  # define cores
        U = scikit_tt.TT(cores)
        U_full = U.matricize()
        n_full = np.linalg.norm(U_full, 1)
        n_tt = U.norm(p=1)
        self.assertLess((n_full - n_tt) / n_full, self.tol)

    def test_2_norm(self):
        """test 2-norm"""
        cores = [np.random.rand(self.T.ranks[i], self.T.row_dims[i], 1, self.T.ranks[i + 1]) for i in
                 range(self.T.order)]  # define cores
        U = scikit_tt.TT(cores)
        U_full = U.matricize()
        n_full = np.linalg.norm(U_full)
        n_tt = U.norm()
        self.assertLess((n_full - n_tt) / n_full, self.tol)

    def test_Frobenius_norm(self):
        """test Frobenius norm"""
        U_full = self.T.matricize()
        n_full = np.linalg.norm(U_full)
        n_tt = self.T.norm()
        self.assertLess((n_full - n_tt) / n_full, self.tol)
