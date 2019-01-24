# -*- coding: utf-8 -*-

from unittest import TestCase
from scikit_tt.tensor_train import TT
import scikit_tt.slim as slim
import numpy as np


class TestSLIM(TestCase):

    def setUp(self):
        """Construct exact TT decomposition for the Markov generator of the CO oxidation model. See [1]_ for details.

        References
        ----------
        .. [1] P. Gelß, S. Matera, C. Schütte, "Solving the master equation without kinetic Monte Carlo: Tensor train
               approximations for a CO oxidation model", Journal of Computational Physics 314 (2016) 489–502 
        """

        # set tolerance for the error (smallest entry in the operator is k_diff_co=6.6e-2)
        self.tol = 1e-7

        # order of the TT operator and reaction rate for CO adsorption
        self.order = 5
        self.k_ad_co = 1e4

        # further parameters of the model
        self.k_ad_o2 = 9.7e7
        self.k_de_co = 9.2e6
        self.k_de_o2 = 2.8e1
        self.k_diff_co = 6.6e-2
        self.k_diff_o = 5.0e-1
        self.k_de_co2 = 1.7e5

        # define core elements
        s_mat = np.array([[-self.k_ad_co, 0, self.k_de_co], [0, 0, 0], [self.k_ad_co, 0, -self.k_de_co]])
        l_mat = [None, None, None, None, None, None, None]
        l_mat[0] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        l_mat[1] = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        l_mat[2] = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        l_mat[3] = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        l_mat[4] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        l_mat[5] = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        l_mat[6] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        m_mat = [None, None, None, None, None, None, None]
        m_mat[0] = np.array([[-self.k_ad_o2, 0, 0], [0, -self.k_diff_o, 0], [0, 0, -self.k_diff_co]])
        m_mat[1] = np.array([[0, self.k_de_o2, self.k_de_co2], [self.k_diff_o, 0, 0], [0, 0, 0]])
        m_mat[2] = np.array([[0, self.k_de_co2, 0], [0, 0, 0], [self.k_diff_co, 0, 0]])
        m_mat[3] = np.array([[0, self.k_diff_o, 0], [self.k_ad_o2, 0, 0], [0, 0, 0]])
        m_mat[4] = np.array([[-self.k_diff_o, 0, 0], [0, -self.k_de_o2, 0], [0, 0, -self.k_de_co2]])
        m_mat[5] = np.array([[0, 0, self.k_diff_co], [0, 0, 0], [0, 0, 0]])
        m_mat[6] = np.array([[-self.k_diff_co, 0, 0], [0, -self.k_de_co2, 0], [0, 0, 0]])
        i_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # construct TT cores for cyclic system
        cores = [np.zeros([1, 3, 3, 16])]
        cores[0][0, :, :, 0] = s_mat
        for i in range(7):
            cores[0][0, :, :, 1 + i] = l_mat[i]
        cores[0][0, :, :, 8] = i_mat
        for i in range(7):
            cores[0][0, :, :, 9 + i] = m_mat[i]
        for k in range(1, self.order - 1):
            cores.append(np.zeros([16, 3, 3, 16]))
            cores[k][0, :, :, 0] = i_mat
            for i in range(7):
                cores[k][1 + i, :, :, 0] = m_mat[i]
            cores[k][8, :, :, 0] = s_mat
            for i in range(7):
                cores[k][8, :, :, 1 + i] = l_mat[i]
            cores[k][8, :, :, 8] = i_mat
            for i in range(7):
                cores[k][9 + i, :, :, 9 + i] = i_mat
        cores.append(np.zeros([16, 3, 3, 1]))
        cores[-1][0, :, :, 0] = i_mat
        for i in range(7):
            cores[-1][1 + i, :, :, 0] = m_mat[i]
        cores[-1][8, :, :, 0] = s_mat
        for i in range(7):
            cores[-1][9 + i, :, :, 0] = l_mat[i]

        # define TT operator for cyclic system
        self.operator_cyclic = TT(cores)

        # construct TT cores for non-cyclic system
        cores = [np.zeros([1, 3, 3, 10])]
        cores[0][0, :, :, 0] = s_mat
        for i in range(7):
            cores[0][0, :, :, 1 + i] = l_mat[i]
        cores[0][0, :, :, 8] = i_mat
        for k in range(1, self.order - 1):
            cores.append(np.zeros([10, 3, 3, 10]))
            cores[k][0, :, :, 0] = i_mat
            for i in range(7):
                cores[k][1 + i, :, :, 0] = m_mat[i]
            cores[k][8, :, :, 0] = s_mat
            for i in range(7):
                cores[k][8, :, :, 1 + i] = l_mat[i]
            cores[k][8, :, :, 8] = i_mat
        cores.append(np.zeros([10, 3, 3, 1]))
        cores[-1][0, :, :, 0] = i_mat
        for i in range(7):
            cores[-1][1 + i, :, :, 0] = m_mat[i]
        cores[-1][8, :, :, 0] = s_mat

        # define TT operator for non-cyclic system
        self.operator_noncyclic = TT(cores)

    def test_slim(self):
        """test for slim_mme"""

        # construct operator using slim_mme_hom

        # define state space
        state_space = [3] * self.order

        # define operator using automatic construction of SLIM decomposition
        # ------------------------------------------------------------------

        # define list of reactions
        single_cell_reactions = [[0, 2, self.k_ad_co], [2, 0, self.k_de_co]]
        two_cell_reactions = [[0, 1, 0, 1, self.k_ad_o2], [1, 0, 1, 0, self.k_de_o2], [2, 0, 1, 0, self.k_de_co2],
                              [1, 0, 2, 0, self.k_de_co2], [1, 0, 0, 1, self.k_diff_o], [0, 1, 1, 0, self.k_diff_o],
                              [0, 2, 2, 0, self.k_diff_co], [2, 0, 0, 2, self.k_diff_co]]

        # define operators for cyclic and non-cyclic systems
        operator_cyclic = slim.slim_mme_hom(state_space, single_cell_reactions, two_cell_reactions, threshold=1e-12)
        operator_noncyclic = slim.slim_mme_hom(state_space, single_cell_reactions, two_cell_reactions, cyclic=False)

        # compute errors
        err_cyclic = np.amax(np.abs(self.operator_cyclic.matricize() - operator_cyclic.matricize()))
        err_noncyclic = np.amax(np.abs(self.operator_noncyclic.matricize() - operator_noncyclic.matricize()))

        # check if errors are smaller than tolerance
        self.assertLess(err_cyclic, self.tol)
        self.assertLess(err_noncyclic, self.tol)
