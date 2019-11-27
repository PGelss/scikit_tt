# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.transform as tdt


class TestMANDy(TestCase):

    def setUp(self):
        """..."""
        self.d = 10
        self.m = 5
        self.data = np.random.rand(self.d, self.m)
        self.phi_1 = [[tdt.constant_function(), tdt.identity(i), tdt.monomial(i,2)] for i in range(self.d)]
        self.psi_1 = [lambda t: 1, lambda t: t, lambda t: t**2]
        self.phi_2 = [[tdt.constant_function()] + [tdt.sin(i,1) for i in range(self.d)], [tdt.constant_function()] + [tdt.cos(i,1) for i in range(self.d)] ]
        self.psi_2 = [lambda t: np.sin(t), lambda t: np.cos(t)]


    def test_basis_functions(self):
        """test basis functions"""

        constant = tdt.constant_function()
        indicator = tdt.indicator_function(0,0,0.5)
        identity = tdt.identity(0)
        monomial = tdt.monomial(0,2)
        sin = tdt.sin(0,1)
        cos = tdt.cos(0,1)
        gauss = tdt.gauss_function(0,1,1)
        periodic_gauss = tdt.periodic_gauss_function(0,1,1)
        
        self.assertEqual(np.sum(np.abs(constant(self.data)-np.ones(self.m))), 0)
        self.assertEqual(np.sum(np.abs(indicator(self.data)-np.logical_and(self.data[0, :]>=0, self.data[0, :]<0.5))), 0)
        self.assertEqual(np.sum(np.abs(identity(self.data)-self.data[0, :])), 0)
        self.assertEqual(np.sum(np.abs(monomial(self.data)-self.data[0, :]**2)), 0)
        self.assertEqual(np.sum(np.abs(sin(self.data)-np.sin(self.data[0,:]))), 0)
        self.assertEqual(np.sum(np.abs(cos(self.data)-np.cos(self.data[0,:]))), 0)
        self.assertEqual(np.sum(np.abs(gauss(self.data)-np.exp(-0.5 * (self.data[0,:] - 1) ** 2))), 0)
        self.assertEqual(np.sum(np.abs(periodic_gauss(self.data)-np.exp(-0.5 * np.sin(0.5 * (self.data[0,:] - 1)) ** 2))), 0)
        
    def test_basis_decomposition(self):
        """test construction of transformed data tensors"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1).transpose(cores=[self.d]).matricize()
        tdt_2 = np.zeros([3**self.d, self.m])
        for j in range(self.m):
            v = [1, self.data[0,j], self.data[0,j]**2]
            for i in range(1,self.d):
                v = np.kron(v, [1, self.data[i,j], self.data[i,j]**2])
            tdt_2[:,j] = v
        self.assertEqual(np.sum(np.abs(tdt_1-tdt_2)), 0)

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1)
        core_0 = tdt.basis_decomposition(self.data, self.phi_1, single_core=0)
        core_1 = tdt.basis_decomposition(self.data, self.phi_1, single_core=1)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[0]-core_0)), 0)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[1]-core_1)), 0)






        
        