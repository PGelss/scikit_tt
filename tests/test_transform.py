# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.transform as tdt


class TestMANDy(TestCase):

    def setUp(self):
        """..."""

        self.data = np.random.rand(10, 20)


        

    def test_basis_functions(self):
        """test basis functions"""

        constant = tdt.constant_function()
        indicator = tdt.indicator_function(0,0,0.5)
        identity = tdt.identity(0)
        sin = tdt.sin(0,1)
        cos = tdt.cos(0,1)
        gauss = tdt.gauss_function(0,1,1)
        periodic_gauss = tdt.periodic_gauss_function(0,1,1)
        
        self.assertEqual(np.sum(np.abs(constant(self.data)-np.ones(20))), 0)
        self.assertEqual(np.sum(np.abs(indicator(self.data)-np.logical_and(self.data[0, :]>=0, self.data[0, :]<0.5))), 0)
        self.assertEqual(np.sum(np.abs(identity(self.data)-self.data[0, :])), 0)
        self.assertEqual(np.sum(np.abs(sin(self.data)-np.sin(self.data[0,:]))), 0)
        self.assertEqual(np.sum(np.abs(cos(self.data)-np.cos(self.data[0,:]))), 0)
        self.assertEqual(np.sum(np.abs(gauss(self.data)-np.exp(-0.5 * (self.data[0,:] - 1) ** 2))), 0)
        self.assertEqual(np.sum(np.abs(periodic_gauss(self.data)-np.exp(-0.5 * np.sin(0.5 * (self.data[0,:] - 1)) ** 2))), 0)
        
        
        
        