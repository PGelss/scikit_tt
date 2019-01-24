# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.models as mdl
from unittest import TestCase


class TestModels(TestCase):

    def setUp(self):
        """Define parameters"""

        # set dimension
        self.order = 3
        self.k_ad_co = 1e4
        self.number_of_snapshots = 100
        self.theta_init = 2 * np.pi * np.random.rand(self.order) - np.pi
        self.frequencies = np.linspace(-5, 5, self.order)
        self.time = 1000
        self.k_1 = 1
        self.k_2 = 2
        self.m = 3

    def test_models(self):
        """tests for model constructions"""

        mdl.co_oxidation(self.order, self.k_ad_co, cyclic=True)
        mdl.fermi_pasta_ulam(self.order, self.number_of_snapshots)
        mdl.fpu_coefficients(self.order)
        mdl.kuramoto(self.theta_init, self.frequencies, self.time, self.number_of_snapshots)
        mdl.kuramoto_coefficients(self.order, self.frequencies)
        mdl.signaling_cascade(self.order)
        mdl.two_step_destruction(self.k_1, self.k_2, self.m)
