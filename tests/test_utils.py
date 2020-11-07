# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lin
import scikit_tt.utils as utl
from unittest import TestCase
import sys
import os
import time


class TestUtils(TestCase):

    def setUp(self):
        """construct random matrix for test of trncated_svd"""

        self.m = np.random.rand(50,100)
        self.threshold = 1e-1
        self.max_rank = 15
        self.tol = 1e-12

    def test_header(self):
        """test header"""

        # suppress print output
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        utl.header(title='test', subtitle='test')
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def test_progress(self):
        """test progress"""

        # suppress print output
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        utl.progress('test', 0)
        utl.progress('test', 50)
        utl.progress('test', 100)
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def test_timer(self):
        """test timer"""

        # wait for 1 second
        with utl.timer() as timer:
            time.sleep(1)
        self.assertLess(np.abs(timer.elapsed - 1), 0.1)

    def test_truncated_svd(self):
        """test truncated_svd"""

        # decompose m using different parameters
        u_1, s_1, v_1 = lin.svd(self.m, full_matrices=False)
        u_2, s_2, v_2 = utl.truncated_svd(self.m)
        u_3, s_3, v_3 = utl.truncated_svd(self.m, threshold=self.threshold)
        u_4, s_4, v_4 = utl.truncated_svd(self.m, max_rank=self.max_rank)

        # no truncation
        self.assertLess(np.linalg.norm(u_1-u_2), self.tol)
        self.assertLess(np.linalg.norm(s_1-s_2), self.tol)
        self.assertLess(np.linalg.norm(v_1-v_2), self.tol)

        # relative truncation
        self.assertLess(np.linalg.norm(u_1[:,:len(s_3)]-u_3), self.tol)
        self.assertLess(np.linalg.norm(s_1[:len(s_3)]-s_3), self.tol)
        self.assertLess(np.linalg.norm(v_1[:len(s_3),:]-v_3), self.tol)

        self.assertGreater(s_3[-1]/s_1[0], self.threshold)
        self.assertLess(s_1[len(s_3)]/s_1[0], self.threshold)

        # absolute truncation
        self.assertLess(np.linalg.norm(u_1[:,:self.max_rank]-u_4), self.tol)
        self.assertLess(np.linalg.norm(s_1[:self.max_rank]-s_4), self.tol)
        self.assertLess(np.linalg.norm(v_1[:self.max_rank,:]-v_4), self.tol)
