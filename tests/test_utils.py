# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.utils as utl
from unittest import TestCase
import sys
import os
import time


class TestUtils(TestCase):

    def test_header(self):
        """test header"""

        # suppress print output
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        utl.header(title='test', subtitle='test')
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def test_plot_parameters(self):
        """test plot_parameters"""

        utl.plot_parameters()

    def test_progress(self):
        """test progress"""

        # suppress print output
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        utl.progress('test', 0)
        utl.progress('test', 100)
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def test_timer(self):
        """test timer"""

        with utl.timer() as timer:
            time.sleep(1)

        self.assertLess(np.abs(timer.elapsed - 1), 0.1)
