# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.utils as utl
from unittest import TestCase
from contextlib import redirect_stdout
import io
import time


class TestUtils(TestCase):

    def test_header(self):
        """test header"""

        # suppress print output
        text_trap = io.StringIO()
        with redirect_stdout(text_trap):
            utl.header(title='test', subtitle='test')

    def test_plot_parameters(self):
        """test plot_parameters"""

        utl.plot_parameters()

    def test_progress(self):
        """test progress"""

        # suppress print output
        text_trap = io.StringIO()
        with redirect_stdout(text_trap):
            utl.progress('test', 0)
            utl.progress('test', 100)

    def test_timer(self):
        """test timer"""

        with utl.timer() as timer:
            time.sleep(1)

        self.assertLess(np.abs(timer.elapsed - 1), 0.1)
