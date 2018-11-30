# -*- coding: utf-8 -*-

import time as _time
import sys


class Timer(object):
    """Measure CPU time"""

    def __enter__(self):
        self.start_time = _time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = _time.time() - self.start_time


def progress(text, percent, dots=3):
    """Show progress in percent"""

    sys.stdout.write('\r' + text + ' ' + dots * '.' + ' ' + str("%.1f" % percent) + '%')

    if percent == 100:
        sys.stdout.write('\n')
