# -*- coding: utf-8 -*-

import time as _time
import sys


class Timer(object):

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start_time = _time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = _time.time() - self.start_time
        if self.name:
            print('%s: ' % self.name, end='')

def progress(text, percent, dots=3):
		sys.stdout.write('\r' + text + ' ' + dots*'.' + ' ' + str("%.1f" % percent) + '%')
		if percent == 100:
			sys.stdout.write('\n')
