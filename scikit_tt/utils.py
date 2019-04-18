# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import time


def header(title=None, subtitle=None):
    """Print scikit_tt header

    Parameters
    ----------
    title: string
        title or name of the procedure
    subtitle: string
        subtitle of the procedure
    """

    print('                                               ')
    print('.  __    __               ___    ___ ___       ')
    print('. /__`  /  `  |  |__/  |   |      |   |        ')
    print('| .__/  \__,  |  |  \  |   |      |   |        ')
    print('o ─────────── o ────── o ─ o ──── o ─ o ── ─  ─')
    if title is not None:
        print('|')
        print('o ─ ' + title)
    if subtitle is not None:
        print('    ' + subtitle)
    print(' ')
    print(' ')


def progress(str_text, percent, cpu_time=0, show=True, width=47):
    """Show progress in percent

    Print strings of the form, e.g., 'Running ... 10%' etc., without line breaks.

    Parameters
    ----------
    str_text: string
        string to print
    percent: float
        current progress; if percent=0, the current time is returned
    cpu_time: float
        current CPU time
    show: bool, optional
        whether to print the progress, default is True
    width: int
        width of the progress bar, default is 47
    """

    if show:
        os.system('setterm -cursor off')
        len_text = len(str_text)
        space_text = ' ' * (width - len_text)
        number_of_boxes = width - 6
        if percent == 100:
            str_percent = '100%'
        else:
            str_percent = str("%.1f" % percent) + '%'
        len_percent = len(str_percent)
        space_percent = ' ' * (6 - len_percent)
        str_cpu = 'CPU time: ' + str("%.1f" % cpu_time) + 's'

        color_done = '\33[42m'
        color_remain = '\33[100m'
        underline = '\033[4m'
        end = '\33[0m'
        up_two = '\u001b[2A\r'

        done = int(number_of_boxes * (np.floor(percent) / 100))
        str_done = ' ' * done
        str_remain = ' ' * (number_of_boxes - done)

        sys.stdout.write(underline + str_text + space_text + end + '\n')
        sys.stdout.write(color_done + underline + str_done + end)
        sys.stdout.write(color_remain + underline + str_remain + end)
        sys.stdout.write(underline + space_percent + str_percent + end + '\n')
        sys.stdout.write(str_cpu + up_two)
        sys.stdout.flush()

        if percent == 100:
            sys.stdout.write(4 * '\n')
            os.system('setterm -cursor on')

    if percent == 0:
        return time.time()


class timer(object):
    """Measure CPU time

    Can be executed using the 'with' statement in order to measure the CPU time needed for calculations.
    """

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self.start_time
