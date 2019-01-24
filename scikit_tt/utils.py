#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time as _time
import sys
import matplotlib.pyplot as plt


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


def plot_parameters(font_size=14):
    """Customized plot parameters

    Parameters
    ----------
    font_size: int
        default font size for title, labels, etc.
    """

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'axes.grid': True})


def progress(text, percent, dots=3, show=True):
    """Show progress in percent

    Print strings of the form, e.g., 'Running ... 10%' etc., without line breaks.

    Parameters
    ----------
    text: string
        string to print
    percent: float
        current progress
    dots: int, optional
        number of dots to print, default is 3
    show: bool, optional
        whether to print the progress, default is True
    """

    if show:
        sys.stdout.write('\r' + text + ' ' + dots * '.' + ' ' + str("%.1f" % percent) + '%')

        if percent == 100:
            sys.stdout.write('\n')


class timer(object):
    """Measure CPU time

    Can be executed using the 'with' statement in order to measure the CPU time needed for calculations.
    """

    def __enter__(self):
        self.start_time = _time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = _time.time() - self.start_time
