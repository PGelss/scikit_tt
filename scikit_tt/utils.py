#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt
import numpy as np
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


def mean_concentrations(series):
    """Mean concentrations of TT series

    Compute mean concentrations of a given time series in TT format representing probability distributions of, e.g., a
    chemical reaction network..

    Parameters
    ----------
    series: list of instances of TT class

    Returns
    -------
    mean: ndarray(#time_steps,#species)
        mean concentrations of the species over time
    """

    # define array
    mean = np.zeros([len(series), series[0].order])

    # loop over time steps
    for i in range(len(series)):

        # loop over species
        for j in range(series[0].order):
            # define tensor train to compute mean concentration of jth species
            cores = [np.ones([1, series[0].row_dims[k], 1, 1]) for k in range(series[0].order)]
            cores[j] = np.zeros([1, series[0].row_dims[j], 1, 1])
            cores[j][0, :, 0, 0] = np.arange(series[0].row_dims[j])
            tensor_mean = TT(cores)

            # define entry of mean
            mean[i, j] = (series[i].transpose() @ tensor_mean).element([0] * 2 * series[0].order)

    return mean


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


def two_cell_tof(t, reactant_states, reaction_rate):
    """Turn-over frequency of a reaction in a cyclic homogeneous nearest-neighbor interaction system

    Parameters
    ----------
    t: instance of TT class
        tensor train representing a probability distribution
    reactant_states: list of ints
        reactant states of the given reaction in the form of [reactant_state_1, reactant_state_2] where the list entries
        represent the reactant states on two neighboring cell
    reaction_rate: float
        reaction rate constant of the given reaction

    Returns
    -------
    tof: float
        turn-over frequency of the given reaction
    """

    tt_left = [None] * t.order
    tt_right = [None] * t.order
    for i in range(t.order):
        tt_left[i] = tt.ones([1] * t.order, t.row_dims)
        tt_right[i] = tt.ones([1] * t.order, t.row_dims)
        tt_left[i].cores[i] = np.zeros([1, 1, t.row_dims[i], 1])
        tt_left[i].cores[i][0, 0, reactant_states[0], 0] = 1
        tt_right[i].cores[i] = np.zeros([1, 1, t.row_dims[i], 1])
        tt_right[i].cores[i][0, 0, reactant_states[0], 0] = 1
        if i > 0:
            tt_left[i].cores[i - 1] = np.zeros([1, 1, t.row_dims[i - 1], 1])
            tt_left[i].cores[i - 1][0, 0, reactant_states[1], 0] = 1
        else:
            tt_left[i].cores[-1] = np.zeros([1, 1, t.row_dims[-1], 1])
            tt_left[i].cores[-1][0, 0, reactant_states[1], 0] = 1
        if i < t.order - 1:
            tt_right[i].cores[i + 1] = np.zeros([1, 1, t.row_dims[i + 1], 1])
            tt_right[i].cores[i + 1][0, 0, reactant_states[1], 0] = 1
        else:
            tt_right[i].cores[0] = np.zeros([1, 1, t.row_dims[0], 1])
            tt_right[i].cores[0][0, 0, reactant_states[1], 0] = 1
    tof = 0
    for i in range(t.order):
        tof = tof + (reaction_rate / t.order) * (tt_left[i] @ t).element([0] * t.order * 2) + \
              (reaction_rate / t.order) * (tt_right[i] @ t).element([0] * t.order * 2)
    return tof
