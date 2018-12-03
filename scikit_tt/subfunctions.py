#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time as _time
import sys
from scikit_tt.tensor_train import TT


class Timer(object):
    """Measure CPU time"""

    def __enter__(self):
        self.start_time = _time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = _time.time() - self.start_time


def progress(text, percent, dots=3):
    """Show progress in percent"""

    sys.stdout.write('\r' + text + ' ' + dots * '.' + ' ' + str("%.1f" % percent) + '%')

    if percent == 100:
        sys.stdout.write('\n')


def unit_vector(dimension, index):
    """Canonical unit vector

    Return specific canonical unit vector in a given given dimension.

    Parameters
    ----------
    dimension: int
        dimension of the unit vector
    index: int
        position of the 1

    Returns
    -------
    v: ndarray
        unit vector
    """
    v = np.zeros(dimension)
    v[index] = 1

    return v


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
