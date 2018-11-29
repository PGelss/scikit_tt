#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tt as tt


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
            tensor_mean = tt.TT(cores)

            # define entry of mean
            mean[i, j] = (series[i].transpose() @ tensor_mean).matricize()[0, 0]

    return mean
