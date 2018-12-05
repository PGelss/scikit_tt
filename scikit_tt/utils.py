#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time as _time
import sys
from scikit_tt.tensor_train import TT
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



class Timer(object):
    """Measure CPU time

    Can be executed using the 'with' statement in order to measure the CPU time needed for calculations.
    """

    def __enter__(self):
        self.start_time = _time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = _time.time() - self.start_time


def progress(text, percent, dots=3):
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
    """

    sys.stdout.write('\r' + text + ' ' + dots * '.' + ' ' + str("%.1f" % percent) + '%')

    if percent == 100:
        sys.stdout.write('\n')

def plot(axes, axes_title = None, axes_projection = None, axes_aspect = None, grid = [], figure_title=' ', figure_title_size = 18, figure_title_location = 0.98, figure_aspect = 1):
    """Customized plot routine

    Parameters
    ----------
    axes: 
    grid:
    axes_title:
    axes_projection:
    axes_aspect:
    figure_title: string, optional
        figure title, default is ' '
    figure_title_size: int, optional
        font size of the figure title, default is 18
    figure_title_location: float, optional
        y location of the figure title, default is 0.98
    figure_aspect: float, optional
        figure aspect, default is 1
    """

    # set text font and size
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 12})

    # autolayout
    plt.rcParams.update({'figure.autolayout': True})

    # create figure
    f = plt.figure(figsize=plt.figaspect(figure_aspect))

    # figure title
    plt.suptitle(figure_title, fontsize=figure_title_size, y=figure_title_location)

    # number of rows and columns of the grid
    if grid==[]:
        grid_rows = 1
        grid_cols = len(axes)
    else:
        grid_rows = grid[0]
        grid_cols = grid[1]

    # set axes titles
    if axes_title == None:
        axes_title = [' ']*len(axes)

    # set axes projections
    if axes_projection == None:
        axes_projection = [None]*len(axes)

    # set axes aspects
    if axes_aspect == None:
        axes_aspect = [1]*len(axes)

    for i in range(len(axes)):
        ax = f.add_subplot(grid_rows, grid_cols, i + 1, projection=axes_projection[i], aspect=axes_aspect[i])
        ax.set_title(axes_title[i])


    # for i in range(number_ev):
    #     indices = np.where(abs(eigentensors_tt[i]/np.amax(abs(eigentensors_tt[i]))) > 0.001)

    #     ax = f.add_subplot(1, 3, i + 1, projection='3d', aspect=1)
    #     ax.set_title(r'$\lambda$=' + str("%.4f" % np.abs(eigenvalues_exact[i])))


    #     im = ax.scatter(indices[0], indices[1], indices[2], c=eigentensors_tt[i][indices], cmap='jet',
    #                     s=abs(eigentensors_tt[i])[indices]/np.amax(abs(eigentensors_tt[i])) * 100, vmin=np.amin(eigentensors_tt[i]),
    #                     vmax=np.amax(eigentensors_tt[i]))
    #     f.colorbar(im, shrink=0.4, aspect=10)
        
    #     ax.xaxis.set_ticklabels([])
    #     ax.xaxis.set_ticks([])
    #     ax.yaxis.set_ticklabels([])
    #     ax.yaxis.set_ticks([])
    #     ax.zaxis.set_ticklabels([])
    #     ax.zaxis.set_ticks([])

    # display figure
    plt.show()


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
