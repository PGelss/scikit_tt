# -*- coding: utf-8 -*-

from scikit_tt.tensor_train import TT
import numpy as np


def mandy_cm(x, y, psi, threshold=0):
    """Multidimensional Approximation of Nonlinear Dynamics (MANDy)

    Coordinate-major approach for construction of the tensor train xi. See [1]_ for details.

    Parameters
    ----------
    x: ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    y: ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    psi: list of lambda functions
        list of basis functions
    threshold: float, optional
        threshold for SVDs, default is 0

    Returns
    -------
    xi: instance of TT class
        tensor train of coefficients for chosen basis functions

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # parameters
    d = x.shape[0]
    m = x.shape[1]
    p = len(psi)

    # define cores as empty arrays
    cores = [np.zeros([1, p, 1, m])] + [np.zeros([m, p, 1, m]) for _ in range(1, d)]

    # insert elements of first core
    for j in range(m):
        cores[0][0, :, 0, j] = np.array([psi[k](x[0, j]) for k in range(p)])

    # insert elements of subsequent cores
    for i in range(1, d):
        for j in range(m):
            cores[i][j, :, 0, j] = np.array([psi[k](x[i, j]) for k in range(p)])

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    # construct tensor train
    xi = TT(cores)

    # compute pseudoinverse of xi
    xi = xi.pinv(d, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[d] = (xi.cores[d].reshape([xi.ranks[d], m]) @ y.transpose()).reshape(xi.ranks[d], d, 1, 1)

    # set new row dimension
    xi.row_dims[d] = d

    return xi


def mandy_fm(x, y, psi, threshold=0, add_one=True):
    """Multidimensional Approximation of Nonlinear Dynamics (MANDy)

    Function-major approach for construction of the tensor train xi. See [1]_ for details.

    Parameters
    ----------
    x: ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    y: ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    psi: list of lambda functions
        list of basis functions
    threshold: float, optional
        threshold for SVDs, default is 0
    add_one: bool, optional
        whether to add the basis function 1 to the cores or not, default is True

    Returns
    -------
    xi: instance of TT class
        tensor train of coefficients for chosen basis functions

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # parameters
    d = x.shape[0]
    m = x.shape[1]
    p = len(psi)

    # define cores as empty arrays
    cores = [np.zeros([1, d + add_one, 1, m])] + [np.zeros([m, d + add_one, 1, m]) for _ in range(1, p)]

    # insert elements of first core
    if add_one is True:
        for j in range(m):
            cores[0][0, 0, 0, j] = 1
            cores[0][0, 1:, 0, j] = np.array([psi[0](x[k, j]) for k in range(d)])
    else:
        for j in range(m):
            cores[0][0, :, 0, j] = np.array([psi[0](x[k, j]) for k in range(d)])

    # insert elements of subsequent cores
    for i in range(1, p):
        if add_one is True:
            for j in range(m):
                cores[i][j, 0, 0, j] = 1
                cores[i][j, 1:, 0, j] = np.array([psi[i](x[k, j]) for k in range(d)])
        else:
            for j in range(m):
                cores[0][0, :, 0, j] = np.array([psi[0](x[k, j]) for k in range(d)])

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    # construct tensor train
    xi = TT(cores)

    # compute pseudoinverse of xi
    xi = xi.pinv(p, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[p] = (xi.cores[p].reshape([xi.ranks[p], m]) @ y.transpose()).reshape(xi.ranks[p], d, 1, 1)

    # set new row dimension
    xi.row_dims[p] = d

    return xi
