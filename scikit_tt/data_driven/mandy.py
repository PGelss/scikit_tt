# -*- coding: utf-8 -*-

import scikit_tt.data_driven.transform as tdt


def mandy_cm(x, y, phi, threshold=0):
    """Multidimensional Approximation of Nonlinear Dynamics (MANDy)

    Coordinate-major approach for construction of the tensor train xi. See [1]_ for details.

    Parameters
    ----------
    x: ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    y: ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    phi: list of lambda functions
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

    # construct transformed data tensor
    psi = tdt.coordinate_major(x, phi)

    # define xi as pseudoinverse of psi
    xi = psi.pinv(d, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[d] = (xi.cores[d].reshape([xi.ranks[d], m]).dot(y.transpose())).reshape(xi.ranks[d], d, 1, 1)

    # set new row dimension
    xi.row_dims[d] = d

    return xi


def mandy_fm(x, y, phi, threshold=0, add_one=True):
    """Multidimensional Approximation of Nonlinear Dynamics (MANDy)

    Function-major approach for construction of the tensor train xi. See [1]_ for details.

    Parameters
    ----------
    x: ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    y: ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    phi: list of lambda functions
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
    p = len(phi)

    # construct transformed data tensor
    psi = tdt.function_major(x, phi, add_one=add_one)

    # define xi as pseudoinverse of psi
    xi = psi.pinv(p, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[p] = (xi.cores[p].reshape([xi.ranks[p], m]).dot(y.transpose())).reshape(xi.ranks[p], d, 1, 1)

    # set new row dimension
    xi.row_dims[p] = d

    return xi
