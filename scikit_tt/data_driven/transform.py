# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.utils as utl
import scipy.linalg as splin
import time as _time
from scikit_tt.tensor_train import TT


def constant_function():
    """Constant function.

    Returns
    -------
    f: function
        constant function
    """

    f = lambda t: 1 + 0 * t[0]

    return f


def indicator_function(index, a, b):
    """Indicator function.

    Parameters
    ----------
    index: int
        define which entry of a snapshot is passed to the indicator function
    a: float
        lower bound of the interval
    b: float
        upper bound of the interval

    Returns
    -------
    f: function
        indicator function
    """

    #f = lambda t: __indicator_function_callback(t[index], a, b)
    f = lambda t: 1 * ((a <= t[index]) & (t[index] < b))

    return f

def identity(index):
    """Identiy function.

    Parameters
    ----------
    index: int
        define which entry of a snapshot is passed to the identity function

    Returns
    -------
    f: function
        identity function at given index
    """

    f = lambda t: t[index]

    return f

def monomial(index, exponent):
    """Monomial function.

    Parameters
    ----------
    index: int
        define which entry of a snapshot is passed to the identity function
    exponent: int
        degree of the monomial

    Returns
    -------
    f: function
        monomial function at given index
    """

    f = lambda t: (t[index])**exponent

    return f


def sin(index, alpha):
    """Sine function.

    Parameters
    ----------
    index: int
        define which entry of a snapshot is passed to the sine function
    alpha: float
        prefactor

    Returns
    -------
    f: function
        sine function at given index
    """

    f = lambda t: np.sin(alpha * t[index])

    return f


def cos(index, alpha):
    """Cosine function.

    Parameters
    ----------
    index: int
        define which entry of a snapshot is passed to the cosine function
    alpha: float
        prefactor

    Returns
    -------
    f: function
        cosine function at given index
    """

    f = lambda t: np.cos(alpha * t[index])

    return f


def gauss_function(index, mean, variance):
    """Gauss function.

    Parameters
    ----------
    index: int
        define which entry of a snapshot is passed to the Gauss function
    mean: float
        mean of the distribution
    variance: float (>0)
        variance

    Returns
    -------
    f: function
        Gauss function
    """

    f = lambda t: np.exp(-0.5 * (t[index] - mean) ** 2 / variance)

    return f


def periodic_gauss_function(index, mean, variance):
    """Periodic Gauss function.

        Parameters
        ----------
        index: int
            define which entry of a snapshot is passed to the periodic Gauss function
        mean: float
            mean of the distribution
        variance: float (>0)
            variance

        Returns
        -------
        f: function
            Gauss function
        """

    f = lambda t: np.exp(-0.5 * np.sin(0.5 * (t[index] - mean)) ** 2 / variance)

    return f


def basis_decomposition(x, phi, single_core=None):
    """Construct a transformed data tensor in TT format.

    Given a set of basis functions phi, construct a TT decomposition psi of the tensor

              -                    -         -                      -
              | phi[0][0](x[:,0])  |         | phi[p-1][0](x[:,0])  |
        psi = |      ...           | x ... x |      ...             | x e_{1}
              | phi[0][-1](x[:,0]) |         | phi[p-1][-1](x[:,0]) |
              -                    -         -                      -

                      -               -         -                 -
                      | phi[0][0](x[:,m-1])  |         | phi[p-1][0](x[:,m-1])  |
              + ... + |      ...             | x ... x |      ...               | x e_{m}
                      | phi[0][-1](x[:,m-1]) |         | phi[p-1][-1](x[:,m-1]) |
                      -                      -         -                        -

    where e_{1}, ... ,e_{m} are the m-dimensional canonical unit vectors. See [1]_ for details.

    Parameters
    ----------
    x: np.ndarray
        snapshot matrix of size d x m
    phi: list[list[function]]
        list of basis functions in every mode
    single_core: None or int, optional
        return only the ith core of psi if single_core=i (<p), default is None

    Returns
    -------
    psi: instance of TT class or np.ndarray
        tensor train of basis function evaluations if single_core=None, 4-dimensional array if single core
        is an integer

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           Journal of Computational and Nonlinear Dynamics 14, 2019
    """

    # number of snapshots
    m = x.shape[1]

    # number of modes
    p = len(phi)

    # mode dimensions
    n = [len(phi[i]) for i in range(p)]

    if single_core is None:

        # define cores as a list of empty arrays
        cores = [np.zeros([1, n[0], 1, m])] + [np.zeros([m, n[i], 1, m]) for i in range(1, p)]

        # insert elements of first core
        for j in range(m):
            # apply first list of basis functions to all snapshots
            cores[0][0, :, 0, j] = np.array([phi[0][k](x[:, j]) for k in range(n[0])])

        # insert elements of subsequent cores
        for i in range(1, p):
            for j in range(m):
                # apply ith list of basis functions to all snapshots
                cores[i][j, :, 0, j] = np.array([phi[i][k](x[:, j]) for k in range(n[i])])

        # append core containing unit vectors
        cores.append(np.eye(m).reshape(m, m, 1, 1))

        # construct tensor train
        psi = TT(cores)

    elif single_core == 0:

        # define core
        psi = np.zeros([1, n[0], 1, m])

        # insert elements
        for j in range(m):
            # apply basis functions
            psi[0, :, 0, j] = np.array([phi[0][k](x[:, j]) for k in range(n[0])])

    else:

        # define core
        psi = np.zeros([m, n[single_core], 1, m])

        # insert elements
        for j in range(m):
            # apply basis functions
            psi[j, :, 0, j] = np.array([phi[single_core][k](x[:, j]) for k in range(n[single_core])])

    return psi


def coordinate_major(x, phi, single_core=None):
    """Construct a transformed data tensor in TT format using the coordinate-major approach.

    Given a set of basis functions phi, construct a TT decomposition psi of the form::

              -                 -         -                  -
              | phi[0](x[0,0])  |         | phi[0](x[-1,0])  |
        psi = |      ...        | x ... x |      ...         | x e_{1}
              | phi[-1](x[0,0]) |         | phi[-1](x[-1,0]) |
              -                 -         -                  -

                      -                   -         -                    -
                      | phi[0](x[0,m-1])  |         | phi[0](x[-1,m-1])  |
              + ... + |      ...          | x ... x |      ...           | x e_{m}
                      | phi[-1](x[0,m-1]) |         | phi[-1](x[-1,m-1]) |
                      -                   -         -                    -

    where e_{1}, ... ,e_{m} are the m-dimensional canonical unit vectors. See [1]_ for details.

    Parameters
    ----------
    x: np.ndarray
        snapshot matrix of size d x m
    phi: list[function]
        list of basis functions
    single_core: None or int, optional
        return only the ith core of psi if single_core=i (<p), default is None

    Returns
    -------
    psi: instance of TT class
        tensor train of basis function evaluations

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # number of snapshots
    m = x.shape[1]

    # number of modes
    p = len(phi)

    # number of dimensions
    d = x.shape[0]

    if single_core is None:

        # define cores as list of empty arrays
        cores = [np.zeros([1, p, 1, m])] + [np.zeros([m, p, 1, m]) for _ in range(1, d)]

        # insert elements of first core
        for j in range(m):
            cores[0][0, :, 0, j] = np.array([phi[k](x[0, j]) for k in range(p)])

        # insert elements of subsequent cores
        for i in range(1, d):
            for j in range(m):
                cores[i][j, :, 0, j] = np.array([phi[k](x[i, j]) for k in range(p)])

        # append core containing unit vectors
        cores.append(np.eye(m).reshape(m, m, 1, 1))

        # construct tensor train
        psi = TT(cores)

    elif single_core == 0:

        # define core
        psi = np.zeros([1, p, 1, m])

        # insert elements
        for j in range(m):
            # apply basis functions
            psi[0, :, 0, j] = np.array([phi[k](x[0, j]) for k in range(p)])

    else:

        # define core
        psi = np.zeros([m, p, 1, m])

        # insert elements
        for j in range(m):
            # apply basis functions
            psi[j, :, 0, j] = np.array([phi[k](x[single_core, j]) for k in range(p)])

    return psi


def function_major(x, phi, add_one=True, single_core=None):
    """Construct a transformed data tensor in TT format using the function-major approach.

    Given a set of basis functions phi, construct a TT decomposition psi of the form::

              -                 -         -                  -
              | phi[0](x[0,0])  |         | phi[-1](x[0,0])  |
        psi = |      ...        | x ... x |      ...         | x e_{1}
              | phi[0](x[-1,0]) |         | phi[-1](x[-1,0]) |
              -                 -         -                  -

                      -                   -         -                    -
                      | phi[0](x[0,m-1])  |         | phi[-1](x[0,m-1])  |
              + ... + |      ...          | x ... x |      ...           | x e_{m}
                      | phi[0](x[-1,m-1]) |         | phi[-1](x[-1,m-1]) |
                      -                   -         -                    -

    where e_{1}, ... ,e_{m} are the m-dimensional canonical unit vectors. See [1]_ for details.

    Parameters
    ----------
    x : np.np.ndarray
        snapshot matrix of size d x m
    phi : list[function]
        list of basis functions
    add_one: bool, optional
        whether to add the basis function 1 to the cores or not, default is True

    Returns
    -------
    psi: instance of TT class
        tensor train of basis function evaluations

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # number of snapeshots
    m = x.shape[1]

    # number of modes
    p = len(phi)

    # number of dimensions
    d = x.shape[0]

    if single_core is None:

        # define cores as list of empty arrays
        cores = [np.zeros([1, d + add_one, 1, m])] + [np.zeros([m, d + add_one, 1, m]) for _ in range(1, p)]

        # insert elements of first core
        if add_one is True:
            for j in range(m):
                cores[0][0, 0, 0, j] = 1
                cores[0][0, 1:, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])
        else:
            for j in range(m):
                cores[0][0, :, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])

        # insert elements of subsequent cores
        for i in range(1, p):
            if add_one is True:
                for j in range(m):
                    cores[i][j, 0, 0, j] = 1
                    cores[i][j, 1:, 0, j] = np.array([phi[i](x[k, j]) for k in range(d)])
            else:
                for j in range(m):
                    cores[i][j, :, 0, j] = np.array([phi[i](x[k, j]) for k in range(d)])

        # append core containing unit vectors
        cores.append(np.eye(m).reshape(m, m, 1, 1))

        # construct tensor train
        psi = TT(cores)

    elif single_core == 0:

        # define core
        psi = np.zeros([1, d + add_one, 1, m])

        # insert elements
        if add_one is True:
            for j in range(m):
                psi[0, 0, 0, j] = 1
                psi[0, 1:, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])
        else:
            for j in range(m):
                psi[0, :, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])

    else:

        # define core
        psi = np.zeros([m, d + add_one, 1, m])

        # insert elements
        if add_one is True:
            for j in range(m):
                psi[j, 0, 0, j] = 1
                psi[j, 1:, 0, j] = np.array([phi[single_core](x[k, j]) for k in range(d)])
        else:
            for j in range(m):
                psi[j, :, 0, j] = np.array([phi[single_core](x[k, j]) for k in range(d)])

    return psi

def gram(x_1, x_2, basis_list):
    """Gram matrix.

    Compute the Gram matrix of two transformed data tensors psi_1=psi(x_1) and psi_2=psi(x_2), i.e., psi_1^T@psi_2. See
    _[1] for details.

    Parameters
    ----------
    x_1: np.ndarray
        data matrix for psi_1
    x_2: np.ndarray
        data matrix for psi_2
    basis_list: list[list[function]]
        list of basis functions in every mode

    Returns
    -------
    gram: np.ndarray
        Gram matrix

    References
    ----------
    .. [1] S. Klus, P. Gelß, "Tensor-Based Algorithms for Image Classification", Algorithms, 2019
    """

    # compute gram by iteratively applying the Hadarmard product
    gram = np.ones([x_1.shape[1], x_2.shape[1]])
    for i in range(len(basis_list)):
        theta_1 = np.array([basis_list[i][k](x_1) for k in range(len(basis_list[i]))])
        theta_2 = np.array([basis_list[i][k](x_2) for k in range(len(basis_list[i]))])
        gram *= (theta_1.T.dot(theta_2))

    return gram


def hocur(x, basis_list, ranks, repeats=1, multiplier=10, progress=True, string=None):
    """Higher-order CUR decomposition of transformed data tensors.

    Given a snapshot matrix x and a list of basis functions in each mode, construct a TT decomposition of the
    transformed data tensor Psi(x) using a higher-order CUR decomposition and maximum-volume subtensors. See [1]_, [2]_
    and [3]_ for details.

    Parameters
    ----------
    x: np.ndarray
        data matrix
    basis_list: list[list[function]]
        list of basis functions in every mode
    ranks: list[int] or int
        maximum TT ranks of the resulting TT representation; if type is int, then the ranks are set to
        [1, ranks, ..., ranks, 1]; note that - depending on the number of linearly independent rows/columns that have
        been found - the TT ranks may be reduced during the decomposition
    repeats: int, optional
        number of repeats, default is 1
    multiplier: int, optional
        multiply the number of initially chosen column indices (given by ranks) in order to increase the probability of
        finding a 'full' set of linearly independent columns; default is 10
    progress: bool, optional
        whether to show the progress of the algorithm or not, default is True
    string: string
        string to print; if None (default), then print 'HOCUR (repeats: <repeats>)'

    Returns
    -------
    psi: instance of TT class
        TT representation of the transformed data tensor

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           Journal of Computational and Nonlinear Dynamics 14, 2019
    .. [2] I. Oseledets, E. Tyrtyshnikov, "TT-cross approximation for multidimensional arrays", Linear Algebra and its
           Applications 432, 2010
    .. [3] S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov, N. L. Zamarashkin, "How to find a
           good submatrix", Matrix Methods: Theory, Algorithms, Applications, 2010
    """

    # parameters
    # ----------

    # number of snapshots
    m = x.shape[1]

    # number of modes
    p = len(basis_list)

    # mode dimensions
    n = [len(basis_list[k]) for k in range(p)] + [m]

    # ranks
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(n) - 1)] + [1]

    # initial definitions
    # -------------------

    # define initial lists of column indices
    col_inds = __hocur_first_col_inds(n, ranks, multiplier)

    # define list of cores
    cores = [None] * (p + 1)

    # show progress
    # -------------

    if string is None:
        string = 'HOCUR'
    start_time = utl.progress(string, 0, show=progress)

    # start decomposition
    # -------------------

    for k in range(repeats):

        row_inds = [None]

        # first half sweep
        for i in range(p):

            # extract submatrix
            y = __hocur_extract_matrix(x, basis_list, row_inds[i], col_inds[i])

            if k == 0:
                # find linearly independent columns
                cols = __hocur_find_li_cols(y)
                cols = cols[:ranks[i + 1]]
                y = y[:, cols]

            # find optimal rows
            rows = __hocur_maxvolume(y)  # type: list

            # adapt ranks if necessary
            ranks[i + 1] = len(rows)

            if i == 0:

                # store row indices for first dimensions
                row_inds.append([[rows[j]] for j in range(ranks[i + 1])])

            else:

                # convert rows to multi indices
                multi_indices = np.array(np.unravel_index(rows, (ranks[i], n[i])))

                # store row indices for dimensions m_1, n_1, ..., m_i, n_i
                row_inds.append([row_inds[i][multi_indices[0, j]] + [multi_indices[1, j]] for j in
                                 range(ranks[i + 1])])

            # define core
            if len(rows) < y.shape[1]:
                y = y[:, :len(rows)]
            u_inv = np.linalg.inv(y[rows, :].copy())
            cores[i] = y.dot(u_inv).reshape([ranks[i], n[i], 1, ranks[i + 1]])

            # show progress
            utl.progress(string + ' ... r=' + str(ranks[i + 1]), 100 * (k * 2 * p + i + 1) / (repeats * 2 * p),
                         cpu_time=_time.time() - start_time,
                         show=progress)

        # second half sweep
        for i in range(p, 0, -1):

            # extract submatrix
            y = __hocur_extract_matrix(x, basis_list, row_inds[i], col_inds[i]).reshape([ranks[i], n[i] * ranks[i + 1]])

            # find optimal rows
            cols = __hocur_maxvolume(y.T)  # type: list

            # adapt ranks if necessary
            ranks[i] = len(cols)

            if i == p:

                # store row indices for first dimensions
                col_inds[p - 1] = [[cols[j]] for j in range(ranks[i])]

            else:

                # convert cols to multi indices
                multi_indices = np.array(np.unravel_index(cols, (n[i], ranks[i + 1])))

                # store col indices for dimensions m_i, n_i, ... , m_d, n_d
                col_inds[i - 1] = [[multi_indices[0, j]] + col_inds[i][multi_indices[1, j]] for j in range(ranks[i])]

            # define TT core
            if len(cols) < y.shape[0]:
                y = y[:len(cols), :]
            u_inv = np.linalg.inv(y[:, cols].copy())
            cores[i] = u_inv.dot(y).reshape([ranks[i], n[i], 1, ranks[i + 1]])

            # show progress
            utl.progress(string, 100 * ((k + 1) * 2 * p - i + 1) / (repeats * 2 * p),
                         cpu_time=_time.time() - start_time, show=progress)

        # define first core
        y = __hocur_extract_matrix(x, basis_list, None, col_inds[0])
        cores[0] = y.reshape([1, n[0], 1, ranks[1]])

    # construct tensor train
    # ----------------------

    psi = TT(cores)

    return psi

# private functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def __hocur_first_col_inds(dimensions, ranks, multiplier):
    """Create random column indices

    Parameters
    ----------
    dimensions: list[int]
        dimensions of a given tensor
    ranks: list[int]
        ranks for decomposition
    multiplier: int
        multiply the number of initially chosen column indices (given by ranks) in order to increase the probability of
        finding a 'full' set of linearly independent columns

    Returns
    -------
    col_inds: list[list[int]]
        array containing single indices
    """

    # define list of column indices
    col_inds = [None]

    # insert column indices for last dimension
    col_inds.insert(0, [[j] for j in range(np.minimum(multiplier * ranks[-2], dimensions[-1]))])

    for i in range(len(dimensions) - 3, -1, -1):
        # define array of flat indices
        flat_inds = np.arange(np.minimum(multiplier * ranks[i + 1], dimensions[i + 1] * ranks[i + 2]))

        # convert flat indices to tuples
        multi_inds = np.array(np.unravel_index(flat_inds, (dimensions[i + 1], ranks[i + 2])))

        # insert column indices
        col_inds.insert(0, [[multi_inds[0, j]] + col_inds[0][multi_inds[1, j]] for j in range(multi_inds.shape[1])])

    return col_inds


def __hocur_extract_matrix(data, basis_list, row_coordinates_list, col_coordinates_list):
    """Extraction of a submatrix of a transformed data tensor.

    Given a set of row and column coordinates, extracts a submatrix from the transformed data tensor corresponding to
    the data matrix x and the set of basis functions stored in basis_list.

    Parameters
    ----------
    data: np.ndarray
        data matrix
    basis_list: list[list[function]]
        list of basis functions in every mode
    row_coordinates_list: list[list[int]]
        list of row indices
    col_coordinates_list: list[list[int]]
        list of column indices

    Returns
    -------
    matrix: np.ndarray
        extracted matrix
    """

    # construction of the first submatrix
    if row_coordinates_list is None:

        # define current index and mode size
        current_index = 0
        current_mode = len(basis_list[0])

        # define number of row and column sets
        n_rows = 1
        n_cols = len(col_coordinates_list)

        # initialize submatrix
        matrix = np.zeros([n_rows * current_mode, n_cols])

        # construct submatrix
        for j in range(n_cols):

            # set current column index set
            col_coordinates = col_coordinates_list[j]

            # set current snapshot
            snapshot = data[:, col_coordinates[-1]]

            # compute right coefficient
            right_part = 1
            for k in range(len(col_coordinates) - 1):
                right_part *= basis_list[k + current_index + 1][col_coordinates[k]](snapshot)

            # compute vector between left and right part
            middle_part = np.array([basis_list[0][k](snapshot) for k in range(current_mode)])

            # insert matrix entries
            matrix[0:current_mode, j] = middle_part * right_part

    # construction of the last submatrix
    elif col_coordinates_list is None:

        # define current mode size
        current_mode = data.shape[1]

        # define number of row and columns sets
        n_rows = len(row_coordinates_list)
        n_cols = 1

        # initialize submatrix
        matrix = np.zeros([n_rows * current_mode, n_cols])

        # construct submatrix
        for i in range(n_rows):

            # set current row index set
            row_coordinates = row_coordinates_list[i]

            # compute left coefficient
            left_part = 1
            for k in range(len(row_coordinates)):
                left_part *= basis_list[k][row_coordinates[k]](data)

            # insert matrix entries
            matrix[i * current_mode:(i + 1) * current_mode, 0] = left_part

    # construction of the intermediate submatrices
    else:

        # define current index and mode size
        current_index = len(row_coordinates_list[0])
        current_mode = len(basis_list[current_index])

        # define number of row and column sets
        n_rows = len(row_coordinates_list)
        n_cols = len(col_coordinates_list)

        # initialize submatrix
        matrix = np.zeros([n_rows * current_mode, n_cols])

        # construct submatrix
        for j in range(n_cols):

            # set current column index set
            col_coordinates = col_coordinates_list[j]

            # set current snapshot
            snapshot = data[:, col_coordinates[-1]]

            # compute right coefficient
            right_part = 1
            for k in range(len(col_coordinates) - 1):
                right_part *= basis_list[k + current_index + 1][col_coordinates[k]](snapshot)

            # loop over rows
            for i in range(n_rows):

                # set current row index set
                row_coordinates = row_coordinates_list[i]

                # compute left coefficient
                left_part = 1
                for k in range(len(row_coordinates)):
                    left_part *= basis_list[k][row_coordinates[k]](snapshot)

                # compute vector between left and right part
                middle_part = np.array([basis_list[current_index][k](snapshot) for k in range(current_mode)])

                # insert matrix entries
                matrix[i * current_mode:(i + 1) * current_mode, j] = left_part * middle_part * right_part

    return matrix


def __hocur_find_li_cols(matrix, tol=1e-14):
    """Find linearly independent columns of a matrix.

    Parameters
    ----------
    matrix: np.ndarray (m,n)
        rectangular matrix

    Returns
    -------
    cols: list[int]
        indices of linearly independent columns
    """

    # define column list
    cols = []

    # apply QR decomposition with pivoting
    _, r, p = splin.qr(matrix, pivoting=True, mode='economic')

    if tol == 0:
        cols = [p[i] for i in range(matrix.shape[0])]
    else:
        for i in range(r.shape[0]):
            if np.abs(r[i, i]) > tol:
                cols.append(p[i])

    return cols


def __hocur_maxvolume(matrix, maximum_iterations=1000, tolerance=1e-5):
    """Find dominant submatrix.

    Find rows of a given rectangular matrix which build a maximum-volume submatrix, see [1]_.

    Parameters
    ----------
    matrix: np.ndarray (n,r)
        rectangular matrix with rank r
    maximum_iterations: int
        maximum number of iterations, default is 100
    tolerance: float
        tolerance for stopping criterion, default is 1e-5

    Returns
    -------
    rows: list[int]
        rows of the matrix which build the dominant submatrix

    References
    ----------
    .. [1] S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov, N. L. Zamarashkin, "How to find a
           good submatrix", Matrix Methods: Theory, Algorithms, Applications, 2010
    """

    # set max_value and iteration_counter
    max_val = np.infty
    iteration_counter = 1

    # find linearly independent rows
    rows = __hocur_find_li_cols(matrix.T, tol=0)  # type: list

    # repeat row swapping until tolerance is reached
    while max_val > 1 + tolerance and iteration_counter <= maximum_iterations:
        # extract submatrix corresponding to given rows and invert
        submatrix = matrix[rows, :]

        submatrix_inv = np.linalg.inv(submatrix)

        # find maximum absolute value and corresponding indices of matrix @ submatrix^-1
        product = matrix.dot(submatrix_inv)
        max_inds = np.unravel_index(np.argmax(np.abs(product)), product.shape)
        max_val = product[max_inds[0], max_inds[1]]

        # replace row
        rows[int(max_inds[1])] = max_inds[0]

        # increase iteration counter
        iteration_counter += 1

    return rows
