# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lin


def exact(x, y, threshold=0, ortho_l=True, ortho_r=True):
    """Exact TDMD

    Tensor-based version of exact DMD. See [1]_ for details.

    Parameters
    ----------
    x: instance of TT class
        tensor train containing the snapshots
    y: instance of TT class
        tensor-train containing the shifted snapshots
    threshold: float, optional
        threshold for SVDs, default is 0
    ortho_l: bool, optional
        whether to left-orthonormalize the first TT cores of x, default is True
    ortho_r: bool, optional
        whether to right-orthonormalize the last TT cores of x, default is True

    Returns
    -------
    dmd_eigenvalues: ndarray
        vector containing the DMD eigenvalues
    dmd_modes: instance of TT class
        tensor train containing the DMD modes

    References
    ----------
    .. [1] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018)
           3359
    """

    # compute pseudoinverse of x
    x = x.pinv(x.order - 1, threshold=threshold, ortho_l=ortho_l, ortho_r=ortho_r)

    # compute reduced matrix
    reduced_matrix = __reduced_matrix(x, y)

    # compute eigenvalues and eigenvectors of the reduced matrix
    # noinspection PyTupleAssignmentBalance
    eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)

    # sort eigenvalues
    ind = np.argsort(eigenvalues)[::-1]
    dmd_eigenvalues = eigenvalues[ind]

    # compute exact DMD modes
    dmd_modes = y.copy()
    dmd_modes.cores[-1] = y.cores[-1][:, :, 0, 0] @ x.cores[-1][:, :, 0, 0].T @ eigenvectors[:, ind] @ np.diag(
        np.reciprocal(dmd_eigenvalues))
    dmd_modes.row_dims[-1] = len(ind)

    return dmd_eigenvalues, dmd_modes


def standard(x, y, threshold=0, ortho_l=True, ortho_r=True):
    """Standard TDMD

    Tensor-based version of standard DMD. See [1]_ for details.

    Parameters
    ----------
    x: instance of TT class
        tensor train containing the snapshots
    y: instance of TT class
        tensor-train containing the shifted snapshots
    threshold: float, optional
        threshold for SVDs, default is 0
    ortho_l: bool, optional
        whether to left-orthonormalize the first TT cores of x, default is True
    ortho_r: bool, optional
        whether to right-orthonormalize the last TT cores of x, default is True

    Returns
    -------
    dmd_eigenvalues: ndarray
        vector containing the DMD eigenvalues
    dmd_modes: instance of TT class
        tensor train containing the DMD modes

    References
    ----------
    .. [1] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018)
           3359
    """

    # compute pseudoinverse of x
    x = x.pinv(x.order - 1, threshold=threshold, ortho_l=ortho_l, ortho_r=ortho_r)

    # compute reduced matrix
    reduced_matrix = __reduced_matrix(x, y)

    # compute eigenvalues and eigenvectors of the reduced matrix
    # noinspection PyTupleAssignmentBalance
    eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)

    # sort eigenvalues
    ind = np.argsort(eigenvalues)[::-1]
    dmd_eigenvalues = eigenvalues[ind]

    # compute standard DMD modes
    dmd_modes = x.copy()
    dmd_modes.cores[-1] = eigenvectors[:, ind, None, None]
    dmd_modes.row_dims[-1] = len(ind)

    return dmd_eigenvalues, dmd_modes


def __reduced_matrix(x, y):
    """Compute the reduced matrix for finding DMD eigenvalues. See [1]_ for details.

    Parameters
    ----------
    x: instance of TT class
        tensor train containing the snapshots
    y: instance of TT class
        tensor-train containing the shifted snapshots

    Returns
    -------
    reduced_matrix: ndarray
        reduced matrix

    References
    ----------
    .. [1] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018)
           3359
    """

    # contract snapshot tensors
    z = x.transpose() @ y

    # construct reduced matrix
    reduced_matrix = z.cores[0][:, 0, 0, :]
    for i in range(1, x.order - 1):
        reduced_matrix = reduced_matrix @ z.cores[i][:, 0, 0, :]
    reduced_matrix = reduced_matrix.reshape([x.ranks[-2], y.ranks[-2]])
    reduced_matrix = reduced_matrix @ z.cores[-1].reshape([x.ranks[-2], y.ranks[-2]]).T

    return reduced_matrix
