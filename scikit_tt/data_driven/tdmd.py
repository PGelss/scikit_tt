# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lin
from scikit_tt.tensor_train import TT


def tdmd_exact(x, y, threshold=0, ortho_l=True, ortho_r=True):
    """
    Exact TDMD.

    Tensor-based version of exact DMD. See [1]_ for details.

    Parameters
    ----------
    x : TT
        tensor train containing the snapshots
    y : TT
        tensor-train containing the shifted snapshots
    threshold : float, optional
        threshold for SVDs, default is 0
    ortho_l : bool, optional
        whether to left-orthonormalize the first TT cores of x, default is True
    ortho_r : bool, optional
        whether to right-orthonormalize the last TT cores of x, default is True

    Returns
    -------
    dmd_eigenvalues : np.ndarray
        vector containing the DMD eigenvalues
    dmd_modes : TT
        tensor train containing the DMD modes

    References
    ----------
    .. [1] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018)
           3359
    """

    # compute pseudoinverse of x
    x = x.pinv(x.order - 1, threshold=threshold, ortho_l=ortho_l, ortho_r=ortho_r)

    # compute reduced matrix
    reduced_matrix = __tdmd_reduced_matrix(x, y)

    # compute eigenvalues and eigenvectors of the reduced matrix
    # noinspection PyTupleAssignmentBalance
    eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)

    # sort eigenvalues
    ind = np.argsort(eigenvalues)[::-1]
    dmd_eigenvalues = eigenvalues[ind]

    # compute exact DMD modes
    dmd_modes = y.copy()
    dmd_modes.cores[-1] = y.cores[-1][:, :, 0, 0].dot(x.cores[-1][:, :, 0, 0].T).dot(eigenvectors[:, ind]).dot(np.diag(
        np.reciprocal(dmd_eigenvalues)))
    dmd_modes.row_dims[-1] = len(ind)

    return dmd_eigenvalues, dmd_modes


def tdmd_standard(x, y, threshold=0, ortho_l=True, ortho_r=True):
    """
    Standard TDMD.

    Tensor-based version of standard DMD. See [1]_ for details.

    Parameters
    ----------
    x : TT
        tensor train containing the snapshots
    y : TT
        tensor-train containing the shifted snapshots
    threshold : float, optional
        threshold for SVDs, default is 0
    ortho_l : bool, optional
        whether to left-orthonormalize the first TT cores of x, default is True
    ortho_r : bool, optional
        whether to right-orthonormalize the last TT cores of x, default is True

    Returns
    -------
    dmd_eigenvalues : np.ndarray
        vector containing the DMD eigenvalues
    dmd_modes : TT
        tensor train containing the DMD modes

    References
    ----------
    .. [1] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018)
           3359
    """

    # compute pseudoinverse of x
    x = x.pinv(x.order - 1, threshold=threshold, ortho_l=ortho_l, ortho_r=ortho_r)

    # compute reduced matrix
    reduced_matrix = __tdmd_reduced_matrix(x, y)

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


def __tdmd_reduced_matrix(x, y):
    """
    Compute the reduced matrix for finding DMD eigenvalues. See [1]_ for details.

    Parameters
    ----------
    x : TT
        tensor train containing the snapshots
    y : TT
        tensor-train containing the shifted snapshots

    Returns
    -------
    np.ndarray
        reduced matrix

    References
    ----------
    .. [1] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018)
           3359
    """

    # construct reduced matrix
    # ------------------------

    # contract first cores of x and y and reshape
    contraction = np.tensordot(x.cores[0], y.cores[0], axes=(1,1)).reshape([1, x.ranks[1]*y.ranks[1]])

    # set reduced_matrix to contraction
    reduced_matrix = contraction

    # loop over all cores except the last
    for i in range(1, x.order - 1):

        # contract ith cores of x and y and reshape
        contraction = np.tensordot(x.cores[i], y.cores[i], axes=(1,1)).transpose([0, 1, 3, 2, 4, 5]).reshape([x.ranks[i]*y.ranks[i], x.ranks[i+1]*y.ranks[i+1]])

        # multiply reduced_matrix with contraction
        reduced_matrix = reduced_matrix.dot(contraction)

    # reshape reduced_matrix to 2-dimensional array
    reduced_matrix = reduced_matrix.reshape([x.ranks[-2], y.ranks[-2]])

    # contract last cores and reshape
    contraction = np.tensordot(x.cores[-1], y.cores[-1], axes=(1, 1)).reshape([x.ranks[-2], y.ranks[-2]]).T

    # multiply reduced_matrix with contraction
    reduced_matrix = reduced_matrix.dot(contraction)

    return reduced_matrix
