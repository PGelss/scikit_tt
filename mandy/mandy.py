# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scikit_tt
import timeit
import tools


def mandy_coordinate_major(X, Y, psi, threshold=0, cpu_time=False):
    """Multidimensional Approximation of Nonlinear Dynamics (MANDy)

    Coordinate-major order is used to construct the tensor train Xi.

    References
    ----------
    ...

    Arguments
    ---------
    X: ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    Y: ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    psi: list of lambda functions
        list of basis functions
    threshold: float, optional
        threshold for SVDs
    cpu_time: bool, optional
        Whether or not to measure CPU time. False by default.

    Returns
    -------
    Xi: instance of TT class
        tensor train of coefficients for chosen basis functions
    time: float
        CPU time needed for computations. Only returned when `cpu_time` is True.
    """
    with tools.Timer() as time: # measure CPU time
        cores = [np.zeros([1, len(psi), 1, X.shape[1]])] + [np.zeros([X.shape[1], len(psi), 1, X.shape[1]]) for i in
                                                            range(1, X.shape[0])] + [
                    np.eye(X.shape[1]).reshape(X.shape[1], X.shape[1], 1, 1)]  # construct TT cores (empty)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if i == 0:
                    cores[i][0, :, 0, j] = np.array(
                        [psi[k](X[i, j]) for k in range(len(psi))])  # insert elements of first core
                else:
                    cores[i][j, :, 0, j] = np.array(
                        [psi[k](X[i, j]) for k in range(len(psi))])  # insert elements of other cores
        Xi = scikit_tt.TT(cores)  # define tensor train
        Xi = Xi.ortho_left(1, Xi.order - 2, threshold)  # left-orthonormalize first cores

        U, S, V = scipy.linalg.svd(
            Xi.cores[-2].reshape(Xi.ranks[-3] * Xi.row_dims[-2] * Xi.col_dims[-2],
                                 Xi.ranks[-2]),
            full_matrices=False)  # left-orthonormalize penultimate core and keep U, S, and V
        if threshold != 0:
            indices = np.where(S / S[0] > threshold)[0]
            U = U[:, indices]
            S = S[indices]
            V = V[indices, :]
        Xi.ranks[-2] = U.shape[1]  # set new TT rank
        Xi.cores[-2] = U.reshape(Xi.ranks[-3], Xi.row_dims[-2], Xi.col_dims[-2],
                                 Xi.ranks[-2])  # replace penultimate core

        Xi.cores[-1] = (np.diag(np.reciprocal(S)) @ V @ Y.transpose()).reshape(Xi.ranks[-2], Y.shape[0], 1,
                                                                               1)  # replace last core
        Xi.row_dims[-1] = Y.shape[0]  # set new row dimension
    if cpu_time:
        return Xi, time
    else:
        return Xi


def mandy_coordinate_major_matrix(X, Y, psi, threshold=0, cpu_time=False):
    """Multidimensional Approximation of Nonlinear Dynamics (MANDy)

    Coordinate-major order is used to construct the matrix Xi.

    Matrix-based counterpart to MANDy method. This routine can be used for comparing CPU times of the tensor- and
    matrix-based approximation of nonlinear dynamics. Here, we only solve the least-squares problem without iterating
    as done in the SINDy algorithm, cf. ...

    References
    ----------
    ...

    Arguments
    ---------
    X: ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    Y: ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    psi: list of lambda functions
        list of basis functions
    threshold: float, optional
        threshold for SVD
    cpu_time: bool, optional
        Whether or not to measure CPU time. False by default.

    Returns
    -------
    Xi: ndarray
        matrix of coefficients for chosen basis functions
    time: float
        CPU time needed for computations. The construction of the basis matrix Psi is excluded. Only returned when
        `cpu_time` is True.
    """
    Psi = np.zeros([len(psi) ** X.shape[0], X.shape[1]])
    for j in range(X.shape[1]):
        P = [psi[k](X[0, j]) for k in range(len(psi))]
        for i in range(1, X.shape[0]):
            P = np.kron([psi[k](X[i, j]) for k in range(len(psi))], P)
        Psi[:, j] = P
    with tools.Timer() as time:
        U, S, V = scipy.linalg.svd(Psi, full_matrices=False, overwrite_a=True)
        if threshold != 0:
            indices = np.where(S / S[0] > threshold)[0]
            U = U[:, indices]
            S = S[indices]
            V = V[indices, :]
        Xi = Y @ V.transpose() @ np.diag(np.reciprocal(S)) @ U.transpose()  # solve least-squares problem
    if cpu_time:
        return Xi, time
    else:
        return Xi


def mandy_function_major(X, Y, psi, threshold=0, add_one=False, cpu_time=False):
    """Multidimensional Approximation of Nonlinear Dynamics (MANDy)

    Function-major order is used to construct the tensor train Xi.

    References
    ----------
    ...

    Arguments
    ---------
    X: ndarray
        snapshot matrix of size d x m (e.g., coordinates)
    Y: ndarray
        corresponding snapshot matrix of size d x m (e.g., derivatives)
    psi: list of lambda functions
        list of basis functions
    threshold: float, optional
        threshold for SVDs
    cpu_time: bool, optional
        Whether or not to measure CPU time. False by default.

    Returns
    -------
    Xi: instance of TT class
        tensor train of coefficients for chosen basis functions
    time: float
        CPU time needed for computations. Only returned when `cpu_time` is True.
    """
    with tools.Timer() as time: # measure CPU time
        cores = [np.zeros([1, X.shape[0]+(add_one==True), 1, X.shape[1]])] + [np.zeros([X.shape[1], X.shape[0]+(add_one==True), 1, X.shape[1]]) for i in
                                                            range(1, len(psi))] + [
                    np.eye(X.shape[1]).reshape(X.shape[1], X.shape[1], 1, 1)]  # construct TT cores (empty)
        for i in range(len(psi)):
            for j in range(X.shape[1]):
                if i == 0:
                    cores[i][0, :, 0, j] = np.array(
                        [1]*(add_one==True)+[psi[i](X[k, j]) for k in range(X.shape[0])])  # insert elements of first core
                else:
                    cores[i][j, :, 0, j] = np.array(
                        [1] * (add_one == True) +[psi[i](X[k, j]) for k in range(X.shape[0])])  # insert elements of other cores
        Xi = scikit_tt.TT(cores)  # define tensor train
        Xi = Xi.ortho_left(1, Xi.order - 2, threshold)  # left-orthonormalize first cores

        U, S, V = scipy.linalg.svd(
            Xi.cores[-2].reshape(Xi.ranks[-3] * Xi.row_dims[-2] * Xi.col_dims[-2],
                                 Xi.ranks[-2]),
            full_matrices=False)  # left-orthonormalize penultimate core and keep U, S, and V
        if threshold != 0:
            indices = np.where(S / S[0] > threshold)[0]
            U = U[:, indices]
            S = S[indices]
            V = V[indices, :]
        Xi.ranks[-2] = U.shape[1]  # set new TT rank
        Xi.cores[-2] = U.reshape(Xi.ranks[-3], Xi.row_dims[-2], Xi.col_dims[-2],
                                 Xi.ranks[-2])  # replace penultimate core

        Xi.cores[-1] = (np.diag(np.reciprocal(S)) @ V @ Y.transpose()).reshape(Xi.ranks[-2], Y.shape[0], 1,
                                                                               1)  # replace last core
        Xi.row_dims[-1] = Y.shape[0]  # set new row dimension
    if cpu_time:
        return Xi, time
    else:
        return Xi



#
# def mandy_b(X, Y, psi):
#     U = np.eye(X.shape[1])
#     T = scikit_tt.TT(
#         [np.array([1] + [psi[j](X[i, 0]) for i in range(X.shape[0])]).reshape(1, X.shape[0] + 1, 1, 1) for j in
#          range(len(psi))] + [U[:, 0].reshape(1, X.shape[1], 1, 1)])
#     for k in range(1, X.shape[1]):
#         T = T + scikit_tt.TT(
#             [np.array([1] + [psi[j](X[i, k]) for i in range(X.shape[0])]).reshape(1, X.shape[0] + 1, 1, 1) for j in
#              range(len(psi))] + [U[:, k].reshape(1, X.shape[1], 1, 1)])
#     T = T.ortho_left(1, T.d - 2)
#
#     [U, S, V] = scipy.linalg.svd(T.cores[T.d - 2].reshape(T.r[T.d - 2] * T.m[T.d - 2] * T.n[T.d - 2], T.r[T.d - 1]),
#                                  full_matrices=False)
#
#     T.r[T.d - 1] = U.shape[1]
#     T.cores[T.d - 2] = U.reshape(T.r[T.d - 2], T.m[T.d - 2], T.n[T.d - 2], T.r[T.d - 1])
#     T.cores[T.d - 1] = np.diag(np.reciprocal(S)) @ V @ Y.transpose()
#     T.cores[T.d - 1] = T.cores[T.d - 1].reshape(T.r[T.d - 1], Y.shape[0], 1, 1)
#
#     return T
