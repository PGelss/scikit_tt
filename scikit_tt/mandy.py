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

    # # left-orthonormalize first d-1 cores
    # xi = xi.ortho_left(end_index=d - 2, threshold=threshold)
    #
    # # decompose dth core
    # [u, s, v] = lin.svd(xi.cores[d - 1].reshape(xi.ranks[d - 1] * xi.row_dims[d - 1], xi.ranks[d]),
    #                     full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
    #
    # # rank reduction
    # if threshold != 0:
    #     indices = np.where(s / s[0] > threshold)[0]
    #     u = u[:, indices]
    #     s = s[indices]
    #     v = v[indices, :]
    #
    # # set new rank
    # xi.ranks[d] = u.shape[1]
    #
    # # update dth core
    # xi.cores[d - 1] = u.reshape(xi.ranks[d - 1], xi.row_dims[d - 1], 1, xi.ranks[d])
    #
    # # replace last core
    # xi.cores[d] = (np.diag(np.reciprocal(s)) @ v @ y.transpose()).reshape(xi.ranks[d], d, 1, 1)
    #
    # # set new row dimension
    # xi.row_dims[d] = d

    return xi

# def mandy_function_major(X, Y, psi, threshold=0, add_one=False, cpu_time=False):
#     """Multidimensional Approximation of Nonlinear Dynamics (MANDy)
#
#     Function-major order is used to construct the tensor train Xi.
#
#     References
#     ----------
#     ...
#
#     Arguments
#     ---------
#     X: ndarray
#         snapshot matrix of size d x m (e.g., coordinates)
#     Y: ndarray
#         corresponding snapshot matrix of size d x m (e.g., derivatives)
#     psi: list of lambda functions
#         list of basis functions
#     threshold: float, optional
#         threshold for SVDs
#     cpu_time: bool, optional
#         Whether or not to measure CPU time. False by default.
#
#     Returns
#     -------
#     Xi: instance of TT class
#         tensor train of coefficients for chosen basis functions
#     time: float
#         CPU time needed for computations. Only returned when `cpu_time` is True.
#     """
#     with tools.Timer() as time:  # measure CPU time
#         cores = [np.zeros([1, X.shape[0] + (add_one == True), 1, X.shape[1]])] + [
#             np.zeros([X.shape[1], X.shape[0] + (add_one == True), 1, X.shape[1]]) for i in
#             range(1, len(psi))] + [
#                     np.eye(X.shape[1]).reshape(X.shape[1], X.shape[1], 1, 1)]  # construct TT cores (empty)
#         for i in range(len(psi)):
#             for j in range(X.shape[1]):
#                 if i == 0:
#                     cores[i][0, :, 0, j] = np.array(
#                         [1] * (add_one == True) + [psi[i](X[k, j]) for k in
#                                                    range(X.shape[0])])  # insert elements of first core
#                 else:
#                     cores[i][j, :, 0, j] = np.array(
#                         [1] * (add_one == True) + [psi[i](X[k, j]) for k in
#                                                    range(X.shape[0])])  # insert elements of other cores
#         Xi = scikit_tt.TT(cores)  # define tensor train
#         Xi = Xi.ortho_left(1, Xi.order - 2, threshold)  # left-orthonormalize first cores
#
#         U, S, V = scipy.linalg.svd(
#             Xi.cores[-2].reshape(Xi.ranks[-3] * Xi.row_dims[-2] * Xi.col_dims[-2],
#                                  Xi.ranks[-2]),
#             full_matrices=False)  # left-orthonormalize penultimate core and keep U, S, and V
#         if threshold != 0:
#             indices = np.where(S / S[0] > threshold)[0]
#             U = U[:, indices]
#             S = S[indices]
#             V = V[indices, :]
#         Xi.ranks[-2] = U.shape[1]  # set new TT rank
#         Xi.cores[-2] = U.reshape(Xi.ranks[-3], Xi.row_dims[-2], Xi.col_dims[-2],
#                                  Xi.ranks[-2])  # replace penultimate core
#
#         Xi.cores[-1] = (np.diag(np.reciprocal(S)) @ V @ Y.transpose()).reshape(Xi.ranks[-2], Y.shape[0], 1,
#                                                                                1)  # replace last core
#         Xi.row_dims[-1] = Y.shape[0]  # set new row dimension
#     if cpu_time:
#         return Xi, time
#     else:
#         return Xi
#
# #
# # def mandy_b(X, Y, psi):
# #     U = np.eye(X.shape[1])
# #     T = scikit_tt.TT(
# #         [np.array([1] + [psi[j](X[i, 0]) for i in range(X.shape[0])]).reshape(1, X.shape[0] + 1, 1, 1) for j in
# #          range(len(psi))] + [U[:, 0].reshape(1, X.shape[1], 1, 1)])
# #     for k in range(1, X.shape[1]):
# #         T = T + scikit_tt.TT(
# #             [np.array([1] + [psi[j](X[i, k]) for i in range(X.shape[0])]).reshape(1, X.shape[0] + 1, 1, 1) for j in
# #              range(len(psi))] + [U[:, k].reshape(1, X.shape[1], 1, 1)])
# #     T = T.ortho_left(1, T.d - 2)
# #
# #     [U, S, V] = scipy.linalg.svd(T.cores[T.d - 2].reshape(T.r[T.d - 2] * T.m[T.d - 2] * T.n[T.d - 2], T.r[T.d - 1]),
# #                                  full_matrices=False)
# #
# #     T.r[T.d - 1] = U.shape[1]
# #     T.cores[T.d - 2] = U.reshape(T.r[T.d - 2], T.m[T.d - 2], T.n[T.d - 2], T.r[T.d - 1])
# #     T.cores[T.d - 1] = np.diag(np.reciprocal(S)) @ V @ Y.transpose()
# #     T.cores[T.d - 1] = T.cores[T.d - 1].reshape(T.r[T.d - 1], Y.shape[0], 1, 1)
# #
# #     return T
