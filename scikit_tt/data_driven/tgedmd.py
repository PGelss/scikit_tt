import numpy as np
import scipy.sparse as sparse

from scikit_tt.tensor_train import TT
import scikit_tt.utils as utl

import os


# This file contains functions related to tensor-based generator EDMD.

# Structure of this file:
# 1. the four main functions
#       amuset_hosvd : AMUSEt for the general case (line 18)
#       amuset_hosvd_reversible : AMUSEt for the reversible case (line 128)
#       generator_on_product : Evaluate the action of the Koopman generator on a product of functions (line 234)
#       generator_on_product_reversible : Analog to generator_on_product in the reversible case, as it can be used
#                                         to calculate the entries of dPsi(x) (line 280)
# 2. private functions related to the general case  (line 314)
# 3. private functions related to the reversible case (line 667)


# def amuset_hosvd(data_matrix, basis_list, b, sigma, num_eigvals=np.infty, threshold=1e-2, max_rank=np.infty,
#                  return_option='eigenfunctionevals'):
#     """
#     AMUSEt algorithm for the calculation of eigenvalues of the Koopman generator.
#     The tensor-trains are created using the exact TT decomposition, whose ranks are reduced using SVDs.
#     An efficient implementation of tensor contractions that exploits the special structure of the cores is used.
#
#     Parameters
#     ----------
#     data_matrix : np.ndarray
#         snapshot matrix, shape (d, m)
#     basis_list : list[list[Function]]
#         list of basis functions in every mode
#     b : np.ndarray
#         drift, shape (d, m)
#     sigma : np.ndarray
#         diffusion, shape (d, d2, m)
#     num_eigvals : int, optional
#         number of eigenvalues and eigentensors that are returned
#         default: return all calculated eigenvalues and eigentensors
#     threshold : float, optional
#         threshold for svd of psi
#     max_rank : int, optional
#         maximal rank of TT representations of psi after svd/ortho
#     return_option : {'eigentensors', 'eigenfunctionevals', 'eigenvectors'}
#         'eigentensors': return a list of the eigentensors of the koopman generator
#         'eigenfunctionevals': return the evaluations of the eigenfunctions of the koopman generator at all snapshots
#         'eigenvectors': eigenvectors of M in AMUSEt
#
#     Returns
#     -------
#     eigvals : np.ndarray
#         eigenvalues of Koopman generator
#     eigtensors : list[TT] or np.ndarray
#         eigentensors of Koopman generator or evaluations of eigenfunctions at snapshots (shape (*, m))
#         (cf. return_option)
#     """
#
#     # Order:
#     p = len(basis_list)
#     # Mode sizes:
#     n = [len(basis_list[k]) for k in range(p)]
#     # Data size:
#     m = data_matrix.shape[1]
#
#     print('calculating Psi(X)...')
#     cores = [None] * p
#     s, v = None, None
#
#     residual = np.ones((1, m))
#     r_i = residual.shape[0]
#     for i in range(len(basis_list)):
#         core_tmp = np.zeros((r_i, n[i], m))
#         for j in range(m):
#             psi_jk = np.array([basis_list[i][k](data_matrix[:, j]) for k in range(n[i])])
#             core_tmp[:, :, j] = np.outer(residual[:, j], psi_jk)
#         u, s, v = utl.truncated_svd(core_tmp.reshape([core_tmp.shape[0] * core_tmp.shape[1], core_tmp.shape[2]]),
#                                     threshold=threshold, max_rank=max_rank)
#         cores[i] = u.reshape([core_tmp.shape[0], core_tmp.shape[1], 1, u.shape[1]])
#         residual = np.diag(s).dot(v)
#         r_i = residual.shape[0]
#
#     # Complete orthonormal part:
#     U = TT(cores)
#     # Build rank-reduced representation of psi(X):
#     psi = U.rank_tensordot(np.diag(s))
#     psi.concatenate([v[:, :, None, None]], overwrite=True)
#     # Also compute inverse of s:
#     s_inv = np.diag(1.0 / s)
#
#     print('Psi(X): {}'.format(psi))
#
#     print('calculating M in AMUSEt')
#     M = _amuset_efficient(U, s, v.T, data_matrix, basis_list, b, sigma)
#
#     print('calculating eigenvalues and eigentensors...')
#     # calculate eigenvalues of M
#     eigvals, eigvecs = np.linalg.eig(M)
#
#     sorted_indices = np.argsort(-eigvals)
#     eigvals = eigvals[sorted_indices]
#     eigvecs = eigvecs[:, sorted_indices]
#
#     if not (eigvals < 0).all():
#         print('WARNING: there are eigenvalues >= 0')
#
#     if len(eigvals) > num_eigvals:
#         eigvals = eigvals[:num_eigvals]
#         eigvecs = eigvecs[:, :num_eigvals]
#
#     U.rank_tensordot(s_inv, mode='last', overwrite=True)
#
#     # calculate eigentensors
#     if return_option == 'eigentensors':
#         eigvecs = eigvecs[:, :, np.newaxis]
#         eigtensors = []
#         for i in range(eigvals.shape[0]):
#             eigtensor = U.copy()
#             eigtensor.rank_tensordot(eigvecs[:, i, :], overwrite=True)
#             eigtensors.append(eigtensor)
#         return eigvals, eigtensors
#
#     elif return_option == 'eigenfunctionevals':
#         eigenfunctionevals = eigvecs.T @ v
#         return eigvals, eigenfunctionevals
#
#     else:
#         return eigvals, eigvecs


def amuset_hosvd_reversible(data_matrix, basis_list, sigma, reweight=None, num_eigvals=np.infty, threshold=1e-2,
                            max_rank=np.infty, return_option='eigenfunctionevals'):
    """
    AMUSEt algorithm for calculation of eigenvalues of the Koopman generator.
    The tensor-trains are created using the exact TT decomposition, whose ranks are reduced using SVDs.
    An efficient implementation of tensor contractions that exploits the special structure of the cores is used.
    Parameters
    ----------
    data_matrix : np.ndarray
        snapshot matrix, shape (d, m)
    basis_list : list[list[Function]]
        list of basis functions in every mode
    sigma : np.ndarray
        diffusion, shape (d, d2, m)
    reweight : np.ndarray or None, optional
        array of importance sampling ratios, shape (m,)
        can be passed to re-weight calculations if off-equilibrium data are used.
    num_eigvals : int, optional
        number of eigenvalues and eigentensors that are returned
        default: return all calculated eigenvalues and eigentensors
    threshold : float, optional
        threshold for svd of psi and dpsi
    max_rank : int, optional
        maximal rank of TT representations of psi and dpsi after svd/ortho
    return_option : {'eigentensors', 'eigenfunctionevals', 'eigenvectors'}
        'eigentensors': return a list of the eigentensors of the koopman generator
        'eigenfunctionevals': return the evaluations of the eigenfunctions of the koopman generator at all snapshots
        'eigenvectors': return eigenvectors of M in AMUSEt
    Returns
    -------
    eigvals : np.ndarray
        eigenvalues of Koopman generator
    eigtensors : list[TT] or np.ndarray
        eigentensors of Koopman generator or evaluations of eigenfunctions at snapshots (shape (*, m))
        (cf. return_option)
    """

    # Order:
    p = len(basis_list)
    # Mode sizes:
    n = [len(basis_list[k]) for k in range(p)]
    # Data size:
    m = data_matrix.shape[1]

    print('calculating Psi(X)...')
    cores = [None] * p

    residual = np.ones((1, m))
    r_i = residual.shape[0]
    for i in range(len(basis_list)):
        core_tmp = np.zeros((r_i, n[i], m))
        for j in range(m):
            psi_jk = np.array([basis_list[i][k](data_matrix[:, j]) for k in range(n[i])])
            core_tmp[:, :, j] = np.outer(residual[:, j], psi_jk)
        if i == (len(basis_list) - 1) and (reweight is not None):
            for j in range(m):
                core_tmp[:, :, j] *= np.sqrt(reweight[j])
        u, s, v = utl.truncated_svd(core_tmp.reshape([core_tmp.shape[0] * core_tmp.shape[1], core_tmp.shape[2]]),
                                    threshold=threshold, max_rank=max_rank)
        cores[i] = u.reshape([core_tmp.shape[0], core_tmp.shape[1], 1, u.shape[1]])
        residual = np.diag(s).dot(v)
        r_i = residual.shape[0]

    # Complete orthonormal part:
    U = TT(cores)
    # Build rank-reduced representation of psi(X):
    psi = U.rank_tensordot(np.diag(s))
    psi.concatenate([v[:, :, None, None]], overwrite=True)
    # Also compute inverse of s:
    s_inv = np.diag(1.0 / s)
    print('Psi(X): {}'.format(psi))

    print('calculating M in AMUSEt')
    M = _amuset_efficient_reversible(U, s, data_matrix, basis_list, sigma, reweight)

    print('calculating eigenvalues and eigentensors...')
    # calculate eigenvalues of M
    eigvals, eigvecs = np.linalg.eig(M)

    sorted_indices = np.argsort(-eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    if not (eigvals < 0).all():
        print('WARNING: there are eigenvalues >= 0')

    if len(eigvals) > num_eigvals:
        eigvals = eigvals[:num_eigvals]
        eigvecs = eigvecs[:, :num_eigvals]

    U.rank_tensordot(s_inv, mode='last', overwrite=True)

    # calculate eigentensors
    if return_option == 'eigentensors':
        eigvecs = eigvecs[:, :, np.newaxis]
        eigtensors = []
        for i in range(eigvals.shape[0]):
            eigtensor = U.copy()
            eigtensor.rank_tensordot(eigvecs[:, i, :], overwrite=True)
            eigtensors.append(eigtensor)

        return eigvals, eigtensors
    elif return_option == 'eigenfunctionevals':
        U.rank_tensordot(eigvecs, overwrite=True)
        U.tensordot(psi, p, mode='first-first', overwrite=True)
        U = U.cores[0][0, :, 0, :].T
        return eigvals, U
    else:
        return eigvals, eigvecs

def amuset_hosvd(data_matrix, basis_list, sigma, b=None, reweight=None, num_eigvals=np.infty, threshold=1e-2,
                 max_rank=np.infty, return_option='eigenfunctionevals', output_freq=None):
    """
    Calculate eigenvalues and eigenfunctions of the Koopman generator, projected on a tensor product basis,
    using simulation data.
    This function uses the tensor train format and the tgEDMD method (Algorithm 2 in [1]) to approximate the
    Galerkin projection of the Koopman generator on a given product basis.

    Parameters
    ----------
    data_matrix : np.ndarray
        snapshot matrix, shape (d, m), where d is the state space dimension and m is the data size.
    basis_list : list[list[Function]]
        list of lists of elementary basis functions for each mode. Functions must be derived from
        data_driven.transform.Function
    sigma : np.ndarray
        shape (d, d2, m), diffusion matrix at each data point.
    b : np.ndarray or None
        shape (d, m), drift vector at each data point. If b equals None, reversible tgEDMD is applied.
    reweight : np.ndarray or None
        shape (m,) array of importance sampling ratios
    num_eigvals : int or np.infty
        number of eigenvalues and eigentensors that are returned
        default: return all calculated eigenvalues and eigentensors
    threshold : float
        absolute truncation threshold for singular values during global SVD of \PSI(X).
        Default value is 1e-2.
    max_rank : int or np.infty
        maximally allowed rank of orthonormal TT representation for data tensor \Psi(X).
    return_option : {'eigentensors', 'eigenfunctionevals', 'eigenvectors'}
        'eigentensors': return a list of the eigentensors of the Koopman generator, not implemented at this time
        'eigenfunctionevals': return the evaluations of the eigenfunctions of the Koopman generator at all snapshots
        'eigenvectors': return eigenvectors of reduced matrix M in tgEDMD.
    output_freq : int or None
        Display progress message every output_freq steps.

    Returns
    -------
    eigvals : np.ndarray
        shape (num_eigvals,) eigenvalues of Koopman generator
    eigtensors : list[TT] or np.ndarray of shape  (num_eigvals, m)
        eigentensors of Koopman generator or evaluations of eigenfunctions at all snapshots.
        (cf. return_option)
    ranks : list
        list of TT ranks obtained by application global SVD to \PSI(X).
    """

    """ Extract information on the data and basis sets: """
    # Order of tensor representation:
    p = len(basis_list)
    # Mode sizes:
    n = [len(basis_list[k]) for k in range(p)]
    # Data size:
    m = data_matrix.shape[1]

    """ Apply global SVD algorithm to \PSI(X): """
    if output_freq is not None:
        print("Computing global SVD of \PSI(X)...")
    # Lists for orthonormal cores and ranks:
    cores_u = []
    ranks_u = []
    # First rank is always equals to one:
    ranks_u.append(1)
    # array to hold residual (non-orthonormal) part of PSI(X):
    residual = np.ones((1, m))
    r_i = residual.shape[0]

    # Loop over modes of the tensor:
    for i in range(p):
        # Evaluate updated next core:
        core_tmp = np.zeros((r_i, n[i], m))
        # For each data point ...
        for j in range(m):
            # ... evaluate basis set for mode i at this data point...
            psi_jk = np.array([basis_list[i][k](data_matrix[:, j]) for k in range(n[i])])
            # ... compute outer product with reduced basis from previous step, stored in
            # rows of residual (see [2]).
            core_tmp[:, :, j] = np.outer(residual[:, j], psi_jk)
        # Apply re-weighting if required:
        if i == (len(basis_list) - 1) and (reweight is not None):
            for j in range(m):
                core_tmp[:, :, j] *= np.sqrt(reweight[j])

        # Compute SVD and update residual:
        u, s, v = utl.truncated_svd(core_tmp.reshape([core_tmp.shape[0] * core_tmp.shape[1], core_tmp.shape[2]]),
                                    threshold=threshold, max_rank=max_rank, rel_truncation=False)
        # Compute non-orthonormal part:
        residual = np.diag(s).dot(v)
        # Re-shape orthonormal part and add to list:
        u = u.reshape([core_tmp.shape[0], core_tmp.shape[1], u.shape[1]])
        cores_u.append(u.copy())
        # Update ranks:
        r_i = residual.shape[0]
        ranks_u.append(r_i)
    # Final rank always equals one:
    ranks_u.append(1)
    if output_freq is not None:
        print("Completed global SVD of Psi(X) with ranks: ", ranks_u)

    """ Compute reduced matrix: """
    if output_freq is not None:
        print('Calculating reduced matrix ...')
    s_inv = np.diag(1.0 / s)
    M = _reduced_matrix_tgedmd(cores_u, s_inv, v.T, ranks_u, data_matrix, basis_list, sigma, b=b,
                               reweight=reweight, output_freq=output_freq)

    """ Compute spectral components of reduced matrix: """
    print('Calculating eigenvalues and eigentensors...')
    # Diagonalize M:
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort in descending order:
    sorted_indices = np.argsort(-eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    # Issue warning if positive eigenvalues occur:
    if not (eigvals < 0).all():
        print('WARNING: there are eigenvalues >= 0')
    # Reduce spectral components to required number:
    if len(eigvals) > num_eigvals:
        eigvals = eigvals[:num_eigvals]
        eigvecs = eigvecs[:, :num_eigvals]

    # Calculate required output according to parameter return_option:
    if return_option == 'eigentensors':
        # Obtain TT representation of eigentensor by contracting last orthonormal
        # core of U with each eigenvector of M:
        eigtensors = []
        eigvecs = eigvecs[:, :, None]
        for i in range(eigvals.shape[0]):
            eigtensor = [cores_u[jj].copy() for jj in range(p)]
            eigtensor[-1] = np.tensordot(eigtensor[-1], eigvecs[:, i, :], axes=([2], [0]))
            eigtensors.append(eigtensor)
        return eigvals, eigtensors, ranks_u

    if return_option == 'eigenfunctionevals':
        # Obtain eigenfunction trajectory by simply dotting eigenvectors and right singular vectors
        # v obtained from globals SVD, see [2].
        eigfun_traj = np.dot(eigvecs.T, v)
        return eigvals, eigfun_traj, ranks_u
    else:
        return eigvals, eigvecs, ranks_u


def generator_on_product(basis_list, s, x, b, sigma):
    """
    Evaluate the Koopman generator operating on the following function
    f = basis_list[1][s[1]] * ... * basis_list[p][s[p]]
    in x.
    It holds dPsi(x)_{s_1,...,s_p} = generator_on_product

    Parameters
    ----------
    basis_list : list[list[Function]]
    s : tuple
        indices of basis functions
    x : np.ndarray
        shape(d,)
    b : np.ndarray
        drift at the snapshot x, shape (d,)
    sigma : np.ndarray
        diffusion at the snapshot x, shape (d, d2)

    Returns
    -------
    float
    """

    p = len(s)
    a = sigma.dot(sigma.T)

    out = 0
    for j in range(p):
        product = 1
        for l in range(p):
            if l == j:
                continue
            product *= basis_list[l][s[l]](x)
        out += product * _generator(basis_list[j][s[j]], x, b, sigma)

        for v in range(j + 1, p):
            product = 1
            for l in range(p):
                if l == j or l == v:
                    continue
                product *= basis_list[l][s[l]](x)
            out += product * _frob_inner(a, np.outer(basis_list[v][s[v]].gradient(x), basis_list[j][s[j]].gradient(x)))
    return out


def generator_on_product_reversible(basis_list, s, i, x, sigma):
    """
    It holds dPsi(x)_{s_1,...,s_p, i} = nabla psi_{s_1,...,s_p} cdot sigma_{:, i} = generator_on_product

    Parameters
    ----------
    basis_list : list[list[Function]]
    s : tuple
        indices of basis functions
    i : int
        index of column of sigma
    x : np.ndarray
        shape(d,)
    sigma : np.ndarray
        diffusion at the snapshot x, shape (d, d2)

    Returns
    -------
    float
        dPsi(x)_{s_1,...,s_p, i}
    """
    p = len(s)
    out = 0
    for j in range(p):
        product = np.inner(sigma[:, i], basis_list[j][s[j]].gradient(x))
        for l in range(p):
            if l == j:
                continue
            product *= basis_list[l][s[l]](x)
        out += product
    return out


# ################ private functions for the general case ##########################################
def _amuset_efficient(u, s, v, x, basis_list, b, sigma):
    """
    Construct the Matrix M in AMUSEt using the efficient implementation (M = sum M_k).

    Parameters
    ----------
    u : TT
    s : np.ndarray
    v : np.ndarray
    x : np.ndarray
        snapshot matrix of size d x m
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : np.ndarray
        drift, shape (d, m)
    sigma : np.ndarray
        diffusion, shape (d, d2, m)

    Returns
    -------
    np.ndarray
        matrix M from AMUSEt
    """

    m = x.shape[1]
    k = 0

    # for outputting progress
    next_print = 0.1

    s_inv = np.diag(1.0 / s)

    dPsi = _tt_decomposition_one_snapshot(x[:, k], basis_list, b[:, k], sigma[:, :, k])
    M = _calc_M_k_amuset(u, v, s_inv, dPsi, k)

    while k + 1 < m:
        k = k + 1
        if k / m > next_print:
            print('progress: {}%'.format(int(round(next_print * 100))))
            next_print += 0.1
        dPsi = _tt_decomposition_one_snapshot(x[:, k], basis_list, b[:, k], sigma[:, :, k])
        M += _calc_M_k_amuset(u, v, s_inv, dPsi, k)

    return M


def _tt_decomposition_one_snapshot(x_k, basis_list, b_k, sigma_k):
    """
    Calculate the exact tt_decomposition of dPsi(x_k).

    Parameters
    ----------
    x_k : np.ndarray
        k-th snapshot, shape (d,)
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b_k : np.ndarray
        drift at the snapshot x_k, shape (d,)
    sigma_k : np.ndarray
        diffusion at the snapshot x_k, shape (d, d2)

    Returns
    -------
    TT
        tt_decomposition of dPsi(x_k)
    """

    # number of modes
    p = len(basis_list)

    # cores of dPsi(x_k)
    cores = [_dPsix(basis_list[0], x_k, b_k, sigma_k, position='first')]
    for i in range(1, p - 1):
        cores.append(_dPsix(basis_list[i], x_k, b_k, sigma_k, position='middle'))
    cores.append(_dPsix(basis_list[p - 1], x_k, b_k, sigma_k, position='last'))

    return TT(cores)


def _dPsix(psi_k, x, b, sigma, position='middle'):
    """
    Computes the k-th core of dPsi(x).

    Parameters
    ----------
    psi_k : list[Function]
        [psi_{k,1}, ... , psi_{k, n_k}]
    x : np.ndarray
        shape (d,)
    b : np.ndarray
        drift at the snapshot x, shape (d,)
    sigma : np.ndarray
        diffusion at the snapshot x, shape (d, d2)
    position : {'first', 'middle', 'last'}, optional
        first core: k = 1
        middle core: 2 <= k <= p-1
        last core: k = p
    Returns
    -------
    np.ndarray
        k-th core of dPsi(x)
    """

    d = x.shape[0]
    nk = len(psi_k)
    psi_kx = [fun(x) for fun in psi_k]
    a = sigma.dot(sigma.T)

    partial_psi_kx = np.zeros((nk, d))

    for i in range(nk):
        partial_psi_kx[i, :] = psi_k[i].gradient(x)

    if position == 'middle':
        core = np.zeros((d + 2, nk, 1, d + 2))

        # diagonal
        for i in range(d + 2):
            core[i, :, 0, i] = psi_kx

        # 1. row
        core[0, :, 0, 1] = [_generator(fun, x, b, sigma) for fun in psi_k]
        core[0, :, 0, 2:] = partial_psi_kx

        # 2. column
        for i in range(2, d + 2):
            core[i, :, 0, 1] = [np.inner(a[i - 2, :], partial_psi_kx[row, :]) for row in range(nk)]

    elif position == 'first':
        core = np.zeros((1, nk, 1, d + 2))
        core[0, :, 0, 0] = psi_kx
        core[0, :, 0, 1] = [_generator(fun, x, b, sigma) for fun in psi_k]
        core[0, :, 0, 2:] = partial_psi_kx

    else:  # position == 'last'
        core = np.zeros((d + 2, nk, 1, 1))
        core[0, :, 0, 0] = [_generator(fun, x, b, sigma) for fun in psi_k]
        core[1, :, 0, 0] = psi_kx

        for i in range(2, d + 2):
            core[i, :, 0, 0] = [np.inner(a[i - 2, :], partial_psi_kx[row, :]) for row in range(nk)]

    return core


def _calc_M_k_amuset(u, v, s_inv, dpsi, k):
    """
    Calculate the Matrix M_k in the efficient AMUSEt formulation using the special tensordot.
    Computes index contraction between (V^T (dPsi(x_k) otimes e_k)^T) and (U Sigma^-1) in AMUSEt
    using the special_kron and special_tensordot functions.
    For the construction of M_k^{(i)} the function _special_kron is used.
    For the contraction of the M_k^{(i)} the function _special_tensordot is used.

    Parameters
    ----------
    u : TT
        tensor u from svd of transformed data tensor
    v : np.ndarray
        v from svd of transformed data tensor, shape = (m, r)
    s_inv : np.ndarray
        Sigma^-1 from svd of transformed data tensor
    dpsi : TT
        TT decomposition of dPsi(x_k) (output of _tt_decomposition_one_snapshot)
    k : int
        index of snapshot

    Returns
    -------
    np.ndarray
        Matrix M in AMUSEt
    """

    # calculate the contraction dpsi.tensordot(u, p, mode='first-first')
    M = _special_kron(dpsi.cores[0], u.cores[0])
    # M.shape (r_0, r_1, s_0, s_1)

    for i in range(1, dpsi.order):
        M_new = _special_kron(dpsi.cores[i], u.cores[i])
        # M_new.shape (r_{i-1}, r_i, s_{i-1}, s_i)
        M = _special_tensordot(M, M_new)

    return np.outer(v[k, :], s_inv.dot(M[0, 0, 0, :]))


def _special_tensordot(A, B):
    """
    Tensordot between arrays A and B with special structure.
    A and B have the structure that arises in the cores of dPsi. As A and B can be the result of a Kronecker product,
    the entries of A and B can be matrices themselves. Thus A and B are modeled as 4D Arrays where the first and second
    index refer to the rows and columns of A and B. The third and fourth index refer to the rows and colums of the
    entries of A and B.
    All nonzero elements of A and B are
    in the diagonal [i,i,:,:], in the first row [0,:,:,:] and in the second column [:,1,:,:].
    Furthermore A and B are quadratic (A.shape[0] = A.shape[1]) or A has only one row or B only one column.
    The tensordot is calculated along both column dimensions of A (1,3) and both row dimensions of B (0,2).
    The resulting array has the same structure as A and B.

    Parameters
    ----------
    A : np.ndarray
    B : np.ndarray

    Returns
    -------
    np.ndarray
        tensordot between A and B along the axis ((1,3), (0,2))
    """
    C = np.zeros((A.shape[0], B.shape[1], A.shape[2], B.shape[3]))

    if B.shape[1] > 1:  # B is a matrix
        # diagonal
        for i in range(min(A.shape[0], B.shape[1])):
            C[i, i, :, :] = A[i, i, :, :].dot(B[i, i, :, :])

        # entry(0, 1)
        for i in range(A.shape[1]):
            C[0, 1, :, :] += A[0, i, :, :].dot(B[i, 1, :, :])

        # first row
        for i in range(2, B.shape[0]):
            C[0, i, :, :] = A[0, 0, :, :].dot(B[0, i, :, :]) + A[0, i, :, :].dot(B[i, i, :, :])

        # second column
        for i in range(2, A.shape[0]):
            C[i, 1, :, :] = A[i, 1, :, :].dot(B[1, 1, :, :]) + A[i, i, :, :].dot(B[i, 1, :, :])

    else:  # B is a vector
        # entry(0, 0)
        for i in range(A.shape[1]):
            C[0, 0, :, :] += A[0, i, :, :].dot(B[i, 0, :, :])

        if A.shape[0] > 1:
            # entry(1, 0)
            C[1, 0, :, :] = A[1, 1, :, :].dot(B[1, 0, :, :])

            # others
            for i in range(2, A.shape[0]):
                C[i, 0, :, :] = A[i, 1, :, :].dot(B[1, 0, :, :]) + A[i, i, :, :].dot(B[i, 0, :, :])

    return C


def _special_kron(dPsi, B):
    """
    Build Matrices M_new = M_k^{(i)} from AMUSEt.
    dPsi and B are TT cores (np.ndarray with order 4).
    dPsi needs to have the special structure of the dPsi(x) cores.

    Parameters
    ----------
    dPsi : np.ndarray
    B : np.ndarray

    Returns
    -------
    np.ndarray
        shape = (dPsi.shape[0], dPsi.shape[3], B.shape[0], B.shape[3])
    """

    C = np.zeros((dPsi.shape[0], dPsi.shape[3], B.shape[0], B.shape[3]))

    if dPsi.shape[0] == 1 or dPsi.shape[3] == 1:  # dPsi is a vector
        for i in range(dPsi.shape[0]):
            for j in range(dPsi.shape[3]):
                C[i, j, :, :] = np.tensordot(dPsi[i, :, 0, j], B[:, :, 0, :], axes=(0, 1))
    else:  # dPsi is a matrix
        # diagonal
        psi = dPsi[0, :, 0, 0]
        psi = np.tensordot(psi, B[:, :, 0, :], axes=(0, 1))
        for i in range(C.shape[0]):
            C[i, i, :, :] = psi

        # first row
        for i in range(1, C.shape[1]):
            C[0, i, :, :] = np.tensordot(dPsi[0, :, 0, i], B[:, :, 0, :], axes=(0, 1))

        # 2nd column
        for i in range(2, C.shape[0]):
            C[i, 1, :, :] = np.tensordot(dPsi[i, :, 0, 1], B[:, :, 0, :], axes=(0, 1))

    return C


def _frob_inner(a, b):
    """
    Frobenius inner product of matrices a and b.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return np.trace(np.inner(a, b))


def _generator(f, x, b, sigma):
    """
    Infinitesimal Koopman Generator applied to f.
    Computes Lf(x) = b(x) cdot nabla f(x) + 0.5 a(x) : nabla^2 f(x).

    Parameters
    ----------
    f : Function
    x : np.ndarray
    b : np.ndarray
        drift at the snapshot x, shape (d,)
    sigma : np.ndarray
        diffusion at the snapshot x, shape (d, d2)

    Returns
    -------
    float
    """
    a = sigma.dot(sigma.T)
    return np.inner(b, f.gradient(x)) + 0.5 * _frob_inner(a, f.hessian(x))


def _is_special(A):
    """
    Check if ndarray A has the special structure from the dpsi cores.
    A.shape = (outer rows, outer coumns, inner rows, inner colums). (not a TT core!)

    Parameters
    ----------
    A : np.ndarray
       A.ndim = 4

    Returns
    -------
    bool
    """
    if not A.ndim == 4:
        raise ValueError('A should have ndim = 4')

    if A.shape[0] == A.shape[1]:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == j or i == 0 or j == 1:
                    continue
                if not (A[i, j, :, :] == 0).all():
                    return False

    elif not (A.shape[0] == 1 or A.shape[1] == 1):
        return False

    return True


# ################ private functions for the reversible case ##########################################
def _amuset_efficient_reversible(u, s, x, basis_list, sigma, reweight=None):
    """
    Construct the Matrix M in AMUSEt using the efficient implementation (M = sum (-0.5 M_k M_k^T)).
    Parameters
    ----------
    u : TT
    s : np.ndarray
    x : np.ndarray
        snapshot matrix of size d x m
    basis_list : list[list[Function]]
        list of basis functions in every mode
    sigma : np.ndarray
        diffusion, shape (d, d2, m)
    reweight : np.ndarray or None, optional
        array of importance sampling ratios, shape (m,)
    Returns
    -------
    np.ndarray
        matrix M from AMUSEt
    """
    m = x.shape[1]
    k = 0

    # Check if re-weighting factors are given:
    if reweight is not None:
        w = reweight
    else:
        w = np.ones(m)

    # for outputting progress
    next_print = 0.1

    s_inv = np.diag(1.0 / s)

    dPsi = _tt_decomposition_one_snapshot_reversible(x[:, k], basis_list, sigma[:, :, k])
    M = _calc_M_k_amuset_reversible(u, s_inv, dPsi)

    while k + 1 < m:
        k = k + 1
        if k / m > next_print:
            print('progress: {}%'.format(int(round(next_print * 100))))
            next_print += 0.1
        dPsi = _tt_decomposition_one_snapshot_reversible(x[:, k], basis_list, sigma[:, :, k])
        M += w[k] * _calc_M_k_amuset_reversible(u, s_inv, dPsi)

    return M


def _tt_decomposition_one_snapshot_reversible(x_k, basis_list, sigma_k):
    """
    Calculate the exact tt_decomposition of dPsi(x_k).

    Parameters
    ----------
    x_k : np.ndarray
        snapshot, size (d,)
    basis_list : list[list[Function]]
        list of basis functions in every mode
    sigma_k : np.ndarray
        diffusion at snapshot x_k, shape (d, d2)
    Returns
    -------
    TT
        tt decomposition of dPsi(x_k)
    """
    # number of modes
    p = len(basis_list)

    # insert elements of core 1
    cores = [_dPsix_reversible(basis_list[0], x_k, position='first')]

    # insert elements of cores 2,...,p-1
    for i in range(1, p - 1):
        cores.append(_dPsix_reversible(basis_list[i], x_k, position='middle'))

    # insert elements of core p
    cores.append(_dPsix_reversible(basis_list[p - 1], x_k, position='last'))

    # insert elements of core p + 1
    cores.append(sigma_k[:, :, None, None])

    return TT(cores)


def _dPsix_reversible(psi_k, x, position='middle'):
    """
    Computes the k-th core of dPsi(x).

    Parameters
    ----------
    psi_k : list[Function]
        [psi_{k,1}, ... , psi_{k, n_k}]
    x : np.ndarray
        shape (d,)
    position : {'first', 'middle', 'last'}, optional
        first core: k = 1
        middle core: 2 <= k <= p-1
        last core: k = p (not the one with sigma!)

    Returns
    -------
    np.ndarray
        k-th core of dPsi(x)
    """

    d = x.shape[0]
    nk = len(psi_k)
    psi_kx = [fun(x) for fun in psi_k]

    if position == 'middle':
        core = np.zeros((d + 1, nk, 1, d + 1))

        # diagonal
        for i in range(d + 1):
            core[i, :, 0, i] = psi_kx

        # partials
        for i in range(nk):
            core[0, i, 0, 1:] = psi_k[i].gradient(x)

    elif position == 'first':
        core = np.zeros((1, nk, 1, d + 1))
        core[0, :, 0, 0] = psi_kx
        for i in range(nk):
            core[0, i, 0, 1:] = psi_k[i].gradient(x)

    else:
        core = np.zeros((d + 1, nk, 1, d))
        for i in range(d):
            core[i + 1, :, 0, i] = psi_kx
        for i in range(nk):
            core[0, i, 0, :] = psi_k[i].gradient(x)

    return core


def _calc_M_k_amuset_reversible(u, s_inv, dpsi):
    """
    Calculate the Matrix M_k in the efficient AMUSEt formulation using the special tensordot.
    Computes index contraction between (Sigma^-1 U^T) and (dPsi(x_k) in AMUSEt
    using the special_kron and special_tensordot functions.
    For the construction of M_k^{(i)} the function _special_kron_reversible is used.
    For the contraction of the M_k^{(i)} the function _special_tensordot_reversible is used.

    Parameters
    ----------
    u : TT
        tensor u from svd of transformed data tensor
    s_inv : np.ndarray
        matrix Sigma^-1 from svd of transformed data tensor
    dpsi : TT
        TT decomposition of dPsi(x_k) (output from _tt_decomposition_one_snapshot_reversible)
    Returns
    -------
    np.ndarray
        Matrix -0.5 * M_k * M_k^T in AMUSEt
    """

    p = dpsi.order - 1
    M = _special_kron_reversible(dpsi.cores[0], u.cores[0])
    # M.shape (r_0, r_1, s_0, s_1)

    for i in range(1, p):
        M_new = _special_kron_reversible(dpsi.cores[i], u.cores[i])
        # M_new.shape (r_{i-1}, r_i, s_{i-1}, s_i)
        M = _special_tensordot_reversible(M, M_new)

    M = M[0, :, 0, :].T
    sigma = dpsi.cores[p][:, :, 0, 0]
    M = s_inv.dot(M).dot(sigma)
    return -0.5 * M.dot(M.T)

#================ memory-efficient versions of all private functions for the reversible case: =============
def _reduced_matrix_tgedmd(u, s_inv, V, ranks, x, basis_list, sigma, b=None, reweight=None,
                           output_freq=None):
    """
    Construct the reduced matrix M by the tensor network contraction Eqs. (16-17) in [1]. As
    described therein, the network is contracted for all data points separately, and then summed
    up.

    NOTE: If the drift vector b is given as an array, non-reversible tgEDMD Eq. (16) is applied.
        Otherwise, reversible tgEDMD Eq. (17) is used.

    Parameters
    ----------
    u :     list of cores of the orthonormal tensor U obtained from global SVD of \Psi(X).
    s_inv:  np.ndarray
            shape (r_p, r_p), inverse of diagonal matrix  obtained from global SVD of \Psi(X).
    V:      np.ndarray
            shape (m, r_p), matrix of right singular vectors obtained from global SVD of \Psi(X).
    ranks:  list of TT ranks for u
    x :     np.ndarray
            shape (d, m), data matrix
    basis_list : list[list[Function]]
            list of lists of basis functions for each mode, derived from data_driven.transform.Function
    sigma : np.ndarray
            shape (d, d2, m), diffusion matrix at all data sites.
    b:      np.ndarray
            shape (d, m) or None, d-dimensional drift vector at all data sites.
    reweight : np.ndarray
            shape (m,) or None, array of importance sampling ratios
    output_freq: int or None,
            Display progress message after every output_freq data sites.

    Returns
    -------
    M : np.ndarray
        shape (r_p, r_p), reduced matrix
    """
    # Obtain dimensions of the data:
    d, m = x.shape
    # Obtain order of the tensor:
    p = len(basis_list)
    # Check if re-weighting factors are given:
    if reweight is not None:
        w = reweight
    else:
        w = np.ones(m)
    # Check whether reversible or non-reversible tgEDMD applies:
    rev = (b is None)
    # Compute co-variance of the diffusion for reversible case:
    a = np.einsum("ikl, jkl -> ijl", sigma, sigma)

    if output_freq is not None:
        print("Contracting tensor network: ")
    # Prepare output matrix:
    M = np.zeros((ranks[p], ranks[p]))
    # Contract tensor network separately for each data point:
    for l in range(m):
        # First step of tensor contraction:
        if rev:
            v = _contraction_step_dPsi_u(basis_list[0], x[:, l], u[0], position='first')
        else:
            v = _contraction_step_LPsi_u(basis_list[0], x[:, l], b[:, l], sigma[:, :, l],
                                         u[0], position="first")

        # All intermediate steps of tensor contraction:
        for k in range(1, p - 1):
            if rev:
                v = _contraction_step_dPsi_u(basis_list[k], x[:, l], u[k], position='middle',
                                             v=v)
            else:
                v = _contraction_step_LPsi_u(basis_list[k], x[:, l], b[:, l], sigma[:, :, l],
                                             u[k], position="middle", v=v)

        # Final step of the contraction:
        if rev:
            v = _contraction_step_dPsi_u(basis_list[p-1], x[:, l], u[p-1], position='last',
                                         v=v)
        else:
            v = _contraction_step_LPsi_u(basis_list[p-1], x[:, l], b[:, l], sigma[:, :, l],
                                         u[p-1], position="last", v=v)

        # Add contribution to M:
        if rev:
            v = np.reshape(v, (d, ranks[p]))
            v = np.dot(v, s_inv)
            M += -0.5 * w[l] * np.dot(v.T, np.dot(a[:, :, l], v))
        else:
            M += np.sqrt(w[l]) * np.outer(V[l, :], np.dot(v, s_inv))
        # Display progress message if required:
        if output_freq is not None and (np.remainder(l+1, output_freq) == 0):
            print("Processed %.2f per cent."%(100 * (l+1) / m))
    return M


def _contraction_step_LPsi_u(psi_k, x, bx, sig_x, u_k, position="middle", v=None):
    """
    Helper function for the tensor network contraction Lpsi(x)^T \times \hat{U}, for
     one data point x, within non-reversible tgEDMD. The contraction is performed using
     Algorithm 3 in [1], leveraging the sparse structure of Lpsi. This function carries
     out a single step in Algorithm 3.

    Parameters:
    -----------
    psi_k: list of callable function (derived from data_driven.transform.Function),
        representing the basis set for mode k.
    x : np.ndarray
        shape (d,): a single data point
    bx: np.ndarray
        shape (d,), drift vector at position x
    sig_x: np.ndarray
        shape (d, d), diffusion matrix at position x
    u_k:  ndarray, shape(r_km, n_k, r_kp)
        orthogonal core from the global SVD
    position : {'first', 'middle', 'last'}, optional
        first core: k = 1
        middle core: 2 <= k <= p-1
        last core: k = p
    v, ndarray, shape (1, (d+2) * r_km) or None.
        the output of the previous step of the contraction.
        v should be set to None if position=='first'

    Returns:
    --------
    array of shape
            position == 'first':   (1, r_kp)
            position == 'middle':   (1, r_kp)
            position == 'last':   (1, r_kp)

        result of k-th step of the contraction
    """
    # Obtain dimensions of the data point:
    d = x.shape[0]
    # Obtain shape of orthonormal core:
    r_km, nk, r_kp = u_k.shape

    # Pre-compute required quantities:
    # Covariance of the diffusion:
    ax = np.dot(sig_x, sig_x.T)
    # Evaluation of the basis set at x:
    psi_kx = np.array([psi_k[ii](x) for ii in range(nk)])
    # Gradients of the basis set at x:
    psi_grad = np.array([psi_k[ii].gradient(x) for ii in range(nk)])
    # Application of the generator to the basis set at x:
    Lpsi_kx = np.array([np.dot(bx, psi_grad[ii, :]) + 0.5 * np.sum(ax * psi_k[ii].hessian(x))
                     for ii in range(nk)])
    # Product of gradient matrix and diffusion matrix, required for cores of Lpsi^k(x):
    psi_grad_sigma = np.dot(psi_grad, sig_x)

    # Middle position:
    if position == 'middle':
        v = v.reshape((1, (d + 2), r_km))
        v_new = np.zeros((1, d + 2, r_kp))
        for ii in range(nk):
            # Multiply v and first column of Lpsi^k(x) x U^k:
            v_new[0, 0, :] += np.dot(v[0, 0, :], psi_kx[ii] * u_k[:, ii, :])
            # Multiply v and second column of Lpsi^k(x) x U^k:
            v_new[0, 1, :] += np.dot(v[0, 0, :], Lpsi_kx[ii] * u_k[:, ii, :]) + np.dot(
                v[0, 1, :], psi_kx[ii] * u_k[:, ii, :])
            for jj in range(d):
                v_new[0, 1, :] += np.dot(v[0, jj + 2, :], psi_grad_sigma[ii, jj] * u_k[:, ii, :])
            # Multiply v and remaining columns of Lpsi^k(x) x U^k:
            for jj in range(d):
                v_new[0, jj + 2, :] += (np.dot(v[0, 0, :], psi_grad_sigma[ii, jj] * u_k[:, ii, :]) +
                                    np.dot(v[0, jj + 2, :], psi_k[ii](x) * u_k[:, ii, :]))
        v_new = v_new.reshape((1, (d + 2) * r_kp))

    # End-of-chain position:
    elif position == 'last':
        v = v.reshape((1, (d + 2), r_km))
        v_new = np.zeros((1, r_kp))
        for ii in range(nk):
            # Multiply v and only column of Lpsi^k(x) x U^k:
            v_new[0, :] += np.dot(v[0, 0, :], Lpsi_kx[ii] * u_k[:, ii, :]) + np.dot(
                v[0, 1, :], psi_kx[ii] * u_k[:, ii, :])
            for jj in range(d):
                v_new[0, :] += np.dot(v[0, jj + 2, :], psi_grad_sigma[ii, jj] * u_k[:, ii, :])

    # Head-of-chain position:
    else:
        v_new = np.zeros((1, (d + 2) * r_kp))
        # Calculate next step of contraction directly summing up Kronecker product:
        for ii in range(nk):
            # Compile slice ii of first core Lpsi^1(x):
            Lpsi_1 = np.zeros((1, d + 2))
            Lpsi_1[0, 0] = psi_kx[ii]
            Lpsi_1[0, 1] = Lpsi_kx[ii]
            Lpsi_1[0, 2:] = psi_grad_sigma[ii, :]
            # Form Kronecker product with orthonormal core U^k:
            v_new += np.kron(Lpsi_1, u_k[:, ii, :])

    return v_new


def _contraction_step_dPsi_u(psi_k, x, u_k, position='middle', v=None):
    """
    Helper function for the tensor network contraction \nabla \psi(x)^T \times \hat{U}, for
     one data point x, within reversible tgEDMD. The contraction is performed using
     Algorithm 3 in [1], leveraging the sparse structure of \nabla \psi. This function carries
     out a single step in Algorithm 3.

    Parameters
    ----------
    psi_k : list of callable function (derived from data_driven.transform.Function),
        representing the basis set for mode k.
    x : np.ndarray
        shape (d,): a single data point
    u_k:  ndarray, shape(r_km, n_k, r_kp)
        orthogonal core from the global SVD
    position : {'first', 'middle', 'last'}, optional
        first core: k = 1
        middle core: 2 <= k <= p-1
        last core: k = p
    v, ndarray, shape (1, (d+1) * r_km) or None.
        the output of the previous step of the contraction.
        v should be set to None if position=='first'

    Returns
    -------
    array of shape
            position == 'first':   (1, (d + 1) * r_kp)
            position == 'middle':   (1, (d + 1) * r_kp)
            position == 'last':   (1, d * r_kp)

        result of k-th step of the contraction
    """
    # Obtain dimensions of the data:
    d = x.shape[0]
    # Obtain shape of orthonormal core:
    r_km, nk, r_kp = u_k.shape

    # Pre-compute required quantities:
    # Evaluation of the basis set at x:
    psi_kx = np.array([psi_k[ii](x) for ii in range(nk)])
    # Gradients of the basis set at x:
    psi_grad = np.array([psi_k[ii].gradient(x) for ii in range(nk)])

    # Middle position:
    if position == 'middle':
        v = v.reshape((1, (d + 1), r_km))
        v_new = np.zeros((1, d + 1, r_kp))
        for ii in range(nk):
            # Multiply v and first column of dPsi^k(x) x U^k:
            v_new[0, 0, :] += np.dot(v[0, 0, :], psi_kx[ii] * u_k[:, ii, :])
            # Multiply v and all remaining columns of dPsi^k(x) x U^k:
            for jj in range(1, d + 1):
                v_new[0, jj, :] += (np.dot(v[0, 0, :], psi_grad[ii, jj - 1] * u_k[:, ii, :]) +
                                np.dot(v[0, jj, :], psi_kx[ii] * u_k[:, ii, :]))
        v_new = v_new.reshape((1, (d + 1) * r_kp))

    # End-of-chain position:
    elif position == 'last':
        v = v.reshape((1, (d + 1), r_km))
        v_new = np.zeros((1, d, r_kp))
        for ii in range(nk):
            # Multiply v and all columns of dPsi^k(x) x U^k:
            for jj in range(d):
                v_new[0, jj, :] += (np.dot(v[0, 0, :], psi_grad[ii, jj] * u_k[:, ii, :]) +
                                     np.dot(v[0, jj + 1, :], psi_kx[ii] * u_k[:, ii, :]))
        v_new = v_new.reshape((1, d * r_kp))

    # Head-of-chain position:
    else:
        v_new = np.zeros((1, (d + 1) * r_kp))
        # Calculate next step of contraction directly summing up Kronecker product:
        for ii in range(nk):
            # Compile slice ii of first core dpsi^1(x):
            dpsi_ii = np.zeros((1, d + 1))
            dpsi_ii[0, 0] = psi_kx[ii]
            dpsi_ii[0, 1:] = psi_grad[ii, :]
            # Form Kronecker product with orthonormal core U^k:
            v_new += np.kron(dpsi_ii, u_k[:, ii, :])

    return v_new
#=====================================

def _special_kron_reversible(dPsi, B):
    """
    Build Matrices M_new = M_k^{(i)} from AMUSEt.
    dPsi and B are TT cores (np.ndarray with order 4).
    dPsi needs to have the special structure of the dPsi(x) cores.

    Parameters
    ----------
    dPsi : np.ndarray
    B : np.ndarray

    Returns
    -------
    np.ndarray
        shape = (dPsi.shape[0], dPsi.shape[3], B.shape[0], B.shape[3])
    """

    C = np.zeros((dPsi.shape[0], dPsi.shape[3], B.shape[0], B.shape[3]))

    if dPsi.shape[0] == 1 or dPsi.shape[3] == 1:  # dPsi is a vector
        for i in range(dPsi.shape[0]):
            for j in range(dPsi.shape[3]):
                C[i, j, :, :] = np.tensordot(dPsi[i, :, 0, j], B[:, :, 0, :], axes=(0, 1))
    else:  # dPsi is a matrix

        if dPsi.shape[0] == dPsi.shape[3]:  # square core (1,...,p-1)
            # diagonal
            psi = dPsi[0, :, 0, 0]
            psi = np.tensordot(psi, B[:, :, 0, :], axes=(0, 1))
            for i in range(C.shape[0]):
                C[i, i, :, :] = psi

            # first row
            for i in range(1, C.shape[1]):
                C[0, i, :, :] = np.tensordot(dPsi[0, :, 0, i], B[:, :, 0, :], axes=(0, 1))
        else:  # core p
            # diagonal
            psi = dPsi[1, :, 0, 0]
            psi = np.tensordot(psi, B[:, :, 0, :], axes=(0, 1))
            for i in range(C.shape[0] - 1):
                C[1 + i, i, :, :] = psi

            # first row
            for i in range(0, C.shape[1]):
                C[0, i, :, :] = np.tensordot(dPsi[0, :, 0, i], B[:, :, 0, :], axes=(0, 1))

    return C


def _special_tensordot_reversible(A, B):
    """
    Tensordot between arrays A and B with special structure.
    A and B have the structure that arises in the cores of dPsi. As A and B can be the result of a Kronecker product,
    the entries of A and B can be matrices themselves. Thus A and B are modeled as 4D Arrays where the first and second
    index refer to the rows and columns of A and B. The third and fourth index refer to the rows and colums of the
    entries of A and B.
    All nonzero elements of A and B are
    in the diagonal [i,i,:,:], in the first row [0,:,:,:] and in the second column [:,1,:,:].
    Furthermore A and B are quadratic (A.shape[0] = A.shape[1]) or A has only one row or B only one column.
    The tensordot is calculated along both column dimensions of A (1,3) and both row dimensions of B (0,2).
    The resulting array has the same structure as A and B.

    Parameters
    ----------
    A : np.ndarray
    B : np.ndarray

    Returns
    -------
    np.ndarray
        tensordot between A and B along the axis ((1,3), (0,2))
    """
    C = np.zeros((A.shape[0], B.shape[1], A.shape[2], B.shape[3]))

    if B.shape[0] == B.shape[1]:
        # diagonal
        for i in range(min(A.shape[0], B.shape[1])):
            C[i, i, :, :] = A[i, i, :, :].dot(B[i, i, :, :])

        # first row
        for i in range(1, B.shape[0]):
            C[0, i, :, :] = A[0, 0, :, :].dot(B[0, i, :, :]) + A[0, i, :, :].dot(B[i, i, :, :])
    else:
        # diagonal
        if A.shape[0] > 1:
            for i in range(B.shape[1]):
                C[i + 1, i, :, :] = A[i + 1, i + 1, :, :].dot(B[i + 1, i, :, :])

        # first row
        for i in range(0, B.shape[0] - 1):
            C[0, i, :, :] = A[0, 0, :, :].dot(B[0, i, :, :]) + A[0, i + 1, :, :].dot(B[i + 1, i, :, :])

    return C
