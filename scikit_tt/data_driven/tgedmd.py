import numpy as np
from scikit_tt.tensor_train import TT
import scikit_tt.utils as utl

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


def amuset_hosvd(data_matrix, basis_list, b, sigma, num_eigvals=np.infty, threshold=1e-2, max_rank=np.infty,
                 return_option='eigenfunctionevals'):
    """
    AMUSEt algorithm for the calculation of eigenvalues of the Koopman generator.
    The tensor-trains are created using the exact TT decomposition, whose ranks are reduced using SVDs.
    An efficient implementation of tensor contractions that exploits the special structure of the cores is used.

    Parameters
    ----------
    data_matrix : np.ndarray
        snapshot matrix, shape (d, m)
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : np.ndarray
        drift, shape (d, m)
    sigma : np.ndarray
        diffusion, shape (d, d2, m)
    num_eigvals : int, optional
        number of eigenvalues and eigentensors that are returned
        default: return all calculated eigenvalues and eigentensors
    threshold : float, optional
        threshold for svd of psi
    max_rank : int, optional
        maximal rank of TT representations of psi after svd/ortho
    return_option : {'eigentensors', 'eigenfunctionevals', 'eigenvectors'}
        'eigentensors': return a list of the eigentensors of the koopman generator
        'eigenfunctionevals': return the evaluations of the eigenfunctions of the koopman generator at all snapshots
        'eigenvectors': eigenvectors of M in AMUSEt

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
    s, v = None, None

    residual = np.ones((1, m))
    r_i = residual.shape[0]
    for i in range(len(basis_list)):
        core_tmp = np.zeros((r_i, n[i], m))
        for j in range(m):
            psi_jk = np.array([basis_list[i][k](data_matrix[:, j]) for k in range(n[i])])
            core_tmp[:, :, j] = np.outer(residual[:, j], psi_jk)
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
    M = _amuset_efficient(U, s, v.T, data_matrix, basis_list, b, sigma)

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
        eigenfunctionevals = eigvecs.T @ v
        return eigvals, eigenfunctionevals

    else:
        return eigvals, eigvecs


def amuset_hosvd_reversible(data_matrix, basis_list, sigma, num_eigvals=np.infty, threshold=1e-2, max_rank=np.infty,
                            return_option='eigenfunctionevals'):
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
    s, v = None, None

    residual = np.ones((1, m))
    r_i = residual.shape[0]
    for i in range(len(basis_list)):
        core_tmp = np.zeros((r_i, n[i], m))
        for j in range(m):
            psi_jk = np.array([basis_list[i][k](data_matrix[:, j]) for k in range(n[i])])
            core_tmp[:, :, j] = np.outer(residual[:, j], psi_jk)
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
    M = _amuset_efficient_reversible(U, s, data_matrix, basis_list, sigma)

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
        eigenfunctionevals = eigvecs.T @ v
        return eigvals, eigenfunctionevals
    else:
        return eigvals, eigvecs


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
def _amuset_efficient_reversible(u, s, x, basis_list, sigma):
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

    dPsi = _tt_decomposition_one_snapshot_reversible(x[:, k], basis_list, sigma[:, :, k])
    M = _calc_M_k_amuset_reversible(u, s_inv, dPsi)

    while k + 1 < m:
        k = k + 1
        if k / m > next_print:
            print('progress: {}%'.format(int(round(next_print * 100))))
            next_print += 0.1
        dPsi = _tt_decomposition_one_snapshot_reversible(x[:, k], basis_list, sigma[:, :, k])
        M += _calc_M_k_amuset_reversible(u, s_inv, dPsi)

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
