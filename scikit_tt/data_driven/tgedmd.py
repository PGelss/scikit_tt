import numpy as np
from typing import List, Union, Tuple
from scikit_tt.data_driven.transform import Function
from scikit_tt.tensor_train import TT

import scikit_tt.utils as utl


# This file contains methods related to tensor-based generator EDMD (tgEDMD) introduced in [1].

# Structure of this file:
# 1. main api functions
#       amuset_hosvd : unified api function to apply the tgEDMD method (algorithm 2 in [1])
#       generator_on_product : Evaluate the action of the Koopman generator on a product of functions (line 234)
#       generator_on_product_reversible : Analog to generator_on_product in the reversible case, as it can be used
#                                         to calculate the entries of dPsi(x) (line 280)
# 2. private functions

# References:
# [1] Lücke, M. and Nüske, F. tgEDMD: Approximation of the Kolmogorov Operator in Tensor Train Format,
#       arxiv 2111.09606 (2022)
# [2] Nüske, F and Gelß, P and Klus, S, and Clementi, C. Tensor-based computation of metastable and
#  coherent sets, Physica D: Nonlinear Phenomena 427, 133018 (2021)


def amuset_hosvd(data_matrix: np.ndarray, basis_list: List[List[Function]],
                 sigma: np.ndarray, b: np.ndarray=None, reweight: np.ndarray=None,
                 num_eigvals: int=np.inf, threshold: float=1e-2,
                 max_rank: int=np.inf, 
                 return_option: str='eigenfunctionevals', 
                 output_freq: int=None,
                 rel_threshold: bool=False) -> Tuple[np.ndarray, Union[List['TT'], np.ndarray], list]: 

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

    num_eigvals : int or np.inf
        number of eigenvalues and eigentensors that are returned
        default: return all calculated eigenvalues and eigentensors

    threshold : float
        absolute truncation threshold for singular values during global SVD of \PSI(X).
        Default value is 1e-2.

    max_rank : int or np.inf
        maximally allowed rank of orthonormal TT representation for data tensor \Psi(X).

    return_option : {'eigentensors', 'eigenfunctionevals', 'eigenvectors'}
        'eigentensors': return a list of the eigentensors of the Koopman generator, not implemented at this time
        'eigenfunctionevals': return the evaluations of the eigenfunctions of the Koopman generator at all snapshots
        'eigenvectors': return eigenvectors of reduced matrix M in tgEDMD.

    output_freq : int or None
        Display progress message every output_freq steps.

    rel_threshold : bool, optional
        If the above threshold is relative (w.r.t. largest singular value), or absolute

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
                                    threshold=threshold, max_rank=max_rank, rel_truncation=rel_threshold)

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


def generator_on_product(basis_list: List[List[Function]], 
                         s: tuple, 
                         x: np.ndarray, b: np.ndarray, sigma: np.ndarray) -> float:
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


def generator_on_product_reversible(basis_list: List[List[Function]],
                                    s: tuple, i: int, 
                                    x: np.ndarray, sigma: np.ndarray) -> float:
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



#================ private functions: =============

def _reduced_matrix_tgedmd(u: List['TT'], 
                           s_inv: np.ndarray, V: np.ndarray, 
                           ranks: list, x, basis_list: List[List[Function]],
                           sigma: np.ndarray, 
                           b: np.ndarray=None, reweight:np.ndarray=None,
                           output_freq: int=None) -> np.ndarray:
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


def _contraction_step_LPsi_u(psi_k: list, 
                             x: np.ndarray, bx: np.ndarray, sig_x: np.ndarray, 
                             u_k: np.ndarray, position: str="middle", 
                             v: np.ndarray=None) -> np.ndarray:
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


def _contraction_step_dPsi_u(psi_k: list, x: np.ndarray, u_k: np.ndarray, 
                             position: str='middle', v: np.ndarray=None
                             ) -> np.ndarray:
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

def _frob_inner(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _generator(f: Function, x: np.ndarray, b: np.ndarray, sigma: np.ndarray) -> float:
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
