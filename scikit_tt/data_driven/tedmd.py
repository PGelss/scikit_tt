# -*- coding: utf-8 -*-

import scikit_tt.data_driven.transform as tdt
from scikit_tt.data_driven.transform import Function
import numpy as np
from scipy import linalg
from typing import List, Tuple, Union
from scikit_tt.tensor_train import TT
import scikit_tt.utils as utl



def amuset_hosvd(data_matrix: np.ndarray, 
                 x_indices: np.ndarray,
                 y_indices: np.ndarray, 
                 basis_list: List[List[Function]],
                 threshold: float=1e-2,
                 max_rank: int=np.inf, 
                 progress: bool=False,
                 ef_tf: bool=False,
                 st_tf: bool=False) -> Tuple[np.ndarray, Union['TT', List['TT']]]:
    """
    AMUSEt (AMUSE on tensors) using HOSVD.

    Apply tEDMD to a given data matrix by using AMUSEt with HOSVD. This procedure is a tensor-based
    version of AMUSE using the tensor-train format. For more details, see [1]_.

    Parameters
    ----------
    data_matrix : np.ndarray
        snapshot matrix

    x_indices : np.ndarray or list[np.ndarray]
        index sets for snapshot matrix x
        
    y_indices : np.ndarray or list[np.ndarray]
        index sets for snapshot matrix y
        
    basis_list : list[list[Function]]
        list of basis functions in every mode
        
    threshold : float, optional
        threshold for SVD/HOSVD, default is 1e-2
        
    max_rank : int
        maximum rank of truncated SVD
        
    progress : boolean, optional
        whether to show progress bar, default is False
        
    et_tf: boolean, optional
        if True, return eigenfunctions evaluated at snapshots
        
    st_tf: boolean, optional
        if True, return singular tensors

    Returns
    -------
    eigenvalues : np.ndarray or list[np.ndarray]
        tEDMD eigenvalues
        
    eigentensors : TT or list[TT]
        tEDMD eigentensors in TT format

    References
    ----------
    ..[1] F. Nüske, P. Gelß, S. Klus, C. Clementi. "Tensor-based EDMD for the Koopman analysis of high-dimensional
          systems", arXiv:1908.04741, 2019
    """

    # define quantities
    eigenvalues = []
    eigentensors = []
    cores = [None] * (len(basis_list)+1)

    # number of snapshots
    m = data_matrix.shape[1]

    # number of modes
    p = len(basis_list)

    # mode dimensions
    n = [len(basis_list[i]) for i in range(p)]

    residual = np.ones((1, m))
    r_i = residual.shape[0]

    for i in range(len(basis_list)):
        # Directly evaluate tensor product between residual and basis for mode i:

        core_tmp = np.zeros((r_i, n[i], m))

        for j in range(m):
            psi_kj = np.array([basis_list[i][k](data_matrix[:, j]) for k in range(n[i])])
            core_tmp[:, :, j] = np.outer(residual[:, j], psi_kj)

        # Truncated SVD:
        u, s, v = utl.truncated_svd(core_tmp.reshape([core_tmp.shape[0]*core_tmp.shape[1], core_tmp.shape[2]]),
                                    threshold=threshold, max_rank=max_rank)
        cores[i] = u.reshape([core_tmp.shape[0], core_tmp.shape[1], 1, u.shape[1]])
        residual = np.diag(s).dot(v)
        r_i      = residual.shape[0]

    cores[-1] = residual.reshape([residual.shape[0], residual.shape[1], 1, 1])
    psi       = TT(cores)


    # # construct transformed data tensor in TT format using direct approach
    # psi = tdt.basis_decomposition(data_matrix, basis_list)

    # # left-orthonormalization
    # psi = psi.ortho_left(threshold=threshold, progress=progress)

    # extract last core
    last_core = psi.cores[-1]

    # convert x_indices and y_indices to lists
    if not isinstance(x_indices, list):
        x_indices = [x_indices]
        y_indices = [y_indices]

    # loop over all index sets
    for i in range(len(x_indices)):
        # compute reduced matrix
        matrix, u, s, v = _reduced_matrix(last_core, x_indices[i], y_indices[i])

        # solve reduced eigenvalue problem
        eigenvalues_reduced, eigenvectors_reduced = np.linalg.eig(matrix)
        idx = (np.abs(eigenvalues_reduced - 1)).argsort()
        
        if ef_tf:
            eigenfunctions = v.T.dot(eigenvectors_reduced)
            
        if st_tf:
            u_reduced, s_reduced, v_reduced = np.linalg.svd(matrix)
            left_singtensors = psi
            left_singtensors.cores[-1] = u.dot(np.diag(np.reciprocal(s))).dot(u_reduced)[:,:,None,None]
            singvalues = s_reduced

        eigenvalues_reduced  = np.real(eigenvalues_reduced[idx])
        eigenvectors_reduced = np.real(eigenvectors_reduced[:, idx])

        # construct eigentensors
        eigentensors_tmp = psi
        eigentensors_tmp.cores[-1] = u.dot(np.diag(np.reciprocal(s))).dot(eigenvectors_reduced)[:, :, None, None]
        eigentensors_tmp.row_dims[-1] = eigentensors_tmp.cores[-1].shape[1]

        # append results
        eigenvalues.append(eigenvalues_reduced)
        eigentensors.append(eigentensors_tmp)

    # only return lists if more than one set of x-indices/y-indices was given
    if len(x_indices) == 1:
        eigenvalues  = eigenvalues[0]
        eigentensors = eigentensors[0]

    if ef_tf and st_tf:
        
        return eigenvalues, eigentensors, eigenfunctions, singvalues, left_singtensors
    
    elif ef_tf:
        
        return eigenvalues, eigentensors, eigenfunctions
    
    elif st_tf:
        
        return eigenvalues, eigentensors, singvalues, left_singtensors
        
    else:
        return eigenvalues, eigentensors


def amuset_hocur(data_matrix: np.ndarray,
                 x_indices: Union[np.ndarray, List[np.ndarray]], 
                 y_indices: Union[np.ndarray, List[np.ndarray]],
                 basis_list: List[List[Function]], 
                 max_rank: int=1000, multiplier: int=2, progress: bool=False
                 ) -> Tuple[Union[np.ndarray, List[np.ndarray]], Union['TT', List['TT']]]:
    """
    AMUSEt (AMUSE on tensors) using HOCUR.

    Apply tEDMD to a given data matrix by using AMUSEt with HOCUR. This procedure is a tensor-based
    version of AMUSE using the tensor-train format. For more details, see [1]_.

    Parameters
    ----------
    data_matrix : np.ndarray
        snapshot matrix
        
    x_indices : np.ndarray or list[np.ndarray]
        index sets for snapshot matrix x
        
    y_indices : np.ndarray or list[np.ndarray]
        index sets for snapshot matrix y
        
    basis_list : list[list[Function]]
        list of basis functions in every mode
        
    max_rank : int, optional
        maximum ranks for HOSVD as well as HOCUR, default is 1000
        
    multiplier : int
        multiplier for HOCUR
        
    progress : boolean, optional
        whether to show progress bar, default is False

    Returns
    -------
    eigenvalues : np.ndarray or list[np.ndarray]
        tEDMD eigenvalues
        
    eigentensors : TT or list[TT]
        tEDMD eigentensors in TT format

    References
    ----------
    ..[1] F. Nüske, P. Gelß, S. Klus, C. Clementi. "Tensor-based EDMD for the Koopman analysis of high-dimensional
          systems", arXiv:1908.04741, 2019
    """

    # define quantities
    eigenvalues = []
    eigentensors = []

    # construct transformed data tensor in TT format using HOCUR
    psi = tdt.hocur(data_matrix, basis_list, max_rank, repeats=1, multiplier=multiplier, progress=progress)

    # left-orthonormalization
    psi = psi.ortho_left(progress=progress)

    # extract last core
    last_core = psi.cores[-1]

    # convert x_indices and y_indices to lists
    if not isinstance(x_indices, list):
        x_indices = [x_indices]
        y_indices = [y_indices]

    # loop over all index sets
    for i in range(len(x_indices)):
        # compute reduced matrix
        matrix, u, s, v = _reduced_matrix(last_core, x_indices[i], y_indices[i])

        # solve reduced eigenvalue problem
        eigenvalues_reduced, eigenvectors_reduced = np.linalg.eig(matrix)
        idx = (np.abs(eigenvalues_reduced - 1)).argsort()
        eigenvalues_reduced = np.real(eigenvalues_reduced[idx])
        eigenvectors_reduced = np.real(eigenvectors_reduced[:, idx])

        # construct eigentensors
        eigentensors_tmp = psi
        eigentensors_tmp.cores[-1] = u.dot(np.diag(np.reciprocal(s))).dot(eigenvectors_reduced)[:, :, None, None]

        # append results
        eigenvalues.append(eigenvalues_reduced)
        eigentensors.append(eigentensors_tmp)

    # only return lists if more than one set of x-indices/y-indices was given
    if len(x_indices) == 1:
        eigenvalues = eigenvalues[0]
        eigentensors = eigentensors[0]

    return eigenvalues, eigentensors


def _reduced_matrix(last_core: np.ndarray,
                    x_indices: np.ndarray, 
                    y_indices: np.ndarray, 
                    threshold: float=1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reduced matrix for AMUSEt.

    Parameters
    ----------
    last_core : np.ndarray
        last TT core of left-orthonormalized psi_z
        
    x_indices : np.ndarray
        index set for snapshot matrix x
        
    y_indices : np.ndarray
        index set for snapshot matrix y
        
    threshold : float, optional
        threshold for SVD, default is 1e-4

    Returns
    -------
    matrix : np.ndarray
        reduced matrix
        
    u : np.ndarray
        left-orthonormal matrix of the SVD of the last core of psi_x
        
    s : np.ndarray
        vector of singular values of the SVD of the last core of psi_x
        
    v : np.ndarray
        right-orthonormal matrix of the SVD of the last core of psi_x
    """

    # extract last cores of psi_x and psi_y
    psi_x_last = np.squeeze(last_core[:, x_indices, :, :])
    psi_y_last = np.squeeze(last_core[:, y_indices, :, :])

    # decompose last core of psi_x
    u, s, v = linalg.svd(psi_x_last, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
    indices = np.where(s / s[0] > threshold)[0]
    u = u[:, indices]
    s = s[indices]
    v = v[indices, :]

    # construct reduced matrix
    matrix = v.dot(psi_y_last.T).dot(u).dot(np.diag(np.reciprocal(s)))

    return matrix, u, s, v
