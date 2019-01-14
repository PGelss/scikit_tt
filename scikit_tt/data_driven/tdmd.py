# -*- coding: utf-8 -*-
import scikit_tt.tensor_train as tt
import numpy as np
import scipy.linalg as lin
def standard(x, y, threshold=0, ortho_l=True, ortho_r=True):
	x = x.pinv(x.order-1, threshold=threshold, ortho_l=ortho_l, ortho_r=ortho_r)
	reduced_matrix = __reduced_matrix(x,y)
	eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)
	ind = np.argsort(eigenvalues)[::-1]
	dmd_eigenvalues = eigenvalues[ind]
	dmd_modes = x
	dmd_modes.cores[-1] = eigenvectors[:, ind, None, None]
	dmd_modes.row_dims[-1] = len(ind)
	return dmd_eigenvalues, dmd_modes


def exact(x, y, threshold=0, ortho_l=True, ortho_r=True):
	x = x.pinv(x.order-1, threshold=threshold, ortho_l=ortho_l, ortho_r=ortho_r)
	reduced_matrix = __reduced_matrix(x,y)
	eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)
	dmd_eigenvalues = eigenvalues
	ind = np.argsort(eigenvalues)[::-1]
	dmd_eigenvalues = eigenvalues[ind]
	dmd_modes = y.copy()
	dmd_modes.cores[-1] = y.cores[-1][:,:,0,0] @ x.cores[-1][:,:,0,0].T @ eigenvectors[:, ind] @ np.diag(np.reciprocal(dmd_eigenvalues))
	dmd_modes.row_dims[-1] = len(ind)
	return dmd_eigenvalues, dmd_modes

def __reduced_matrix(x, y):
	z = x.transpose() @ y
	m = z.cores[0][:,0,0,:]
	for i in range(1, x.order-1):
		m = m @ z.cores[i][:,0,0,:]
	m = m.reshape([x.ranks[-2], y.ranks[-2]])
	m = m @ y.cores[-1][:,:,0,0] @ x.cores[-1][:,:,0,0].T
	return m
