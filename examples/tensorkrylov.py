from typing import List
import numpy as np
import scikit_tt
from numpy import linalg as la
from scipy.sparse import diags_array
from numpy.random import seed, rand
from scikit_tt.tensor_train import TT, build_core, residual_error, uniform, residual_error 
from scikit_tt.solvers.sle import als

class MatrixCollection(object):

    def __init__(self, A: List[np.ndarray]):

        self.order    = len(A)
        self.shapes   = [ A[s].shape[0] for s in range(self.order) ]
        self.matrices = A

        return

    def __getitem__(self, s: int):

        return self.matrices[s]


class KrusalTensor(object):

    def __init__(self, matrices: List[np.ndarray], gamma: np.ndarray):

        self.fmatrices = matrices
        self.gamma     = gamma

        return

class TensorLanczos(object):

    def __init__(self, A: "MatrixCollection", b: List[np.ndarray]):

        self.A = A
        self.V = MatrixCollection([np.zeros((A.shapes[s], A.shapes[s] + 1)) for s in range(A.order)])
        self.T = MatrixCollection([np.zeros((A.shapes[s] + 1, A.shapes[s] + 1)) for s in range(A.order)])

        # Perform first Lanczos step
        for s in range(A.order):

            self.V[s][:, 0] = b[s] / la.norm(b[s]) 
            v               = self.V[s][:, 0]
            u               = A[s] @ v
            self.T[s][0, 0] = np.dot(u, v)
            u              -= (self.T[s][0, 0] * v)
            beta            = la.norm(u)
            self.V[s][:, 1] = u / beta
            self.T[s][[1,0], [0, 1]] = beta

        return
        

def lanczos_step(A, V, H, k: int): 

    beta = H[k - 1, k]

    u = A @ V[:, k] - beta * V[:, k - 1]

    H[k, k] = np.dot(u, V[:, k])

    u   -= (H[k, k] * V[:, k])

    beta = la.norm(u)

    if beta == 0.0:

        V[:, k + 1] = np.zeros(A.shape[0])

    else:

        V[:, k + 1] = (1 / beta) * u 

    H[[k + 1, k], [k, k + 1]] = beta

    return

def _tensor_lanczos_step(tensor_lanczos: "TensorLanczos", k: int):

    for s in range(tensor_lanczos.A.order):

        lanczos_step(tensor_lanczos.A[s], 
                     tensor_lanczos.V[s],
                     tensor_lanczos.T[s],
                     k)
    return


def _principal_minors(v: List[np.ndarray], i: int, j = None):
    
    if j == None:

        if v[0].ndim == 1: # Vector

            return [ v[s][0:i+1] for s in range(len(v)) ]

        else: # Matrix
            
            return MatrixCollection([v[s][0:i+1, 0:i+1] for s in range(v.order)])

    else: 

        return MatrixCollection([ v[s][0:i+1, 0:j+1] for s in range(v.order)])

def _update_rhs(comp_rhs: List[np.ndarray], V: "MatrixCollection", rhs: List[np.ndarray], k: int):

    columns = [ V[s][:, k] for s in range(V.order)]

    for s in range(V.order):

        comp_rhs[s][k] = np.dot(columns[s], rhs[s])

    return

def _initialize_compressed_rhs(b: List[np.ndarray], V: "MatrixCollection"):

    b_tilde  = [ np.zeros( b[s].shape ) for s in range(len(b))]
    b_minors = _principal_minors(b_tilde, 0) 
    _update_rhs(b_minors, V, b, 0)

    return b_tilde

def _TT_operator(A: MatrixCollection, k: int):

    identity = np.eye(k + 1)

    cores    = [None] * A.order
    cores[0] = build_core([ [A[0], identity] ])
    
    for s in range(1, A.order - 1):

        cores[s] = build_core([[identity, 0], [A[s], identity]])

    cores[-1] = build_core([ [identity], [A[-1]] ] )

    return TT(cores)

def _TT_rhs(rhs: List[np.ndarray]):

    cores = [np.zeros((1, rhs[s].shape[0], 1, 1)) for s in range(len(rhs))]

    for s in range(len(rhs)):

        cores[s][0, :, 0, 0] = rhs[s]

    return TT(cores)

def _TT_krylov_approximation(rank: int, shapes: List[int], d: int):

    cores = [None] * d

    cores[0] = np.zeros((1, shapes[0], 1, rank))

    for s in range(1, d - 1):

        cores[s] = np.zeros((rank, shapes[s], 1, rank))

    cores[-1] = np.zeros((rank, shapes[-1], 1, 1))

    return TT(cores)

def _update_approximation(x_TT: "TT", V: "MatrixCollection", y_TT: "TT"):

    for s in range(x_TT.order):

        x_TT.cores[s] = np.sum(V[s][None, :, :, None, None] @ y_TT.cores[s][:, None, :, :, :], axis = 2)

    return


def symmetric_tensorkrylov(A: "MatrixCollection", b: List[np.ndarray], rank: int, nmax: int, tol = 1e-9):

    d      = A.order
    n      = A.shapes[0]
    b_norm = np.prod([la.norm(b[s]) for s in range(d)])
    TT_solution_shape = [n for _ in range(d)]
    col_dims       = [1 for _ in range(d)]
    solution_ranks = [1] + [rank for _ in range(d - 1)] + [1]

    tensor_lanczos = TensorLanczos(A, b)

    b_tilde = _initialize_compressed_rhs(b, tensor_lanczos.V)
    r_norm = np.inf

    A_TT = _TT_operator(A, n - 1)
    b_TT = _TT_rhs(b)

    x_TT = _TT_krylov_approximation(rank, TT_solution_shape, d)

    for k in range(1, nmax):

        _tensor_lanczos_step(tensor_lanczos, k)

        print("Iteration : ", k)

        T_minors = _principal_minors(tensor_lanczos.T, k)
        V_minors = _principal_minors(tensor_lanczos.V, n, k)
        b_minors = _principal_minors(b_tilde, k)

        _update_rhs(b_minors, tensor_lanczos.V, b, k)

        TT_operator = _TT_operator(T_minors, k)
        TT_rhs      = _TT_rhs(b_minors)

        row_dims = [k + 1 for _ in range(d)]
        TT_guess = scikit_tt.tensor_train.rand(
            row_dims, 
            col_dims,
            solution_ranks)

        y_TT = als(TT_operator, TT_guess, TT_rhs)
        _update_approximation(x_TT, V_minors, y_TT)
        print(A_TT)
        print(x_TT)
        print(b_TT)
        r_norm = residual_error(A_TT, x_TT, b_TT)

        if r_norm <= tol:

            return x_TT

    return r_norm

def random_rhs(n: int):

    rhs  = rand(n)
    rhs *= (1 / la.norm(rhs)) 

    return rhs

seed(1234)
d  = 5
n  = 200
As = diags_array([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], offsets=[-1, 0, 1]).todense()
bs = random_rhs(n)

A = MatrixCollection([ As for _ in range(d) ])
b = [ bs for _ in range(d) ]
rank = 5
ranks = [1] + ([rank] * (d - 1)) + [1]


print(symmetric_tensorkrylov(A, b, rank, n, tol = 1e-9))
