from typing import List, Union
import numpy as np
from numpy import linalg as la
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

            self.V[s][:, 0] = (1 / la.norm(b[s])) * b[s]
            v               = self.V[s][:, 0]
            u               = A[s] @ v
            self.T[s][0, 0] = np.dot(u, v)
            u              -= self.T[s][0, 0] * v
            beta            = la.norm(u)
            self.V[s][:, 1] = v / beta
            self.T[s][[1,0], [0, 1]] = beta


        return
        

def lanczos_step(A, V, H, k: int): 

    beta = H[k - 1, k]

    u = A @ V[:, k] - beta * V[:, k - 1]

    H[k, k] = np.dot(u, V[:, k])

    v = u - (H[k, k] * V[:, k])

    beta = la.norm(v)

    if beta == 0.0:

        V[:, k + 1] = np.zeros(A.shape[0])

    else:

        V[:, k + 1] = (1 / beta) * v 

    H[[k + 1, k], [k, k + 1]] = beta

def _tensor_lanczos_step(tensor_lanczos: "TensorLanczos", k):

    for s in range(tensor_lanczos.A.order):

        lanczos_step(tensor_lanczos.A[s], 
                     tensor_lanczos.V[s],
                     tensor_lanczos.T[s],
                     k)
    return


def _principal_minors(v: List[np.ndarray], i: int, j = None):
    
    lv = len(v)

    if j == None:

        if v[0].ndim == 1: # Vector

            return [ v[s][0:i+1] for s in range(lv) ]

        else: # Matrix
            
            return MatrixCollection([v[s][0:i+1, 0:i+1] for s in range(lv)])

    else: 

        return MatrixCollection([ v[s][0:i+1, 0:j+1] for s in range(lv)])

def _update_rhs(comp_rhs: List[np.ndarray], V: "MatrixCollection", rhs: List[np.ndarray], k: int):

    columns = [ V[s][:, k] for s in range(V.order)]

    for s in range(V.order):

        comp_rhs[s][k] = np.dot(columns[s], rhs[s])

    return

def _initialize_compressed_rhs(b: List[np.ndarray], V: "MatrixCollection"):

    b_tilde  = [ np.zeros( b[s].shape ) for s in range(len(b))]
    b_minors = [ _principal_minors(b_tilde, 1) ]
    _update_rhs(b_minors, V, b, 1)

    return

def _TT_operator(A: MatrixCollection, k: int):

    identity = np.eye(k)

    cores    = [None] * A.order
    cores[0] = build_core([ A[0], identity ])
    
    for s in range(1, A.order - 1):

        cores[s] = build_core([[identity, 0], [A[s], identity]])

    cores[-1] = build_core([[identity], [A[-1]]] )

    return TT(cores)

def _TT_rhs(rhs: List[np.ndarray], k: int):

    cores = [np.zeros((1, rhs[s].shape[0], 1, 1)) for s in range(len(rhs))]

    for s in range(len(rhs)):

        cores[s][0, :, 0, 0] = rhs[s]

    return TT(cores)

#def _residual_norm(H, y, b):

    



def symmetric_tensorkrylov(A: "MatrixCollection", b: List[np.ndarray], rank: int, nmax: int, tol = 1e-9):

    d      = A.order
    n      = A.shapes[0]
    b_norm = np.prod([la.norm(b[s]) for s in range(d)])
    TT_solution_shape = [n for _ in range(A.order)]
    solution_ranks    = [rank for _ in range(A.order - 2)]

    tensor_lanczos = TensorLanczos(A, b)

    b_tilde = _initialize_compressed_rhs(b, tensor_lanczos.V)

    r_comp = np.inf
    r_norm = np.inf

    A_TT = _TT_operator(A)
    b_TT = _TT_operator(b)

    for k in range(1, nmax):

        _tensor_lanczos_step(tensor_lanczos, k)

        T_minors = _principal_minors(tensor_lanczos.T, k)
        V_minors = _principal_minors(tensor_lanczos.V, n, k)
        b_minors = _principal_minors(b_tilde, k)

        _update_rhs(b_minors, tensor_lanczos.V, b, k)

        TT_operator = _TT_operator(T_minors, k)
        TT_rhs      = _TT_rhs(b_minors, k)
        TT_guess    = uniform(TT_solution_shape, solution_ranks, 2.0)
        TT_solution = als(TT_operator, TT_guess, TT_rhs)
        #r_norm      = _residual_norm()
        #r_comp      = residual_error(TT_operator, TT_solution, TT_rhs)
        krylov_approximantion = 
        r_norm = residual_error(A_TT, krylov_approximantion, b_TT)




    




