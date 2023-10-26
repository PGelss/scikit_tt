import numpy as np
from scipy.sparse import spdiags, issparse, kron
from scipy.sparse.linalg import spsolve
from scikit_tt.solvers.sle import als
from scikit_tt.tensor_train import rand, TT

# Parameters
d = 2
n = 200

# Matrices
diags = np.array([-1 * np.ones(n), 2 * np.ones(n), -1 * np.ones(n)])
As    = spdiags(diags, [-1, 0, 1], (n,n))
b     = [ np.random.rand(n) for _ in range(d) ]

def initialize_operator(As, d):

    n = As.shape[0]

    # Tensor train
    cores = [None] * d

    cores[0] = np.zeros((1, n, n, 2))
    cores[0][0, :, :, 0] = As
    cores[0][0, :, :, 1] = np.identity(n)

    for i in range(1, d - 1):

        cores[i] = np.zeros((2, n, n, 2))
       
        cores[i][0, :, :, 0] = np.identity(n)
        cores[i][0, :, :, 1] = np.zeros((n, n))
        cores[i][1, :, :, 0] = As
        cores[i][1, :, :, 1] = np.identity(n)
    
    cores[d - 1] = np.zeros((2, n, n, 1))

    cores[d - 1][0, :, :, 0] = np.identity(n)
    cores[d - 1][1, :, :, 0] = As

    return TT(cores)

def initialize_rhs(b, d):

    #cores = [ np.zeros((1, 1, n, 1)) for _ in range(d) ]
    cores = [ np.zeros((1, n, 1, 1)) for _ in range(d) ]

    for i in range(0, d):

        cores[i][0, :, 0, 0] = b[i]
        
    return TT(cores)

def recursive_kronecker(As, n):

    return As

A_sparse = kron(As, np.identity(n)) + kron(np.identity(n), As) 
x_exact  = spsolve(A_sparse, np.kron(b[0], b[1]))

ranks = [1] + ([3] * (d - 1)) + [1]

As_dense = As.toarray()
A_TT     = initialize_operator(As_dense, d)
b_TT     = initialize_rhs(b, d)

x_init   = rand([n] * d, [1] * d, ranks)
#x_init   = rand([n] * d, [1] * d)
print(x_init)
x_als    = als(A_TT, x_init, b_TT, 190)

error = np.linalg.norm(x_exact - x_als.full().reshape(n * n)) / np.linalg.norm(x_exact)

print("Relative error: ", error)




