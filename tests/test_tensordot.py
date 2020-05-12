from scikit_tt.tensor_train import TT
import numpy as np


"""
Running this file showcases the use of TT.tensordot. Furthermore the correctness of TT.tensordot is shown by performing
5 tests with random tensors and comparing the results to numpy.tensordot. The tests are
1. contraction over 1 axis
2. contraction over multiple axis
3. complete contraction over the first tensor
4. complete contraction over the second tensor
5. complete contraction over both
"""


def simple_test_case():
    # initialize random tensors
    T = np.random.random((2, 3, 4, 5))
    U = np.random.random((4, 5, 6))

    # calculate tensordot using numpy
    TU = np.tensordot(T, U, axes=([2, 3], [0, 1]))

    # convert to TT format
    T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
    print('T:')
    print(T_tt)
    U_tt = TT(np.reshape(U, (4, 5, 6, 1, 1, 1)))
    print('\n\nU:')
    print(U_tt)

    # calculate tensordot in TT format
    T_tt.tensordot(U_tt, 2)
    print('\n\ntensordot <T,U>:')
    print(T_tt)

    # convert back to full tensor format and compare the results
    TU_back = np.squeeze(T_tt.full())
    print('\n\nerror: {}'.format(np.linalg.norm(TU_back - TU)))


def check_1():
    # contraction over 1 axis
    T = np.random.random((2, 3, 4, 5))
    U = np.random.random((5, 6, 7))
    TU = np.tensordot(T, U, axes=([3], [0]))

    T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
    U_tt = TT(np.reshape(U, (5, 6, 7, 1, 1, 1)))
    T_tt.tensordot(U_tt, 1)

    TU_back = np.squeeze(T_tt.full())
    error = np.linalg.norm(TU_back - TU)
    if error < 1e-8:
        return True
    return False


def check_2():
    # contraction over multiple axis
    T = np.random.random((2, 3, 4, 5))
    U = np.random.random((3, 4, 5, 6, 7))
    TU = np.tensordot(T, U, axes=([1, 2, 3], [0, 1, 2]))

    T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
    U_tt = TT(np.reshape(U, (3, 4, 5, 6, 7, 1, 1, 1, 1, 1)))
    T_tt.tensordot(U_tt, 3)

    TU_back = np.squeeze(T_tt.full())
    error = np.linalg.norm(TU_back - TU)
    if error < 1e-8:
        return True
    return False


def check_3():
    # contraction over full T
    T = np.random.random((3, 4, 5))
    U = np.random.random((3, 4, 5, 6, 7))
    TU = np.tensordot(T, U, axes=([0, 1, 2], [0, 1, 2]))

    T_tt = TT(np.reshape(T, (3, 4, 5, 1, 1, 1)))
    U_tt = TT(np.reshape(U, (3, 4, 5, 6, 7, 1, 1, 1, 1, 1)))
    T_tt.tensordot(U_tt, 3)

    TU_back = np.squeeze(T_tt.full())
    error = np.linalg.norm(TU_back - TU)
    if error < 1e-8:
        return True
    return False


def check_4():
    # contraction over full U
    T = np.random.random((2, 3, 4, 5))
    U = np.random.random((3, 4, 5))
    TU = np.tensordot(T, U, axes=([1, 2, 3], [0, 1, 2]))

    T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
    U_tt = TT(np.reshape(U, (3, 4, 5, 1, 1, 1)))
    T_tt.tensordot(U_tt, 3)

    TU_back = np.squeeze(T_tt.full())
    error = np.linalg.norm(TU_back - TU)
    if error < 1e-8:
        return True
    return False


def check_5():
    # contraction over both full
    T = np.random.random((3, 4, 5))
    U = np.random.random((3, 4, 5))
    TU = np.tensordot(T, U, axes=([0, 1, 2], [0, 1, 2]))

    T_tt = TT(np.reshape(T, (3, 4, 5, 1, 1, 1)))
    U_tt = TT(np.reshape(U, (3, 4, 5, 1, 1, 1)))
    T_tt.tensordot(U_tt, 3)

    TU_back = np.squeeze(T_tt.full())
    error = np.linalg.norm(TU_back - TU)
    if error < 1e-8:
        return True
    return False


if __name__ == '__main__':
    simple_test_case()
    print('\n\n--------------------------------------------\nperforming some tests...')
    print('check_1 successful: {}'.format(check_1()))
    print('check_2 successful: {}'.format(check_2()))
    print('check_3 successful: {}'.format(check_3()))
    print('check_4 successful: {}'.format(check_4()))
    print('check_5 successful: {}'.format(check_5()))
