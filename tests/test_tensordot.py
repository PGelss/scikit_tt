from scikit_tt.tensor_train import TT, rand
import numpy as np
import unittest as ut
from unittest import TestCase


"""
The correctness of TT.tensordot is shown by performing 5 tests with random tensors and comparing the results to
numpy.tensordot. The tests are
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
    TU_back = T_tt.tensordot(U_tt, 2)
    print('\n\ntensordot <T,U>:')
    print(TU_back)

    # convert back to full tensor format and compare the results
    TU_back = np.squeeze(TU_back.full())
    print('\n\nerror: {}'.format(np.linalg.norm(TU_back - TU)))


class TestTensordot(TestCase):
    def setUp(self):
        self.tol = 1e-8

    def test_exceptions(self):
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((5, 6, 7))
        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (5, 6, 7, 1, 1, 1)))

        with self.assertRaises(ValueError):
            T_tt.tensordot(U_tt, 1, mode='bla')

        with self.assertRaises(ValueError):
            T_tt.tensordot(U_tt, 1, mode='last-last')

        with self.assertRaises(ValueError):
            T_tt.tensordot(U_tt, 1, mode='firs-last')

        with self.assertRaises(ValueError):
            T_tt.tensordot(U_tt, 1, mode='first-first')

    def test_1_axis(self):
        # test contraction over 1 axis
        # last-first contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((5, 6, 7))
        TU = np.tensordot(T, U, axes=([3], [0]))

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (5, 6, 7, 1, 1, 1)))
        TU_back = T_tt.tensordot(U_tt, 1)

        TU_back = np.squeeze(TU_back.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-last contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((6, 7, 5))
        TU = np.tensordot(T, U, axes=([3], [2]))
        TU = np.transpose(TU, [0, 1, 2, 4, 3])

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (6, 7, 5, 1, 1, 1)))
        T_tt.tensordot(U_tt, 1, mode='last-last', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # first-last contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((6, 7, 2))
        TU = np.tensordot(T, U, axes=([0], [2]))
        TU = np.transpose(TU, [3, 4, 0, 1, 2])

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (6, 7, 2, 1, 1, 1)))
        T_tt.tensordot(U_tt, 1, mode='first-last', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # first-first contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((2, 6, 7))
        TU = np.tensordot(T, U, axes=([0], [0]))
        TU = np.transpose(TU, [4, 3, 0, 1, 2])

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (2, 6, 7, 1, 1, 1)))
        T_tt.tensordot(U_tt, 1, mode='first-first', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-first contraction for operator
        T_tt = rand([2, 3, 4], [5, 6, 7], ranks=3)
        U_tt = rand([4, 8, 9], [7, 8, 9], ranks=2)
        T = T_tt.full()
        U = U_tt.full()

        TU = np.tensordot(T, U, axes=([2, 5], [0, 3]))
        TU = np.transpose(TU, [0, 1, 4, 5, 2, 3, 6, 7])
        T_tt.tensordot(U_tt, 1, overwrite=True)
        TU_back = T_tt.full()
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

    def test_multiple_axes(self):
        # contraction over multiple axis
        # last-first contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((3, 4, 5, 6, 7))
        TU = np.tensordot(T, U, axes=([1, 2, 3], [0, 1, 2]))

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (3, 4, 5, 6, 7, 1, 1, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-last contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((6, 7, 4, 5))
        TU = np.tensordot(T, U, axes=([2, 3], [2, 3]))
        TU = np.transpose(TU, [0, 1, 3, 2])

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (6, 7, 4, 5, 1, 1, 1, 1)))
        TU_back = T_tt.tensordot(U_tt, 2, mode='last-last')

        TU_back = np.squeeze(TU_back.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # first-last contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((6, 7, 2, 3))
        TU = np.tensordot(T, U, axes=([0, 1], [2, 3]))
        TU = np.transpose(TU, [2, 3, 0, 1])

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (6, 7, 2, 3, 1, 1, 1, 1)))
        TU_back = T_tt.tensordot(U_tt, 2, mode='first-last')

        TU_back = np.squeeze(TU_back.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # first-first contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((2, 3, 6, 7))
        TU = np.tensordot(T, U, axes=([0, 1], [0, 1]))
        TU = np.transpose(TU, [3, 2, 0, 1])

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (2, 3, 6, 7, 1, 1, 1, 1)))
        TU_back = T_tt.tensordot(U_tt, 2, mode='first-first')

        TU_back = np.squeeze(TU_back.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-last contraction for operator
        T_tt = rand([3, 4, 5], [6, 7, 8], ranks=3)
        U_tt = rand([8, 9, 4, 5], [2, 3, 7, 8], ranks=2)
        T = T_tt.full()
        U = U_tt.full()

        TU = np.tensordot(T, U, axes=([1, 2, 4, 5], [2, 3, 6, 7]))
        TU = np.transpose(TU, [0, 3, 2, 1, 5, 4])
        T_tt.tensordot(U_tt, 2, mode='last-last', overwrite=True)
        TU_back = T_tt.full()
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

    def test_full_self(self):
        # contraction over full T
        # last-first contraction
        T = np.random.random((3, 4, 5))
        U = np.random.random((3, 4, 5, 6, 7))
        TU = np.tensordot(T, U, axes=([0, 1, 2], [0, 1, 2]))

        T_tt = TT(np.reshape(T, (3, 4, 5, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (3, 4, 5, 6, 7, 1, 1, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-last contraction
        T = np.random.random((5, 6, 7))
        U = np.random.random((3, 4, 5, 6, 7))
        TU = np.tensordot(T, U, axes=([0, 1, 2], [2, 3, 4]))

        T_tt = TT(np.reshape(T, (5, 6, 7, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (3, 4, 5, 6, 7, 1, 1, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, mode='last-last', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - np.transpose(TU))
        self.assertLess(error, self.tol)

        # first-last contraction
        T = np.random.random((5, 6, 7))
        U = np.random.random((3, 4, 5, 6, 7))
        TU = np.tensordot(T, U, axes=([0, 1, 2], [2, 3, 4]))

        T_tt = TT(np.reshape(T, (5, 6, 7, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (3, 4, 5, 6, 7, 1, 1, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, mode='first-last', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # first-first contraction
        T = np.random.random((3, 4, 5))
        U = np.random.random((3, 4, 5, 6, 7))
        TU = np.tensordot(T, U, axes=([0, 1, 2], [0, 1, 2]))
        TU = np.transpose(TU)

        T_tt = TT(np.reshape(T, (3, 4, 5, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (3, 4, 5, 6, 7, 1, 1, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, 'first-first', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-first contraction for operator
        T_tt = rand([3, 4], [5, 6], ranks=3)
        U_tt = rand([3, 4, 5, 6], [5, 6, 7, 8], ranks=2)
        T = T_tt.full()
        U = U_tt.full()

        TU = np.tensordot(T, U, axes=([0, 1, 2, 3], [0, 1, 4, 5]))
        T_tt.tensordot(U_tt, 2, overwrite=True)
        TU_back = T_tt.full()
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

    def test_full_other(self):
        # contraction over full U
        # last-first contraction
        T = np.random.random((2, 3, 4, 5))
        U = np.random.random((3, 4, 5))
        TU = np.tensordot(T, U, axes=([1, 2, 3], [0, 1, 2]))

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (3, 4, 5, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-last contraction
        T = np.random.random((2, 3, 4, 5, 6))
        U = np.random.random((4, 5, 6))
        TU = np.tensordot(T, U, axes=([2, 3, 4], [0, 1, 2]))

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 6, 1, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (4, 5, 6, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, mode='last-last', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # first-last contraction
        T = np.random.random((2, 3, 4, 5, 6))
        U = np.random.random((2, 3, 4))
        TU = np.tensordot(T, U, axes=([0, 1, 2], [0, 1, 2]))

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 6, 1, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (2, 3, 4, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, mode='first-last', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # first-first contraction
        T = np.random.random((2, 3, 4, 5, 6))
        U = np.random.random((2, 3, 4))
        TU = np.tensordot(T, U, axes=([0, 1, 2], [0, 1, 2]))

        T_tt = TT(np.reshape(T, (2, 3, 4, 5, 6, 1, 1, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (2, 3, 4, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, mode='first-first', overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # last-first contraction for operator
        T_tt = rand([2, 3, 4, 5], [5, 6, 7, 8], ranks=2)
        U_tt = rand([4, 5], [7, 8], ranks=3)
        T = T_tt.full()
        U = U_tt.full()

        TU = np.tensordot(T, U, axes=([2, 3, 6, 7], [0, 1, 2, 3]))
        T_tt.tensordot(U_tt, 2, overwrite=True)
        TU_back = T_tt.full()
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

    def test_full_both(self):
        # contraction over both full
        T = np.random.random((3, 4, 5))
        U = np.random.random((3, 4, 5))
        TU = np.tensordot(T, U, axes=([0, 1, 2], [0, 1, 2]))

        T_tt = TT(np.reshape(T, (3, 4, 5, 1, 1, 1)))
        U_tt = TT(np.reshape(U, (3, 4, 5, 1, 1, 1)))
        T_tt.tensordot(U_tt, 3, overwrite=True)

        TU_back = np.squeeze(T_tt.full())
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)

        # operator
        T_tt = rand([2, 3, 4], [5, 6, 7], ranks=2)
        U_tt = rand([2, 3, 4], [5, 6, 7], ranks=3)
        T = T_tt.full()
        U = U_tt.full()

        TU = np.tensordot(T, U, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
        T_tt.tensordot(U_tt, 3, overwrite=True)
        TU_back = T_tt.full()
        error = np.linalg.norm(TU_back - TU)
        self.assertLess(error, self.tol)


if __name__ == '__main__':
    # simple_test_case()
    ut.main()
