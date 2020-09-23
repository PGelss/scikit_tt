import unittest as ut
from copy import deepcopy
from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.tgedmd as tgedmd
import scikit_tt.data_driven.transform as tdt
# from examples.lemon_slice import LemonSlice
from scikit_tt.tensor_train import TT


class TestAmuse(TestCase):
    def setUp(self):
        self.d = 3
        self.m = 5
        self.n = 2
        self.basis_list = []
        for i in range(self.d):
            self.basis_list.append([tdt.Monomial(i, j) for j in range(self.n)])
        self.x = np.random.random((self.d, self.m))
        self.drift = np.random.random((self.d, self.m))
        self.diffusion = np.random.random((self.d, self.d, self.m))

    def test_amuse(self):
        eigvals, eigtensors = tgedmd.amuset_hosvd(self.x, self.basis_list, self.drift, self.diffusion,
                                                  return_option='eigentensors', threshold=1e-16)

        PsiX = np.zeros([self.n for _ in range(self.d)] + [self.m])
        dPsiX = np.zeros([self.n for _ in range(self.d)] + [self.m])

        for s0 in range(self.n):
            for s1 in range(self.n):
                for s2 in range(self.n):
                    for k in range(self.m):
                        PsiX[s0, s1, s2, k] = self.basis_list[0][s0](self.x[:, k]) *\
                                              self.basis_list[1][s1](self.x[:, k]) *\
                                              self.basis_list[2][s2](self.x[:, k])
                        dPsiX[s0, s1, s2, k] = tgedmd.generator_on_product(self.basis_list, (s0, s1, s2), self.x[:, k],
                                                                           self.drift[:, k], self.diffusion[:, :, k])

        G = np.tensordot(PsiX, PsiX, axes=(3, 3))
        print(G.shape)


class TestHelperFunctions(TestCase):
    def setUp(self):
        self.tol = 1e-8

    def test_frobenius_inner(self):
        a = np.random.random((4, 4))
        b = np.random.random((4, 4))

        frob = 0
        for i in range(4):
            for j in range(4):
                frob += a[i, j] * b[i, j]

        self.assertLess(abs(frob - tgedmd._frob_inner(a, b)), self.tol)

    def test_generator(self):
        f = tdt.Sin(0, 0.5)
        x = np.array([1, 2, 3])

        def b(y):
            return np.array([y.sum(), 0, y.sum() ** 3])

        def sigma(y):
            return np.eye(3)

        generator = tgedmd._generator(f, x, b, sigma)
        generator2 = np.inner(f.gradient(x), b(x))
        generator2 += 0.5 * np.trace(f.hessian(x))

        self.assertLess(abs(generator - generator2), self.tol)

    def test_is_special(self):
        # ---- special ----
        A = np.zeros((10, 10, 3, 4))
        # diagonal
        for i in range(min(A.shape[0], A.shape[1])):
            A[i, i, :, :] = np.random.random((3, 4))
        # first row
        A[0, :, :, :] = np.random.random((10, 3, 4))
        # second column
        A[:, 1, :, :] = np.random.random((10, 3, 4))
        self.assertTrue(tgedmd._is_special(A))

        A = np.random.random((10, 1, 3, 4))
        self.assertTrue(tgedmd._is_special(A))

        A = np.random.random((1, 10, 3, 4))
        self.assertTrue(tgedmd._is_special(A))

        # ---- not special ----
        A = np.random.random((10, 10, 3, 4))
        self.assertFalse(tgedmd._is_special(A))

        A = np.zeros((10, 10, 3, 4))
        # diagonal
        for i in range(min(A.shape[0], A.shape[1])):
            A[i, i, :, :] = np.random.random((3, 4))
        # first row
        A[0, :, :, :] = np.random.random((10, 3, 4))
        # second column
        A[:, 1, :, :] = np.random.random((10, 3, 4))
        # false entry
        A[1, 3, :, :] = np.random.random((3, 4))
        self.assertFalse(tgedmd._is_special(A))

    def test_special_tensordot(self):
        dimsA = [(10, 10, 3, 4), (1, 10, 3, 4), (10, 10, 3, 4), (1, 10, 3, 4)]
        dimsB = [(10, 10, 4, 3), (10, 10, 4, 3), (10, 1, 4, 3), (10, 1, 4, 3)]

        for idx in range(len(dimsA)):
            dimAi, dimAj, dimA1, dimA2 = dimsA[idx]
            dimBi, dimBj, dimB1, dimB2 = dimsB[idx]
            A = np.zeros((dimAi, dimAj, dimA1, dimA2))
            B = np.zeros((dimBi, dimBj, dimB1, dimB2))

            if A.shape[0] == 1:
                A = np.random.random(A.shape)
            else:
                # diagonal
                for i in range(min(A.shape[0], A.shape[1])):
                    A[i, i, :, :] = np.random.random((dimA1, dimA2))
                # first row
                A[0, :, :, :] = np.random.random((dimAj, dimA1, dimA2))
                # second column
                A[:, 1, :, :] = np.random.random((dimAi, dimA1, dimA2))

            if B.shape[1] == 1:
                B = np.random.random(B.shape)
            else:
                # diagonal
                for i in range(min(B.shape[0], B.shape[1])):
                    B[i, i, :, :] = np.random.random((dimB1, dimB2))
                # first row
                B[0, :, :, :] = np.random.random((dimBj, dimB1, dimB2))
                # second column
                B[:, 1, :, :] = np.random.random((dimBi, dimB1, dimB2))

            C1 = np.tensordot(A, B, axes=((1, 3), (0, 2)))
            C1 = np.transpose(C1, [0, 2, 1, 3])
            C2 = tgedmd._special_tensordot(A, B)

            self.assertTrue((np.abs(C1 - C2) < self.tol).all())


class TestCores(TestCase):
    """
    Test if tgedmd.dPsix constructs cores correctly.
    """
    def drift(self, x):
        return np.zeros((self.d,))

    def diffusion(self, x):
        return np.zeros((self.d, self.d))

    def setUp(self):
        self.tol = 1e-8
        self.d = 4
        self.p = self.d

        # self.ls = LemonSlice(k=4, beta=1, c=1, d=self.d, alpha=10)

        self.basis_list = []
        for i in range(self.d):
            self.basis_list.append([tdt.Identity(i)] + [tdt.Monomial(i, j) for j in range(2, 6)])
        self.n = [len(mode) for mode in self.basis_list]

        self.x = np.random.random(self.d)
        self.a = self.diffusion(self.x) @ self.diffusion(self.x).T

        self.cores = [tgedmd._dPsix(self.basis_list[0], self.x, self.drift, self.diffusion, position='first')]
        self.cores = self.cores + [tgedmd._dPsix(self.basis_list[i], self.x, self.drift, self.diffusion,
                                                position='middle') for i in range(1, self.p - 1)]
        self.cores = self.cores + [tgedmd._dPsix(self.basis_list[-1], self.x, self.drift, self.diffusion,
                                                position='last')]

    def test_core0(self):
        core = self.cores[0]

        self.assertEqual(core.shape, (1, self.n[0], 1, self.d + 2))

        self.assertTrue((core[0, :, 0, 0] == np.array([fun(self.x) for fun in self.basis_list[0]])).all())
        self.assertTrue((core[0, :, 0, 1] == np.array([tgedmd._generator(fun, self.x, self.drift, self.diffusion)
                                                       for fun in self.basis_list[0]])).all())
        self.assertTrue((core[0, :, 0, 2] == np.array([fun.partial(self.x, 0) for fun in self.basis_list[0]])).all())
        self.assertTrue((core[0, :, 0, 3] == np.array([fun.partial(self.x, 1) for fun in self.basis_list[0]])).all())

        self.assertTrue(tgedmd._is_special(core.transpose((0, 3, 1, 2))))

    def test_core1(self):
        core = self.cores[1]

        self.assertEqual(core.shape, (self.d + 2, self.n[1], 1, self.d + 2))

        self.assertTrue((core[0, :, 0, 0] == np.array([fun(self.x) for fun in self.basis_list[1]])).all())
        self.assertTrue((core[2, :, 0, 2] == np.array([fun(self.x) for fun in self.basis_list[1]])).all())

        self.assertTrue((core[0, :, 0, 1] == np.array([tgedmd._generator(fun, self.x, self.drift, self.diffusion)
                                                       for fun in self.basis_list[1]])).all())
        self.assertTrue((core[0, :, 0, 2] == np.array([fun.partial(self.x, 0) for fun in self.basis_list[1]])).all())
        self.assertTrue((core[0, :, 0, 3] == np.array([fun.partial(self.x, 1) for fun in self.basis_list[1]])).all())

        self.assertTrue((core[3, :, 0, 1] == np.array([np.inner(self.a[1, :], fun.gradient(self.x))
                                                       for fun in self.basis_list[1]])).all())

        self.assertTrue(tgedmd._is_special(core.transpose((0, 3, 1, 2))))

    def test_core_last(self):
        core = self.cores[-1]

        self.assertEqual(core.shape, (self.d + 2, self.n[-1], 1, 1))

        self.assertTrue((core[0, :, 0, 0] == np.array([tgedmd._generator(fun, self.x, self.drift, self.diffusion)
                                                       for fun in self.basis_list[-1]])).all())
        self.assertTrue((core[1, :, 0, 0] == np.array([fun(self.x) for fun in self.basis_list[-1]])).all())

        self.assertTrue((core[2, :, 0, 0] == np.array([np.inner(self.a[0, :], fun.gradient(self.x))
                                                       for fun in self.basis_list[-1]])).all())
        self.assertTrue((core[-1, :, 0, 0] == np.array([np.inner(self.a[-1, :], fun.gradient(self.x))
                                                        for fun in self.basis_list[-1]])).all())

        self.assertTrue(tgedmd._is_special(core.transpose((0, 3, 1, 2))))

    def test_tensor(self):
        """
        Check if the full tensor dPsi(x) is correct.
        """
        tensor = TT(self.cores)
        tensor = np.squeeze(tensor.full())

        # create reference tensor from scratch
        t_ref = np.zeros(self.n)
        for s0 in range(self.n[0]):
            for s1 in range(self.n[1]):
                for s2 in range(self.n[2]):
                    for s3 in range(self.n[3]):
                        t_ref[s0, s1, s2, s3] = tgedmd.generator_on_product(self.basis_list, (s0, s1, s2, s3), self.x,
                                                                            self.drift, self.diffusion)
        self.assertTrue((np.abs(tensor - t_ref) < self.tol).all())
