from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.tgedmd as tgedmd
import scikit_tt.data_driven.transform as tdt


class TestAmuse(TestCase):
    """
    Tests for the two functions amuset_hosvd and amuset_hosvd_reversible.
    """
    def setUp(self):
        self.tol = 1e-5
        self.d = 3
        self.m = 10
        self.n = 2
        self.basis_list = []
        for i in range(self.d):
            self.basis_list.append([tdt.Monomial(i, j) for j in range(self.n)])
        self.x = np.random.random((self.d, self.m))
        self.drift = np.random.random((self.d, self.m))
        self.diffusion = np.random.random((self.d, self.d, self.m))

    def test_amuse(self):
        # calculate eigenvalues of Koopman generator with AMUSEt
        eigvals, eigtensors = tgedmd.amuset_hosvd(self.x, self.basis_list, self.drift, self.diffusion,
                                                  return_option='eigentensors', threshold=1e-16)

        # calculate eigenvalues of Koopman generator by constructing the full tensor L from data
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
        G = np.reshape(G, (8, 8))
        A = np.tensordot(PsiX, dPsiX, axes=(3, 3))
        A = np.reshape(A, (8, 8))
        L = np.dot(np.linalg.inv(G), A)
        eigvals2, eigenvalues2 = np.linalg.eig(L)

        # check if the eigenvalues match
        eigvals.sort()
        eigvals2.sort()
        self.assertTrue((np.abs(eigvals - eigvals2) < self.tol).all())

    def test_amuse_reversible(self):
        # calculate eigenvalues of Koopman generator with AMUSEt
        eigvals, eigtensors = tgedmd.amuset_hosvd_reversible(self.x, self.basis_list, self.diffusion,
                                                             return_option='eigentensors', threshold=1e-16)

        # calculate eigenvalues of Koopman generator by constructing the full tensor L from data
        PsiX = np.zeros([self.n for _ in range(self.d)] + [self.m])
        dPsiX = np.zeros([self.n for _ in range(self.d)] + [self.d, self.m])

        for s0 in range(self.n):
            for s1 in range(self.n):
                for s2 in range(self.n):
                    for k in range(self.m):
                        PsiX[s0, s1, s2, k] = self.basis_list[0][s0](self.x[:, k]) * \
                                              self.basis_list[1][s1](self.x[:, k]) * \
                                              self.basis_list[2][s2](self.x[:, k])
                        for i in range(self.d):
                            dPsiX[s0, s1, s2, i, k] = tgedmd.generator_on_product_reversible(
                                self.basis_list, (s0, s1, s2), i, self.x[:, k], self.diffusion[:, :, k])

        G = np.tensordot(PsiX, PsiX, axes=(3, 3))
        G = np.reshape(G, (8, 8))
        A = -0.5 * np.tensordot(dPsiX, dPsiX, axes=((3, 4), (3, 4)))
        A = np.reshape(A, (8, 8))
        L = np.dot(np.linalg.inv(G), A)
        eigvals2, eigenvalues2 = np.linalg.eig(L)

        # check if the eigenvalues match
        eigvals.sort()
        eigvals2.sort()
        
        self.assertTrue((np.abs(eigvals - eigvals2) < self.tol).all())

    def test_is_special(self):
        A = np.zeros((3, 3, 2, 2))
        for i in range(A.shape[0]):
            A[i, i, :, :] = np.random.random((2, 2))
            A[0, i, :, :] = np.random.random((2, 2))
            A[i, 1, :, :] = np.random.random((2, 2))
        self.assertTrue(tgedmd._is_special(A))

        A[1, 2, 0, 0] = 1
        self.assertFalse(tgedmd._is_special(A))

        A = np.zeros((2, 3, 2, 2))
        self.assertFalse(tgedmd._is_special(A))
