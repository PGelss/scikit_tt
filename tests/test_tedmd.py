from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.tedmd as tedmd
import scikit_tt.data_driven.transform as tdt


class TestAmuse(TestCase):
    def setUp(self):
        self.tol = 1e-8
        self.d = 4
        self.m = 10
        self.n = 3
        self.basis_list = []
        for i in range(self.d):
            self.basis_list.append([tdt.Monomial(i, j) for j in range(self.n)])
        self.x = np.random.random((self.d, self.m))

    def test_amuse(self):
        eigvals, eigtensors = tedmd.amuset_hosvd(self.x, np.arange(0, self.m - 1), np.arange(1, self.m),
                                                 self.basis_list, threshold=1e-10)
        # todo: fix hocur?
        eigvals2, eigtensors2 = tedmd.amuset_hocur(self.x, np.arange(0, self.m - 1), np.arange(1, self.m),
                                                   self.basis_list)
        eigvals.sort()
        eigvals2.sort()
        self.assertTrue((np.abs(eigvals - eigvals2) < self.tol).all())
