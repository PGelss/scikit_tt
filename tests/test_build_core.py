import numpy as np
from scikit_tt.tensor_train import build_core
import unittest as ut
from unittest import TestCase
from numpy.random import randint, rand

np.random.seed(1234)

class TestBuildCore(TestCase):

    def input_data(self):

        self.length       = randint(3, 6) # Left  rank of core
        self.n_matrices   = randint(3, 6) # Right rank of core
        self.matrix_order = (randint(3, 6), randint(3, 6)) # Order of matrices
        self.matrix_list  = [[np.zeros(self.matrix_order)] * self.n_matrices] * self.length

    def random_indices(self):

        rand_outer = randint(0, self.length     - 1)
        rand_inner = randint(0, self.n_matrices - 1)

        return rand_outer, rand_inner

    def build_matrix_list(self):

        self.input_data()

        for i in range(self.length):

            self.matrix_list[i] = [ rand(self.matrix_order[0], self.matrix_order[1]) for _ in range(self.n_matrices) ]

        
    def test_shape(self):

        self.build_matrix_list()

        core = build_core(self.matrix_list)

        self.assertEqual(core.shape, (self.length, self.matrix_order[0], self.matrix_order[1], self.n_matrices))
        
        with self.assertRaises(ValueError):

           matrix_list = self.matrix_list

           rand_outer, rand_inner = self.random_indices()
           
           matrix_list[rand_outer][rand_inner] = rand(3, 3, 3) # Fail when element in inner list is not matrix

           build_core(matrix_list) 

    def test_valid_type(self):

        self.build_matrix_list()

        rand_outer, rand_inner  = self.random_indices()

        self.matrix_list[rand_outer][rand_inner] = 0

        core = build_core(self.matrix_list)

        self.assertEqual(True, np.all(core[rand_outer, :, :, rand_inner] == np.zeros(self.matrix_order)))

        with self.assertRaises(TypeError):

            self.build_matrix_list()
            self.matrix_list[rand_outer][rand_inner] = "random_string"
            
            build_core(self.matrix_list)

    def test_valid_matrix_shape(self):

        with self.assertRaises(ValueError):

            self.build_matrix_list()

            rand_outer, rand_inner  = self.random_indices()

            self.matrix_list[rand_outer][rand_inner] = rand(2, 2)

            build_core(self.matrix_list)

    def test_matrix_number(self):

        self.build_matrix_list()

        rand_outer, _ = self.random_indices()

        self.matrix_list[rand_outer] = [ rand(self.matrix_order[0], self.matrix_order[1]) for _ in range(2)]
        
        with self.assertRaises(IndexError):

            build_core(self.matrix_list)


if __name__ == '__main__':
    ut.main()
