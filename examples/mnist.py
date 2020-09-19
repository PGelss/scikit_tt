# -*- coding: utf-8 -*-

"""
This is an example of tensor-based image classification. See [1]_ for details.

References
----------
.. [1] S. Klus, P. Gelß, "Tensor-Based Algorithms for Image Classification", Algorithms, 2019
"""

from scikit_tt.tensor_train import TT
import scikit_tt.data_driven.transform as tdt
import scikit_tt.data_driven.regression as reg
import scikit_tt.utils as utl
import numpy as np
import scipy.linalg as lin
import time as _time


def classification_mandy(data_path, m_start, m_final, m_step):
    """Kernel-based MANDy for classification.

    Parameters
    ----------
    data_path: string
        path of data to load
    m_start: int
        minimum number of images
    m_final: int
        maximum number of images
    m_step: int
        step size for number of images

    Returns
    -------
    classification_rates: list of floats
        amount of correctly identified images
    cpu_times: list of floats
        run times of training phases
    """

    # load data
    data = np.load(data_path)
    x_train = data['tr_img']
    y_train = data['tr_lbl']
    x_test = data['te_img']
    y_test = data['te_lbl']

    # order of the transformed data tensor
    order = x_train.shape[0]

    # define basis functions
    alpha = 19 / 100 * np.pi
    basis_list = []
    for i in range(order):
        basis_list.append([tdt.Cos(i, alpha), tdt.Sin(i, alpha)])

    # define lists
    classification_rates = []
    cpu_times = []

    # output
    print('Images' + 8 * ' ' + 'Classification rate' + 6 * ' ' + 'CPU time')
    print(47 * '-')

    # loop over image numbers
    for m in range(m_start, m_final, m_step):

        # training phase (apply kernel-based MANDy)
        with utl.timer() as timer:
            z = reg.mandy_kb(x_train[:, :m], y_train[:, :m], basis_list)
        cpu_time = timer.elapsed
        
        # test phase (multiply z with gram matrix)
        gram = tdt.gram(x_train[:, :m], x_test, basis_list)
        solution = z.dot(gram)

        # compute classification rate
        n = y_test.shape[1]
        sol = np.zeros(y_test.shape)
        sol[np.argmax(solution, axis=0), np.arange(0, n)] = 1
        classification_rate = 100 - 50 * np.sum(np.abs(sol - y_test)) / n

        # print results
        str_m = str(m)
        len_m = len(str_m)
        str_c = str("%.2f" % classification_rate + '%')
        len_c = len(str_c)
        str_t = str("%.2f" % cpu_time + 's')
        len_t = len(str_t)
        print(str_m + (20 - len_m) * ' ' + str_c + (27 - len_c - len_t) * ' ' + str_t)
        classification_rates.append(classification_rate)
        cpu_times.append(cpu_time)

    print(' ')

    return classification_rates, cpu_times


def classification_arr(data_path, m_start, m_final, m_step, rank):
    """Alternating ridge regression for classification.

    Parameters
    ----------
    data_path: string
        path of data to load
    m_start: int
        minimum number of images
    m_final: int
        maximum number of images
    m_step: int
        step size for number of images
    rank: int
        TT rank of coefficient tensor
    Returns
    -------
    classification_rates: list of floats
        amount of correctly identified images
    cpu_times: list of floats
        run times of training phases
    """

    # load data
    data = np.load(data_path)
    x_train = data['tr_img']
    y_train = data['tr_lbl']
    x_test = data['te_img']
    y_test = data['te_lbl']

    # order of the transformed data tensor
    order = x_train.shape[0]
    
    # define basis functions
    alpha = 19 / 100 * np.pi
    basis_list = []
    for i in range(order):
        basis_list.append([tdt.Cos(i, alpha), tdt.Sin(i, alpha)])

    # initial guess
    ranks = [1] + [rank for _ in range(order - 1)] + [1]
    cores = [0.001*np.ones([ranks[i], 2, 1, ranks[i + 1]]) for i in range(order)]
    initial_guess = TT(cores).ortho()

    # define lists
    classification_rates = []
    cpu_times = []

    # output
    print('Images' + 8 * ' ' + 'Classification rate' + 6 * ' ' + 'CPU time')
    print(47 * '-')

    # loop over image numbers
    for m in range(m_start, m_final, m_step):

        # training phase (apply ARR)
        with utl.timer() as timer:
            xi = reg.arr(x_train[:, :m], y_train[:, :m], basis_list, initial_guess, repeats=5, rcond=10**(-2), progress=False)
        cpu_time = timer.elapsed
    
        # test phase (contract xi with transformed data tensor)
        d = y_test.shape[0]
        solution = []
        for k in range(d):
            solution_vector = np.ones([1, 1])
            for l in range(order):
                n = len(basis_list[l])
                theta = np.array([basis_list[l][k](x_test) for k in range(n)])
                solution_vector = np.einsum('ij,kj->ijk', solution_vector, theta)
                solution_vector = np.tensordot(xi[k].cores[l], solution_vector, axes=([0, 1], [0, 2]))[0, :, :]
            solution.append(solution_vector)
        solution = np.vstack(solution)

        # compute classification rate
        n = y_test.shape[1]
        sol = np.zeros(y_test.shape)
        sol[np.argmax(solution, axis=0), np.arange(0, n)] = 1
        classification_rate = 100 - 50 * np.sum(np.abs(sol - y_test)) / n
        
        # print results
        str_m = str(m)
        len_m = len(str_m)
        str_c = str("%.2f" % classification_rate + '%')
        len_c = len(str_c)
        str_t = str("%.2f" % cpu_time + 's')
        len_t = len(str_t)
        print(str_m + (20 - len_m) * ' ' + str_c + (27 - len_c - len_t) * ' ' + str_t)
        classification_rates.append(classification_rate)
        cpu_times.append(cpu_time)

    print(' ')

    return classification_rates, cpu_times


utl.header(title='MNIST/FMNIST')

# data paths
mnist_reduced = '/srv/public/data/mnist/MNIST_reduced.npz'
mnist_full = '/srv/public/data/mnist/MNIST_full.npz'
fmnist_reduced = '/srv/public/data/mnist/FMNIST_reduced.npz'
fmnist_full = '/srv/public/data/mnist/FMNIST_full.npz'

print('MNIST(14x14) with kernel-based MANDy:\n')
classification_rates, cpu_times = classification_mandy(mnist_reduced, 5000, 60001, 5000)

print('MNIST(28x28) with kernel-based MANDy:\n')
classification_rates, cpu_times = classification_mandy(mnist_full, 5000, 60001, 5000)

print('MNIST(14x14) with ARR:\n')
classification_rates, cpu_times = classification_arr(mnist_reduced, 5000, 60001, 5000, 10)

print('FMNIST(14x14) with kernel-based MANDy:\n')
classification_rates, cpu_times = classification_mandy(fmnist_reduced, 5000, 60001, 5000)

print('FMNIST(28x28) with kernel-based MANDy:\n')
classification_rates, cpu_times = classification_mandy(fmnist_full, 5000, 60001, 5000)

print('FMNIST(14x14) with ARR:\n')
classification_rates, cpu_times = classification_arr(fmnist_reduced, 5000, 60001, 5000, 10)
