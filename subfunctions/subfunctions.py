#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tt as tt
import matplotlib.pyplot as plt

#
# def shiftmatrix(n, k):
#     M = np.diag(np.ones(n - np.absolute(k)), k)
#     return M


def unit_vector(dimension, index):
    v = np.zeros(dimension)
    v[index] = 1
    return v

def mean_concentrations(solution, time_steps=[]):
    mean = np.zeros([len(solution), solution[0].order])
    for i in range(len(solution)):
        for j in range(solution[0].order):
            cores = [None] * solution[0].order
            for k in range(solution[0].order):
                cores[k] = np.ones([1, solution[0].row_dims[k], 1, 1])
            cores[j] = np.zeros([1, solution[0].row_dims[j], 1, 1])
            cores[j][0, :, 0, 0] = np.arange(solution[0].row_dims[j])
            tensor_mean = tt.TT(cores)
            mean[i,j] = (solution[i].transpose() @ tensor_mean).element([0]*2*tensor_mean.order)
    if len(time_steps) > 0:
        plt.plot(time_steps,mean)
        plt.show()
    return mean

