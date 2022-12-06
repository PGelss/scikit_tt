#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from scikit_tt.tensor_train import TT # pip install git+https://github.com/PGelss/scikit_tt
from typing import List


def sampling(quantum_state: 'TT', measure_list: List[float], number_of_samples: int, plot_tf: bool=False):
    """Sampling in MPS format.
    
    Parameters
    ----------
    quantum_state: TT
        MPS/TT representation of quantum state, has to be right-orthonormal

    measure_list: list
        ...
    number_of_samples: int
        number of samples to draw
    plot_tf: bool
        whether to plot results or not (default is False)

    Returns
    -------
    measurements: ndarray
        ...
    probabilities: ndarray
        ...
        
    References
    ----------
    .. [1] P. Gelß, S. Klus, Z. Shakibaei, S. Pokutta, "Low-rank tensor decompositions of 
           quantum circuits", arXiv:2205.09882, 2022
    """

    # compute probabilities as TT decomposition
    qubits_diag = TT.diag(quantum_state, measure_list)
    probabilities = (qubits_diag.transpose(conjugate=True)@quantum_state)

    # squeeze probability tensor
    probabilities = TT.squeeze(probabilities)

    # initialize sample matrix
    sample_matrix = np.zeros([number_of_samples, len(measure_list)])

    # generate random numbers in [0,1] for faster sampling
    samples = np.random.rand(number_of_samples, len(measure_list))

    # contract last mode of each core with reshaped indentity matrix,
    # since qubit_out is right-orthonormal
    cores_tmp = [None]*len(measure_list)
    for i in range(len(measure_list)):
        cores_tmp[i] = probabilities.cores[i][:,:,0,:]@(np.eye(int(np.sqrt(probabilities.ranks[i+1]))).flatten())

    ## sampling
    
    # left part of the network
    theta = np.ones([number_of_samples,1])
    
    for i in range(probabilities.order):
        
        # compute conditional probabilities for sampling
        cond_prob = theta@cores_tmp[i]
        
        # given PROB(0) and PROB(1), the sample is chosen to be 1 if the random
        # number is larger than PROB(0) (which has a probability of 1-PROB(0)=PROB(1))
        sample_matrix[:,i] = (samples[:,i]>cond_prob[:,0]/np.sum(cond_prob,axis=1))
        
        # update left part of the network
        theta = np.einsum('ij,jil->il', theta, probabilities.cores[i][:,sample_matrix[:,i].astype('int'),0,:])
        
    # sort outcomes and compute probabilities
    samples, counts = np.unique(sample_matrix, return_counts=True, axis=0)
    probabilities = counts/number_of_samples
        
    return samples, probabilities

def plot_histogram(samples, probabilities):
    """Plot results of quantum sampling.
    
    Parameters
    ----------
    samples: nd.array
        samples measurements
    probabilities: nd.array
        probabilities of measurements
    """
        
    plt.rcParams.update({"text.usetex":True, "font.family":"serif", "font.size":14})
    plt.figure(figsize=(6,4), dpi=300)
    plt.ylim([0,np.max(probs)*1.2])
    plt.ylabel('\\textsf{probabilities}', fontsize=16)
    plt.xlabel('\\textsf{measurements}', fontsize=16)
    tick_label=[np.array2string(samples[i,:].astype(int), separator='')[-2:0:-1] for i in range(rows.shape[0])]
    for i, v in enumerate(probabilities):
        plt.text(i, v+0.01, str('%.3f' %v), horizontalalignment='center', fontsize=12)
    plt.bar(np.arange(rows.shape[0]), probs, tick_label=tick_label)
    plt.tight_layout()
    plt.show()
