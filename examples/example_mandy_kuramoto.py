#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import scipy.integrate
import mandy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import scikit_tt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

# example for the application of MANDy (function major)
# Reference: P. Gelß, S. Klus, C. Schütte, "MANDy ..."

d = 100#200  # number of oscillators
K = 2  # coupling strength
h = 0.1 # external forcing parameter
w = np.linspace(-5,5, d)  # natural frequencies equidistantly distributed in [0,0.5]
psi = [lambda x: np.sin(x), lambda x: np.cos(x)]  # basis functions for MANDy
t_end = 1020#300
m = 10201#6000  # number of snapshots



def kuramoto(t, theta):
    """Governing equations for the Kuramoto model with external forcing, see [1]_.

    Parameters
    ----------
    t : float
        time
    theta : ndarray, shape(d,)
        positions of the oscillators

    Returns
    -------
    derivative: ndarray, shape(d,)
        derivatives corresponding to data given in theta

    References
    ----------
    .. [1] J. A. Acebrón, L. L. Bonilla, C. J. Pérez Vicente, F. Ritort, R. Spigler, "The Kuramoto model: A simple
           paradigm for synchronization phenomena", Rev. Mod. Phys. 77, pp. 137-185 , 2005
    """
    [theta_i, theta_j] = np.meshgrid(theta, theta)
    return w + K / d * np.sin(theta_j - theta_i).sum(0) + h*np.sin(theta)

def reconstruct_kuramoto(t,theta):
    X = theta
    cores = [np.zeros([1, X.shape[0] + 1, 1, 1])] + [
        np.zeros([1, X.shape[0] + 1, 1, 1]) for i in range(1, len(psi))]
    for i in range(len(psi)):
        ABC = np.hstack([1]  + [psi[i](X[k]) for k in range(X.shape[0])]) # insert elements of other cores
        cores[i][0, :, 0, 0] = ABC
    Psi = scikit_tt.TT(cores)
    Psi = Psi.full().reshape(np.prod(Psi.row_dims), 1, order='F')

    gov = Psi.T @ Xi_tt
    gov = gov.reshape(gov.size)
    return gov


def reconstruct_samples(theta_init):
    #sol = scipy.integrate.solve_ivp(wrapper, t_span = [0, t_end], y0=theta_init, method='BDF', t_eval=np.linspace(0, t_end, m))

    sol = scipy.integrate.solve_ivp(reconstruct_kuramoto, [0, t_end], theta_init, method='BDF', t_eval=np.linspace(0, t_end, m))

    X = sol.y
    Y = np.zeros([d,m])
    for i in range(m):
        Y[:, i] = kuramoto(0, X[:, i])
    return X, Y

def generate_samples():
    """Generate time-series data of the Kuramoto model.

    Returns
    -------
    X : ndarray, shape(d,m)
        snaphot matrix
    Y : ndarray, shape(d,m)
        snapshot matrix (derivatives)
    """
    theta_init = np.linspace(0,2*np.pi,num=d,endpoint=False)
    sol = scipy.integrate.solve_ivp(kuramoto, [0, t_end], theta_init, method='BDF', t_eval=np.linspace(0, t_end, m))
    X = sol.y
    Y = np.zeros([d,m])
    for i in range(m):
        Y[:, i] = kuramoto(0, X[:, i])
    return X, Y, theta_init

def generate_samples_random():
    """Generate time-series data of the Kuramoto model.

    Returns
    -------
    X : ndarray, shape(d,m)
        snaphot matrix
    Y : ndarray, shape(d,m)
        snapshot matrix (derivatives)
    """
    theta_init = 2 * np.pi * np.random.rand(d)  # initial angles uniformly distributed in [-pi,pi]
    sol = scipy.integrate.solve_ivp(kuramoto, [0, t_end], theta_init, method='BDF', t_eval=np.linspace(0, t_end, m))
    X = sol.y
    Y = np.zeros([d,m])
    for i in range(m):
        Y[:, i] = kuramoto(0, X[:, i])
    return X, Y, theta_init



def construct_Xi_exact():
    """Construct exact coefficient tensor for the Kuramoto example presented in [1]_.

    Returns
    -------
    Xi_exact : TT tensor

    References
    ----------
    .. [1] P. Gelß, S. Klus, C. Schütte, "MANDy ..."

    """
    cores = [None] * 3
    cores[0] = np.zeros([1, d + 1, 1, 2 * d + 1])
    cores[1] = np.zeros([2 * d + 1, d + 1, 1, 2 * d + 1])
    cores[2] = np.zeros([2 * d + 1, d, 1, 1])
    cores[0][0, 0, 0, 0] = 1
    cores[1][0, 0, 0, 0] = 1
    cores[2][0, :, 0, 0] = w
    for i in range(d):
        cores[0][0, 1:, 0, 2 * i + 1] = (K / d) * np.ones([d])
        cores[0][0, i + 1, 0, 2 * i + 1] = 0
        cores[1][2 * i + 1, i + 1, 0, 2 * i + 1] = 1
        cores[2][2 * i + 1, i, 0, 0] = 1
        cores[0][0, i + 1, 0, 2 * i + 2] = 1
        cores[1][2 * i + 2, 0, 0, 2 * i + 2] = h
        cores[1][2 * i + 2, 1:, 0, 2 * i + 2] = -(K / d) * np.ones([d])
        cores[1][2 * i + 2, i + 1, 0, 2 * i + 2] = 0
        cores[2][2 * i + 2, i, 0, 0] = 1
    Xi_exact = scikit_tt.TT(cores)
    return Xi_exact










X, Y, theta_init = generate_samples_random()  # generate time-series data
print('d:  ',X.shape[0])
print('m:  ',X.shape[1])
Xi_tt = mandy.mandy_function_major(X, Y, psi, add_one=True, threshold=-16)  # approximate coefficient tensor
Xi_exact = construct_Xi_exact()  # construct exact coefficient tensor
error = scikit_tt.TT.norm(Xi_tt - Xi_exact) / scikit_tt.TT.norm(Xi_exact)  # compute error
print('relative error:  ',error)

# np.savez('../data/example_mandy_kuramoto',Xi_tt=Xi_tt)

# npzfile = np.load('../data/example_mandy_kuramoto.npz') # load data

# Xi_tt = npzfile['Xi_tt']
print(Xi_tt)
Xi_tt= (Xi_tt.full().reshape(np.prod(Xi_tt.row_dims[:-1]), Xi_tt.row_dims[-1], order='F'))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.autolayout': True})

X, Y , theta_init= generate_samples_random()

print('reconstruction:')
print(' ')

X2 , Y2 = reconstruct_samples(theta_init)
print('ready')

pos = np.zeros([2,10])
pos2 = np.zeros([2,10])
plt.figure(dpi=300)

for k in range(6):
    plt.subplot(2,3,k+1)

    for i in range(0,10):
        pos[0,i] = np.cos(X[i*int(d/10),k*200])
        pos[1,i] = np.sin(X[i*int(d/10),k*200])
        pos2[0, i] = np.cos(X2[i * int(d / 10), k * 200])
        pos2[1, i] = np.sin(X2[i * int(d / 10), k * 200])

    circ = pat.Circle((0,0), radius = 1, edgecolor='gray', facecolor='aliceblue')
    plt.gca().add_patch(circ)
    for i in range(10):
        plt.plot(pos[0, i], pos[1, i], 'o', markersize=10, markeredgecolor='C'+str(i), markerfacecolor='white')
        plt.plot([0,pos2[0,i]],[0,pos2[1,i]],'--',color='gray',linewidth=0.5)
        plt.plot(pos2[0,i],pos2[1,i],'o',markeredgecolor='C'+str(i),markerfacecolor='C'+str(i))


    plt.gca().set_xlim([-1.1,1.1])
    plt.gca().set_ylim([-1.1,1.1])
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.title(r't='+str(k*20)+'s', y=1, fontsize=12)
plt.show()
