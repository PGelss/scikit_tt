# -*- coding: utf-8 -*-

"""
These are several examples for the tensor-construction of fractal patterns. For more details,
see [1]_.

References
----------
..[1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for creating 
      self-similar patterns", arXiv:1812.00814, 2018
"""

import numpy as np
import scikit_tt.fractals as frac
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import axes3d


def plot1d(vector):
    """Plot 1-dimensional fractals.

    Parameters
    ----------
    vector: ndarray
        1-dimensional binary tensor representing the fractal pattern
    """
    for i in range(vector.shape[0]):
        if vector[i]==1:
            ax.plot([i/vector.shape[0], (i+1)/vector.shape[0]], [0, 0], color='0.33')
    plt.xlim(-1/3,4/3)
    plt.axis('off')

def plot2d(matrix):
    """Plot 2-dimensional fractals.

    Parameters
    ----------
    matrix: ndarray
        2-dimensional binary tensor representing the fractal pattern
    """

    ax.imshow(matrix, cmap = LinearSegmentedColormap.from_list('_', ['1','0.33']))
    plt.xlim(-0.5-(1/3)*matrix.shape[0],-0.5+(4/3)*matrix.shape[0])
    plt.axis('off')

def plot3d(tensor):
    """Plot 3-dimensional fractals.

    Parameters
    ----------
    tensor: ndarray
        3-dimensional binary tensor representing the fractal pattern
    """

    eps = 0.01
    X = np.array([[0-eps,1+eps],[0-eps,1+eps]])
    Y = np.array([[1+eps,1+eps],[0-eps,0-eps]])
    Z = np.array([[0,0],[0,0]])
    n = tensor.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if tensor[i,j,k]==1:
                    if (tensor[i,j,np.mod(k+1,n-1)] == 0) or (k == n-1):
                        ax.plot_surface(X+i,Y+j,Z+k+1, color='1')
                    if (tensor[np.mod(i+1,n-1),j,k] == 0) or (i == n-1):
                        ax.plot_surface(Z+i+1,X+j,Y+k, color='0.67')
                    if (tensor[i,j-1,k] == 0) or (j == 0):
                        ax.plot_surface(Y+i,Z+j,X+k, color='0.33')
    plt.axis('off')

def plotrgb(tensor):
    """Plot RGB fractals.

    Parameters
    ----------
    tensor: ndarray
        3-dimensional tensor representing the RGB image
    """

    ax.imshow(rgb_fractals[i])
    plt.axis('off')

utl.header(title='Tensor-generated fractals')

# multisponges
# ------------

utl.progress('Generating multisponges', 0, dots=6)
multisponge = []
for i in range(2,4):
    for j in range(1,4):
        multisponge.append(frac.multisponge(i,j))
        utl.progress('Generating multisponges', 100*((i-2)*3+j)/6, dots=6)

# Cantor dusts
# ------------

utl.progress('Generating Cantor dusts', 0, dots=6)
cantor_dust = []
for i in range(1,4):
    for j in range(1,4):
        cantor_dust.append(frac.cantor_dust(i,j))
        utl.progress('Generating Cantor dusts', 100*((i-1)*3+j)/9, dots=6)

# Vicsek fractals
# ---------------

utl.progress('Generating Vicsek fractals', 0)
vicsek = []
for i in range(2,4):
    for j in range(1,4):
        vicsek.append(frac.vicsek_fractal(i,j))
        utl.progress('Generating Vicsek fractals', 100*((i-2)*3+j)/6)

# RGB fractals
# ------------

utl.progress('Generating RGB fractals', 0, dots=6)

level = 5
rgb_fractals = []

matrix_r = np.array([[0.5,1,0.5],[1,0.5,1],[0.5,1,0.5]])
matrix_g = np.array([[0.75,1,0.75],[1,1,1],[0.75,1,0.75]])
matrix_b = np.array([[1,0.75,1],[0.75,1,0.75],[1,0.75,1]])
rgb_fractals.append(frac.rgb_fractal(matrix_r, matrix_g, matrix_b, level))

utl.progress('Generating RGB fractals', 33.3, dots=6)

matrix_r = np.array([[0.5,0.75,0.75,0.5],[0.75,1,1,0.75],[0.75,1,1,0.75],[0.5,0.75,0.75,0.5]])
matrix_g = np.array([[1,0.5,0.5,1],[0.5,0.75,0.75,0.5],[0.5,0.75,0.75,0.5],[1,0.5,0.5,1]])
matrix_b = np.array([[0.75,1,1,0.75],[1,0.5,0.5,1],[1,0.5,0.5,1],[0.75,1,1,0.75]])
rgb_fractals.append(frac.rgb_fractal(matrix_r, matrix_g, matrix_b, level))

utl.progress('Generating RGB fractals', 66.6, dots=6)

matrix_r = np.array([[0.25,0.5,1,0.5,0.25],[0.5,1,1,1,0.5],[1,1,0.5,1,1],[0.5,0.5,0.25,0.5,0.5],[0.5,0.25,0.25,0.25,0.5]])
matrix_g = np.array([[0.25,0.25,0.5,0.25,0.25],[0.25,0.5,1,0.5,0.25],[0.5,1,1,1,0.5],[1,1,0.5,1,1],[0.5,0.5,0.25,0.5,0.5]])
matrix_b = np.array([[0.25,0.25,0.25,0.25,0.25],[0.25,0.25,0.5,0.25,0.25],[0.25,0.5,1,0.5,0.25],[0.5,1,1,1,0.5],[1,1,0.5,1,1]])
rgb_fractals.append(frac.rgb_fractal(matrix_r, matrix_g, matrix_b, level))

utl.progress('Generating RGB fractals', 100, dots=6)

print(' ')

# plot fractals
# -------------

utl.plot_parameters()

utl.progress('Plotting patterns', 0, dots=12)

# multisponges
f = plt.figure(figsize=plt.figaspect(0.65))
for i in range(3):
    ax = f.add_subplot(2, 3, i+1, aspect=1)
    plot2d(multisponge[i])
    if i == 1:
        plt.title('Sierpinski carpet', y=1.2)
for i in range(3,6):
    ax = f.add_subplot(2, 3, i+1, projection='3d', aspect=1)
    plot3d(multisponge[i])
    if i == 4:
        plt.title('Menger sponge', y=1.1)
plt.show()

utl.progress('Plotting patterns', 25, dots=12)

# Cantor dusts
f = plt.figure(figsize=plt.figaspect(1))
for i in range(3):
    ax = f.add_subplot(3, 3, i+1, aspect=1)
    plot1d(cantor_dust[i])
    if i == 1:
        plt.title('Cantor set', y=1.2)
for i in range(3,6):
    ax = f.add_subplot(3, 3, i+1, aspect=1)
    plot2d(cantor_dust[i])
    if i == 4:
        plt.title('Cantor dust (2D)', y=1.2)
for i in range(6,9):
    ax = f.add_subplot(3, 3, i+1, projection='3d', aspect=1)
    plot3d(cantor_dust[i])
    if i == 7:
        plt.title('Cantor dust (3D)', y=1.1)
plt.show()

utl.progress('Plotting patterns', 50, dots=12)

# Vicsek fractals
f = plt.figure(figsize=plt.figaspect(0.65))
for i in range(3):
    ax = f.add_subplot(2, 3, i+1, aspect=1)
    plot2d(vicsek[i])
    if i == 1:
        plt.title('Vicsek fractal (2D)', y=1.2)
for i in range(3,6):
    ax = f.add_subplot(2, 3, i+1, projection='3d', aspect=1)
    plot3d(vicsek[i])
    if i == 4:
        plt.title('Vicsek fractal (3D)', y=1.1)
plt.show()

utl.progress('Plotting patterns', 75, dots=12)

# RGB fractals
f = plt.figure(figsize=plt.figaspect(0.45))
for i in range(3):
    ax = f.add_subplot(1, 3, i + 1, aspect=1)
    plotrgb(rgb_fractals[i])
    if i == 1:
        plt.title('RGB fractals', y=1.1)
plt.show()

utl.progress('Plotting patterns', 100, dots=12)
print(' ')
