# -*- coding: utf-8 -*-

from scikit_tt.tensor_train import TT
import numpy as np


def cantor_dust(dimension, level):
    """Construction of a (multidimensional) Cantor dust

    Generate a binary tensor representing a Cantor dust, see [1]_, by exploiting the 
    tensor-train format and Kronecker products.

    Parameters
    ----------
    dimension: int
        dimension of the Cantor dust
    level: int 
        level of the fractal construction to generate

    Returns
    -------
    fractal: ndarray
        tensor representing the Cantor dust

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for 
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    # construct generating tensor
    cores = []
    for _ in range(dimension):
        cores.append(np.zeros([1, 3, 1, 1]))
        cores[-1][0, :, 0, 0] = [1, 0, 1]
    generator = TT(cores)
    generator = generator.full().reshape(generator.row_dims)

    # construct fractal in the form of a binary tensor
    fractal = generator
    for i in range(2, level + 1):
        fractal = np.kron(fractal, generator)
    fractal = fractal.astype(int)

    return fractal


def multisponge(dimension, level):
    """Construction of a multisponge

    Generate a binary tensor representing a multisponge fractal (e.g., Sierpinski carpet, 
    Menger sponge, etc.), see [1]_, by exploiting the tensor-train format and Kronecker 
    products.

    Parameters
    ----------
    dimension: int (>1)
        dimension of the multisponge
    level: int 
        level of the fractal construction to generate

    Returns
    -------
    fractal: ndarray
        tensor representing the multisponge fractal

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for 
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    if dimension > 1:

        # construct generating tensor
        cores = [np.zeros([1, 3, 1, 2])]
        cores[0][0, :, 0, 0] = [1, 1, 1]
        cores[0][0, :, 0, 1] = [1, 0, 1]
        for _ in range(1, dimension - 1):
            cores.append(np.zeros([2, 3, 1, 2]))
            cores[-1][0, :, 0, 0] = [1, 0, 1]
            cores[-1][1, :, 0, 0] = [0, 1, 0]
            cores[-1][1, :, 0, 1] = [1, 0, 1]
        cores.append(np.zeros([2, 3, 1, 1]))
        cores[-1][0, :, 0, 0] = [1, 0, 1]
        cores[-1][1, :, 0, 0] = [0, 1, 0]
        generator = TT(cores)
        generator = generator.full().reshape(generator.row_dims)

        # construct fractal in the form of a binary tensor
        fractal = generator
        for i in range(2, level + 1):
            fractal = np.kron(fractal, generator)
        fractal = fractal.astype(int)

    else:

        raise ValueError('dimension must be larger than 1')

    return fractal


def rgb_fractal(matrix_r, matrix_g, matrix_b, level):
    """Construction of an RGB fractal

    Generate a 3-dimensional tensor representing an RGB fractal, see [1]_, by exploiting
    the tensor-train format.

    Parameters
    ----------
    matrix_r: ndarray
        matrix representing red primaries
    matrix_g: ndarray
        matrix representing green primaries
    matrix_b: ndarray
        matrix representing blue primaries
    level: int 
        level of the fractal construction to generate

    Returns
    -------
    fractal: ndarray
        tensor representing the RGB fractal

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for 
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    # dimension of RGB matrices
    n = matrix_r.shape[0]

    # construct RGB fractal
    cores = [np.zeros([1, n, n, 3])]
    cores[0][0, :, :, 0] = matrix_r
    cores[0][0, :, :, 1] = matrix_g
    cores[0][0, :, :, 2] = matrix_b
    for _ in range(1, level):
        cores.append(np.zeros([3, n, n, 3]))
        cores[-1][0, :, :, 0] = matrix_r
        cores[-1][1, :, :, 1] = matrix_g
        cores[-1][2, :, :, 2] = matrix_b
    cores.append(np.zeros([3, 3, 1, 1]))
    cores[-1][0, :, 0, 0] = [1, 0, 0]
    cores[-1][1, :, 0, 0] = [0, 1, 0]
    cores[-1][2, :, 0, 0] = [0, 0, 1]
    fractal = TT(cores).full().reshape([n ** level, 3, n ** level]).transpose([0, 2, 1])

    return fractal


def vicsek_fractal(dimension, level):
    """Construction of a Vicsek fractal

    Generate a binary tensor representing a Vicsek fractal, see [1]_, by exploiting the 
    tensor-train format and Kronecker products.

    Parameters
    ----------
    dimension: int (>1)
        dimension of the Vicsek fractal
    level: int 
        level of the fractal construction to generate

    Returns
    -------
    fractal: ndarray
        tensor representing the Vicsek fractal

    References
    ----------
    .. [1] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for 
           creating self-similar patterns", arXiv:1812.00814, 2018
    """

    if dimension > 1:

        # construct generating tensor
        cores = [np.zeros([1, 3, 1, 2])]
        cores[0][0, :, 0, 0] = [1, 1, 1]
        cores[0][0, :, 0, 1] = [0, 1, 0]
        for _ in range(1, dimension - 1):
            cores.append(np.zeros([2, 3, 1, 2]))
            cores[-1][0, :, 0, 0] = [0, 1, 0]
            cores[-1][1, :, 0, 0] = [1, 0, 1]
            cores[-1][1, :, 0, 1] = [0, 1, 0]
        cores.append(np.zeros([2, 3, 1, 1]))
        cores[-1][0, :, 0, 0] = [0, 1, 0]
        cores[-1][1, :, 0, 0] = [1, 0, 1]
        generator = TT(cores)
        generator = generator.full().reshape(generator.row_dims)

        # construct fractal in the form of a binary tensor
        fractal = generator
        for i in range(2, level + 1):
            fractal = np.kron(fractal, generator)
        fractal = fractal.astype(int)

    else:

        raise ValueError('dimension must be larger than 1')

    return fractal
