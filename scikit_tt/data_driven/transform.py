# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scikit_tt.utils as utl
import scipy.linalg as splin
from typing import Union, List
import time as _time
from scikit_tt.tensor_train import TT
from scipy.special import legendre
from scipy.interpolate import BSpline


# ################################## basis functions ###################################
class Function(object):
    """
    Function from R^n -> R.
    All implemented basis functions should inherit from this class.
    Can be called to evaluate function.
    Features first and second order partial derivatives, as well as gradient and hessian matrix.
    Can be initialized with or without dimension n. For every call, checks validity of input.
    """

    def __init__(self, dimension: int=None):
        if dimension is None:
            self.dimension = 1
            self.initialized = False

        else:
            if dimension < 1:
                raise ValueError('dimension has to be >= 1')

            self.dimension = dimension
            self.initialized = True

    def __call__(self, t):
        self.check_call_input(t)

        return 0

    def partial(self, t, direction):
        self.check_partial_input(t, direction)

        return 0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)

        return 0

    def gradient(self, t):
        self.check_call_input(t)

        return np.array([self.partial(t, i) for i in range(self.dimension)])

    def hessian(self, t):
        self.check_call_input(t)
        hess = np.zeros((self.dimension, self.dimension))

        for i in range(self.dimension):
            for j in range(self.dimension):
                hess[i, j] = self.partial2(t, i, j)

        return hess

    def check_call_input(self, t):
        if not self.initialized:
            self.dimension = len(t)
            self.initialized = True
        # elif len(t) != self.dimension:
        #     raise ValueError('wrong dimension of t')

    def check_partial_input(self, t, direction):
        if not self.initialized:
            self.dimension = len(t)
            self.initialized = True

        elif len(t) != self.dimension:
            raise ValueError('wrong dimension of t')

        elif not 0 <= direction < self.dimension:
            raise ValueError('direction has to be >= 0 and < self.dimension')

    def check_partial2_input(self, t, direction1, direction2):
        if not self.initialized:
            self.dimension = len(t)
            self.initialized = True

        elif len(t) != self.dimension:
            raise ValueError('wrong dimension of t')

        elif not 0 <= direction1 < self.dimension or not 0 <= direction2 < self.dimension:
            raise ValueError('direction has to be >= 0 and < self.dimension')


class OneCoordinateFunction(Function):
    """
    Function from R^n -> R, that only depends on one coordinate.
    All implemented functions, that only depend on one coordinate,
    should inherit from this class.
    The checks are modified to secure that the coordinate is valid.
    """

    def __init__(self, index: int, dimension: int=None):

        super(OneCoordinateFunction, self).__init__(dimension)

        if self.initialized and not 0 <= index < self.dimension:
            raise ValueError('index has to be >= 0 and < dimension')

        self.index = index

    def check_call_input(self, t):
        if not self.initialized:
            self.dimension = len(t)

            if not 0 <= self.index < self.dimension:
                raise ValueError('index has to be >= 0 and < dimension')

            self.initialized = True
        # elif len(t) != self.dimension:
        #     raise ValueError('wrong dimension of t')

    def check_partial_input(self, t, direction):

        if not self.initialized:
            self.dimension = len(t)

            if not 0 <= self.index < self.dimension:
                raise ValueError('index has to be >= 0 and < dimension')

            self.initialized = True

        elif len(t) != self.dimension:
            raise ValueError('wrong dimension of t')

        elif not 0 <= direction < self.dimension:
            raise ValueError('direction has to be >= 0 and < self.dimension')

    def check_partial2_input(self, t, direction1, direction2):
        if not self.initialized:
            self.dimension = len(t)

            if not 0 <= self.index < self.dimension:
                raise ValueError('index has to be >= 0 and < dimension')

            self.initialized = True

        elif len(t) != self.dimension:
            raise ValueError('wrong dimension of t')

        elif not 0 <= direction1 < self.dimension or not 0 <= direction2 < self.dimension:
            raise ValueError('direction has to be >= 0 and < self.dimension')

    def gradient(self, t):
        partial = self.partial(t, self.index)
        out = np.zeros((self.dimension,))
        out[self.index] = partial
        return out

    def hessian(self, t):
        partial2 = self.partial2(t, self.index, self.index)
        hess = np.zeros((self.dimension, self.dimension))
        hess[self.index, self.index] = partial2
        return hess


class ConstantFunction(OneCoordinateFunction):
    """
    Constant 1 function.
    """

    def __init__(self, index: int, dimension: int=None):
        super(ConstantFunction, self).__init__(index, dimension)

    def __call__(self, t):
        self.check_call_input(t)
        if np.isscalar(t[self.index]):
            return 1.0
        else:
            return np.ones(len(t[self.index]))

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        return 0.0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)
        return 0.0

    def gradient(self, t):
        self.check_call_input(t)
        return np.zeros((self.dimension,))

    def hessian(self, t):
        self.check_call_input(t)
        return np.zeros((self.dimension, self.dimension))


class IndicatorFunction(OneCoordinateFunction):
    def __init__(self, index: int, a: float, b: float, dimension: int=None):
        """
        Indicator function in one coordiante.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the indicator function
        a : float
            lower bound of the interval
        b : float
            upper bound of the interval
        """
        super(IndicatorFunction, self).__init__(index, dimension)
        self.a = a
        self.b = b

    def __call__(self, t):
        self.check_call_input(t)
        return 1 * ((self.a <= t[self.index]) & (t[self.index] < self.b))

    def partial(self, t, direction):
        raise NotImplementedError('indicator function is not differentiable')

    def partial2(self, t, direction1, direction2):
        raise NotImplementedError('indicator function is not differentiable')


class Identity(OneCoordinateFunction):
    def __init__(self, index: int, dimension: int=None):
        """
        Identiy function.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the identity function
        """
        super(Identity, self).__init__(index, dimension)

    def __call__(self, t):
        self.check_call_input(t)
        return t[self.index]

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        if direction == self.index:
            return 1.0
        return 0.0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)
        return 0.0


class Monomial(OneCoordinateFunction):
    def __init__(self, index: int, exponent: int, prefactor=1, dimension: int=None):
        """
        Monomial function.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the Monomial function
        exponent : int
            degree of the monomial, >= 0
        """
        super(Monomial, self).__init__(index, dimension)
        if exponent < 0:
            raise ValueError('exponent needs to be >= 0')
        self.exponent = exponent
        self.prefactor = prefactor

    def __call__(self, t):
        self.check_call_input(t)
        return self.prefactor * t[self.index] ** self.exponent

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        if direction == self.index:
            if self.exponent > 0:
                return self.prefactor * self.exponent * t[self.index] ** (self.exponent - 1)
        return 0.0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)
        if direction1 == self.index and direction2 == self.index:
            if self.exponent > 1:
                return self.prefactor * self.exponent * (self.exponent - 1) * t[self.index] ** (self.exponent - 2)
        return 0.0


class Legendre(OneCoordinateFunction):
    def __init__(self, index: int, degree: int, domain: float=1.0, dimension: int=None):
        """
        Legendre Polynomial.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the polynomial
        degree : int
            degree of the polynomial, >= 0
        domain : float
                scale the polynomial to the domain [-domain, domain]
        """
        super(Legendre, self).__init__(index, dimension)
        if degree < 0:
            raise ValueError('exponent needs to be >= 0')
        self.degree = degree
        self.domain = domain

    def __call__(self, t):
        self.check_call_input(t)
        return legendre(self.degree)(t[self.index]/self.domain)

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        if direction == self.index:
            return ((1/self.domain)*(legendre(self.degree).deriv(1)))(t[self.index]/self.domain)
        return 0.0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)
        if direction1 == self.index and direction2 == self.index:
            return ((1/self.domain**2)*(legendre(self.degree).deriv(2)))(t[self.index]/self.domain)
        return 0.0


class Sin(OneCoordinateFunction):
    def __init__(self, index: int, alpha: float, dimension: int=None):
        """
        Sine function.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the sine function
        alpha : float
            prefactor
        """
        super(Sin, self).__init__(index, dimension)
        self.index = index
        self.alpha = alpha

    def __call__(self, t):
        self.check_call_input(t)
        return np.sin(self.alpha * t[self.index])

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        if direction == self.index:
            return self.alpha * np.cos(self.alpha * t[self.index])
        return 0.0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)
        if direction1 == self.index and direction2 == self.index:
            return -(self.alpha ** 2) * np.sin(self.alpha * t[self.index])
        return 0.0


class Cos(OneCoordinateFunction):
    def __init__(self, index: int, alpha: float, dimension: int=None):
        """
        Cosine function.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the cosine function

        alpha : float
            prefactor
        """
        super(Cos, self).__init__(index, dimension)
        self.alpha = alpha

    def __call__(self, t):
        self.check_call_input(t)
        return np.cos(self.alpha * t[self.index])

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        if direction == self.index:
            return -self.alpha * np.sin(self.alpha * t[self.index])
        return 0.0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)
        if direction1 == self.index and direction2 == self.index:
            return -(self.alpha ** 2) * np.cos(self.alpha * t[self.index])
        return 0.0


class GaussFunction(OneCoordinateFunction):
    def __init__(self, index: int, mean: float, variance: float, dimension: int=None):
        """
        Gauss function.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the Gauss function

        mean : float
            mean of the distribution

        variance : float

        dimension : int, optional
        """
        super(GaussFunction, self).__init__(index, dimension)
        self.mean = mean
        if variance <= 0:
            raise ValueError('variance must be > 0')
        self.variance = variance

    def __call__(self, t):
        self.check_call_input(t)
        return np.exp(-0.5 * (t[self.index] - self.mean) ** 2 / self.variance)

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        if direction == self.index:
            return -np.exp(-(0.5 * (self.mean - t[self.index]) ** 2) / self.variance) * \
                   (-self.mean + t[self.index]) / self.variance
        return 0.0

    def partial2(self, t, direction1, direction2):
        self.check_partial2_input(t, direction1, direction2)
        if direction1 == self.index and direction2 == self.index:
            return ((1.0/self.variance**2) * (t[self.index] - self.mean)**2 - (1.0/self.variance)) * \
                   np.exp(-0.5 * (t[self.index] - self.mean) ** 2 / self.variance)
        return 0.0


class PeriodicGaussFunction(OneCoordinateFunction):
    def __init__(self, index: int, mean: float, variance: float, dimension: int=None):
        """
        Periodic Gauss function.

        Parameters
        ----------
        index : int
            define which entry of a snapshot is passed to the periodic Gauss function
            
        mean : float
            mean of the distribution
            
        variance : float
        
        dimension : int, optional
        """
        super(PeriodicGaussFunction, self).__init__(index, dimension)
        self.mean = mean
        if variance <= 0:
            raise ValueError('variance must be > 0')
        self.variance = variance

    def __call__(self, t):
        self.check_call_input(t)
        return np.exp(-0.5 * np.sin(0.5 * (t[self.index] - self.mean)) ** 2 / self.variance)

    def partial(self, t, direction):
        self.check_partial_input(t, direction)
        if direction == self.index:
            return (0.5 * np.exp(-(0.5 * np.sin(0.5 * self.mean - 0.5 * t[self.index]) ** 2) / self.variance) *
                    np.cos(0.5 * self.mean - 0.5 * t[self.index]) * np.sin(0.5 * self.mean - 0.5 * t[self.index])) \
                   / self.variance
        return 0.0

    def partial2(self, t, direction1, direction2):
        raise NotImplementedError('not yet implemented')


class Bspline(OneCoordinateFunction):
    def __init__(self, index: int, knots: Union[list, np.ndarray], 
                 degree: int, coeff: Union[list, np.ndarray], dimension: int=None):
        """
        B-Spline basis function.

        Parameters:
        ------------
        index : int
            define which entry of a snapshot is passed to the spline function

        knots : list or np.ndarray(n+1,)
            grid points where the pieces of the spline meet
            NOTE: this is not the extended knot vector, which is constructed internally

        degree : int
            degree of each piecewise polynomial

        coeff : list or np.ndarray(n + degree,)
            coefficient vector with respect to B-Spline basis

        """
        super(Bspline, self).__init__(index, dimension)

        self.knots  = knots
        self.n      = len(self.knots) - 1
        self.degree = degree

        if not len(coeff) == (self.n + self.degree):
            raise ValueError('Coefficient vector does not match dimension of spline space, which is %d'%(
                    self.n + self.degree))

        self.coeff = coeff

        # Construct extended knot vector:
        self.t = np.concatenate((self.degree * [self.knots[0]], self.knots, self.degree * [self.knots[-1]]))

        # Construct spline:
        self.bsp = BSpline(self.t, self.coeff, self.degree)
        self.bsp1 = self.bsp.derivative(1)

    def __call__(self, t):
        self.check_call_input(t)
        return self.bsp(t[self.index])

    def partial(self, t, direction):
        self.check_partial_input(t, direction)

        if direction == self.index:
            return self.bsp1(t[self.index])
        return 0.0

    def partial2(self, t, direction1, direction2):
        raise NotImplementedError('not yet implemented')

# ################################## decompositions ###################################
def basis_decomposition(x: np.ndarray, 
                        phi: List[List[Function]], 
                        single_core: int=None) -> Union['TT', np.ndarray]:

    """Construct a transformed data tensor in TT format.

    Given a set of basis functions phi, construct a TT decomposition psi of the tensor

              -                    -         -                      -
              | phi[0][0](x[:,0])  |         | phi[p-1][0](x[:,0])  |
        psi = |      ...           | x ... x |      ...             | x e_{1}
              | phi[0][-1](x[:,0]) |         | phi[p-1][-1](x[:,0]) |
              -                    -         -                      -

                      -               -         -                 -
                      | phi[0][0](x[:,m-1])  |         | phi[p-1][0](x[:,m-1])  |
              + ... + |      ...             | x ... x |      ...               | x e_{m}
                      | phi[0][-1](x[:,m-1]) |         | phi[p-1][-1](x[:,m-1]) |
                      -                      -         -                        -

    where e_{1}, ... ,e_{m} are the m-dimensional canonical unit vectors. See [1]_ for details.

    Parameters
    ----------
    x: np.ndarray
        snapshot matrix of size d x m

    phi: list[list[Function]]
        list of basis functions in every mode

    single_core: None or int, optional
        return only the ith core of psi if single_core=i (<p), default is None

    Returns
    -------
    psi: TT or np.ndarray
        tensor train of basis function evaluations if single_core=None, 4-dimensional array if single core
        is an integer

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           Journal of Computational and Nonlinear Dynamics 14, 2019
    """

    # number of snapshots
    m = x.shape[1]

    # number of modes
    p = len(phi)

    # mode dimensions
    n = [len(phi[i]) for i in range(p)]

    if single_core is None:

        # define cores as a list of empty arrays
        cores = [np.zeros([1, n[0], 1, m])] + [np.zeros([m, n[i], 1, m]) for i in range(1, p)]

        # insert elements of first core
        for j in range(m):
            # apply first list of basis functions to all snapshots
            cores[0][0, :, 0, j] = np.array([phi[0][k](x[:, j]) for k in range(n[0])])

        # insert elements of subsequent cores
        for i in range(1, p):

            for j in range(m):
                # apply ith list of basis functions to all snapshots
                cores[i][j, :, 0, j] = np.array([phi[i][k](x[:, j]) for k in range(n[i])])

        # append core containing unit vectors
        cores.append(np.eye(m)[:,:,None,None])

        # construct tensor train
        psi = TT(cores)

    elif single_core == 0:

        # define core
        psi = np.zeros([1, n[0], 1, m])

        # insert elements
        for j in range(m):

            # apply basis functions
            psi[0, :, 0, j] = np.array([phi[0][k](x[:, j]) for k in range(n[0])])

    else:

        # define core
        psi = np.zeros([m, n[single_core], 1, m])

        # insert elements
        for j in range(m):

            # apply basis functions
            psi[j, :, 0, j] = np.array([phi[single_core][k](x[:, j]) for k in range(n[single_core])])

    return psi


def coordinate_major(x: np.ndarray, phi: List[Function], single_core: int=None) -> 'TT':
    """Construct a transformed data tensor in TT format using the coordinate-major approach.

    Given a set of basis functions phi, construct a TT decomposition psi of the form::

              -                 -         -                  -
              | phi[0](x[0,0])  |         | phi[0](x[-1,0])  |
        psi = |      ...        | x ... x |      ...         | x e_{1}
              | phi[-1](x[0,0]) |         | phi[-1](x[-1,0]) |
              -                 -         -                  -

                      -                   -         -                    -
                      | phi[0](x[0,m-1])  |         | phi[0](x[-1,m-1])  |
              + ... + |      ...          | x ... x |      ...           | x e_{m}
                      | phi[-1](x[0,m-1]) |         | phi[-1](x[-1,m-1]) |
                      -                   -         -                    -

    where e_{1}, ... ,e_{m} are the m-dimensional canonical unit vectors. See [1]_ for details.

    Parameters
    ----------
    x: np.ndarray
        snapshot matrix of size d x m

    phi: list[Function]
        list of basis functions

    single_core: None or int, optional
        return only the ith core of psi if single_core=i (<p), default is None

    Returns
    -------
    psi: TT
        tensor train of basis function evaluations

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # number of snapshots
    m = x.shape[1]

    # number of modes
    p = len(phi)

    # number of dimensions
    d = x.shape[0]

    if single_core is None:

        # define cores as list of empty arrays
        cores = [np.zeros([1, p, 1, m])] + [np.zeros([m, p, 1, m]) for _ in range(1, d)]

        # insert elements of first core
        for j in range(m):
            cores[0][0, :, 0, j] = np.array([phi[k](x[0, j]) for k in range(p)])

        # insert elements of subsequent cores
        for i in range(1, d):
            for j in range(m):
                cores[i][j, :, 0, j] = np.array([phi[k](x[i, j]) for k in range(p)])

        # append core containing unit vectors
        cores.append(np.eye(m).reshape(m, m, 1, 1))

        # construct tensor train
        psi = TT(cores)

    elif single_core == 0:

        # define core
        psi = np.zeros([1, p, 1, m])

        # insert elements
        for j in range(m):
            # apply basis functions
            psi[0, :, 0, j] = np.array([phi[k](x[0, j]) for k in range(p)])

    else:

        # define core
        psi = np.zeros([m, p, 1, m])

        # insert elements
        for j in range(m):
            # apply basis functions
            psi[j, :, 0, j] = np.array([phi[k](x[single_core, j]) for k in range(p)])

    return psi


def function_major(x: np.ndarray, phi: List[Function], add_one: bool=True, 
                   single_core: int=None) -> 'TT':

    """Construct a transformed data tensor in TT format using the function-major approach.

    Given a set of basis functions phi, construct a TT decomposition psi of the form::

              -                 -         -                  -
              | phi[0](x[0,0])  |         | phi[-1](x[0,0])  |
        psi = |      ...        | x ... x |      ...         | x e_{1}
              | phi[0](x[-1,0]) |         | phi[-1](x[-1,0]) |
              -                 -         -                  -

                      -                   -         -                    -
                      | phi[0](x[0,m-1])  |         | phi[-1](x[0,m-1])  |
              + ... + |      ...          | x ... x |      ...           | x e_{m}
                      | phi[0](x[-1,m-1]) |         | phi[-1](x[-1,m-1]) |
                      -                   -         -                    -

    where e_{1}, ... ,e_{m} are the m-dimensional canonical unit vectors. See [1]_ for details.

    Parameters
    ----------
    x : np.ndarray
        snapshot matrix of size d x m

    phi : list[Function]
        list of basis functions

    add_one: bool, optional
        whether to add the basis function 1 to the cores or not, default is True

    Returns
    -------
    psi: TT
        tensor train of basis function evaluations

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # number of snapeshots
    m = x.shape[1]

    # number of modes
    p = len(phi)

    # number of dimensions
    d = x.shape[0]

    if single_core is None:

        # define cores as list of empty arrays
        cores = [np.zeros([1, d + add_one, 1, m])] + [np.zeros([m, d + add_one, 1, m]) for _ in range(1, p)]

        # insert elements of first core
        if add_one is True:
            for j in range(m):
                cores[0][0, 0, 0, j] = 1
                cores[0][0, 1:, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])
        else:
            for j in range(m):
                cores[0][0, :, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])

        # insert elements of subsequent cores
        for i in range(1, p):
            if add_one is True:

                for j in range(m):
                    cores[i][j, 0, 0, j] = 1
                    cores[i][j, 1:, 0, j] = np.array([phi[i](x[k, j]) for k in range(d)])

            else:
                for j in range(m):
                    cores[i][j, :, 0, j] = np.array([phi[i](x[k, j]) for k in range(d)])

        # append core containing unit vectors
        cores.append(np.eye(m).reshape(m, m, 1, 1))

        # construct tensor train
        psi = TT(cores)

    elif single_core == 0:

        # define core
        psi = np.zeros([1, d + add_one, 1, m])

        # insert elements
        if add_one is True:

            for j in range(m):
                psi[0, 0, 0, j] = 1
                psi[0, 1:, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])
        else:
            for j in range(m):
                psi[0, :, 0, j] = np.array([phi[0](x[k, j]) for k in range(d)])

    else:

        # define core
        psi = np.zeros([m, d + add_one, 1, m])

        # insert elements
        if add_one is True:

            for j in range(m):
                psi[j, 0, 0, j] = 1
                psi[j, 1:, 0, j] = np.array([phi[single_core](x[k, j]) for k in range(d)])
        else:
            for j in range(m):
                psi[j, :, 0, j] = np.array([phi[single_core](x[k, j]) for k in range(d)])

    return psi


def gram(x_1: np.ndarray, x_2: np.ndarray, basis_list: List[List[Function]]) -> np.ndarray:
    """Gram matrix.

    Compute the Gram matrix of two transformed data tensors psi_1=psi(x_1) and psi_2=psi(x_2), i.e., psi_1^T@psi_2. See
    _[1] for details.

    Parameters
    ----------
    x_1: np.ndarray
        data matrix for psi_1

    x_2: np.ndarray
        data matrix for psi_2

    basis_list: list[list[Function]]
        list of basis functions in every mode

    Returns
    -------
    gram: np.ndarray
        Gram matrix

    References
    ----------
    .. [1] S. Klus, P. Gelß, "Tensor-Based Algorithms for Image Classification", Algorithms, 2019
    """
    # compute gram by iteratively applying the Hadarmard product
    gram = np.ones([x_1.shape[1], x_2.shape[1]])

    for i in range(len(basis_list)):
        theta_1 = np.array([[basis_list[i][k](x_1[:, j]) for j in range(x_1.shape[1])]
                            for k in range(len(basis_list[i]))])

        theta_2 = np.array([[basis_list[i][k](x_2[:, j]) for j in range(x_2.shape[1])]
                            for k in range(len(basis_list[i]))])

        # theta_1 = np.zeros((len(basis_list[i]), x_1.shape[1]))
        # theta_2 = np.zeros((len(basis_list[i]), x_2.shape[1]))
        # for k in range(len(basis_list[i])):
        #     theta_1[k, :] = [basis_list[i][k](x_1[:, j]) for j in range(x_1.shape[1])]
        #     theta_2[k, :] = [basis_list[i][k](x_2[:, j]) for j in range(x_2.shape[1])]
        gram *= (theta_1.T.dot(theta_2))

    return gram


def hocur(x: np.ndarray, basis_list: List[List[Function]], 
          ranks: Union[List[int], int], repeats: int=1, multiplier: int=10,
          progress: bool=True, string: str=None) -> 'TT':

    """Higher-order CUR decomposition of transformed data tensors.

    Given a snapshot matrix x and a list of basis functions in each mode, construct a TT decomposition of the
    transformed data tensor Psi(x) using a higher-order CUR decomposition and maximum-volume subtensors. See [1]_, [2]_
    and [3]_ for details.

    Parameters
    ----------
    x: np.ndarray
        data matrix

    basis_list: list[list[Function]]
        list of basis functions in every mode

    ranks: list[int] or int
        maximum TT ranks of the resulting TT representation; if type is int, then the ranks are set to
        [1, ranks, ..., ranks, 1]; note that - depending on the number of linearly independent rows/columns that have
        been found - the TT ranks may be reduced during the decomposition

    repeats: int, optional
        number of repeats, default is 1

    multiplier: int, optional
        multiply the number of initially chosen column indices (given by ranks) in order to increase the probability of
        finding a 'full' set of linearly independent columns; default is 10

    progress: bool, optional
        whether to show the progress of the algorithm or not, default is True

    string: string
        string to print; if None (default), then print 'HOCUR (repeats: <repeats>)'

    Returns
    -------
    psi: TT
        TT representation of the transformed data tensor

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           Journal of Computational and Nonlinear Dynamics 14, 2019
    .. [2] I. Oseledets, E. Tyrtyshnikov, "TT-cross approximation for multidimensional arrays", Linear Algebra and its
           Applications 432, 2010
    .. [3] S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov, N. L. Zamarashkin, "How to find a
           good submatrix", Matrix Methods: Theory, Algorithms, Applications, 2010
    """

    # parameters
    # ----------

    # number of snapshots
    m = x.shape[1]

    # number of modes
    p = len(basis_list)

    # mode dimensions
    n = [len(basis_list[k]) for k in range(p)] + [m]

    # ranks
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(n) - 1)] + [1]

    # cut ranks larger than number of snapshots
    ranks = [np.minimum(ranks[k], m) for k in range(len(n)+1)]

    # initial definitions
    # -------------------

    # define initial lists of column indices
    col_inds = __hocur_first_col_inds(n, ranks, multiplier)

    # define list of cores
    cores = [None] * (p + 1)

    # show progress
    # -------------

    if string is None:
        string = 'HOCUR'

    start_time = utl.progress(string, 0, show=progress)

    # start decomposition
    # -------------------

    for k in range(repeats):

        row_inds = [None]

        # first half sweep
        for i in range(p):

            # extract submatrix
            y = __hocur_extract_matrix(x, basis_list, row_inds[i], col_inds[i])

            if k == 0:
                # find linearly independent columns
                cols = __hocur_find_li_cols(y)
                cols = cols[:ranks[i + 1]]
                y = y[:, cols]

            # find optimal rows
            rows = __hocur_maxvolume(y)  # type: list

            # adapt ranks if necessary
            ranks[i + 1] = len(rows)

            if i == 0:

                # store row indices for first dimensions
                row_inds.append([[rows[j]] for j in range(ranks[i + 1])])

            else:

                # convert rows to multi indices
                multi_indices = np.array(np.unravel_index(rows, (ranks[i], n[i])))

                # store row indices for dimensions m_1, n_1, ..., m_i, n_i
                row_inds.append([row_inds[i][multi_indices[0, j]] + [multi_indices[1, j]] for j in
                                 range(ranks[i + 1])])

            # define core
            if len(rows) < y.shape[1]:
                y = y[:, :len(rows)]

            u_inv = np.linalg.inv(y[rows, :].copy())
            cores[i] = y.dot(u_inv).reshape([ranks[i], n[i], 1, ranks[i + 1]])

            # show progress
            utl.progress(string + ' ... r=' + str(ranks[i + 1]), 100 * (k * 2 * p + i + 1) / (repeats * 2 * p),
                         cpu_time=_time.time() - start_time,
                         show=progress)

        # second half sweep
        for i in range(p, 0, -1):

            # extract submatrix
            y = __hocur_extract_matrix(x, basis_list, row_inds[i], col_inds[i]).reshape([ranks[i], n[i] * ranks[i + 1]])

            # find optimal rows
            cols = __hocur_maxvolume(y.T)  # type: list

            # adapt ranks if necessary
            ranks[i] = len(cols)

            if i == p:

                # store row indices for first dimensions
                col_inds[p - 1] = [[cols[j]] for j in range(ranks[i])]

            else:

                # convert cols to multi indices
                multi_indices = np.array(np.unravel_index(cols, (n[i], ranks[i + 1])))

                # store col indices for dimensions m_i, n_i, ... , m_d, n_d
                col_inds[i - 1] = [[multi_indices[0, j]] + col_inds[i][multi_indices[1, j]] for j in range(ranks[i])]

            # define TT core
            if len(cols) < y.shape[0]:
                y = y[:len(cols), :]
            u_inv = np.linalg.inv(y[:, cols].copy())
            cores[i] = u_inv.dot(y).reshape([ranks[i], n[i], 1, ranks[i + 1]])

            # show progress
            utl.progress(string, 100 * ((k + 1) * 2 * p - i + 1) / (repeats * 2 * p),
                         cpu_time=_time.time() - start_time, show=progress)

        # define first core
        y = __hocur_extract_matrix(x, basis_list, None, col_inds[0])
        cores[0] = y.reshape([1, n[0], 1, ranks[1]])

    # construct tensor train
    # ----------------------

    psi = TT(cores)

    return psi

# private functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def __hocur_first_col_inds(dimensions: List[int], 
                           ranks: List[int], 
                           multiplier: int) -> List[List[int]]:

    """Create random column indices

    Parameters
    ----------
    dimensions: list[int]
        dimensions of a given tensor

    ranks: list[int]
        ranks for decomposition

    multiplier: int
        multiply the number of initially chosen column indices (given by ranks) in order to increase the probability of
        finding a 'full' set of linearly independent columns

    Returns
    -------
    col_inds: list[list[int]]
        array containing single indices
    """

    # define list of column indices
    col_inds = [None]

    # insert column indices for last dimension
    col_inds.insert(0, [[j] for j in range(np.minimum(multiplier * ranks[-2], dimensions[-1]))])

    for i in range(len(dimensions) - 3, -1, -1):
        # define array of flat indices
        flat_inds = np.arange(np.minimum(multiplier * ranks[i + 1], dimensions[i + 1] * ranks[i + 2]))

        # convert flat indices to tuples
        multi_inds = np.array(np.unravel_index(flat_inds, (dimensions[i + 1], ranks[i + 2])))

        # insert column indices
        col_inds.insert(0, [[multi_inds[0, j]] + col_inds[0][multi_inds[1, j]] for j in range(multi_inds.shape[1])])

    return col_inds


def __hocur_extract_matrix(data: np.ndarray, 
                           basis_list: List[List[Function]], 
                           row_coordinates_list: List[List[int]], 
                           col_coordinates_list: List[List[int]]) -> np.ndarray:

    """Extraction of a submatrix of a transformed data tensor.

    Given a set of row and column coordinates, extracts a submatrix from the transformed data tensor corresponding to
    the data matrix x and the set of basis functions stored in basis_list.

    Parameters
    ----------
    data: np.ndarray
        data matrix

    basis_list: list[list[Function]]
        list of basis functions in every mode

    row_coordinates_list: list[list[int]]
        list of row indices

    col_coordinates_list: list[list[int]]
        list of column indices

    Returns
    -------
    matrix: np.ndarray
        extracted matrix
    """

    # construction of the first submatrix
    if row_coordinates_list is None:

        # define current index and mode size
        current_index = 0
        current_mode = len(basis_list[0])

        # define number of row and column sets
        n_rows = 1
        n_cols = len(col_coordinates_list)

        # initialize submatrix
        matrix = np.zeros([n_rows * current_mode, n_cols])

        # construct submatrix
        for j in range(n_cols):

            # set current column index set
            col_coordinates = col_coordinates_list[j]

            # set current snapshot
            snapshot = data[:, col_coordinates[-1]]

            # compute right coefficient
            right_part = 1
            for k in range(len(col_coordinates) - 1):
                right_part *= basis_list[k + current_index + 1][col_coordinates[k]](snapshot)

            # compute vector between left and right part
            middle_part = np.array([basis_list[0][k](snapshot) for k in range(current_mode)])

            # insert matrix entries
            matrix[0:current_mode, j] = middle_part * right_part

    # construction of the last submatrix
    elif col_coordinates_list is None:

        # define current mode size
        current_mode = data.shape[1]

        # define number of row and columns sets
        n_rows = len(row_coordinates_list)
        n_cols = 1

        # initialize submatrix
        matrix = np.zeros([n_rows * current_mode, n_cols])

        # construct submatrix
        for i in range(n_rows):

            # set current row index set
            row_coordinates = row_coordinates_list[i]

            # compute left coefficient
            left_part = 1
            for k in range(len(row_coordinates)):
                left_part *= basis_list[k][row_coordinates[k]](data)

            # insert matrix entries
            matrix[i * current_mode:(i + 1) * current_mode, 0] = left_part

    # construction of the intermediate submatrices
    else:

        # define current index and mode size
        current_index = len(row_coordinates_list[0])
        current_mode = len(basis_list[current_index])

        # define number of row and column sets
        n_rows = len(row_coordinates_list)
        n_cols = len(col_coordinates_list)

        # initialize submatrix
        matrix = np.zeros([n_rows * current_mode, n_cols])

        # construct submatrix
        for j in range(n_cols):

            # set current column index set
            col_coordinates = col_coordinates_list[j]

            # set current snapshot
            snapshot = data[:, col_coordinates[-1]]

            # compute right coefficient
            right_part = 1
            for k in range(len(col_coordinates) - 1):
                right_part *= basis_list[k + current_index + 1][col_coordinates[k]](snapshot)

            # loop over rows
            for i in range(n_rows):

                # set current row index set
                row_coordinates = row_coordinates_list[i]

                # compute left coefficient
                left_part = 1
                for k in range(len(row_coordinates)):
                    left_part *= basis_list[k][row_coordinates[k]](snapshot)

                # compute vector between left and right part
                middle_part = np.array([basis_list[current_index][k](snapshot) for k in range(current_mode)])

                # insert matrix entries
                matrix[i * current_mode:(i + 1) * current_mode, j] = left_part * middle_part * right_part

    return matrix


def __hocur_find_li_cols(matrix: np.ndarray, tol: float=1e-14) -> List[int]:
    """Find linearly independent columns of a matrix.

    Parameters
    ----------
    matrix: np.ndarray
        rectangular matrix, (m,n)

    Returns
    -------
    cols: list[int]
        indices of linearly independent columns
    """

    # define column list
    cols = []

    # apply QR decomposition with pivoting
    _, r, p = splin.qr(matrix, pivoting=True, mode='economic')

    if tol == 0:
        cols = [p[i] for i in range(matrix.shape[0])]
    else:
        for i in range(r.shape[0]):
            if np.abs(r[i, i]) > tol:
                cols.append(p[i])

    return cols


def __hocur_maxvolume(matrix: np.ndarray, 
                      maximum_iterations: int=1000,
                      tolerance: float=1e-5) -> List[int]:

    """Find dominant submatrix.

    Find rows of a given rectangular matrix which build a maximum-volume submatrix, see [1]_.

    Parameters
    ----------
    matrix: np.ndarray (n,r)
        rectangular matrix with rank r

    maximum_iterations: int
        maximum number of iterations, default is 100

    tolerance: float
        tolerance for stopping criterion, default is 1e-5

    Returns
    -------
    rows: list[int]
        rows of the matrix which build the dominant submatrix

    References
    ----------
    .. [1] S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov, N. L. Zamarashkin, "How to find a
           good submatrix", Matrix Methods: Theory, Algorithms, Applications, 2010
    """

    # set max_value and iteration_counter
    max_val = np.inf
    iteration_counter = 1

    # find linearly independent rows
    rows = __hocur_find_li_cols(matrix.T, tol=0)  # type: list

    # repeat row swapping until tolerance is reached
    while max_val > 1 + tolerance and iteration_counter <= maximum_iterations:
        # extract submatrix corresponding to given rows and invert
        submatrix = matrix[rows, :]

        submatrix_inv = np.linalg.inv(submatrix)

        # find maximum absolute value and corresponding indices of matrix @ submatrix^-1
        product = matrix.dot(submatrix_inv)
        max_inds = np.unravel_index(np.argmax(np.abs(product)), product.shape)
        max_val = product[max_inds[0], max_inds[1]]

        # replace row
        rows[int(max_inds[1])] = max_inds[0]

        # increase iteration counter
        iteration_counter += 1

    return rows
