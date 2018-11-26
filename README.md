# scikit_tt

A toolbox for tensor train calculations.

## Short description



## Content

1. TT classs
2. TT solvers
3. SLIM decomposition
4. Multidimensional approximation of nonlinear dynamical systems (MANDy)
5. Models
6. Examples
7. Tests

## 1. TT class

Tensor trains [1]_ are defined in terms of different attributes. In particular, the attribute 'cores' is a list of
4-dimensional tensors representing the corresponding TT cores. There is no distinguish between tensor trains and
tensor trains operators, i.e. a classical tensor train is represented by cores with column dimensions equal to 1.
An instance of the tensor train class can be initialized from a full tensor representation (in this case, the tensor
is decomposed into the TT format) or from a list of cores. For more information on the implemented tensor
operations, we refer to [2]_.
