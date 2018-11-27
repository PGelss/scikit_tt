# scikit_tt

A toolbox for tensor train calculations.

## Short description

The simulation and analysis of high-dimensional problems is often infeasible due to the curse of dimensionality. Using the TT format and tensor-based solvers, scikit_tt can be applied to various numerical problems in order to reduce the memory consumption and the computational costs compared to classical approaches significantly. Possible application areas are the computation of low-rank approximations for high-dimensional systems, solving systems of linear equations and eigenvalue problems in the TT format, representing operators based on nearest-neighbor interactions in the TT format, constructing pseudoinverses for tensor-based reformulations of dimensionality reduction methods, and the approximation of transfer operators as well as governing equations of dynamical systems.

## Content

1. TT classs
2. TT solvers
3. SLIM decomposition
4. Multidimensional approximation of nonlinear dynamical systems (MANDy)
5. Models
6. Examples
7. Tests
8. Subfunctions and tools
9. References

## 1. TT class

Tensor trains [1]_ are defined in terms of different attributes. In particular, the attribute 'cores' is a list of
4-dimensional tensors representing the corresponding TT cores. There is no distinguish between tensor trains and
tensor trains operators, i.e. a classical tensor train is represented by cores with column dimensions equal to 1.
An instance of the tensor train class can be initialized from a full tensor representation (in this case, the tensor
is decomposed into the TT format) or from a list of cores. For more information on the implemented tensor
operations, we refer to [2]_.

## 2. TT solvers

## 3. SLIM decomposition

## 4. Multidimensional approximation of nonlinear dynamical systems (MANDy)

## 5. Models

## 6. Examples

## 7. Tests

## 8. Subfunctions and tools

## 9. References

[1] I. V. Oseledets, "Tensor-Train Decomposition", SIAM Journal on Scientific Computing, 2009

[2] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
