# scikit_tt

A toolbox for tensor train calculations.

## Short description

The simulation and analysis of high-dimensional problems is often infeasible due to the curse of dimensionality. Using the *tensor-train format* (TT format) [1,2] and tensor-based solvers, **scikit_tt** can be applied to various numerical problems in order to reduce the memory consumption and the computational costs compared to classical approaches significantly. Possible application areas are the computation of low-rank approximations for high-dimensional systems, solving systems of linear equations and eigenvalue problems in the TT format, representing operators based on nearest-neighbor interactions in the TT format, constructing pseudoinverses for tensor-based reformulations of dimensionality reduction methods, and the approximation of transfer operators as well as governing equations of dynamical systems.

## Content

1. Getting started
2. TT class
3. TT solvers
4. SLIM decomposition
5. Multidimensional approximation of nonlinear dynamical systems (MANDy)
6. Models
7. Examples
8. Tests
9. Subfunctions and tools
10. Additional information
11. References

## 1. Installing

To install scikit_tt type

`...`

## 2. TT class

Tensor trains [1] are defined in terms of different attributes. In particular, the attribute 'cores' is a list of 4-dimensional tensors representing the corresponding TT cores. There is no distinguish between tensor trains and tensor-train operators, i.e. a classical tensor train is represented by cores with column dimensions equal to 1. An instance of the tensor train class can be initialized from a full tensor representation (in this case, the tensor is decomposed into the TT format) or from a list of cores. For more information on the implemented tensor operations, we refer to [2].

... implemented in tensor_train.py ...
... list of modules ...

## 3. TT solvers

### 3.1 Systems of linear equations

In order to approximate the solution of a system of linear equations in the TT format, the *alternating linear scheme* (ALS) and the *modified alternating linear scheme* (MALS) are implemented in **scikit_tt**. The basic idea is to fix all components of the tensor network except for one (or two in the MALS case). This yields a series of low-dimensional problems, which can then be solved using classical numerical methods. For details, see [3].

### 3.2 Eigenvalue problems

ALS and MALS can also be used to find approximations of eigenvalues and corresponding eigentensors of TT operators. The basic procedures of ALS and MALS for eigenvalue problems are similar to the ones for systems of linear equations. The main difference is the type of optimization problem which has to be solved in the iteration steps. For details, see [3].

**TODO: _add MALS for eigenvalue problems_**

### 3.3 Ordinary differential equations

In order to compute time-dependent or stationary distributions of ordinary differential equations given by Markovian master equations, **scikit_tt** uses implicit integration schemes such as the implicit Euler method or the trapezoidal rule. In order to approximate the solutions at each time step, ALS and MALS, respectively, are used. In addition to the fact that implicit integration schemes are more suitable for the solution of stiff equations than explicit methods, we have control over the ranks of the TT approximations without implementing computationally expensive tensor multiplications and subsequent rank truncations.

**TODO: _revise code_**

## 4. SLIM decomposition

## 5. Multidimensional approximation of nonlinear dynamical systems (MANDy)

## 6. Models

## 7. Examples

Numerical experiments from different application areas are included in **scikit_tt**. For instance, the application of the TT format to the chemical master equation, heterogeneous catalytic process, fluid dynamics, and molecular dynamics can be found in the directory scikit_tt/examples.

**TODO: _revise code_**

## 8. Tests

## 9. Subfunctions and tools

## 10. Additional information

### 10.1 Authors 

* **Patrick Gelß** - _initial work_ - Freie Universität Berlin
* **Stefan Klus**
* **Martin Scherer**

### 10.2 Built with

* PyCharm (+Link)

### 10.3 License

This project is licensed under the ... license - see LICENSE (+Link) for details.

### 10.4 Acknowledgments

* ...

## 11. References

[1] I. V. Oseledets, "Tensor-Train Decomposition", SIAM Journal on Scientific Computing 33 (5) (2011) 2295-2317

[2] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017

[3] S. Holtz, T. Rohwedder, R. Schneider, "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", SIAM Journal on Scientific Computing 34 (2) (2012) A683-A713

[4] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-Neighbor Interaction Systems in the Tensor-Train Format", Journal of Computational Physics 341 (2017) 140-162
