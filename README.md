# scikit_tt

A toolbox for tensor train computations.

## Short description

The simulation and analysis of high-dimensional problems is often infeasible due to the curse of dimensionality. Using the *tensor-train format* (TT format) [[1](README.md#11-references), [2](README.md#11-references)] and tensor-based solvers, **scikit_tt** can be applied to various numerical problems in order to reduce the memory consumption and the computational costs compared to classical approaches significantly. Possible application areas are the computation of low-rank approximations for high-dimensional systems, solving systems of linear equations and eigenvalue problems in the TT format, representing operators based on nearest-neighbor interactions in the TT format, constructing pseudoinverses for tensor-based reformulations of dimensionality reduction methods, and the approximation of transfer operators as well as governing equations of dynamical systems.

## Content

1. [Installing](README.md#1-installing)
2. [TT class](README.md#2-tt-class) 
3. [TT solvers](README.md#3-tt-solvers)
   - [Systems of linear equations](README.md#31-systems-of-linear-equations)
   - [Eigenvalue problems](README.md#32-eigenvalue-problems)
   - [Linear differential equations](README.md#33-linear-differential-equations)
4. [SLIM decomposition](README.md#4-slim-decomposition)
5. [Multidimensional approximation of nonlinear dynamical systems (MANDy)](README.md#5-multidimensional-approximation-of-nonlinear-dynamical-systems-mandy)
6. [Models](README.md#6-models)
7. [Examples](README.md#7-examples)
8. [Tests](README.md#8-tests)
9. [Subfunctions and tools](README.md#9-subfunctions-and-tools)
10. [Additional information](README.md#10-additional-information)
    - [Authors & contact](README.md#101-authors--contact)
    - [Built with](README.md#102-built-with)
    - [License](README.md#103-license)
    - [Versioning](README.md#104-versioning)
11. [References](README.md#11-references)

## 1. Installing

A [*setup.py*](setup.py) is included in the package. To install **scikit_tt** simply enter:

```
python setup.py install --user
```

## 2. TT class

The tensor-train class - implemented in the module [*tensor-train.py*](scikit_tt/tensor_train.py) - is the core of **scikit_tt** and enables us to work with the tensor-train format. We define tensor trains in terms of different attributes such as *order*, *row_dims*, *col_dims*, *ranks*, and *cores*. An overview of the member functions of the class is shown in the following list.

```
TT ................... construct tensor train from array or list of cores
print ................ print the attributes of a given tensor train
+,-,*,@ .............. basic operations on tensor trains 
transpose ............ transpose of a tensor train
isoperator ........... check if a tensor train is an operator
copy ................. deep copy of a tensor train
element .............. compute single element of a tensor train
full ................. convert a tensor train to full format
matricize ............ matricize a tensor train
ortho_left ........... left-orthonormalize a tensor train
ortho_right .......... right-orthonormalize a tensor train
norm ................. compute the norm of a tensor train
tt2qtt ............... convert from TT to QTT format
qtt2tt ............... convert from QTT to TT format
```

Further functions defined in [*tensor-train.py*](scikit_tt/tensor_train.py) are:

```
zeros ................ construct a tensor train of all zeros
ones ................. construct a tensor train of all ones
eye .................. construct an identity tensor train
rand ................. construct a random tensor train
uniform .............. construct a uniformly distributed tensor train
```

## 3. TT solvers

### 3.1 Systems of linear equations

In order to approximate the solution of a system of linear equations in the TT format, a series of low-dimensional problems can be solved by fixing certain components of the tensor network. For this purpose, the *alternating linear scheme* (ALS) and the *modified alternating linear scheme* (MALS) [[3](README.md#11-references)] are implemented in [*solvers/sle.py*](scikit_tt/solvers/sle.py).

```
als .................. alternating linear scheme for systems of linear equations in the TT format
mals ................. modified ALS for systems of linear equations in the TT format
```

### 3.2 Eigenvalue problems

ALS and MALS can also be used to find approximations of eigenvalues and corresponding eigentensors of TT operators. The basic procedures of ALS and MALS - implemented in [*solvers/evp.py*](scikit_tt/solvers/evp.py) - for eigenvalue problems are similar to the ones for systems of linear equations. The main difference is the type of optimization problem which has to be solved in the iteration steps. See [[3](README.md#11-references)]  for details. 

```
als .................. alternating linear scheme for eigenvalue problems in the TT format
```

**TODO: _add MALS for eigenvalue problems_ / _implement solvers for generalized EVPs_**

### 3.3 Linear differential equations

In order to compute time-dependent or stationary distributions of linear differential equations in the TT format, **scikit_tt** uses implicit integration schemes such as the implicit Euler method or the trapezoidal rule. In order to approximate the solutions at each time step, ALS and MALS, respectively, are used. The methods can be found in [*solvers/ode.py*](scikit_tt/solvers/ode.py).

```
implicit_euler ....... implicit Euler method for linear differential equations in the TT format
trapezoidal_rule ..... trapezoidal rule for linear differential equations in the TT format
adaptive_step_size ... adaptive step size method for linear differential equations in the TT format
```

**TODO: _revise code_ / _combine ALS/MALS methods_ / _explicit methods?_**

## 4. SLIM decomposition

The SLIM decomposition is a specific form of TT decompositions which represent tensors networks with a certain structure. For instance, tensor operators corresponding to nearest-neighbor interaction systems can be systematicly decomposed into a tensor-train operator using the algorithms in [*slim.py*](scikit_tt/slim.py). See [[4](README.md#11-references)] for details.

```
slim_gen ............. general SLIM decomposition
slim_mme ............. SLIM decomposition for Markov generators
slim_mme_hom ......... SLIM decomposition for homogeneous Markov generators
```

**TODO: _write routines_**

## 5. Multidimensional approximation of nonlinear dynamical systems (MANDy)

**TODO: _rewrite and check_**

## 6. Models

The construction of several models from various fields is included in [*models.py*](scikit_tt/models.py). 

```
co_oxidation ......... CO oxidation on a RuO2 surface
signaling_cascade .... cascading process on a genetic network consisting of genes of different species
two_step_destruction . two-step mechanism for the destruction of molecules
```

**TODO: _include other models_**

## 7. Examples

Numerical experiments from different application areas are included in **scikit_tt**. For instance, the application of the TT format to chemical master equations, heterogeneous catalytic processes [[5](README.md#11-references)], fluid dynamics, and molecular dynamics can be found in the directory [*examples*](examples/).

```
chua_circuit ......... apply MANDy to Chua's circuit
co_oxidation ......... compute stationary distributions of a catalytic process
fermi_pasta_ulam ..... apply MANDy to the Fermi-Pasta-Ulam problem
fractals ............. create self-similar patterns using tensors
kuramoto ............. apply MANDy to the Kuramoto model
quadruple_well ....... approximate eigenfunctions of the Perron-Frobenius operator in 3D
signaling_cascade .... compute mean concentrations of a 20-dimensional signaling cascade
triple_well .......... approximate eigenfunctions of the Perron-Frobenius operator in 2D
two_step_destruction . apply QTT and MALS to a two-step destruction process
```

**TODO: _revise code_**

## 8. Tests

Modules containing unit tests are provided in the directory [*tests*](tests/).

```
test_tensor_train .... unit tests for tensor_train.py
test_mandy ........... unit tests for mandy.py
test_slim ............ unit tests for slim.py
```

**TODO: _write other tests_**

## 9. Subfunctions and tools

In [*utils.py*](scikit_tt/utils.py) we collect algorithms and tools which are employed at several points in **scikit_tt** and/or helpful for data analysis, comparisons, and visualization.

```
mean_concentration ... mean concentrations of TT series
progress ............. show progress in percent
sindy ................ SINDy for the analysis of nonlinear dynamical systems
timer ................ measure CPU time
turn_over_frequency .. turn-over frequency of probabilty distributions for catalytic reactions
unit_vector .......... canonical unit vector
```

**TODO: _write routines_**

## 10. Additional information

### 10.1 Authors & contact

* **Dr. Patrick Gelß** - _major contribution_ - CRC 1114, Freie Universität Berlin, Germany
  - address: Arnimallee 9, 14195 Berlin, Germany
  - email: p.gelss@fu-berlin.de
* **Dr. Stefan Klus** - _initial work_ - CRC 1114, Freie Universität Berlin, Germany
* **Martin Scherer** - _setup_ - Computational Molecular Biology, Freie Universität Berlin, Germany

### 10.2 Built with

* [PyCharm](https://www.jetbrains.com/pycharm/)

### 10.3 License

This project is licensed under the [LGPLv3+](https://www.gnu.org/licenses/lgpl-3.0.en.html) license - see [LICENSE.txt](LICENSE.txt) for details.

### 10.4 Versioning

We use ... for versioning. For the available versions, see ...

## 11. References

[1] I. V. Oseledets, "Tensor-Train Decomposition", SIAM Journal on Scientific Computing 33 (5) (2011) 2295-2317

[2] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin (2017)

[3] S. Holtz, T. Rohwedder, R. Schneider, "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", SIAM Journal on Scientific Computing 34 (2) (2012) A683-A713

[4] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-Neighbor Interaction Systems in the Tensor-Train Format", Journal of Computational Physics 341 (2017) 140-162

[5] P. Gelß, S. Matera, C. Schütte, "Solving the Master Equation without Kinetic Monte Carlo: Tensor Train Approximations for a CO Oxidation Model", Journal of Computational Physics 314 (2016) 489502

[6] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018) 3359

[7] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems", arXiv:1809.02448 (2018)
