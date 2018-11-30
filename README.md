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
11. Acknowledgments
12. References

## 1. Installing

A *setup.py* is included in the package. To install **scikit_tt** simply enter:

```
python setup.py install --user
```

## 2. TT class

The tensor-train class - implemented in the module *tensor-train.py* - is the core of **scikit_tt** and enables us to work with the tensor-train format. We define tensor trains in terms of different attributes such as *order*, *row_dims*, *col_dims*, *ranks*, and *cores*. An overview of the member functions of the class is shown in the following list.

```
TT ................... construct tensor train from array or list of cores
print ................ print the attributes of a given tensor train
+,-,*,@ .............. basic operations on tensor trains 
copy ................. deep copy of a tensor train
full ................. convert a tensor train to full format
element .............. compute single element of a tensor train
transpose ............ transpose of a tensor train
isoperator ........... check if a tensor train is an operator
zeros ................ construct a tensor train of all zeros
ones ................. construct a tensor train of all ones
eye .................. construct an identity tensor train
rand ................. construct a random tensor train
uniform .............. construct a uniformly distributed tensor train
ortho_left ........... left-orthonormalize a tensor train
ortho_right .......... right-orthonormalize a tensor train
matricize ............ matricize a tensor train
norm ................. compute the norm of a tensor train
tt2qtt ............... convert from TT to QTT format
qtt2tt ............... convert from QTT to TT format
```

## 3. TT solvers

### 3.1 Systems of linear equations

In order to approximate the solution of a system of linear equations in the TT format, a series of low-dimensional problems can be solved by fixing certain components of the tensor network. For this purpose, the *alternating linear scheme* (ALS) and the *modified alternating linear scheme* (MALS) [3] are implemented in *solvers/sle.py*.

```
als .................. alternating linear scheme for solving systems of linear equations in the TT format
mals ................. modified alternating linear scheme for solving systems of linear equations in the TT format
```

### 3.2 Eigenvalue problems

ALS and MALS can also be used to find approximations of eigenvalues and corresponding eigentensors of TT operators. The basic procedures of ALS and MALS - implemented in *solvers/evp.py* - for eigenvalue problems are similar to the ones for systems of linear equations. The main difference is the type of optimization problem which has to be solved in the iteration steps. See [3] for details. 

```
als .................. alternating linear scheme for solving eigenvalue problems in the TT format
```

**TODO: _add MALS for eigenvalue problems_ / _implement solvers for generalized EVPs_**

### 3.3 Ordinary differential equations

In order to compute time-dependent or stationary distributions of linear differential equations in the TT format, **scikit_tt** uses implicit integration schemes such as the implicit Euler method or the trapezoidal rule. In order to approximate the solutions at each time step, ALS and MALS, respectively, are used. The methods can be found in *solvers/ode.py*.

**TODO: _revise code_ / _combine ALS/MALS methods_ / _explicit methods?_**

## 4. SLIM decomposition

## 5. Multidimensional approximation of nonlinear dynamical systems (MANDy)

## 6. Models

```
signaling_cascade .... cascading process on a genetic network consisting of genes of different species
two_step_destruction . two-step mechanism for the destruction of molecules
co_oxidation ......... CO oxidation on a RuO2 surface
```

## 7. Examples

Numerical experiments from different application areas are included in **scikit_tt**. For instance, the application of the TT format to the chemical master equation, heterogeneous catalytic process, fluid dynamics, and molecular dynamics can be found in the directory *examples*.

**TODO: _revise code_**

## 8. Tests

## 9. Subfunctions and tools

## 10. Additional information

### 10.1 Authors 

* **Patrick Gelß** - _major contribution_ - Freie Universität Berlin, CRC 1114
* **Stefan Klus** - _initial work_ - Freie Universität Berlin, CRC 1114
* **Martin Scherer** - _setup_ - Freie Universität Berlin, Computational Molecular Biology

### 10.2 Built with

* [PyCharm](https://www.jetbrains.com/pycharm/)

### 10.3 License

This project is licensed under the [LGPLv3+](https://www.gnu.org/licenses/lgpl-3.0.en.html) license - see [LICENSE](LICENSE) for details.

### 10.4 Versioning

We use ... for versioning. For the available versions, see ...

## 11. Acknowledgments

* ...

## 12. References

[1] I. V. Oseledets, "Tensor-Train Decomposition", SIAM Journal on Scientific Computing 33 (5) (2011) 2295-2317

[2] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017

[3] S. Holtz, T. Rohwedder, R. Schneider, "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", SIAM Journal on Scientific Computing 34 (2) (2012) A683-A713

[4] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-Neighbor Interaction Systems in the Tensor-Train Format", Journal of Computational Physics 341 (2017) 140-162
