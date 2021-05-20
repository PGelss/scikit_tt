![scikit-tt - A toolbox for tensor-train computations](https://raw.githubusercontent.com/PGelss/scikit_tt/master/logo.png)

<p align="center">
  <a href="https://travis-ci.org/PGelss/scikit_tt"><img src="https://img.shields.io/travis/PGelss/scikit_tt.svg"><a>
  <a href="https://codecov.io/gh/PGelss/scikit_tt/branch/master"><img src="https://img.shields.io/codecov/c/github/PGelss/scikit_tt.svg"><a>
</p>

## Short description

The simulation and analysis of high-dimensional problems is often infeasible due to the curse of dimensionality. Using the *tensor-train format* (TT format) [[1](README.md#11-references), [2](README.md#11-references)], **Scikit-TT** can be applied to various numerical problems in order to reduce the memory consumption and the computational costs compared to classical approaches significantly. Possible application areas are:

- the computation of low-rank approximations for high-dimensional systems [[3](README.md#11-references)],
- solving systems of linear equations and eigenvalue problems in the TT format [[4](README.md#11-references)],
- representing operators based on nearest-neighbor interactions in the TT format [[5](README.md#11-references)],
- constructing pseudoinverses for tensor-based reformulations of dimensionality reduction methods [[6](README.md#11-references)],
- recovery of governing equations of dynamical systems [[7](README.md#11-references)],
- creating fractal patterns with tensor products [[9](README.md#11-references)],
- computation of metastable and coherent sets [[10](README.md#11-references)],
- [[11](README.md#11-references)]
- tensor-based image classification [[12](README.md#11-references)].

**Scikit-TT** provides a powerful TT class as well as different modules comprising solvers for algebraic problems, the automatic construction of tensor trains, and data-driven methods. Furthermore, several examples for the diverse application areas are included.

## Content

1. [Installing](README.md#1-installing)
2. [TT class](README.md#2-tt-class) 
3. [TT solvers](README.md#3-tt-solvers)
   - [Systems of linear equations](README.md#31-systems-of-linear-equations)
   - [Generalized eigenvalue problems](README.md#32-generalized-eigenvalue-problems)
   - [Linear differential equations](README.md#33-linear-differential-equations)
4. [SLIM decomposition](README.md#4-slim-decomposition)
5. [Data analysis](README.md#5-data-analysis)
   - [Tensor-based dynamic mode decomposition (tDMD)](README.md#51-tensor-based-dynamic-mode-decomposition-tdmd)
   - [Transformed data tensors](README.md#52-transformed-data-tensors)
   - [Regression methods](README.md#53-regression-methods)
     - [Multidimensional approximation of nonlinear dynamical systems (MANDy)](README.md#531-multidimensional-approximation-of-nonlinear-dynamical-systems-mandy)
     - [Kernel-based MANDy](README.md#532-kernel-based-mandy)
     - [Alternating ridge regression (ARR)](README.md#533-alternating-ridge-regression-arr)
   - [Tensor-based extended dynamic mode decomposition (tEDMD)](README.md#54-tensor-based-extended-dynamic-mode-decomposition-tedmd)
     - [Tensor-based infinitesimal generator EDMD (tgEDMD)](README.md#541-tensor-based-infinitesimal-generator-edmd-tgedmd)
   - [Ulam's method](README.md#55-ulams-method)
6. [Models](README.md#6-models)
7. [Examples](README.md#7-examples)
8. [Tests](README.md#8-tests)
9. [Utilities](README.md#9-utilities)
10. [Additional information](README.md#10-additional-information)
    - [Authors & contact](README.md#101-authors--contact)
    - [Built with](README.md#102-built-with)
    - [License](README.md#103-license)
    - [Versions](README.md#104-versions)
11. [References](README.md#11-references)

## 1. Installing

A [*setup.py*](setup.py) is included in the package. To install **Scikit-TT** simply enter:

```
python setup.py install --user
```

or install the latest version directly from GitHub:

```
pip install git+https://github.com/PGelss/scikit_tt
```

## 2. TT class

The tensor-train class - implemented in the module [*tensor-train.py*](scikit_tt/tensor_train.py) - is the core of **Scikit-TT** and enables us to work with the tensor-train format. We define tensor trains in terms of different attributes such as *order*, *row_dims*, *col_dims*, *ranks*, and *cores*. That is, a tensor train (operator) 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?T%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%28m_1%20%5Ctimes%20n_1%29%20%5Ctimes%20%5Cdots%20%5Ctimes%20%28m_d%5Ctimes%20n_d%29%7D">
</p>

<!-- T \in \mathbb{R}^{(m_1 \times n_1) \times \dots \times (m_d\times n_d)} -->

with graphical representation

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5C%2C%5Ctext%7B---%7D%5C%2C%7B%5Cfootnotesize%20r_0%7D%5C%2C%5Ctext%7B---%7D%5C%2C%20%5Cbegin%7Bmatrix%7D%20%7C%5C%5C%20%7B%5Cfootnotesize%20n_1%7D%5C%5C%20%7C%20%5C%5C%20%5Cboxed%7BT_1%7D%20%5C%5C%20%7C%20%5C%5C%20%7B%5Cfootnotesize%20m_1%7D%5C%5C%20%7C%20%5Cend%7Bmatrix%7D%20%5C%2C%5Ctext%7B---%7D%5C%2C%7B%5Cfootnotesize%20r_1%7D%5C%2C%5Ctext%7B---%7D%5C%2C%20%5Cbegin%7Bmatrix%7D%20%7C%5C%5C%20%7B%5Cfootnotesize%20n_2%7D%5C%5C%20%7C%20%5C%5C%20%5Cboxed%7BT_2%7D%20%5C%5C%20%7C%20%5C%5C%20%7B%5Cfootnotesize%20m_2%7D%5C%5C%20%7C%20%5Cend%7Bmatrix%7D%20%5C%2C%5Ctext%7B---%7D%5C%2C%7B%5Cfootnotesize%20r_2%7D%5C%2C%5Ctext%7B---%7D%5C%2C%20%5Cbegin%7Bmatrix%7D%20%7C%5C%5C%20%7B%5Cfootnotesize%20n_3%7D%5C%5C%20%7C%20%5C%5C%20%5Cboxed%7BT_3%7D%20%5C%5C%20%7C%20%5C%5C%20%7B%5Cfootnotesize%20m_3%7D%5C%5C%20%7C%20%5Cend%7Bmatrix%7D%20%5C%2C%5Ctext%7B------%7D%5Ccdots%5Ccdots%5Ctext%7B------%7D%5C%2C%20%5Cbegin%7Bmatrix%7D%20%7C%5C%5C%20%7B%5Cfootnotesize%20n_d%7D%5C%5C%20%7C%20%5C%5C%20%5Cboxed%7BT_d%7D%20%5C%5C%20%7C%20%5C%5C%20%7B%5Cfootnotesize%20m_d%7D%5C%5C%20%7C%20%5Cend%7Bmatrix%7D%20%5C%2C%5Ctext%7B---%7D%5C%2C%7B%5Cfootnotesize%20r_d%7D%5C%2C%5Ctext%7B---%7D%5C%2C">
</p>

<!-- \,\text{---}\,{\footnotesize r_0}\,\text{---}\,
\begin{matrix}
|\\
{\footnotesize n_1}\\
|   \\
\boxed{T_1}  \\
| \\
{\footnotesize m_1}\\
|
\end{matrix}
\,\text{---}\,{\footnotesize r_1}\,\text{---}\,
\begin{matrix}
|\\
{\footnotesize n_2}\\
|   \\
\boxed{T_2}   \\
| \\
{\footnotesize m_2}\\
|
\end{matrix}
\,\text{---}\,{\footnotesize r_2}\,\text{---}\,
\begin{matrix}
|\\
{\footnotesize n_3}\\
|   \\
\boxed{T_3}  \\
| \\
{\footnotesize m_3}\\
|
\end{matrix}
\,\text{------}\cdots\cdots\text{------}\,
\begin{matrix}
|\\
{\footnotesize n_d}\\
|   \\
\boxed{T_d}  \\
| \\
{\footnotesize m_d}\\
|
\end{matrix}
\,\text{---}\,{\footnotesize r_d}\,\text{---}\, -->

is given by ```order=d```, ```row_dims=[m_1,...,m_d]```, ```col_dims=[n_1,...,n_d]```, ```ranks=[r_0,...,r_d]```, and ```cores=[T_1,...,T_d]```. An overview of the member functions of the class is shown in the following list.

```
TT ....................... construct tensor train from array or list of cores
print .................... print the attributes of a given tensor train
+,-,*,@ .................. basic operations on tensor trains
tensordot ................ mode contractions of tensor trains
rank_tensordot ........... contraction of first/last mode with a matrix
concatenate .............. expand list of tensor cores
transpose ................ transpose of a tensor train
rank_transpose ........... rank-transpose of a tensor train
conj ..................... complex conjugate of a tensor train
isoperator ............... check if a tensor train is an operator
copy ..................... deep copy of a tensor train
element .................. compute single element of a tensor train
full ..................... convert a tensor train to full format
matricize ................ matricize a tensor train
ortho_left ............... left-orthonormalize a tensor train
ortho_right .............. right-orthonormalize a tensor train
ortho .................... left- and right-orthonormalize a tensor train
norm ..................... compute the norm of a tensor train
tt2qtt ................... convert from TT to QTT format
qtt2tt ................... convert from QTT to TT format
svd ...................... compute global SVDs of tensor trains
pinv ..................... compute pseudoinverses of tensor trains
```

Further functions defined in [*tensor_train.py*](scikit_tt/tensor_train.py) are:

```
zeros .................... construct a tensor train of all zeros
ones ..................... construct a tensor train of all ones
eye ...................... construct an identity tensor train
unit ..................... construct canonical unit tensor
rand ..................... construct a random tensor train
canonical ................ non-compressible tensor train of basis products
uniform .................. construct a uniformly distributed tensor train
residual_error ........... compute residual errors of systems of linear equations
```

## 3. TT solvers

Different methods for solving systems of linear equations, eigenvalue problems, and linear differential equations in the TT format are implemented in **Scikit-TT**. These methods - which can be found in the directory [*scikit_tt/solvers*](scikit_tt/solvers) - are based on the alternating optimization of the TT cores.

### 3.1 Systems of linear equations

In order to approximate the solution of a system of linear equations in the TT format, a series of low-dimensional problems can be solved by fixing certain components of the tensor network. For this purpose, the *alternating linear scheme* (ALS) and the *modified alternating linear scheme* (MALS) [[4](README.md#11-references)] are implemented in [*sle.py*](scikit_tt/solvers/sle.py).

```
als ...................... alternating linear scheme for systems of linear equations in the TT format
mals ..................... modified ALS for systems of linear equations in the TT format
```

### 3.2 Generalized eigenvalue problems

Besides power iteration methods [[8](README.md#11-references)], ALS and MALS can also be used to find approximations of eigenvalues and corresponding eigentensors of TT operators. The basic procedures of ALS and MALS - implemented in [*evp.py*](scikit_tt/solvers/evp.py) - for (generalized) eigenvalue problems are similar to the ones for systems of linear equations. The main difference is the type of optimization problem which has to be solved in the iteration steps. See [[4](README.md#11-references)] for details. The implemented version of ALS can either be used to compute single eigenpairs or sets of eigenpairs by using the block-TT format. Additionally, the Wielandt deflation technique can be used to compute the eigenvalues (and corresponding eigentensors) close to a given estimation.

```
als ...................... ALS for generalized eigenvalue problems in the TT format
power_method ............. inverse power iteration method for eigenvalue problems in the TT format
```

### 3.3 Linear differential equations

In order to compute time-dependent or stationary distributions of linear differential equations in the TT format, **Scikit-TT** uses explicit as well as implicit integration schemes such as the Euler methods or the trapezoidal rule. In order to approximate the solutions at each time step using implicit methods, ALS and MALS, respectively, are used. The methods can be found in [*ode.py*](scikit_tt/solvers/ode.py).

```
explicit_euler ........... explicit Euler method for linear differential equations in the TT format
errors_expl_euler ........ compute approximation errors of the explicit Euler method
symmetric_euler .......... second order differencing for quantum mechanics
implicit_euler ........... implicit Euler method for linear differential equations in the TT format
errors_impl_euler ........ compute approximation errors of the implicit Euler method
trapezoidal_rule ......... trapezoidal rule for linear differential equations in the TT format
errors_trapezoidal ....... compute approximation errors of the trapezoidal rule
adaptive_step_size ....... adaptive step size method for linear differential equations in the TT format
split .................... Strang splitting for ODEs with SLIM operator
```

## 4. SLIM decomposition

The SLIM decomposition is a specific form of TT decompositions which represent tensors networks with a certain structure. For instance, tensor operators corresponding to nearest-neighbor interaction systems can be systematicly decomposed into a tensor-train operator using the algorithms in [*slim.py*](scikit_tt/slim.py). See [[5](README.md#11-references)] for details.

```
slim_mme ................. SLIM decomposition for Markov generators
slim_mme_hom ............. SLIM decomposition for homogeneous Markov generators
```

## 5. Data analysis

**Scikit-TT** combines data-driven methods with tensor network decompositions in order to significantly reduce the computational costs and/or storage consumption for high-dimensional data sets. Different methods can be found in the directory [*scikit_tt/data_driven*](scikit_tt/data_driven/).

### 5.1 Tensor-based dynamic mode decomposition (tDMD)

tDMD is an extension of the classical dynamic mode decomposition which exploits the TT format to compute DMD modes and eigenvalues. The algorithms below can be found in [*tdmd.py*](scikit_tt/data_driven/tdmd.py). See [[6](README.md#11-references)] for details.

```
tdmd_exact ............... exact tDMD algorithm
tdmd_standard ............ standard tDMD algorithm
```

### 5.2 Transformed data tensors

Given time-series data and a set of basis functions, **Scikit-TT** provides methods to construct the counterparts of tranformed basis matrices, so-called transformed data tensors. The algorithms in [*transform.py*](scikit_tt/data_driven/transform.py) include general basis decompositions [[10](README.md#11-references)], coordinate- and function-major decompositions [[7](README.md#11-references)], as well as an approach to construct transformed data tensors using higher-order CUR decompositions (HOCUR) [[10, 13](README.md#11-references)]. Furthermore, different basis functions which are typicall used can be found in the module.

```
ConstantFunction ......... constant function
IndicatorFunction ........ indicator function
Identity ................. identity function
Monomial ................. monomial function
Legendre ................. Legendre polynomial
Sin ...................... sine function
Cos ...................... cosine function
GaussFunction ............ Gauss function
PeriodicGaussFunction .... periodic Gauss function
basis_decomposition ...... construct general transformed data tensors
coordinate_major ......... construct transformed data tensors with coordinate-major basis
function_major ........... construct transformed data tensors with function-major basis
gram ..................... compute Gram matrix of transformed data tensors
hocur .................... construct general transformed data tensors using HOCUR
```

### 5.3 Regression methods

Our toolbox provides different tensor-based methods to solve regression problems on transformed data tensors in the least-squares sense. The algorithms can be found in [*regression.py*](scikit_tt/data_driven/regression.py). The following functions have been implemented so far.

```
arr ...................... alternating ridge regression
mandy_cm ................. MANDy using coordinate-major decompositions
mandy_fm ................. MANDy using function-major decompositions
mandy_kb ................. kernel-based MANDy
```

#### 5.3.1 Multidimensional approximation of nonlinear dynamical systems (MANDy)

MANDy combines the data-driven recovery of dynamical systems with tensor decompositions. It can be used for, e.g., the recovery of unknown governing equations from measurement data only. MANDy computes an exact TT decomposition of involved coefficient tensors. See [[7](README.md#11-references)] for details. 


#### 5.3.2 Kernel-based MANDy

Instead of computing the coefficient tensors explicitly, kernel-based MANDy can be used to indirectly represent those tensors by inverting gram matrices corresponding to the given transformed data tensors. Then, a sequence of Hadamard products is exploited to speed up computations. See [[12](README.md#11-references)] for details.

#### 5.3.3 Alternating ridge regression

The exact computation of the coefficient tensors (indirectly or directly represented) may lead to high TT ranks. An alternative is the application of ARR to the regression problem in order to compute a low-rank representation of the coefficient tensor by iteratively solving low-dimensional regression problems. See [[12](README.md#11-references)] for details.

### 5.4 Tensor-based extended dynamic mode decomposition (tEDMD)

As described in [[10](README.md#11-references)], a tensor-based counterpart of EDMD is implemented in **Scikit-TT**. Given a data set and list of basis functions, tEDMD can be used to approximate eigenvalues and eigenfunctions of evolution operators, i.e., Perron-Frobenius and Koopman operators. The basic procedures of tEDMD - combinations of the TT format and so-called AMUSE - are implemented in [*tedmd.py*](scikit_tt/data_driven/tedmd.py).

```
amuset_hosvd ............. tEDMD using AMUSEt with HOSVD
amuset_hocur ............. tEDMD using AMUSEt with HOCUR
```

#### 5.4.1 Tensor-based infinitesimal generator EDMD (tgEDMD)

A very similar approach to tEDMD can be used to approximate the infinitesimal Koopman generator if the underlying dynamics are given by a stochastic differential equation, see [[11](README.md#11-references)]. The algorithms can be found in [*tgedmd.py*](scikit_tt/data_driven/tgedmd.py).

```
amuset_hosvd ........................ tgEDMD using AMUSEt with HOSVD
amuset_hosvd_reversible ............. tgEDMD for reversible systems
generator_on_product ................ evaluation of the Koopman generator
generator_on_product_reversible ..... evaluation for the reversible case
```

### 5.5 Ulam's method

Given transitions of particles in a 2- or 3-dimensional potentials, **Scikit-TT** can be used to approximate the corresponding Perron-Frobenius operator in TT format. The algorithms can be found in [*ulam.py*](scikit_tt/data_driven/ulam.py). See [[2](README.md#11-references)] for details.

```
ulam_2d .................. approximate Perron-Frobenius operators for 2-dimensional systems
ulam_3d .................. approximate Perron-Frobenius operators for 3-dimensional systems
```

## 6. Models

The construction of several models from various fields such as heterogeneous catalysis [[3](README.md#11-references)], chemical reaction networks [[2](README.md#11-references)], and fractal geometry [[9](README.md#11-references)] is included in [*models.py*](scikit_tt/models.py). 

```
cantor_dust .............. generalization of the Cantor set and the Cantor dust
co_oxidation ............. CO oxidation on a RuO2 surface
fpu_coefficients ......... coefficient tensor for the Fermi-Pasta_ulam problem
kuramoto_coefficients .... coefficient tensor for the kuramoto model
multisponge .............. generalization of the Sierpinski carpet and the Menger sponge
rgb_fractal .............. generate RGB fractals
signaling_cascade ........ cascading process on a genetic network
toll_station ............. queuing problem at a toll station
two_step_destruction ..... two-step mechanism for the destruction of molecules
vicsek_fractal ........... generalization of the Vicsek fractal
```

## 7. Examples

Numerical experiments from different application areas are included in **Scikit-TT**. For instance, the application of the TT format to chemical master equations [[2](README.md#11-references)], heterogeneous catalytic processes [[3](README.md#11-references)], fluid dynamics [[6](README.md#11-references)], and dynamical systems [[6](README.md#11-references), [7](README.md#11-references)] can be found in the directory [*examples*](examples/).

```
ala10 .................... apply tEDMD to time series data of deca-alanine
co_oxidation ............. compute stationary distributions of a catalytic process
fermi_pasta_ulam_1 ....... apply MANDy to the Fermi-Pasta-Ulam problem
fermi_pasta_ulam_2 ....... apply MANDy to the Fermi-Pasta-Ulam problem
fractals ................. use tensor decompositions for generating fractal patterns
karman ................... apply tDMD to the von Kármán vortex street
kuramoto ................. apply MANDy to the Kuramoto model
lemon_slice .............. apply tgEDMD to Langevin dynamics in Lemon-Slice potential
mnist .................... tensor-based image classification of theMNIST and FMNIST data set
ntl9 ..................... apply tEDMD to time series data of NTL9
quadruple_well ........... approximate eigenfunctions of the Perron-Frobenius operator in 3D
radial_potential ......... apply tEDMD to time series data of particles in a radial potential
signaling_cascade ........ compute mean concentrations of a 20-dimensional signaling cascade
toll_station ............. compute distribution of cars at a toll station
triple_well .............. approximate eigenfunctions of the Perron-Frobenius operator in 2D
two_step_destruction ..... apply QTT and MALS to a two-step destruction process
```

## 8. Tests

Modules containing unit tests are provided in the directory [*tests*](tests/).

```
test_evp ................. unit tests for solvers/evp.py
test_models .............. unit tests for models.py
test_ode ................. unit tests for solvers/ode.py
test_regression .......... unit tests for data_driven/regression.py
test_sle ................. unit tests for solvers/sle.py
test_slim ................ unit tests for slim.py
test_tdmd ................ unit tests for data_driven/tdmd.py
test_tedmd ............... unit tests for data_driven/tedmd.py
test_tensor_train ........ unit tests for tensor_train.py
test_tensordot ........... unit tests for tensordot in tensor_train.py
test_tgedmd .............. unit tests for data_driven/tgedmd.py
test_transform ........... unit tests for data_driven/transform.py
test_ulam ................ unit tests for data_driven/ulam.py
test_utils ............... unit tests for utils.py
```

## 9. Utilities

In [*utils.py*](scikit_tt/utils.py) we collect routines which are employed at several points in **Scikit-TT**.

```
header ................... ASCII header for scikit-tt
progress ................. show progress in percent
timer .................... measure CPU time
truncated_svd ............ compute truncated SVD 
```

## 10. Additional information

### 10.1 Authors & contact

* **Dr. Patrick Gelß** - CRC 1114, Freie Universität Berlin, Germany - _Main developer_
  - address: Arnimallee 9, 14195 Berlin, Germany
  - email: p.gelss@fu-berlin.de
* **Marvin Lücke** - Institute of Mathematics, Paderborn University, Germany - _TT class methods, development of tgEDMD, code compatibility, unit tests_
* **Dr. Stefan Klus** - CRC 1114, Freie Universität Berlin, Germany - _initial work_
* **Martin Scherer** - _development advisor_
* **Dr. Feliks Nüske** - Institute of Mathematics, Paderborn University, Germany - _development of tEDMD_


### 10.2 Built with

* [PyCharm](https://www.jetbrains.com/pycharm/)

### 10.3 License

This project is licensed under the [LGPLv3+](https://www.gnu.org/licenses/lgpl-3.0.en.html) license - see [LICENSE.txt](LICENSE.txt) for details.

### 10.4 Versions

The current version of **Scikit-TT** is [1.1](https://github.com/PGelss/scikit_tt/releases/latest). For a list of previous versions, click [here](https://github.com/PGelss/scikit_tt/releases).

## 11. References

[1] I. V. Oseledets, "Tensor-Train Decomposition", SIAM Journal on Scientific Computing 33 (5) (2011)

[2] P. Gelß, "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin (2017)

[3] P. Gelß, S. Matera, C. Schütte, "Solving the Master Equation without Kinetic Monte Carlo: Tensor Train Approximations for a CO Oxidation Model", Journal of Computational Physics 314 (2016)

[4] S. Holtz, T. Rohwedder, R. Schneider, "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", SIAM Journal on Scientific Computing 34 (2) (2012)

[5] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-Neighbor Interaction Systems in the Tensor-Train Format", Journal of Computational Physics 341 (2017)

[6] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7) (2018)

[7] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems", Journal of Computational and Nonlinear Dynamics, 14 (6) (2019)

[8] S. Klus, C. Schütte, "Towards tensor-based methods for the numerical approximation of the Perron-Frobenius and Koopman operator", Journal of Computational Dynamics 3 (2), 2016

[9] P. Gelß, C. Schütte, "Tensor-generated fractals: Using tensor decompositions for creating self-similar patterns", arXiv:1812.00814 (2018)

[10] F. Nüske, P. Gelß, S. Klus, C. Clementi, "Tensor-based computation of metastable and coherent sets", arXiv:1908.04741v2 (2019)

[11] M. Lücke, "Tensor-based Extended Dynamic Mode Decomposition for approximating the infinitesimal Koopman Generator", Universität Paderborn, 2020

[12] S. Klus, P. Gelß, "Tensor-Based Algorithms for Image Classification", Algorithms 12 (11), 2019

[13] I. Oseledets, E. Tyrtyshnikov, "TT-cross approximation for multidimensional arrays", Linear Algebra and its Applications 432 (1) (2010)

