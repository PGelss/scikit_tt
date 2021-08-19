import numpy as np
from scipy.integrate import quad, nquad
import scikit_tt.data_driven.transform as tdt
from scikit_tt.data_driven import tgedmd
from msmtools.analysis.dense.pcca import _pcca_connected_isa
from matplotlib import pyplot as plt


class LemonSlice:
    def __init__(self, k, beta, c=1.0, d=2, alpha=1.0):
        """
        Stochastic dynamics in two-dimensional Lemon Slice potential, given by:
        V(x, y) = cos(k*phi) + 1.0/np.cos(0.5*phi) + 10*(r-1)**2 + (1.0/r)
        where r, phi are the polar coordinates of x, y.

        Parameters
        ----------
        k : int
            number of minima along the polar axis.
        beta : float
            inverse temperature
        c : float
            scaling factor for cosine part of the potential.
        d : int, optional
            dimension of the system. If not equal to two, the potential
            will just be harmonic along all additional dimensions.
        """
        self.k = k
        self.beta = beta
        self.c = c
        self.d = d
        self.alpha = alpha

        # Compute partition function:
        # start with "Lemon Slice" part:
        dims = np.array([[-2.51, 2.51],
                         [-2.51, 2.51]])
        Z = nquad(lambda x, y: np.exp(-self.beta * self._V_lemon_slice(x, y)), dims)[0]
        # Multiply by partition functions of the Gaussians along all other directions:
        for ii in range(2, self.d):
            Z *= np.sqrt(self.beta * self.alpha / (2 * np.pi))
        self.Z = Z

        # Calculate normalization constants for effective dynamics:
        self.C1 = quad(lambda r: np.exp(-(10 * (r - 1) ** 2 + (1.0 / r))) * (1.0 / r), 0, 6.0)[0]
        self.C2 = quad(lambda r: np.exp(-(10 * (r - 1) ** 2 + (1.0 / r))) * r, 0, 6.0)[0]

    def simulate(self, x0, m, dt):
        """
        Generate trajectory of stochastic dynamics in Lemon Slice potential, using Euler scheme.

        Parameters
        ----------
        x0 : np.ndarray
            initial values for the simulation, shape (d,)
        m : int
            number time steps (including initial value)
        dt : float
            integration time step

        Returns
        --------
        X : np.ndarray
            simulation trajectory, shape (d, m)
        """
        # Initialize:
        X = np.zeros((self.d, m))
        X_old = x0[:, None]
        # print(X_old)
        X[:, 0] = X_old[:, 0]
        # Run simulation:
        for t in range(1, m):
            # print("t = %d:"%t)
            # print(self.gradient(X_old))
            X_new = X_old - self.gradient(X_old) * dt + \
                    np.sqrt(2 * dt / self.beta) * np.random.randn(self.d, 1)
            # print(X_new)
            X[:, t] = X_new[:, 0]
            X_old = X_new
            # print("")
        return X

    def potential(self, x):
        """
        Evaluate potential energy at Euclidean positions x

        Parameters
        ----------
        x : np.ndarray
            array of Euclidean coordinates, shape (d, m)

        Returns
        --------
        V : np.ndarray
            Values of the potential, shape (m,)
        """
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the potential:
        r, phi = self._polar_rep(x[0, :], x[1, :])
        V = self.c * np.cos(self.k * phi) + 1.0 / np.cos(0.5 * phi) + 10 * (r - 1) ** 2 + (1.0 / r)
        # Add harmonic terms for all remaining dimensions:
        for ii in range(2, self.d):
            V += 0.5 * self.alpha * x[ii, :] ** 2

        return V

    def gradient(self, x):
        """
        Evaluate gradient of potential energy at Euclidean positions x

        Parameters
        ----------
        x : np.ndarray
            array of Euclidean coordinates, shape (d, m)

        Returns
        --------
        dx : np.ndarray
            gradient of the potential for all m data points in x, shape (d, m)
        """
        dV = np.zeros((self.d, x.shape[1]))
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the gradient:
        r, phi = self._polar_rep(x[0, :], x[1, :])
        dV[0, :] = -(0.5 * np.sin(0.5 * phi) / np.cos(0.5 * phi) ** 2 -
                     self.c * self.k * np.sin(self.k * phi)) * (x[1, :] / r ** 2) + 20 * (r - 1) * (x[0, :] / r) - (
                               1.0 / r ** 2) * (x[0, :] / r)
        dV[1, :] = (0.5 * np.sin(0.5 * phi) / np.cos(0.5 * phi) ** 2 -
                    self.c * self.k * np.sin(self.k * phi)) * (x[0, :] / r ** 2) + 20 * (r - 1) * (x[1, :] / r) - (
                               1.0 / r ** 2) * (x[1, :] / r)
        # Add harmonic contributions to all remaining dimensions:
        for ii in range(2, self.d):
            dV[ii, :] = self.alpha * x[ii, :]
        return dV

    def drift(self, x):
        """
        Evaluate drift at position x.

        Parameters
        ----------
        x : np.ndarray
            single point, shape (d,)

        Returns
        -------
        np.ndarray
            drift at x
        """
        return -self.gradient(x[:, np.newaxis])[:, 0]

    def diffusion(self, x):
        """
        Evaluate diffusion at position x.

        Parameters
        ----------
        x : np.ndarray
            single point, shape (d,)

        Returns
        -------
        np.ndarray
            diffusion at x
        """
        return (2.0 / self.beta) ** 0.5 * np.eye(self.d)

    def stat_dist(self, x):
        """
        Evaluate stationary density at Euclidean positions x

        Parameters
        ----------
        x : np.ndarray
            array of Euclidean coordinates, shape (d, m)

        Returns
        --------
        mu : np.ndarray
            values of the stationary density for all m data points in x, shape (m,)
        """
        return (1.0 / self.Z) * np.exp(-self.beta * self.potential(x))

    def _V_lemon_slice(self, x, y):
        """ Return only the "Lemon Slice" part of the potential."""
        r, phi = self._polar_rep(x, y)
        return self.c * np.cos(self.k * phi) + 1.0 / np.cos(0.5 * phi) + 10 * (r - 1) ** 2 + (1.0 / r)

    @staticmethod
    def _polar_rep(x, y):
        """
        Compute polar coordinates from 2d Euclidean coordinates:

        Parameters
        ----------
        x, y : np.ndarray
            array of euclidean coordinates to be transformed, shape (m,)

        Returns
        --------
        r, phi : np.ndarray
            array of polar coordinates corresponding to x and y, shape (m,)
        """
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return r, phi


"""  System Settings: """
# Number of dimensions:
d = 4
# Diffusion constant:
beta = 1.0
# Spring constant for harmonic parts of the potential
alpha = 10.0
# Pre-factor for Lemon Slice:
c = 1.0
# Number of minima for Lemon Slice:
k = 4

""" Simulation settings: """
# Integration time step:
dt = 1e-3
# Number of time steps:
m = 300000
m_tgedmd = 3000
# Initial position:
x0 = np.random.rand(d)
# Maximal rank for HOSVD:
max_rank = np.infty
# Truncation threshold for HOSVD:
epsilon = 1e-3

""" For use with Gaussian basis functions """
mean_ls = np.arange(-1.2, 1.21, 0.2)
sig_ls = 0.2
mean_quad = np.arange(-1.0, 1.01, 0.2)
sig_quad = 0.2

# Define basis sets:
basis_list = []
for i in range(2):
    basis_list.append([tdt.GaussFunction(i, mean_ls[j], sig_ls) for j in range(len(mean_ls))])
for i in range(2, 4):
    basis_list.append([tdt.GaussFunction(i, mean_quad[j], sig_quad) for j in range(len(mean_quad))])

""" For use with B-Spline basis functions: """
# # Set boundary values for Lemon Slice and quadratic parts:
# bounds_ls = [-1.7, 1.7]
# bounds_q = [-1.2, 1.2]
# # Set number of subintervals for both parts:
# nsub_ls = 6
# nsub_q = 6
# # Set degree of B-spline to be used:
# degree = 3
#
# # Define knot vectors:
# knots_ls = np.linspace(bounds_ls[0], bounds_ls[1], nsub_ls+1)
# knots_q = np.linspace(bounds_q[0], bounds_q[1], nsub_q+1)
# # Define basis sets:
# basis_list = []
#
# for i in range(2):
#     cbsp = np.eye(nsub_ls + degree, nsub_ls + degree)
#     basis_list.append([tdt.Bspline(i, knots_ls, degree, cbsp[j, :]) for j in range(nsub_ls + degree)])
# for i in range(2, 4):
#     cbsp = np.eye(nsub_q + degree, nsub_q + degree)
#     basis_list.append([tdt.Bspline(i, knots_q, degree, cbsp[j, :]) for j in range(nsub_q + degree)])

""" Run Simulation """
print('Running Simulation...')
LS = LemonSlice(k, beta, c=c, d=d, alpha=alpha)
data = LS.simulate(x0, m, dt)  # data.shape = (d, m)

""" plot potential """
LS2 = LemonSlice(k, beta, c=c, d=2, alpha=alpha)
nx, ny = (100, 100)
x = np.linspace(-1.5, 1.5, nx)
y = np.linspace(-1.5, 1.5, ny)
x, y = np.meshgrid(x, y)

XY = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
C = LS2.potential(XY.T)
C = C.reshape(nx, ny)
Cm = np.ma.masked_where(C > 10, C)

plt.figure(figsize=(8, 6))
cf = plt.pcolormesh(x, y, Cm, shading='gouraud', cmap='jet')
plt.colorbar(cf)
plt.title(r"Potential in $x_1$ and $x_2$")
plt.xlabel(r"$x_1$", fontsize=12)
plt.ylabel(r"$x_2$", fontsize=12)

""" Define basis functions for tgEDMD """

# for i in range(2):
#     basis_list.append([tdt.Legendre(i, j, domain=1.5) for j in range(0, 10)])
# for i in range(2, 4):
#     basis_list.append([tdt.Legendre(i, j, domain=1.5) for j in range(0, 8)])

""" Run tgEDMD """
delta = int(round(m / m_tgedmd))
data_tgedmd = data[:, ::delta]

drift = np.zeros(data_tgedmd.shape)
for k in range(drift.shape[1]):
    drift[:, k] = LS.drift(data_tgedmd[:, k])

diffusion = np.zeros((data_tgedmd.shape[0], data_tgedmd.shape[0], data_tgedmd.shape[1]))
for k in range(diffusion.shape[2]):
    diffusion[:, :, k] = LS.diffusion(data_tgedmd[:, k])

# AMUSEt for the general case
#eigvals, traj_eigfuns = tgedmd.amuset_hosvd(data[:, ::delta], basis_list, drift, diffusion, num_eigvals=5,
#                                            return_option='eigenfunctionevals',
#                                            threshold=epsilon, max_rank=max_rank)

# AMUSEt for the reversible case
eigvals, traj_eigfuns = tgedmd.amuset_hosvd_reversible(data_tgedmd, basis_list, diffusion, num_eigvals=5,
                                                       return_option='eigenfunctionevals',
                                                       threshold=epsilon, max_rank=max_rank)

eigvals = eigvals.real
print('Eigenvalues of Koopman generator: {}'.format(eigvals))
its = [-1 / kappa for kappa in eigvals[1:]]
print('Implied time scales: {}'.format(its))

""" Identify metastable areas """
diffs = np.abs(np.max(traj_eigfuns.T, axis=0) - np.min(traj_eigfuns.T, axis=0))
if diffs[0] > 1e-6:
    traj_eigfuns[0, :] = traj_eigfuns[0, 0] * np.ones((traj_eigfuns.shape[1]))
chi, _ = _pcca_connected_isa(traj_eigfuns.T, 4)
chi = chi.T
for i in range(chi.shape[1]):
    ind = np.argmax(chi[:, i])
    chi[:, i] = np.zeros((chi.shape[0],))
    chi[ind, i] = 1
chi = chi.astype(bool)
tgedmd_data = data[:, ::delta]

plt.figure(figsize=(8, 6))
plt.plot(tgedmd_data[0, :][chi[0, :]], tgedmd_data[1, :][chi[0, :]], 'bx')
plt.plot(tgedmd_data[0, :][chi[1, :]], tgedmd_data[1, :][chi[1, :]], 'r*')
plt.plot(tgedmd_data[0, :][chi[2, :]], tgedmd_data[1, :][chi[2, :]], 'g2')
plt.plot(tgedmd_data[0, :][chi[3, :]], tgedmd_data[1, :][chi[3, :]], 'y+')
plt.grid()
plt.xlabel(r"$x_1$", fontsize=12)
plt.ylabel(r"$x_2$", fontsize=12)
plt.title("Clustering of the state space obtained by tgEDMD and PCCA")
plt.show()
