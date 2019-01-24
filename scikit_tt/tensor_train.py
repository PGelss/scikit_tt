# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg


class TT(object):
    """Tensor train class

    Tensor trains [1]_ are defined in terms of different attributes. In particular, the attribute 'cores' is a list of
    4-dimensional tensors representing the corresponding TT cores. There is no distinguish between tensor trains and
    tensor trains operators, i.e. a classical tensor train is represented by cores with column dimensions equal to 1.
    An instance of the tensor train class can be initialized from a full tensor representation (in this case, the tensor
    is decomposed into the TT format) or from a list of cores. For more information on the implemented tensor
    operations, we refer to [2]_.

    Parameters
    ----------
    x: ndarray or list of ndarrays
        either a full tensor or a list of TT cores

    Attributes
    ----------
    order: int
        order of the tensor train
    row_dims: list of ints
        list of the row dimensions of the tensor train
    col_dims: list of ints
        list of the column dimensions of the tensor train
    ranks: list of ints
        list of the ranks of the tensor train
    cores: list of ndarrays
        list of the cores of the tensor train (core[i] has dimensions ranks[i] x row_dims[i] x col_dims[i] x ranks[i+1])

    References
    ----------
    .. [1] I. V. Oseledets, "Tensor-Train Decomposition", SIAM Journal on Scientific Computing 33 (5), 2011
    .. [2] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
           Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
    """

    def __init__(self, x):

        # initialize from full array
        # --------------------------

        # X is an array with dimensions m_1 x ... x m_d x n_1 x ... x n_d

        if isinstance(x, np.ndarray):

            # define order, row dimensions, column dimensions, ranks, and cores
            order = len(x.shape) // 2
            row_dims = x.shape[:order]
            col_dims = x.shape[order:]
            ranks = [1] * (order + 1)
            cores = []

            # permute dimensions, e.g., for order = 4: p = [0, 4, 1, 5, 2, 6, 3, 7]
            p = [order * j + i for i in range(order) for j in range(2)]
            y = np.transpose(x, p).copy()

            # decompose the full tensor
            for i in range(order - 1):
                # reshape residual tensor
                m = ranks[i] * row_dims[i] * col_dims[i]
                n = np.prod(row_dims[i + 1:]) * np.prod(col_dims[i + 1:])
                y = np.reshape(y, [m, n])

                # apply SVD in order to isolate modes
                [u, s, v] = linalg.svd(y, full_matrices=False)

                # define new TT core
                ranks[i + 1] = u.shape[1]
                cores.append(np.reshape(u, [ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]))

                # set new residual tensor
                y = np.diag(s) @ v

            # define last TT core
            cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))

            # initialize tensor train
            self.__init__(cores)

        # initialize from list of cores
        # -----------------------------

        else:

            # define order, row dimensions, column dimensions, ranks, and cores
            self.order = len(x)
            self.row_dims = [x[i].shape[1] for i in range(self.order)]
            self.col_dims = [x[i].shape[2] for i in range(self.order)]
            self.ranks = [x[i].shape[0] for i in range(self.order)] + [1]
            self.cores = x
            self.nbytes = np.sum([self.cores[i].nbytes for i in range(self.order)])

    def __repr__(self):
        """string representation of tensor trains

        Print the attributes of a given tensor train.
        """

        return ('\n'
                'Tensor train with order    = {d}, \n'
                '                  row_dims = {m}, \n'
                '                  col_dims = {n}, \n'
                '                  ranks    = {r}'.format(d=self.order, m=self.row_dims, n=self.col_dims, r=self.ranks))

    def __add__(self, tt_add):
        """sum of two tensor trains

        Add two given tensor trains.

        Parameters
        ----------
        tt_add: instance of TT class
            tensor train which is added to self

        Returns
        -------
        tt_sum: instance of TT class
            sum of tt_add and self
        """

        if isinstance(tt_add, TT):

            # define order, ranks, and cores
            order = self.order
            ranks = [1] + [self.ranks[i] + tt_add.ranks[i] for i in range(1, order)] + [1]
            cores = []

            # construct cores
            for i in range(order):
                # set core to zero array
                cores.append(np.zeros([ranks[i], self.row_dims[i], self.col_dims[i], ranks[i + 1]]))

                # insert core of self
                cores[i][0:self.ranks[i], :, :, 0:self.ranks[i + 1]] = self.cores[i]

                # insert core of tt_add

                cores[i][ranks[i] - tt_add.ranks[i]:ranks[i], :, :, ranks[i + 1] - tt_add.ranks[i + 1]:ranks[i + 1]] = \
                    tt_add.cores[i]

            # define tt_sum
            tt_sum = TT(cores)

            return tt_sum
        else:
            raise TypeError('unsupported argument')

    def __sub__(self, tt_sub):
        """difference of two tensor trains

        Subtract two given tensor trains.

        Parameters
        ----------
        tt_sub: instance of TT class
            tensor train which is subtracted from self

        Returns
        -------
        tt_diff: instance of TT class
            difference of tt_add and self
        """

        # define difference in terms of addition and left-multiplication
        tt_diff = self + (-1) * tt_sub.copy()

        return tt_diff

    def __mul__(self, scalar):
        """left-multiplication of tensor trains and scalars

        Parameters
        ----------
        scalar: float
            scalar value for the left-multiplication

        Returns
        -------
        tt_prod: instance of TT class
            product of scalar and self
        """

        # multiply first core with scalar
        tt_prod = self.copy()
        tt_prod.cores[0] *= scalar

        return tt_prod

    def __rmul__(self, scalar):
        """right-multiplication of tensor trains and scalars

        Parameters
        ----------
        scalar: float
            scalar value for the right-multiplication

        Returns
        -------
        tt_prod: instance of TT class
            product of self and scalar
        """

        # define product in terms of left-multiplication
        tt_prod = self.copy() * scalar

        return tt_prod

    def __matmul__(self, tt_mul):
        """multiplication of tensor trains

        Parameters
        ----------
        tt_mul: instance of TT class
            tensor train which is multiplied with self

        Returns
        -------
        tt_prod: instance of TT class or float
            product of self and tt_mul
        """

        if isinstance(tt_mul, TT):

            # multiply TT cores
            cores = [np.tensordot(self.cores[i], tt_mul.cores[i], axes=(2, 1)).transpose([0, 3, 1, 4, 2, 5]).reshape(
                self.ranks[i] * tt_mul.ranks[i], self.row_dims[i], tt_mul.col_dims[i],
                self.ranks[i + 1] * tt_mul.ranks[i + 1]) for i in range(self.order)]

            # define product tensor
            tt_prod = TT(cores)

            # set tt_prod to scalar if all dimensions are equal to 1
            m = np.prod(tt_prod.row_dims)
            n = np.prod(tt_prod.col_dims)
            if m == 1 and n == 1:
                tt_prod = tt_prod.matricize()[0]

            return tt_prod
        else:
            raise ValueError('unsupported argument')

    # dot = __matmul__

    def transpose(self):
        """transpose of tensor trains

        Returns
        -------
        tt_transpose: instance of TT class
            transpose of self
        """

        # copy self
        tt_transpose = self.copy()

        for i in range(self.order):
            # permute second and third dimension of each core
            tt_transpose.cores[i] = np.transpose(tt_transpose.cores[i], [0, 2, 1, 3])

            # interchange row and column dimensions
            tt_transpose.row_dims[i] = self.col_dims[i]
            tt_transpose.col_dims[i] = self.row_dims[i]

        return tt_transpose

    def isoperator(self):
        """operator check

        Returns
        -------
        op_bool: boolean
            true if self is a TT operator
        """

        # check if all row dimensions or column dimensions of self are equal to 1
        op_bool = not (all([i == 1 for i in self.row_dims]) or all([i == 1 for i in self.col_dims]))

        return op_bool

    def copy(self):
        """deep copy of tensor trains

        Returns
        -------
        tt_copy: instance of TT class
            deep copy of self
        """

        # copy TT cores
        cores = [self.cores[i].copy() for i in range(self.order)]

        # define copied version of self
        tt_copy = TT(cores)

        return tt_copy

    def element(self, coordinates):
        """single element of tensor trains

        Parameters
        ----------
        coordinates: list of ints
            coordinates of a single entry of self ([x_1, ..., x_d, y_1, ..., y_d])

        Returns
        -------
        entry: float
            single entry of self
        """

        # construct matrix for first row and column coordinates
        entry = np.squeeze(self.cores[0][:, coordinates[0], coordinates[self.order], :]).reshape(1, self.ranks[1])

        # multiply with respective matrices for the following coordinates
        for i in range(1, self.order):
            entry = entry @ np.squeeze(self.cores[i][:, coordinates[i], coordinates[self.order + i], :]).reshape(
                self.ranks[i], self.ranks[i + 1])

        entry = entry[0, 0]

        return entry

    def full(self):
        """conversion to full format

        Returns
        -------
        full_tensor : ndarray
            full tensor representation of self (dimensions: m_1 x ... x m_d x n_1 x ... x n_d)
        """

        # reshape first core
        full_tensor = self.cores[0].reshape(self.row_dims[0] * self.col_dims[0], self.ranks[1])

        for i in range(1, self.order):
            # contract full_tensor with next TT core and reshape
            full_tensor = full_tensor @ self.cores[i].reshape(self.ranks[i],
                                                              self.row_dims[i] * self.col_dims[i] * self.ranks[
                                                                  i + 1])
            full_tensor = full_tensor.reshape(np.prod(self.row_dims[:i + 1]) * np.prod(self.col_dims[:i + 1]),
                                              self.ranks[i + 1])

        # reshape and transpose full_tensor
        p = [None] * 2 * self.order
        p[::2] = self.row_dims
        p[1::2] = self.col_dims
        q = [2 * i for i in range(self.order)] + [1 + 2 * i for i in range(self.order)]
        full_tensor = full_tensor.reshape(p).transpose(q)

        return full_tensor

    def matricize(self):
        """matricization of tensor trains

        If self is a TT operator, then tt_mat is a matrix. Otherwise, the result is a vector.

        Returns
        -------
        tt_mat: ndarray
            matricization of self
        """

        # copy self
        tt_mat = self.copy()

        # conversion to full format and reshape into matrix
        m = np.prod(self.row_dims)
        n = np.prod(self.col_dims)
        if n == 1:
            tt_mat = tt_mat.full().reshape(m)
        else:
            tt_mat = tt_mat.full().reshape(m, n)

        return tt_mat

    def ortho_left(self, start_index=0, end_index=None, threshold=0, max_rank=np.infty):
        """left-orthonormalization of tensor trains

        Parameters
        ----------
        start_index: int, optional
            start index for orthonormalization, default is 0
        end_index: int, optional
            end index for orthonormalization, default is the index of the penultimate core
        threshold: float, optional
            threshold for reduced SVD decompositions, default is 0
        max_rank: int
            maximum rank of the left-orthonormalized tensor train

        Returns
        -------
        tt_ortho: instance of TT class
            copy of self with left-orthonormalized cores
        """

        # set end_index to the index of the penultimate core if not otherwise defined
        if end_index is None:
            end_index = self.order - 2

        # left-orthonormalization
        # -----------------------

        for i in range(start_index, end_index + 1):

            # apply SVD to ith TT core
            [u, s, v] = linalg.svd(
                self.cores[i].reshape(self.ranks[i] * self.row_dims[i] * self.col_dims[i], self.ranks[i + 1]),
                full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

            # rank reduction
            if threshold != 0:
                indices = np.where(s / s[0] > threshold)[0]
                u = u[:, indices]
                s = s[indices]
                v = v[indices, :]
            if max_rank != np.infty:
                u = u[:, :np.minimum(u.shape[1], max_rank)]
                s = s[:np.minimum(s.shape[0], max_rank)]
                v = v[:np.minimum(v.shape[0], max_rank), :]

            # define updated rank and core
            self.ranks[i + 1] = u.shape[1]
            self.cores[i] = u.reshape(self.ranks[i], self.row_dims[i], self.col_dims[i], self.ranks[i + 1])

            # shift non-orthonormal part to next core
            self.cores[i + 1] = np.diag(s) @ v @ self.cores[i + 1].reshape(self.cores[i + 1].shape[0],
                                                                           self.row_dims[i + 1] *
                                                                           self.col_dims[i + 1] *
                                                                           self.ranks[i + 2])
            self.cores[i + 1] = self.cores[i + 1].reshape(self.ranks[i + 1], self.row_dims[i + 1],
                                                          self.col_dims[i + 1],
                                                          self.ranks[i + 2])

        return self

    def ortho_right(self, start_index=None, end_index=1, threshold=0, max_rank=np.infty):
        """right-orthonormalization of tensor trains

        Parameters
        ----------
        start_index: int, optional
            start index for orthonormalization, default is the index of the last core
        end_index: int, optional
            end index for orthonormalization, default is 1
        threshold: float, optional
            threshold for reduced SVD decompositions, default is 0
        max_rank: int
            maximum rank of the right-orthonormalized tensor train

        Returns
        -------
        tt_ortho: instance of TT class
            copy of self with right-orthonormalized cores
        """

        # set start_index to the index of the last core if not otherwise defined
        if start_index is None:
            start_index = self.order - 1

        # right-orthonormalization
        # ------------------------

        for i in range(start_index, end_index - 1, -1):

            # apply SVD to ith TT core
            [u, s, v] = linalg.svd(
                self.cores[i].reshape(self.ranks[i], self.row_dims[i] * self.col_dims[i] * self.ranks[i + 1]),
                full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

            # rank reduction
            if threshold != 0:
                indices = np.where(s / s[0] > threshold)[0]
                u = u[:, indices]
                s = s[indices]
                v = v[indices, :]
            if max_rank != np.infty:
                u = u[:, :np.minimum(u.shape[1], max_rank)]
                s = s[:np.minimum(s.shape[0], max_rank)]
                v = v[:np.minimum(v.shape[0], max_rank), :]

            # define updated rank and core
            self.ranks[i] = v.shape[0]
            self.cores[i] = v.reshape(self.ranks[i], self.row_dims[i], self.col_dims[i], self.ranks[i + 1])

            # shift non-orthonormal part to previous core
            self.cores[i - 1] = self.cores[i - 1].reshape(
                self.ranks[i - 1] * self.row_dims[i - 1] * self.col_dims[i - 1],
                self.cores[i - 1].shape[3]) @ u @ np.diag(s)
            self.cores[i - 1] = self.cores[i - 1].reshape(self.ranks[i - 1], self.row_dims[i - 1],
                                                          self.col_dims[i - 1], self.ranks[i])

        return self

    def ortho(self, threshold=0, max_rank=np.infty):
        """left- and right-orthonormalization of tensor trains

        Parameters
        ----------
        threshold: float, optional
            threshold for reduced SVD decompositions, default is 0
        max_rank: int
            maximum rank of the right-orthonormalized tensor train

        Returns
        -------
        tt_ortho: instance of TT class
            copy of self with right-orthonormalized cores
        """

        # left- and right-orthonormalize self
        self.ortho_left(threshold=threshold, max_rank=max_rank).ortho_right(threshold=threshold, max_rank=max_rank)

        return self

    def norm(self, p=2):
        """norm of tensor trains

        If self is a TT operator, the cores will be reshaped, i.e., the modes of each core are converted from
        (r x m x n x r') to (r x m*n x 1 x r'). For the Manhattan norm, it is assumed that all entries of the tensor
        train are non-negative.

        Parameters
        ----------
        p: int
            if p = 1 compute Manhattan norm (all entries of self should be positive and all column dimensions have to be
            equal to 1)
            if p = 2 (default) compute Euclidean norm of self (all column dimensions have to be equal to 1)

        Returns
        -------
        norm: float
            norm of self
        """

        norm = None

        # copy self
        tt_tensor = self.copy()

        # reshape cores if self is a TT operator
        if self.isoperator():
            cores = [tt_tensor.cores[i].reshape(tt_tensor.ranks[i], tt_tensor.row_dims[i] * tt_tensor.col_dims[i], 1,
                                                tt_tensor.ranks[i + 1]) for i in range(tt_tensor.order)]
            tt_tensor = TT(cores)

        # Manhattan norm
        # --------------

        if p == 1:
            # sum over row axes
            tt_tensor.cores = [
                np.sum(tt_tensor.cores[i], axis=1).reshape(tt_tensor.ranks[i], 1, 1, tt_tensor.ranks[i + 1]) for i in
                range(tt_tensor.order)]

            # define new row dimensions
            tt_tensor.row_dims = [1] * tt_tensor.order  # define new row dimensions

            # compute single element
            norm = tt_tensor.element([0] * 2 * tt_tensor.order)

        # Euclidean norm
        # --------------

        if p == 2:
            # right-orthonormalize tt_tensor
            tt_tensor = tt_tensor.ortho_right()

            # compute norm from first core
            norm = np.linalg.norm(tt_tensor.cores[0].reshape(tt_tensor.row_dims[0] * tt_tensor.ranks[1]))

        return norm

    def tt2qtt(self, row_dims, col_dims, threshold=0):
        """conversion from TT format into QTT format

        Split the TT cores of a given tensor train in order to obtain a QTT representation.

        Example:

        Given a tensor train t with row dimensions m_1, ..., m_d and column dimensions n_1, ...,n_d,

            t_qtt = tt2qtt(t, [M_1, ..., M_d], [N_1, ..., N_d])

        defines a new instance of the TT class with dimensions given by the lists M_1, ..., M_d and N_1, ..., N_d,
        respectively. M_i and N_i are lists of ints such that np.prod(M_i) = m_i and np.prod(N_i)=n_i.

        Parameters
        ----------
        row_dims: list of lists of ints
            row dimensions for the QTT representation
        col_dims: list of lists of ints
            col dimensions for the QTT representation
        threshold: float, optional
            threshold for reduced SVD decompositions, default is 0

        Returns
        -------
        qtt_tensor: instance of TT class
            QTT representation of self
        """

        # define QTT cores and copy self
        qtt_cores = []
        tt_tensor = self.copy()

        # split TT cores of self
        # ----------------------

        for i in range(self.order):

            # define core, rank, row_dim, and col_dim
            core = tt_tensor.cores[i]
            rank = tt_tensor.ranks[i]
            row_dim = tt_tensor.row_dims[i]
            col_dim = tt_tensor.col_dims[i]

            # begin splitting
            for j in range(len(row_dims[i]) - 1):

                # set new row_dim and col_dim for reshape
                row_dim = int(row_dim / row_dims[i][j])
                col_dim = int(col_dim / col_dims[i][j])

                # reshape and transpose core
                core = core.reshape(rank, row_dims[i][j], row_dim, col_dims[i][j], col_dim,
                                    tt_tensor.ranks[i + 1]).transpose([0, 1, 3, 2, 4, 5])

                # apply SVD in order to split core
                [u, s, v] = linalg.svd(
                    core.reshape(rank * row_dims[i][j] * col_dims[i][j], row_dim * col_dim * tt_tensor.ranks[i + 1]),
                    full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

                # rank reduction
                if threshold != 0:
                    indices = np.where(s / s[0] > threshold)[0]
                    u = u[:, indices]
                    s = s[indices]
                    v = v[indices, :]

                # define QTT core
                qtt_cores.append(u.reshape(rank, row_dims[i][j], col_dims[i][j], s.shape[0]))

                # update residual core and rank
                core = np.diag(s) @ v
                rank = s.shape[0]

            # define last QTT core
            qtt_cores.append(core.reshape(rank, row_dim, col_dim, tt_tensor.ranks[i + 1]))

        # define quantized tensor train
        qtt_tensor = TT(qtt_cores)

        return qtt_tensor

    def qtt2tt(self, merge_numbers):
        """conversion from QTT format into TT format

        Contract the QTT cores of a given quantized tensor train in order to obtain a TT representation.

        Example:

        Given a quantized tensor train t with d cores,

            t_tt = qtt2tt(t, [c_1, ..., c_e])

        defines a new instance of the TT class with order e, i.e. qtt2tt merges the first c_1 cores, the second c_2
        cores, and so on.

        Parameters
        ----------
        merge_numbers: list of ints
            list of core numbers for contractions

        Returns
        -------
        tt_tensor: instance of TT class
            TT representation of self
        """

        # define TT cores and copy self
        tt_cores = []
        qtt_tensor = self.copy()

        # contract cores of self
        # ----------------------

        # first index
        k = 0

        for i in range(len(merge_numbers)):

            # set new QTT core
            core = qtt_tensor.cores[k]

            # begin contractions
            for j in range(k + 1, k + merge_numbers[i]):
                # contract with next core and reshape
                core = np.tensordot(core, qtt_tensor.cores[j], axes=(3, 0)).transpose(0, 1, 3, 2, 4, 5)
                core = core.reshape(core.shape[0], core.shape[1] * core.shape[2], core.shape[3] * core.shape[4],
                                    core.shape[5])

            # define TT core
            tt_cores.append(core)

            # increase contraction index
            k = k + merge_numbers[i]

        # define tensor train
        tt_tensor = TT(tt_cores)

        return tt_tensor

    def pinv(self, index, threshold=0, ortho_l=True, ortho_r=True):
        """Computation of the pseudoinverse of a tensor train

        Construct the pseudoinverse of a (non-operator) tensor train by a global SVD. See [1]_, [2]_ and [3]_ for
        details.

        Parameters
        ----------
        index: int
            the cores 0 to index-1 represent the row dimensions and index to order-1 the column dimensions of the
            unfolded version of self
        threshold: float, optional
            threshold for reduced SVD decompositions, default is 0
        ortho_l: bool, optional
            whether to apply left-orthonormalization or not, default is True
        ortho_r: bool, optional
            whether to apply right-orthonormalization or not, default is True

        References
        ----------
        .. [1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
               Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
        .. [2] S. Klus, P. Gelß, S. Peitz, C. Schütte, "Tensor-based Dynamic Mode Decomposition", Nonlinearity 31 (7),
               2018
        .. [3] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
               arXiv:1809.02448, 2018
        """

        # copy self
        p_inv = self.copy()

        # left-orthonormalize cores 0 to index-2
        if ortho_l is True:
            p_inv = p_inv.ortho_left(end_index=index - 2, threshold=threshold)

        # right-orthonormalize cores index to order -1
        if ortho_r is True:
            p_inv = p_inv.ortho_right(end_index=index, threshold=threshold)

        # decompose (index-1)th core
        [u, s, v] = linalg.svd(
            p_inv.cores[index - 1].reshape(p_inv.ranks[index - 1] * p_inv.row_dims[index - 1], p_inv.ranks[index]),
            full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

        # rank reduction
        if threshold != 0:
            indices = np.where(s / s[0] > threshold)[0]
            u = u[:, indices]
            s = s[indices]
            v = v[indices, :]

        # set new rank
        p_inv.ranks[index] = u.shape[1]

        # update (index-1)th core
        p_inv.cores[index - 1] = u.reshape(p_inv.ranks[index - 1], p_inv.row_dims[index - 1], 1, p_inv.ranks[index])

        # update (index)th core
        p_inv.cores[index] = np.tensordot(np.diag(np.reciprocal(s)) @ v, p_inv.cores[index], axes=(1, 0))

        return p_inv


# construction of specific tensor-train decompositions
# ----------------------------------------------------

def zeros(row_dims, col_dims, ranks=1):
    """tensor train of all zeros

    Parameters
    ----------
    row_dims: list of ints
        list of the row dimensions of the tensor train of all zeros
    col_dims: list of ints
        list of the column dimensions of the tensor train of all zeros
    ranks: list of ints, optional
        list of the ranks of the tensor train of all zeros, default is [1, ..., 1]

    Returns
    -------
    tt_zeros: instance of TT class
        tensor train of all zeros
    """

    # set ranks of tt_zeros
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(row_dims) - 1)] + [1]

    # define TT cores of tt_zeros
    cores = [np.zeros([ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]) for i in range(len(row_dims))]

    # define tensor train
    tt_zeros = TT(cores)

    return tt_zeros


def ones(row_dims, col_dims, ranks=1):
    """tensor train of all ones

    Parameters
    ----------
    row_dims: list of ints
        list of the row dimensions of the tensor train of all ones
    col_dims: list of ints
        list of the column dimensions of the tensor train of all ones
    ranks: list of ints, optional
        list of the ranks of the tensor train of all ones, default is [1, ..., 1]

    Returns
    -------
    tt_ones: instance of TT class
        tensor train of all ones
    """

    # set ranks of tt_ones
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(row_dims) - 1)] + [1]

    # define TT cores of tt_ones
    cores = [np.ones([ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]) for i in range(len(row_dims))]

    # define tensor train
    tt_ones = TT(cores)

    return tt_ones


def eye(dims):
    """identity tensor train

    Parameters
    ----------
    dims: list of ints
        list of row/column dimensions of the identity tensor train

    Returns
    -------
    tt_eye: instance of TT class
        identity tensor train
    """

    # define cores of tt_eye
    cores = [np.zeros([1, dims[i], dims[i], 1]) for i in range(len(dims))]
    for i in range(len(dims)):
        cores[i][0, :, :, 0] = np.eye(dims[i])

    # define tensor train
    tt_eye = TT(cores)

    return tt_eye


def unit(dims, inds):
    """Canonical unit tensor

    Return specific canonical unit tensor in given dimensions.

    Parameters
    ----------
    dims: list of ints
        dimensions of the tensor train
    inds: list of ints
        positions of the 1s

    Returns
    -------
    t: instance of TT class
        unit tensor train
    """

    t = zeros(dims, [1] * len(dims))
    for i in range(t.order):
        t.cores[i][0, inds[i], 0, 0] = 1
    return t


def rand(row_dims, col_dims, ranks=1):
    """random tensor train

    Parameters
    ----------
    row_dims: list of ints
        list of row dimensions of the random tensor train
    col_dims: list of ints
        list of column dimensions of the random tensor train
    ranks: list of ints, optional
        list of the ranks of the random tensor train, default is [1, ..., 1]

    Returns
    -------
    tt_rand: instance of TT class
        random tensor train
    """

    # set ranks of tt_rand
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(row_dims) - 1)] + [1]

    # define TT cores of tt_rand
    cores = [np.random.rand(ranks[i], row_dims[i], col_dims[i], ranks[i + 1]) for i in range(len(row_dims))]

    # define tensor train
    tt_rand = TT(cores)

    return tt_rand


def uniform(row_dims, ranks=1, norm=1):
    """uniformly distributed tensor train

    Parameters
    ----------
    row_dims: list of ints
        list of row dimensions of the random tensor train
    ranks: list of ints, optional
        list of the ranks of the uniformly distributed tensor train, default is [1, ..., 1]
    norm: float, optional
        norm of the uniformly distributed tensor train, default is 1

    Returns
    -------
    tt_uni: instance of TT class
        uniformly distributed tensor train
    """

    # set ranks of tt_uni
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(row_dims) - 1)] + [1]

    # compute factor for each core such that tt_uni has given norm
    factor = (norm / (np.sqrt(np.prod(row_dims)) * np.prod(ranks))) ** (1 / len(row_dims))

    # define TT cores of tt_uni
    cores = [factor * np.ones([ranks[i], row_dims[i], 1, ranks[i + 1]]) for i in range(len(row_dims))]

    # define tensor train
    tt_uni = TT(cores)

    return tt_uni
