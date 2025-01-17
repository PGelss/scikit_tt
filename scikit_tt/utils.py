# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy as sp
import time
from typing import List


def header(title=None, subtitle=None):
    """
    Print scikit_tt header.

    Parameters
    ----------
    title : string
        title or name of the procedure
    subtitle : string
        subtitle of the procedure
    """

    print('                                               ')
    print('.  __    __               ___    ___ ___       ')
    print('. /__`  /  `  |  |__/  |   |      |   |        ')
    print('| .__/  \__,  |  |  \  |   |      |   |        ')
    print('o ─────────── o ────── o ─ o ──── o ─ o ── ─  ─')

    if title is not None:
        print('|')
        print('o ─ ' + title)
    if subtitle is not None:
        print('    ' + subtitle)
    print(' ')
    print(' ')


def progress(str_text: str, percent: float, cpu_time: float=0, show: bool=True, width: int=47):
    """
    Show progress in percent.

    Print strings of the form, e.g., 'Running ... 10%' etc., without line breaks.

    Parameters
    ----------
    str_text : string
        string to print
    percent : float
        current progress; if percent=0, the current time is returned
    cpu_time : float
        current CPU time
    show : bool, optional
        whether to print the progress, default is True
    width : int
        width of the progress bar, default is 47
    """

    up_two = '\u001b[2A\r'

    if show:
        if percent >0:
            sys.stdout.write(up_two)
        len_text = len(str_text)
        space_text = ' ' * (width - len_text)
        number_of_boxes = width - 6
        if percent == 100:
            str_percent = '100%'
        else:
            str_percent = str("%.1f" % percent) + '%'
        len_percent = len(str_percent)
        space_percent = ' ' * (6 - len_percent)
        str_cpu = 'CPU time: ' + str("%.1f" % cpu_time) + 's'

        color_done = '\33[42m'
        color_remain = '\33[100m'
        underline = '\033[4m'
        end = '\33[0m'


        done = int(number_of_boxes * (np.floor(percent) / 100))
        str_done = ' ' * done
        str_remain = ' ' * (number_of_boxes - done)

        sys.stdout.write(underline + str_text + space_text + end + '\n')
        sys.stdout.write(color_done + underline + str_done + end)
        sys.stdout.write(color_remain + underline + str_remain + end)
        sys.stdout.write(underline + space_percent + str_percent + end + '\n')
        sys.stdout.write(str_cpu + ' ')
        sys.stdout.flush()

        if percent == 100:
            sys.stdout.write(2*'\n')

    if percent == 0:
        return time.time()


class timer(object):
    """
    Measure CPU time.

    Can be executed using the 'with' statement in order to measure the CPU time needed for calculations.
    """

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self.start_time

def truncated_svd(matrix: np.ndarray, threshold: float=0, max_rank: int=np.inf, rel_truncation: bool=True):
    """
    Compute truncated SVD.

    Parameters
    ----------
    matrix : ndarray
        matrix to be decomposed
    threshold : float, optional
        threshold for truncated SVD, default is 0
    max_rank : int
        maximum rank of truncated SVD
    rel_truncation: bool
        truncate singular values relative to largest singular value. If False,
        parameter threshold is used as absolute truncation threshold.
        Only applies if threshold is non-zero.

    Returns
    -------
    u : ndarray
        matrix of left singular vectors
    s : ndarray
        vector of singular values
    v : ndarray
        matrix of right singular vectors
    """

    # try different Lapack driver if necessary
    try:
        [u, s, v] = sp.linalg.svd(matrix, full_matrices=False, overwrite_a=True, check_finite=False)
    except:
        [u, s, v] = sp.linalg.svd(matrix, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

    # rank reduction
    if threshold != 0:
        if rel_truncation:
            indices = np.where(s / s[0] > threshold)[0]
        else:
            indices = np.where(s > threshold)[0]
        u = u[:, indices]
        s = s[indices]
        v = v[indices, :]
    if max_rank != np.inf:
        u = u[:, :np.minimum(u.shape[1], max_rank)]
        s = s[:np.minimum(s.shape[0], max_rank)]
        v = v[:np.minimum(v.shape[0], max_rank), :]

    return u, s, v
