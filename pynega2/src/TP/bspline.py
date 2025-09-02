# -*- coding: utf-8 -*-
"""
pyNega
Bsplines module
Developed @ICARUS by Ramon Dalmau, M.Sc
2018
"""

__author__ = 'Ramon Dalmau'
__copyright__ = "Copyright 2018, ICARUS"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ramon Dalmau"
__email__ = "ramon.dalmau@upc.edu"
__status__ = "Development"
__docformat__ = 'reStructuredText'

from casadi import *
from scipy.interpolate import splrep


def fit(x, y, **kwargs):
    # get sorted and unique x values
    x, idx = np.unique(x, return_index=True)
    # get optional arguments
    k = kwargs.get('k', 3)
    t = kwargs.get('t', None)
    w = kwargs.get('w', None)
    # modify arguments if required
    if t is not None:
        kwargs['t'] = t[(k + 1):-(k + 1)]
    if w is not None:
        kwargs['w'] = w[idx]
    return splrep(x=x, y=y[idx], **kwargs)


def basis(t, x, k, i):
    """
    Evaluate the B-Spline basis function using Cox-de Boor recursion.

    :param x: Point at which to evaluate.
    :param k: Order of the basis function.
    :param i: Knot number.

    :returns: The B-Spline basis function of the given order, at the given knot, evaluated at the given point.
    """
    if k == 0:
        return if_else(logic_and(t[i] <= x, x < t[i + 1]), 1.0, 0.0)
    else:
        if t[i] < t[i + k]:
            a = (x - t[i]) / (t[i + k] - t[i]) * basis(t, x, k - 1, i)
        else:
            a = 0.0
        if t[i + 1] < t[i + k + 1]:
            b = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * \
                basis(t, x, k - 1, i + 1)
        else:
            b = 0.0
        return a + b


def eval(t, c, k, x):
    y = 0.0
    for i in range(len(t) - k - 1):
        y += if_else(logic_and(x >= t[i], x <= t[i + k + 1]), c[
            i] * basis(t, x, k, i), 0.0)

    return y
