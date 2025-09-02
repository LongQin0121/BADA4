# -*- coding: utf-8 -*-
"""
pyNega
Utilities module
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

import datetime


def formatArray(x, format=None):
    # x = list(np.array(x))
    # if format is None: format = [[lambda x: x]*len(x)]
    # for f in format:
    #    if len(f) != len(x): raise ValueError
    #    for i in range(len(x)):
    #        x[i] = f[i](x[i])
    return str(x)


def deltaTime(x):
    return datetime.timedelta(seconds=int(x))
