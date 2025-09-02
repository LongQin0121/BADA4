# -*- coding: utf-8 -*-
"""
pyNega
Generic model class
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


class Model(object):
    x = None  # states vector
    u = None  # controls vector
    p = None
    p0 = None
    xdot = None  # state derivatives
    L = None  # lagrangian term
    opts = None
    phases = None  # phases dictionary
    N = None
    phi = None

    def __init__(self):
        pass
