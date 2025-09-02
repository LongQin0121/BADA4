# -*- coding: utf-8 -*-
"""
pyBada
common constants module
2021
"""

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"

import datetime as dt
import numpy as np
import pickle as pckl


# Datetime object to POSIX seconds
def time2sec(time):

    t_sec = (time - dt.datetime(1970, 1, 1)).total_seconds()

    return t_sec


# POSIX seconds to datetime object
def sec2time(t):
    date_time = dt.datetime.utcfromtimestamp(t)

    return date_time


# Call the interpolation function
def interp_gen(x_1, x_2, y_1, y_2, pos):

    # F* casadi
    x_1 = float(x_1)
    x_2 = float(x_2)
    y_1 = float(y_1)
    y_2 = float(y_2)
    pos = float(pos)

    steps = [x_1, pos, x_2]
    x = [x_1, x_2]
    y = [y_1, y_2]
    time_int = np.interp(steps, x, y)

    return time_int


# Pickle operations

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pckl.dump(obj, f, pckl.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        try:
            return pckl.load(f)

        # Handled exception in case trying to load a pickle python 2.7 object (encoding changed from latin1 to ascii)
        except UnicodeDecodeError:
            return pckl.load(f, encoding="latin1")
