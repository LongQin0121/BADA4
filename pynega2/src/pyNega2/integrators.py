# -*- coding: utf-8 -*-
"""
pyNega
ODE Integrators module
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

from casadi.tools import *


def euler(ode, opts):
    """This function integrates the ODE system using forwards euler integration scheme

    :param ode: ODE definition
    :param opts: integrator options
    :type ode: dict
    :type opts: dict
    :returns: CasADi integrator function

    """
    T = SX.sym('T')
    DT = T / opts['M']
    f = Function('f', [ode['x'], ode['p']], [ode['ode'], ode['quad']])
    X0 = SX.sym('X0', ode['x'].shape[0])
    U = SX.sym('U', ode['p'].shape[0])
    X = X0
    Q = 0
    for j in range(opts['M']):
        k1, k1_q = f(X, U)
        X = X + DT * k1
        Q = Q + DT * k1_q

    return Function('F', [X0, U, T], [X, Q], ['x0', 'p', 'dt'], ['xf', 'qf'])


def trapezoidal(ode, opts):
    """This function integrates the ODE system using trapezoidal integration scheme

    :param ode: ODE definition
    :param opts: integrator options
    :type ode: dict
    :type opts: dict
    :returns: CasADi integrator function

    """
    T = SX.sym('T')
    DT = T / opts['M']
    f = Function('f', [ode['x'], ode['p']], [ode['ode'], ode['quad']])
    X0 = SX.sym('X0', ode['x'].shape[0])
    U = SX.sym('U', ode['p'].shape[0])
    X = X0
    Q = 0
    for j in range(opts['M']):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT * k1, U)
        X = X + DT / 2 * (k1 + k2)
        Q = Q + DT / 2 * (k1_q + k2_q)

    return Function('F', [X0, U, T], [X, Q], ['x0', 'p', 'dt'], ['xf', 'qf'])


def rungeKutta4(ode, opts):
    """This function integrates the ODE system using Runge-Kutta 4 integration scheme

    :param ode: ODE definition
    :param opts: integrator options
    :type ode: dict
    :type opts: dict
    :returns: CasADi integrator function

    """
    T = SX.sym('T')
    DT = T / opts['M']
    f = Function('f', [ode['x'], ode['p']], [ode['ode'], ode['quad']])
    X0 = SX.sym('X0', ode['x'].shape[0])
    U = SX.sym('U', ode['p'].shape[0])
    X = X0
    Q = 0
    for j in range(opts['M']):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT / 2 * k1, U)
        k3, k3_q = f(X + DT / 2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

    return Function('F', [X0, U, T], [X, Q], ['x0', 'p', 'dt'], ['xf', 'qf'])


integrators = {'rungeKutta4': rungeKutta4, 'euler': euler, 'trapezoidal': trapezoidal}
