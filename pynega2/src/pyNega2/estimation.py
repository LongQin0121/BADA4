# -*- coding: utf-8 -*-
"""
pyNega
Optimisation module
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

import logging
from casadi.tools import *
import bspline as spline
import numpy as np
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
from scipy.stats import poisson


class Estimator:
    """pyNega estimator class
    #TODO
    The optimiser class allows to define any kind of parameter estimation problem
    using a moving horizon estimation approach
    """
    y = DM([])

    def __init__(self, optimiser, L=5, **kwargs):
        self.integrator = optimiser.integrator
        self.nxpnu = optimiser.nxpnu
        self.nx = optimiser.nx
        self.nu = optimiser.nu
        self.p = optimiser.model.p
        self.lbp = -DM_inf(self.p.shape[0])
        self.ubp = DM_inf(self.p.shape[0])
        self._y = DM([])
        self._inputs = DM([])
        self._p0 = optimiser.model.p0
        self._x0 = DM([])

        self.logger = logging.getLogger("pyNega." + __name__)
        self.outdir = kwargs.get('out', None)

    @property
    def x0(self):
        return self._x0

    @property
    def p0(self):
        return self._p0

    def save(self, **kwargs):
        pass

    def estimate(self, **kwargs):
        self._x0 = self._y[-self.nx:]

    def predict(self, delta, config=None):
        if type(self.integrator) is dict:
            if config in self.integrator.keys():
                phaseIntegrator = self.integrator[config]
            else:
                raise KeyError
        else:
            phaseIntegrator = self.integrator

        self._x0 = phaseIntegrator(self._x0, vertcat(self._inputs[-self.nu:], self._p0), delta)[0]

    def observe(self, x, u, p):
        self._y = vertcat(self._y, x)
        self._inputs = vertcat(self._inputs, u)


class WindEstimator(Estimator):
    """pyNega estimator class
    #TODO
    The optimiser class allows to define any kind of parameter estimation problem
    using a moving horizon estimation approach
    """
    y = DM([])

    def __init__(self, optimiser, L=5, **kwargs):
        Estimator.__init__(self, optimiser=optimiser, L=L, **kwargs)
        self.observer = optimiser.model.y
        self.wind = kwargs.get('data', np.empty((0, 2)))
        self.wind = np.hstack((np.zeros((self.wind.shape[0], 1)), self.wind))
        self.splinepars = kwargs.get('splinepar', {})
        self.tck = kwargs.get('tck', spline.fit(x=self.wind[:, 1], y=self.wind[:, 2], **self.splinepars))
        self.mu = float(kwargs.get('mu', 1.0))  # forgetting factor
        self.frac = float(kwargs.get('frac', 0.0))
        self.loc = float(kwargs.get('loc', 0.0))
        self.scale = float(kwargs.get('scale', 0.0))
        self.lam = float(kwargs.get('lam', 0.0))
        self.tau = 0

    def save(self, **kwargs):
        k = kwargs.get('k', 0)
        estimation = dict(zip(['t', 'c', 'k'], self.tck))
        if self.outdir is not None:
            outFile = os.path.join(self.outdir, str(k).zfill(2) + '_estimation.xml')
            with open(outFile, 'w') as f:
                f.write(parseString(dicttoxml(estimation, custom_root='spline')).toprettyxml())

    def estimate(self, **kwargs):
        t, c, k = self.tck
        # compute weights using the forgetting factor
        self.w = self.mu ** (self.tau - self.wind[:, 0])
        # fit control points using new observations
        self.tck = spline.fit(x=self.wind[:, 1], y=self.wind[:, 2], t=t, w=self.w, **self.splinepars)
        # update estimator parameters
        self._p0 = self.tck[1]
        # update estimator initial condition
        self._x0 = self._y[-self.nx:]

    def observe(self, x, u, p):
        self.tau = self.tau + 1
        obs = self.observer(x, u, p)
        dev = np.random.normal(loc=self.loc, scale=self.scale)
        self._y = vertcat(self._y, x)
        self._inputs = vertcat(self._inputs, u)
        self.wind = np.vstack((self.wind, np.append(self.tau, np.array(obs)).reshape((-1, 3))))
        self.wind[-1, 2] = self.wind[-1, 2] + dev

        # inject = np.random.binomial(1, self.frac)
        # if inject == 1:
        num = poisson.rvs(self.lam, size=1)
        for dummy in range(num):
            dev = np.random.normal(loc=self.loc, scale=self.scale)
            h = np.random.uniform(low=np.min(self.wind[:, 1]), high=x[-1])
            obs = self.observer(DM([0, 0, h]), u, p)
            self.wind = np.vstack((self.wind, np.append(self.tau, np.array(obs)).reshape((-1, 3))))
            self.wind[-1, 2] = self.wind[-1, 2] + dev


class MHE(Estimator):
    """pyNega estimator class
    #TODO
    The optimiser class allows to define any kind of parameter estimation problem
    using a moving horizon estimation approach
    """
    y = DM([])

    def __init__(self, optimiser, L=5, **kwargs):
        self.t = kwargs.get('t')
        self.h = kwargs.get('h', DM([]))
        self.w = kwargs.get('w', DM([]))
        self.weight = 0.5 * np.ones(self.w.shape[0])

        self.j = 0
        self.L = L
        self.R = DM([0.5, 0.5, 10.0])
        self.integrator = optimiser.integrator
        self.nxpnu = optimiser.nxpnu
        self.nx = optimiser.nx
        self.nu = optimiser.nu
        self.p = optimiser.model.p
        self._p0 = optimiser.model.p0
        self.lbp = -DM_inf(self.p.shape[0])
        self.ubp = DM_inf(self.p.shape[0])

        self.x = SX([])
        self.u = SX([])
        self.lbx = DM([])
        self.ubx = DM([])

        for key in optimiser.xstruct.keys():
            self.x = vertcat(self.x, *optimiser.xstruct[key, 'x'])
            self.lbx = vertcat(self.lbx, *optimiser.lbxstruct[key, 'x'])
            self.ubx = vertcat(self.ubx, *optimiser.ubxstruct[key, 'x'])
            self.u = vertcat(self.u, *optimiser.xstruct[key, 'u'])

        self.y = DM([])
        self.inputs = DM([])
        self.d = optimiser.d
        self.e = optimiser.e
        self.g = optimiser.g

        self.lbg = optimiser.lbg
        self.g = optimiser.g
        self.ubg = optimiser.ubg

        self._y = DM([])
        self._x = SX([])
        self._g = SX([])
        self._lbx = DM([])
        self._ubx = DM([])
        self._lbg = DM([])
        self._ubg = DM([])
        self._u = SX([])
        self._in = DM([])
        self._x0 = DM([])
        self.logger = logging.getLogger("pyNega." + __name__)

    @property
    def x0(self):
        return self._x0

    @property
    def p0(self):
        return self._p0

    @property
    def states(self):
        return self._x

    @property
    def controls(self):
        return self._u

    @property
    def observations(self):
        return self._y

    def shrink(self):
        if self.j - self.L < 0:
            self.logger.warning("Not enough observations to perform moving horizon estimation")
            return False

        self._y = self.y[(self.j - self.L) * self.nx:self.j * self.nx]
        self._in = self.inputs[(self.j - self.L) * self.nu:self.j * self.nu]

        self._x = self.x[(self.j - self.L) * self.nx:self.j * self.nx]
        self._u = self.u[(self.j - self.L) * self.nu:self.j * self.nu]
        self._lbx = self.lbx[(self.j - self.L) * self.nx:self.j * self.nx]
        self._ubx = self.ubx[(self.j - self.L) * self.nx:self.j * self.nx]

        indexes = [
            i
            for i in range(self.g.shape[0])
            if (depends_on(self.g[i], self._x)
                and not depends_on(self.g[i], self.x[self.j * self.nx:])
                and not depends_on(self.g[i], self.x[:(self.j - self.L) * self.nx]))
        ]

        self._g = self.g[indexes]
        self._lbg = self.lbg[indexes]
        self._ubg = self.ubg[indexes]
        return True

    def createSolver(self, method):
        """Create solver instance
           note: opts not used at all
           note: solver not used at all
        """
        nlp = {'x': self._z, 'f': self._f, 'g': self._g}

        # Allocate an NLP solver
        if method == 'ipopt':
            self.solver = nlpsol('solver', 'ipopt', nlp, dict(ipopt=dict(tol=1e-9), calc_multipliers=False))
        elif method == 'snopt':
            self.solver = nlpsol('solver', 'snopt', nlp, {})
        elif method == 'sqpmethod':
            self.solver = nlpsol('solver', 'sqpmethod', nlp, dict(min_iter=1, qpsol='activeset'))

    def estimate(self, **kwargs):
        if self.shrink() is True:
            # TODO matrix based on observations
            V = repmat(1.0 / self.R, self.L)
            Qp = repmat(1.0 / DM(100.0), self.p.shape[0])

            e = self._x - self._y
            dp = self.p - self.p0

            self._f = mtimes(mtimes(dp.T, diag(Qp)), dp) + mtimes(mtimes(e.T, diag(V)), e)
            self._z = vertcat(self.p, self._x)

            self._g = substitute(self._g, self._u, self._in)
            silent = kwargs.get('silent', False)

            self.createSolver(kwargs.get('method', 'ipopt'))
            self.createSolverArgs()

            self.res = self.solver(**self.arg)
            solver_stats = self.solver.stats()

            if solver_stats['success']:
                self.p0 = self.res['x'][:self.p.shape[0]]
                self.x0 = self.res['x'][-self.nx:]
            else:
                if self.y.shape[0] >= self.nx:
                    self.x0 = self.y[-self.nx:]
                self.logger.warning("Optimiser re-plan has not beet updated due to an infeasible solution")
        else:
            if self.y.shape[0] >= self.nx:
                self.x0 = self.y[-self.nx:]

    def predict(self, delta, config=None):
        self._x0 = self.integrator(self._x0, vertcat(self.inputs[-self.nu:], self.p0), delta)[0]

    def createSolverArgs(self):
        self.arg = {
            "x0": vertcat(self.p0, self._y),   # initial guess
            "lbx": vertcat(self.p0 - fabs(self.p0) * 0.1, self._lbx),  # Bounds on x
            "ubx": vertcat(self.p0 + fabs(self.p0) * 0.1, self._ubx),  # Bounds on x
            "lbg": self._lbg,   # Bounds on g
            "ubg": self._ubg   # Bounds on g
        }

    def observe(self, y, u, p=None):
        self.y = vertcat(self.y, y)
        self.inputs = vertcat(self.inputs, u)
        self.j += 1
