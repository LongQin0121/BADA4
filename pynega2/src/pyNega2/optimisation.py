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

from integrators import integrators
import pyBada3.conversions as conv
import logging
from casadi.tools import *
from collections import OrderedDict
import pandas as pd

__opts__ = {}
__opts__["ipopt.print_level"] = 5
__opts__["expand"] = True
# __opts__["ipopt.linear_solver"] = 'ma27'
# __opts__["ipopt.linear_solver"] = 'ma77'
__opts__["ipopt.linear_solver"] = 'ma86'


# __opts__["ipopt.linear_solver"] = 'ma97'


class NlpsolInput:
    NLPSOL_X0, NLPSOL_P, NLPSOL_LBX, NLPSOL_UBX, NLPSOL_LBG, NLPSOL_UBG, NLPSOL_LAM_X0, NLPSOL_LAM_G0, \
        NLPSOL_NUM_IN = range(9)


class NlpsolOutput:
    NLPSOL_X, NLPSOL_F, NLPSOL_G, NLPSOL_LAM_X, NLPSOL_LAM_G, NLPSOL_LAM_P, \
        NLPSOL_NUM_OUT = range(7)


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def silentRun(arg):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if arg:
                with suppress_stdout_stderr():
                    return function(*args, **kwargs)
            else:
                return function(*args, **kwargs)

        return wrapper

    return decorator


def isDigit(x):
    try:
        return float(x)
    except ValueError:
        return x


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Optimiser:
    """pyNega optimiser class

    The optimiser class allows to define any kind of optimal control problem
    using direct multiple or single-shooting methods

    :ivar x0: initial guess or optimal of variables
    :ivar x: optimal control variables
    :ivar g: vector of NLP constraints
    :ivar lbx: lower bound of variables
    :ivar ubx: upper bound of variables
    :ivar model: model definition
    :ivar integrator: single or multiple shooting integrator
    :ivar L: Lagrangian term of the cost function
    :ivar phi: Mayer term of the cost function
    :ivar lam_x0: Lagrange multipliers at the optimal solution for the variable bounds
    :ivar lam_g0: Lagrange multipliers at the optimal solution for the NLP constraints
    :ivar lam_p: Lagrange multipliers at the optimal solution for the parameters
    :ivar nx: number of states
    :ivar nu: number of controls
    :ivar N: number of control discretisation points
    :ivar t: independent variable at the different discretisation points
    :ivar f: NLP cost function
    :ivar duration: duration variables of those phases whose duration is flexible
    """
    # NLP variables
    z = SX([])
    z0 = DM([])
    lbz = DM([])
    ubz = DM([])
    lam_z0 = DM([])
    # NLP states and controls variables
    xi = SX([])
    x = SX([])
    x0 = DM([])
    lbx = DM([])
    ubx = DM([])
    lam_x0 = DM([])
    # NLP parameter variables
    d = SX([])
    d0 = DM([])
    ubd = DM([])
    lbd = DM([])
    lam_d0 = DM([])
    # NLP fixed parameters
    p = SX([])
    p0 = DM([])
    lam_p0 = DM([])
    # NLP slack variables
    e = SX([])
    e0 = DM([])
    lbe = DM([])
    ube = DM([])
    lam_e0 = DM([])
    alpha = DM([])
    W = DM([])
    # NLP constraints
    g = SX([])
    lbg = DM([])
    ubg = DM([])
    lam_g0 = DM([])
    g0 = DM([])
    # cost function terms
    Lagrange = SX([])
    phi = 0.0
    f = SX([])
    f0 = DM(0)
    # others
    model = None
    integrator = None
    nx = 0
    nxpnu = 0
    nu = 0
    N = 0
    N0 = 0
    t = np.empty(0)
    solver = None
    # _plan      = np.empty((0))
    _plan = None

    def __init__(self, model, **kwargs):
        """The optimiser class allows to define any kind of optimal control problem
        using direct multiple or single-shooting methods

        :param model: model definition
        :type model: pyNega Model class
        """
        self.model = model
        self.nx = self.model.x.shape[0]  # number of states
        self.nu = self.model.u.shape[0]  # number of controls
        self.nxpnu = self.nx + self.nu  # number of variables per discretisation point, using direct collocation
        self.method = 'collocation'
        self.standarize = kwargs.get('standarise', False)

        self.fun = {}
        for key in self.model.x.keys():
            self.fun[key] = Function(key, [self.model.x, self.model.u], [self.model.x[key]])
        for key in self.model.u.keys():
            self.fun[key] = Function(key, [self.model.x, self.model.u], [self.model.u[key]])
        # self.fun = dict(x=Function('modelVars', [self.model.x, self.model.u], [vertsplit(self.model.x), vertsplit(self.model.u)]))

        # create optimiser logger
        self.logger = logging.getLogger("pyNega." + __name__)

        # the initial state is a parameter
        self.xi = SX.sym('xi', self.model.x.sparsity())

        # the vector of NLP parameters include the initial state and the other model parameters
        self.p = vertcat(self.xi, self.model.p)
        self.p0 = vertcat(GenDM_zeros(self.model.x.sparsity()), self.model.p0)

        # initialise plan
        self._plan = pd.DataFrame(columns=self.model.u.keys() + self.model.x.keys())
        self._plan.index.name = str(self.model.iv)

        # get integrator type
        integrator = kwargs.get('integrator', 'trapezoidal')

        # get penalty for the slack variables
        self.rho = float(kwargs.get('rho', 100.0))

        # create integrator(s)
        if integrator in integrators.keys():
            if type(self.model.ode) is dict:
                self.integrator = dict.fromkeys(self.model.ode.keys())
                for key, value in self.model.ode.iteritems():
                    self.integrator[key] = integrators[integrator](value, self.model.opts)
            else:
                self.integrator = integrators[integrator](self.model.ode, self.model.opts)
        else:
            raise ValueError

        # initialise optimiser
        self.initX()
        self.initG()
        self.initBounds()
        self.initGuess(**self.model.guess)

        # WARNING
        # last control of each phase should not be a variable of the NLP !
        # for phase in self.model.phases:
        #    self.g    = substitute(self.g, self.x[phase['@name'], 'u', -1], self.x[phase['@name'], 'u', -2])

        self.lbxstruct = self.lbx
        self.ubxstruct = self.ubx
        self.x0struct = self.x0
        self.xstruct = self.x

        self.x = self.x.cat
        self.d = self.d.cat
        self.x0 = self.x0.cat
        self.d0 = self.d0.cat
        self.lbd = self.lbd.cat
        self.ubd = self.ubd.cat

        self.lbx = self.lbx.cat
        self.ubx = self.ubx.cat

        # normalise weights of the slack variables
        self.alpha = self.alpha / sum1(self.alpha)

        # create NLP variables vector, bounds and initial guess
        self.z = vertcat(self.x, self.d, self.e)
        self.lbz = vertcat(self.lbx, self.lbd, self.lbe)
        self.ubz = vertcat(self.ubx, self.ubd, self.ube)
        self.z0 = vertcat(self.x0, self.d0, self.e0)

        self.initX0(**self.model.x0)

    @staticmethod
    def getValue(b, default=inf):
        value = default
        if type(b) is OrderedDict:
            if '@units' in b.keys():
                value = conv.convertFrom[b['@units']](isDigit(b['#text']))
            else:
                value = isDigit(b['#text'])
        else:
            value = isDigit(b)

        return value

    @classmethod
    def getBound(cls, c, bound, default=inf):
        """Get constraint/bound from definition

        :param c: constraint definition
        :param bound: lower bound (lb) or upper bound (ub)
        :param default: default value of bound/constraint
        :type c: dict
        :type bound: str
        :type default: float
        :returns value: value for bound/constraint
        :rtype: float
        """
        # check if the bound type is in C, otherwise return default (unbounded) value
        value = default
        if bound in c.keys():
            b = c[bound]
            value = cls.getValue(b, default)

        return value

    @staticmethod
    def isSlacked(c, bound, **kwargs):
        """Check if bound or constraint is slacked

        :param c: constraint definition
        :param bound: lower bound (lb) or upper bound (ub)
        :param slacked: default value
        :param alpha: default alpha value (check documentation)
        :param Delta: default Delta value (check documentation)
        :type c: dict
        :type bound: str
        :type slacked: bool
        :type alpha: float
        :type Delta: float
        :returns value, alpha, Delta: slack parameters for the given bound or constraint
        :rtype: tuple
        """
        value = kwargs.get('slacked', False)
        alpha = kwargs.get('alpha', 1.0)
        Delta = kwargs.get('Delta', 1.0)
        units = kwargs.get('units', None)

        if bound in c.keys():
            b = c[bound]
            if type(b) is OrderedDict:
                if '@units' in b.keys():
                    units = b['@units']
                if '@slacked' in b.keys():
                    value = str2bool(b['@slacked'])
                if '@alpha' in b.keys():
                    alpha = isDigit(b['@alpha'])
                if '@Delta' in b.keys():
                    if units is not None:
                        Delta = conv.convertFrom[units](isDigit(b['@Delta']))
                    else:
                        Delta = isDigit(b['@Delta'])

        return value, alpha, Delta

    @staticmethod
    def compteW(lb, ub, Delta=None):
        """Compute weight of a slack from the bounds

        :param lb: lower bound
        :param ub: upper bound
        :param Delta: delta of the slack (optinal)
        :returns W: weight of a slack from the bounds
        :rtype: float
        """
        if Delta is None:
            if lb == -inf:
                lb = 0.0
            if ub == inf:
                ub = 0.0
            if ub == lb and ub != 0.0:
                Delta = ub
            elif ub == lb and ub == 0.0:
                Delta = 1.0
            else:
                Delta = ub - lb

        return 1.0 / Delta ** 2

    @property
    def plan(self):
        return self._plan

    @property
    def config(self):
        return self._plan['config'].iloc[0]

    @property
    def state(self):
        return self._plan[self.model.x.keys()].iloc[0]

    @property
    def control(self):
        return self._plan[self.model.u.keys()].iloc[0]

    @property
    def time(self):
        return self._plan.index[0]

    @plan.deleter
    def plan(self):
        del self._plan

    @plan.setter
    def plan(self, value):
        self._plan = value

    def NCO(self):
        fun = self.sensitivity()
        L_z = fun['nlp_jac_l'](z=self.z0, p=self.p0, lam_g=self.lam_g0)['L_z'].T + self.lam_z0
        L_p = fun['nlp_jac_l'](z=self.z0, p=self.p0, lam_g=self.lam_g0)['L_p'].T + self.lam_p0
        g = self.g0
        e_optZ = norm_inf(L_z)
        e_optP = norm_inf(L_p)
        e_infs = mmax(g)
        return e_optZ, e_optP, e_infs

    def shrink(self):
        if self.N == 2:
            self.logger.warning("The plan cannot be shrinked because it only contains two points")
        else:
            if self.plan.shape[0] > 0 and self.plan.index[0] == self.plan.index[1]:
                remove = False
            else:
                remove = True

            # reduce number of remaining points
            self._plan.reset_index(inplace=True)
            self._plan.drop(self._plan.index[0], inplace=True)
            self._plan.set_index(str(self.model.iv), inplace=True)

            self.Lagrange = self.Lagrange[1:]
            self.N -= 1

            if remove:
                # remove ONLY initial control
                self.x = self.x[self.nu:]
                self.x0 = self.x0[self.nu:]
                self.lbx = self.lbx[self.nu:]
                self.ubx = self.ubx[self.nu:]
                self.lam_x0 = self.lam_x0[self.nu:]

    def activeSet(self, **kwargs):
        lam_x = kwargs.get('lam_x', self.lam_z0)
        lam_g = kwargs.get('lam_g', self.lam_g0)
        bIx = np.where(lam_x != 0)[0]
        bIg = np.where(lam_g != 0)[0]
        return dict(x=set(bIx), g=set(bIg))

    def setX0(self, x0=None):
        # get current state, otherwise it is the current state plan,
        # which may change if trajectory has been shrinked before
        if x0 is None:
            x0 = self.state

        if self.plan.shape[0] > 0 and self.plan.index[0] == self.plan.index[1]:
            # substitute initial state by parameters in the constraints vector
            self.g = substitute(self.g, self.x[:self.nx], self.xi)
            self.Lagrange = substitute(self.Lagrange, self.x[:self.nx], self.xi)

            # set parameter to the current state
            self.p0[:self.nx] = x0

            # remove initial state from the variables, since it is a parameter!
            self.x = self.x[self.nx:]
            self.x0 = self.x0[self.nx:]
            self.lbx = self.lbx[self.nx:]
            self.ubx = self.ubx[self.nx:]
            if self.lam_x0.shape[0] > 0:
                self.lam_x0 = self.lam_x0[self.nx:]
        else:
            # substitute initial state by parameters in the constraints vector
            self.g = substitute(self.g, self.x[self.nu:self.nxpnu], self.xi)
            self.Lagrange = substitute(self.Lagrange, self.x[self.nu:self.nxpnu], self.xi)

            # set parameter to the current state
            self.p0[:self.nx] = x0

            # remove initial state from the variables, since it is a parameter!
            self.x = vertcat(self.x[:self.nu], self.x[self.nxpnu:])
            self.x0 = vertcat(self.x0[:self.nu], self.x0[self.nxpnu:])
            self.lbx = vertcat(self.lbx[:self.nu], self.lbx[self.nxpnu:])
            self.ubx = vertcat(self.ubx[:self.nu], self.ubx[self.nxpnu:])
            if self.lam_x0.shape[0] > 0:
                self.lam_x0 = vertcat(self.lam_x0[:self.nu], self.lam_x0[self.nxpnu:])

        # remove constraints depending UNIQUELY on the parameters (initial conditions of the problem)
        indexes = [i for i in range(self.g.shape[0]) if depends_on(self.g[i], self.x)]

        self.g = self.g[indexes]
        self.lbg = self.lbg[indexes]
        self.ubg = self.ubg[indexes]
        if self.g0.shape[0] > 0:
            self.g0 = self.g0[indexes]
        if self.lam_g0.shape[0] > 0:
            self.lam_g0 = self.lam_g0[indexes]

        # Add constraint to the sum of phase distances (it is removed otherwise)!!
        self.g = vertcat(self.g, sum1(self.d))
        self.lbg = vertcat(self.lbg, self.model.dist2golow)
        self.ubg = vertcat(self.ubg, self.model.dist2goup)

        if self.model.phasefix > 0:
            self.g = vertcat(self.g, sum1(self.d[-self.model.phasefix:]))
            self.lbg = vertcat(self.lbg, self.model.distfix)
            self.ubg = vertcat(self.ubg, self.model.distfix)

        # remove parameters, if required.
        if self.d.shape[0]:
            indexes = np.where(which_depends(self.g, self.d))[0]
            if self.lam_d0.shape[0] > 0:
                self.lam_d0 = self.lam_d0[indexes]
            self.d = self.d[indexes]
            self.d0 = self.d0[indexes]
            self.lbd = self.lbd[indexes]
            self.ubd = self.ubd[indexes]

        # remove slack variables, if required.
        indexes = np.where(which_depends(self.g, self.e))[0]
        if self.lam_e0.shape[0] > 0:
            self.lam_e0 = self.lam_e0[indexes]
        self.e = self.e[indexes]
        self.e0 = self.e0[indexes]
        self.lbe = self.lbe[indexes]
        self.ube = self.ube[indexes]
        self.W = self.W[indexes]
        self.alpha = self.alpha[indexes]

        # create NLP variables vector, bounds and initial guess method
        self.z = vertcat(self.x, self.d, self.e)
        self.lbz = vertcat(self.lbx, self.lbd, self.lbe)
        self.ubz = vertcat(self.ubx, self.ubd, self.ube)
        self.z0 = vertcat(self.x0, self.d0, self.e0)
        if self.lam_z0.shape[0] > 0:
            self.lam_z0 = vertcat(self.lam_x0, self.lam_d0, self.lam_e0)

        # update cost function with the new terms
        self.f = sum1(self.Lagrange) + self.phi + self.rho * mtimes(
            mtimes(transpose(self.e), diag(self.W) * diag(self.alpha)), self.e)

        # compute new cost function and constraints value after shrinking horizon and setting new state
        self.f0, self.g0 = Function('f', [self.z, self.p], [self.f, self.g])(self.z0, self.p0)

    def generateIV(self, ref=0.0):
        """generates vector for the independent variable at the different shooting
        discretisation points
        """
        # only create vector if the vector does not exist or any of the phases has a flexible duration
        # if self._plan.index.size and self.d.shape[0] == 0.0: return
        if self._plan.index.size:
            return

        # get current phase
        tv = np.empty(0)
        cv = np.empty(0)

        # end of the trajectory is always the reference
        t = ref

        # look for the current phase
        curPhase = str(self.x[0]).split('_')[0]
        pstr = [str(i) for i in vertsplit(self.d)]

        # iterate backwards over the different phases of the model
        for phase in self.model.phases[::-1]:
            name = phase['@name']
            N = int(phase['N'])

            if name + '_d' in pstr:
                duration = float(self.d0[pstr.index(name + '_d')])
            else:
                duration = phase['d']
                if type(duration) is OrderedDict:
                    if '@units' in duration.keys():
                        duration = conv.convertFrom[duration['@units']](isDigit(duration['#text']))
                    else:
                        try:
                            duration = isDigit(duration['#text'])
                        except:
                            duration = 0.0
                else:
                    try:
                        duration = isDigit(duration)
                    except:
                        duration = 0.0

            if 'config' in phase.keys():
                config = phase['config']
            else:
                config = 'unknown'

            # independent variable at the different discretisation points of the phase
            tj = np.linspace(start=t, stop=t + duration, num=N)
            cj = np.full(tj.shape, config)

            tv = np.concatenate((tv, tj))
            cv = np.concatenate((cv, cj))

            # break when arriving at the current phase
            if name == curPhase:
                break

            # last independent variable becomes current independent variable
            t = tv[-1]

        # select only the last N nodes
        self._plan = self._plan.reindex(tv[:self.N][::-1])
        self._plan['config'] = cv[:self.N][::-1]

    def initGuess(self, **kwargs):
        """Initialise guess for first optimisation

        .. note:: this function should be improved to accept complete trajectories

        :param x0: state guess [1 x nx]
        :param u0: control guess [1 x nu]
        :type x0: list
        :type u0: list
        """
        # default initialisation to 0
        self.x0 = self.x(0)
        self.d0 = self.d(self.model.guess_d0)
        self.e0 = self.lbe  # initialise slack variables to 0!!

        for key, value in kwargs.iteritems():
            if key in self.model.x.keys():
                self.x0[..., 'x', :, key] = repeated(self.getValue(value))
            elif key in self.model.u.keys():
                self.x0[..., 'u', :, key] = repeated(self.getValue(value))
            elif key == 'd':
                self.x0[..., 'd'] = repeated(self.getValue(value))

    def update(self, **kwargs):
        """Modify values of the NLP solution

        :param z0: new value for the primal variables
        :param lam_z0: new value for the dual variables associated with varialbes
        :param lam_g0: new value for the dual variables associated with constraints
        :param lam_p0: new value for the dual variables associated with parameters
        :param p0: new parameters vector (optional)
        :type z0: DM list
        :type lam_z0: DM list
        :type lam_g0: DM list
        :type lam_p0: DM list
        :type p0: DM list
        """
        self.lam_g0 = kwargs.get('lam_g0', self.lam_g0)
        self.lam_p0 = kwargs.get('lam_p0', self.lam_p0)
        self.p0 = kwargs.get('p0', self.p0)
        self.f0 = kwargs.get('f0', self.f0)
        self.g0 = kwargs.get('g0', self.g0)

        z0 = kwargs.get('z0', None)
        lam_z0 = kwargs.get('lam_z0', None)

        if z0 is not None:
            self.x0 = z0[:self.x0.shape[0]]
            self.d0 = z0[self.x0.shape[0]:self.d0.shape[0] + self.x0.shape[0]]
            self.e0 = z0[self.x0.shape[0] + self.d0.shape[0]:]
            self.z0 = z0

            self.updatePlan()
            # numpyplan = np.array(vertcat(self.x0[:self.nu], self.p0[:self.nx], self.x0[self.nu:])).reshape(-1,self.nxpnu)
            # plan = pd.DataFrame(data=numpyplan, columns=self._plan.columns, index=self._plan.index)
            # self._plan.update(plan)
        if lam_z0 is not None:
            self.lam_x0 = lam_z0[:self.x0.shape[0]]
            self.lam_d0 = lam_z0[self.x0.shape[0]:self.d0.shape[0] + self.x0.shape[0]]
            self.lam_e0 = lam_z0[self.x0.shape[0] + self.d0.shape[0]:]
            self.lam_z0 = lam_z0

    def createSolver(self, solver):
        """Create solver instance
           note: opts not used at all
           note: solver not used at all
        """
        nlp = {'x': self.z, 'p': self.p, 'f': self.f, 'g': self.g}

        # Allocate an NLP solver
        if solver == 'ipopt':
            opts = {}
            # opts["ipopt.linear_solver"] = 'ma27'
            # opts["ipopt.linear_solver"] = 'ma77'
            opts["ipopt.linear_solver"] = 'ma86'
            # opts["ipopt.linear_solver"] = 'ma97'

            opts['ipopt.max_iter'] = 1000  # Setting the maximum number of iterations

            if self.lam_z0.shape[0] > 0:
                opts['ipopt.warm_start_init_point'] = 'yes'
                opts['ipopt.warm_start_entire_iterate'] = 'yes'

            self.solver = silentRun(True)(nlpsol)('solver', 'ipopt', nlp, opts)
        elif solver == 'snopt':
            self.solver = silentRun(True)(nlpsol)('solver', 'snopt', nlp)
        elif solver == 'sqpmethod':
            # self.solver = silentRun(True)(nlpsol)('solver', 'sqpmethod', nlp, dict(min_iter=1, max_iter=50, qpsol='qrqp',
            #                                                          qpsol_options=dict(verbose=False)))
            self.solver = silentRun(True)(nlpsol)('solver', 'sqpmethod', nlp,
                                                  dict(min_iter=1, max_iter=50, qpsol='qpoases',
                                                       qpsol_options=dict(verbose=False)))

    def createSolverArgs(self):
        self.arg = {}

        # initial guess
        self.arg["x0"] = self.z0

        # Bounds on x
        self.arg["lbx"] = self.lbz
        self.arg["ubx"] = self.ubz
        self.arg["p"] = self.p0

        # Bounds on g
        self.arg["lbg"] = self.lbg
        self.arg["ubg"] = self.ubg

        if self.lam_z0.shape[0] > 0:
            self.arg['lam_x0'] = self.lam_z0
        if self.lam_g0.shape[0] > 0:
            self.arg['lam_g0'] = self.lam_g0

    def getSolverResults(self):
        self.x0 = self.res['x'][:self.x0.shape[0]]
        self.d0 = self.res['x'][self.x0.shape[0]:self.d0.shape[0] + self.x0.shape[0]]
        self.e0 = self.res['x'][self.x0.shape[0] + self.d0.shape[0]:]
        self.z0 = vertcat(self.x0, self.d0, self.e0)
        self.lam_x0 = self.res['lam_x'][:self.x0.shape[0]]
        self.lam_d0 = self.res['lam_x'][self.x0.shape[0]:self.d0.shape[0] + self.x0.shape[0]]
        self.lam_e0 = self.res['lam_x'][self.x0.shape[0] + self.d0.shape[0]:]
        self.lam_z0 = vertcat(self.lam_x0, self.lam_d0, self.lam_e0)
        self.lam_g0 = self.res['lam_g']
        self.lam_p0 = self.res['lam_p']
        self.f0 = self.res['f']
        self.g0 = self.res['g']

    def updatePlan(self):
        # generate plan
        indexes = self._plan.index
        cols = self._plan.columns[:-1]
        N = len(indexes)
        count = 0
        # fill plan row by row
        for k in range(N):
            # end of phase
            if k == 0 and indexes[0] == indexes[1]:  # first point of the trajectory has not state
                x = self.p0[:self.nx]
                u = self.x0[:self.nu]
                count = 0
            elif k == 0:
                x = self.p0[:self.nx]
                u = self.x0[:self.nu]
                count = self.nu
            elif k == N - 1:  # last point of the trajectory has not control
                x = self.x0[self.x0.shape[0] - self.nx:]
            else:  # other points have state and control except if end of phase!!!
                if indexes[k] == indexes[k + 1]:  # end of phase, use previous control
                    x = self.x0[count:count + self.nx]
                    count += self.nx
                else:
                    u = self.x0[count:count + self.nu]
                    count += self.nu
                    x = self.x0[count:count + self.nx]
                    count += self.nx

            dfk = pd.DataFrame(data=np.array(vertcat(u, x)).T, columns=cols, index=[indexes[k]])
            self._plan.update(dfk)

    def solve(self, **kwargs):
        """Solve optimal control using NLP algorithmStat

        :returns solver_stats: solver statistics
        :rtype: dict
        """
        silent = kwargs.get('silent', False)

        self.createSolver(solver=kwargs.get('solver', 'ipopt'))
        self.createSolverArgs()

        # solve the problem with interior point method
        self.res = silentRun(silent)(self.solver)(**self.arg)

        solver_stats = self.solver.stats()

        if solver_stats['success']:
            self.getSolverResults()
        else:
            self.logger.warning("Optimiser re-plan has not beet updated due to an infeasible solution")

        # generate independent variables vector
        self.generateIV()

        self.updatePlan()

        # numpyplan = np.array(vertcat(self.x0[:self.nu], self.p0[:self.nx], self.x0[self.nu:])).reshape(-1,self.nxpnu)
        # plan = pd.DataFrame(data=numpyplan, columns=self._plan.columns[:-1], index=self._plan.index)
        # self._plan.update(plan)

        return solver_stats

    def addG(self, c, x, u, fun=None):
        """Add constraint to constraints vector g and associated bounds lbg and ubg

        :param c: constraint definition
        :param x: states affected by the constraint
        :param u: controls affected by the constraint
        :param fun: ...
        :type c: dict
        :type x: list
        :type u: list
        """
        if type(c) == OrderedDict:
            if ('@type' in c.keys() and c['@type'] in self.model.g.keys()) or fun is not None:
                # get expression of the constraint, upper and lower bounds
                if fun is None:
                    fun = self.model.g[c['@type']]
                val = fun(x, u)
                lb = self.getBound(c, 'lb', -inf)
                ub = self.getBound(c, 'ub', +inf)

                # ------------------------------------- Standarised mode ----------------------------------------------
                if self.standarize:
                    # --------------------------------- Fixed value constraint mode -----------------------------------
                    if ub == lb:
                        # set expression to LHS
                        expr = val - lb

                        # check whether constraint is slacked or not
                        slacked, alpha, Delta = self.isSlacked(c, 'lb', Delta=None)

                        # if slacked, define a new slack variable
                        if slacked is True:
                            slack = SX.sym('_'.join(str(x[0]).split('_')[::2] + [c['@type'], 'lb']))
                            self.e = vertcat(self.e, slack)
                            self.alpha = vertcat(self.alpha, alpha)
                            self.W = vertcat(self.W, self.compteW(lb, ub, Delta=Delta))
                            expr += slack

                        # check whether constraint is slacked or not
                        slacked, alpha, Delta = self.isSlacked(c, 'ub', Delta=None)

                        # if slacked, define a new slack variable
                        if slacked is True:
                            slack = SX.sym('_'.join(str(x[0]).split('_')[::2] + [c['@type'], 'ub']))
                            self.e = vertcat(self.e, slack)
                            self.alpha = vertcat(self.alpha, alpha)
                            self.W = vertcat(self.W, self.compteW(lb, ub, Delta=Delta))
                            expr -= slack

                        # add constraint
                        self.g = vertcat(self.g, expr)
                        self.lbg = vertcat(self.lbg, 0.0)
                        self.ubg = vertcat(self.ubg, 0.0)
                    else:
                        if lb != -inf:
                            # set expression to LHS
                            expr = lb - val

                            # constraint on the lower bound is set
                            # check whether constraint is slacked or not
                            slacked, alpha, Delta = self.isSlacked(c, 'lb', Delta=None)

                            # if slacked, define a new slack variable
                            if slacked is True:
                                slack = SX.sym('_'.join(str(x[0]).split('_')[::2] + [c['@type'], 'lb']))
                                self.e = vertcat(self.e, slack)
                                self.alpha = vertcat(self.alpha, alpha)
                                self.W = vertcat(self.W, self.compteW(lb, ub, Delta=Delta))
                                expr = expr - slack

                            # add constraint
                            self.g = vertcat(self.g, expr)
                            self.lbg = vertcat(self.lbg, -inf)
                            self.ubg = vertcat(self.ubg, 0.0)

                        if ub != +inf:
                            # Constraint on th upper bound is set.
                            # Set expression to LHS
                            expr = val - ub

                            # check whether constraint is slacked or not
                            slacked, alpha, Delta = self.isSlacked(c, 'ub', Delta=None)

                            # if slacked, define a new slack variable
                            if slacked is True:
                                slack = SX.sym('_'.join(str(x[0]).split('_')[::2] + [c['@type'], 'ub']))
                                self.e = vertcat(self.e, slack)
                                self.alpha = vertcat(self.alpha, alpha)
                                self.W = vertcat(self.W, self.compteW(lb, ub, Delta=Delta))
                                expr = expr - slack

                            # add constraint
                            self.g = vertcat(self.g, expr)
                            self.lbg = vertcat(self.lbg, -inf)
                            self.ubg = vertcat(self.ubg, 0.0)

                # ------------------------------------- Default mode ------------------------------------------------
                else:
                    # check whether constraint is slacked or not
                    slacked, alpha, Delta = self.isSlacked(c, 'lb', Delta=None)

                    # if slacked, define a new slack variable
                    if slacked is True:
                        slack = SX.sym('_'.join(str(x[0]).split('_')[::2] + [c['@type'], 'lb']))
                        self.e = vertcat(self.e, slack)
                        self.alpha = vertcat(self.alpha, alpha)
                        self.W = vertcat(self.W, self.compteW(lb, ub, Delta=Delta))
                        val += slack

                    # check whether constraint is slacked or not
                    slacked, alpha, Delta = self.isSlacked(c, 'ub', Delta=None)

                    # if slacked, define a new slack variable
                    if slacked is True:
                        slack = SX.sym('_'.join(str(x[0]).split('_')[::2] + [c['@type'], 'ub']))
                        self.e = vertcat(self.e, slack)
                        self.alpha = vertcat(self.alpha, alpha)
                        self.W = vertcat(self.W, self.compteW(lb, ub, Delta=Delta))
                        val -= slack

                    # add constraint
                    self.g = vertcat(self.g, val)
                    self.lbg = vertcat(self.lbg, lb)
                    self.ubg = vertcat(self.ubg, ub)

    def addBound(self, c, name, k):
        """Add bound into bounds vectors lbx and ubx

           note: if the bound type is defined in the dictionary of constraints,
        the bound will be modelled as a constraint and as a bound. i.e. it will be included in the vector g
        and the lower and upper bounds of the associated variable will be -inf and +inf, respectively.

        :param c: constraint definition
        :param name: phase affected by the bound
        :param k: index of discretisation point in the concerned phase
        :type c: dict
        :type name: str
        :type k: int
        """
        if type(c) == OrderedDict:
            # ------------------------------------- State constraints ------------------------------------------------
            if '@type' in c.keys() and c['@type'] in self.model.x.keys() and c['@type'] not in self.model.g.keys():
                lb = self.getBound(c, 'lb', self.lbx[name, 'x', k, c['@type']])
                ub = self.getBound(c, 'ub', self.ubx[name, 'x', k, c['@type']])

                # if slacked, this is not a bound, but a constraint with associated slack variable
                slackedLb, alpha, Delta = self.isSlacked(c, 'lb', Delta=None)
                if slackedLb is True:
                    self.addG(c, self.x[name, 'x', k], self.x[name, 'u', k], fun=self.fun[c['@type']])
                else:
                    self.lbx[name, 'x', k, c['@type']] = lb

                # if slacked, this is not a bound, but a constraint with associated slack variable
                slackedUb, alpha, Delta = self.isSlacked(c, 'ub', Delta=None)
                if slackedUb is True:
                    if slackedLb is False:
                        self.addG(c, self.x[name, 'x', k], self.x[name, 'u', k], fun=self.fun[c['@type']])
                else:
                    self.ubx[name, 'x', k, c['@type']] = ub

            # ------------------------------------- Control constraints ------------------------------------------------
            elif '@type' in c.keys() and c['@type'] in self.model.u.keys() and c['@type'] not in self.model.g.keys():
                lb = self.getBound(c, 'lb', self.lbx[name, 'u', k, c['@type']])
                ub = self.getBound(c, 'ub', self.ubx[name, 'u', k, c['@type']])

                # if slacked, this is not a bound, but a constraint with associated slack variable
                slackedLb, alpha, Delta = self.isSlacked(c, 'lb', Delta=None)
                if slackedLb is True:
                    self.addG(c, self.x[name, 'x', k], self.x[name, 'u', k], fun=self.fun[c['@type']])
                else:
                    self.lbx[name, 'u', k, c['@type']] = lb

                # if slacked, this is not a bound, but a constraint with associated slack variable
                slackedUb, alpha, Delta = self.isSlacked(c, 'ub', Delta=None)
                if slackedUb is True:
                    if slackedLb is False:
                        self.addG(c, self.x[name, 'x', k], self.x[name, 'u', k], fun=self.fun[c['@type']])
                else:
                    self.ubx[name, 'u', k, c['@type']] = ub

    def initX0(self, **kwargs):
        x0 = self.model.x(0)
        for key, value in kwargs.iteritems():
            if key in x0.keys():
                x0[key] = DM(self.getValue(value, 0.0))
        self.setX0(x0)

    def initX(self):
        """Initialise variables vector

        .. note: first state vector is always fixed to the initial state vector parameter. The first state vector will be initially defined, but not used at all
        during the optimisation process. It will be removed when fixing the initial state. Similarly, all the constraints and bounds depending uniquely on this
        state will be removed.

        .. note: last control is not needed because x_{k+1} = f(x_{k}, u_{k}, p) \forall k \in {0, ..., N-1}
        """
        entries = [entry(phase['@name'], struct=struct_symSX([
            (
                entry("u", repeat=[int(phase['N']) - 1], struct=self.model.u),
                # TODO last control of each phase should be removed,
                #  is an additional variable not necessary that could lead to numerical difficulties.
                entry("x", repeat=[int(phase['N'])], struct=self.model.x),
            )]
        )) for phase in self.model.phases]

        # create state structure from entries
        self.x = struct_symSX(entries)

        # add parameters for phase duration (if flexible)
        # NOTE: d parameters are different from p parameters in the sense that THEY ARE VARIABLES
        # (i.e. optimised) but are independent of time. Instead, p is a vector
        # of FIXED parameters, not optimised at all
        entries = [entry(phase['@name'], struct=struct_symSX([entry("d")]))
                   for phase in self.model.phases if
                   (type(phase['d']) is OrderedDict and ('lb' in phase['d'].keys() or 'ub' in phase['d'].keys()))]

        self.d = struct_symSX(entries)

        # compute number of control intervals
        self.N = 0
        for phase in self.model.phases:
            self.N += int(phase['N'])
        self.N0 = self.N

    def initG(self):
        """Initialise constraints vector g and associated bounds lbg and ubg

           note: constraints differ from bounds because they are FUNCTIONS of x, while bounds set upper
        and lower limits on x

        """
        # iterate over phases
        for j, phase in enumerate(self.model.phases):
            name = phase['@name']
            u, x = self.x[name, ...]
            N = int(phase['N'])
            duration = phase['d']
            if 'config' in phase.keys():
                config = phase['config']
            else:
                config = None

            if type(self.integrator) is dict:
                if config in self.integrator.keys():
                    phaseIntegrator = self.integrator[config]
                else:
                    raise KeyError
            else:
                phaseIntegrator = self.integrator

            # check if duration of the phase is flexible, otherwise get the associated scalar
            if name in self.d.keys():
                duration = self.d[name, 'd']
            else:
                if type(duration) is OrderedDict:
                    if '@units' in duration.keys():
                        duration = conv.convertFrom[duration['@units']](isDigit(duration['#text']))
                    else:
                        duration = isDigit(duration['#text'])
                else:
                    duration = isDigit(duration)

            # Parse initial constraints (affecting ONLY the first node of the phase).
            # NOTE: initial constraints of a phase correspond to final constraints of the preceding one.
            # It is preferable to NEVER define initial constraints.
            # It is better to define constraints always in terms of terminal phase constraints
            if 'initial' in phase.keys():
                self.logger.warning(
                    "Initial constraints found for phase " + name + ". " +
                    "It is strongly suggested to model initial constraints as terminal constraints "
                    "of the preceding phase, if possible.")
                # initial event constraints
                if type(phase['initial']['constraint']) == OrderedDict:
                    self.addG(phase['initial']['constraint'], x=x[0], u=u[0])
                else:
                    for c in phase['initial']['constraint']:
                        self.addG(c, x=x[0], u=u[0])

            # parse path constraints (affecting ALL nodes of the phase)
            for i in range(N - 1):
                if 'path' in phase.keys():
                    if type(phase['path']['constraint']) == OrderedDict:
                        self.addG(phase['path']['constraint'], x=x[i], u=u[i])
                    else:
                        for c in phase['path']['constraint']:
                            self.addG(c, x=x[i], u=u[i])

                if i == N - 1:
                    xp1 = self.x[self.model.phases[j + 1]['@name'], 'x', 0]
                else:
                    xp1 = x[i + 1]

                # shooting / collocation constraints
                # TODO add flexibility to generate plan with multiple/single-shooting also
                xf_k, qf_k = phaseIntegrator(x[i], vertcat(u[i], self.model.p), duration / (N - 1))
                val = vertcat(xp1 - xf_k)
                self.g = vertcat(self.g, val)
                self.lbg = vertcat(self.lbg, DM.zeros(self.nx))
                self.ubg = vertcat(self.ubg, DM.zeros(self.nx))

                # contribution of node i to Lagrangian term (last node of phase does not contribute at all in
                # the lagrange term)
                self.Lagrange = vertcat(self.Lagrange, qf_k)

            # parse terminal constraints (affecting ONLY the last node of the phase)
            if 'final' in phase.keys():
                # path constraints
                if type(phase['final']['constraint']) == OrderedDict:
                    self.addG(phase['final']['constraint'], x=x[-1], u=u[-1])
                else:
                    for c in phase['final']['constraint']:
                        self.addG(c, x=x[-1], u=u[-1])

            # link constraints between phases (required for both single/multiple-shooting and direct collocation)
            if j < len(self.model.phases) - 1:
                self.g = vertcat(self.g, vertcat(self.x[self.model.phases[j + 1]['@name'], 'x', 0] - x[-1]))
                self.lbg = vertcat(self.lbg, DM.zeros(self.nx))
                self.ubg = vertcat(self.ubg, DM.zeros(self.nx))
                self.Lagrange = vertcat(self.Lagrange, DM(0.0))
            else:
                # Mayer cost
                self.phi = self.model.phi(self.x[name, 'x', -1])

    def initBounds(self):
        """Initialise bounds vector lbx and ubx
        """
        self.lbx = self.x(-inf)  # default lower bounds on state and control variables
        self.ubx = self.x(inf)  # default upper bounds on state and control variables
        self.lbd = self.d(-inf)  # default lower bounds on parameter variables
        self.ubd = self.d(inf)  # default upper bounds on parameter variables

        # iterate over phases
        for j, phase in enumerate(self.model.phases):
            name = phase['@name']
            N = int(phase['N'])

            # add bounds on the phase duration (if defined)
            if name in self.d.keys():
                duration = phase['d']
                self.lbd[name, 'd'] = self.getBound(duration, 'lb', -inf)
                self.ubd[name, 'd'] = self.getBound(duration, 'ub', +inf)

            for i in range(N - 1):
                # path bounds (affecting ALL nodes of the phase)
                # (except las one?? which is specified by terminal constraints?)
                if 'path' in phase.keys():
                    if type(phase['path']['constraint']) == OrderedDict:
                        self.addBound(phase['path']['constraint'], name, i)
                    else:
                        for c in list(phase['path']['constraint']):
                            self.addBound(c, name, i)

            # initial event bounds (affecting ONLY the first node of the phase)
            if 'initial' in phase.keys():
                if type(phase['initial']['constraint']) == OrderedDict:
                    self.addBound(phase['initial']['constraint'], name, 0)
                else:
                    for c in list(phase['initial']['constraint']):
                        self.addBound(c, name, 0)

            # terminal bounds (affecting ONLY the last node of the phase)
            if 'final' in phase.keys():
                if type(phase['final']['constraint']) == OrderedDict:
                    self.addBound(phase['final']['constraint'], name, -1)
                else:
                    for c in list(phase['final']['constraint']):
                        self.addBound(c, name, -1)

        # slack variables can be as high as possible to satisfy constraints, but cannot soften the constraints.
        self.ube = DM_inf(self.e.shape)
        self.lbe = GenDM_zeros(self.e.shape)

    def sensitivity(self):
        lam_g = SX.sym('lam_g', self.g.sparsity())
        L = self.f + mtimes(transpose(lam_g), self.g)
        fun = {}

        fun["nlp_jac_f"] = Function('nlp_jac_f', [self.z, self.p],
                                    [self.f, jacobian(self.f, self.z), jacobian(self.f, self.p)], ["z", "p"],
                                    ["f", "f_z", "f_p"])
        fun["nlp_hess_l"] = Function('nlp_hess_l', [self.z, lam_g, self.p],
                                     [L, hessian(L, self.z)[0], jacobian(jacobian(L, self.z), self.p)],
                                     ["z", "lam_g", "p"], ["L", "L_zz", "L_zp"])
        fun["nlp_jac_g"] = Function('nlp_jac_g', [self.z, self.p],
                                    [self.g, jacobian(self.g, self.z), jacobian(self.g, self.p)], ["z", "p"],
                                    ["g", "g_z", "g_p"])
        fun["nlp_jac_l"] = Function('nlp_jac_l', [self.z, lam_g, self.p], [L, jacobian(L, self.z), jacobian(L, self.p)],
                                    ["z", "lam_g", "p"], ["L", "L_z", "L_p"])

        # self.solver.factory('nlp_jac_f' , self.solver.name_in(), ['f', 'grad:f:x0',  'grad:f:p'])
        # self.solver.factory('nlp_jac_g' , self.solver.name_in(), ['g', 'jac:g:x0' ,  'jac:g:p'])
        # self.solver.factory('nlp_hess_f', self.solver.name_in(), ['sym:hess:f:x0:x0','sym:hess:f:p:p'])
        return fun

    def getForward2(self, fwd_p, nfwd=1):
        fwd_solver = self.solver.forward(nfwd)

        fwd_ubx = fwd_lbx = DM.zeros(self.z0.sparsity())
        fwd_ubg = fwd_lbg = DM.zeros(self.g.sparsity())

        # sol_ad = fwd_solver(out_x=self.res['x'], out_lam_g=self.res['lam_g'], out_lam_x=self.res['lam_x'],
        #            out_f=self.res['f'], out_g=self.res['g'], lbx=self.arg['lbx'], ubx=self.arg['ubx'],
        #            lbg=self.arg['lbg'], ubg=self.arg['ubg'],
        #            fwd_lbx=fwd_lbx, fwd_ubx=fwd_ubx,
        #            fwd_lbg=fwd_lbg, fwd_ubg=fwd_ubg,
        #            p=self.p0, fwd_p=fwd_p)

        sol_ad = fwd_solver(out_x=self.z0, out_lam_g=self.lam_g0, out_lam_x=self.lam_z0,
                            out_f=self.f0, out_g=self.g0, lbx=self.lbz, ubx=self.ubz, lbg=self.lbg, ubg=self.ubg,
                            fwd_lbx=fwd_lbx, fwd_ubx=fwd_ubx,
                            fwd_lbg=fwd_lbg, fwd_ubg=fwd_ubg,
                            p=self.p0, fwd_p=fwd_p)

        return sol_ad

    def getForward(self, **kwargs):
        """Get forward sensitivities of the NLP solution with respect to the parameters vector
        """
        nfwd = kwargs.get('nfwd', 1)
        silent = kwargs.get('silent', 'ipopt')
        p0 = kwargs.get('p0', self.p0)

        forward = self.solver.factory('nlp_fwd', ['x0', 'lam_x0', 'lbx', 'ubx', 'lam_g0', 'lbg', 'ubg', 'p', 'fwd:p'],
                                      ['f', 'g', 'lam_g', 'lam_p', 'lam_x', 'x', 'fwd:x', 'fwd:lam_g', 'fwd:lam_x',
                                       'fwd:f', 'fwd:lam_p'])
        if silent is True:
            with suppress_stdout_stderr():
                # Sensitivity of the solver at the optimum solution, with respect to p
                sensitivities = forward(x0=self.z0, lam_x0=self.lam_z0, lbx=self.lbz, ubx=self.ubz, lam_g0=self.lam_g0,
                                        lbg=self.lbg, ubg=self.ubg, p=p0, fwd_p=nfwd)
        else:
            sensitivities = forward(x0=self.z0, lam_x0=self.lam_z0, lbx=self.lbz, ubx=self.ubz, lam_g0=self.lam_g0,
                                    lbg=self.lbg, ubg=self.ubg, p=p0, fwd_p=nfwd)

        return sensitivities
