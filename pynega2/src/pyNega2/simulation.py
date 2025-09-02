# -*- coding: utf-8 -*-
"""
pyNega
Simulation module
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
import pyNega2.guidance as guid
import logging
import pandas as pd
import re
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString


class Simulator:
    def __init__(self, optimiser, **kwargs):
        self.i = None
        self.t0 = None
        self.x0 = None
        self.optimiser = optimiser
        self.model = self.optimiser.model
        self.estimator = kwargs.get('estimator', None)
        self.guidance = kwargs.get('guidance', guid.OpenLoop(optimiser=self.optimiser, model=self.model))
        self.opts = kwargs.get('opts', {'t0': 0.0})
        # default parameters are those defined in the model!! Assuming no perturbations but only errors to the
        # discretisation method itself
        # set p0 in kwargs to model mismatches between model and reality!
        self.p0 = kwargs.get('p0', self.model.p0)
        self.logger = logging.getLogger("pyNega." + __name__)
        # initialise flown datalog (includes controls, states and independent variable)
        self.format = kwargs.get('format', None)
        self.q = 0.0
        self.outdir = kwargs.get('out', None)
        self.solvers = kwargs.get('solvers', ['ipopt', 'sqpmethod'])

        if self.outdir is not None:
            with open(os.path.join(self.outdir, 'params.csv'), 'w') as f:
                for p in self.p0:
                    f.write(str(p) + '\n')

    def __iter__(self):
        # compute initial plan
        # -----------------------------------------------------------------------------------------------------
        # first solve with ipopt, which seems to be more stable to dummy guess
        for solver in self.solvers:
            solverStats = self.optimiser.solve(solver=solver, silent=False)
            if not solverStats['success']:
                raise Exception("Error computing initial plan with " + solver)
            else:
                if self.outdir is not None:
                    outFile = os.path.join(self.outdir, '00_stats.xml')
                    with open(outFile, 'w') as f:
                        f.write(parseString(dicttoxml(solverStats, custom_root='solver_stats')).toprettyxml())

        if self.outdir is not None:
            self.optimiser.plan.to_csv(os.path.join(self.outdir, '00_plan.csv'), sep=',')

        if self.estimator is not None:
            self.estimator.save(k=0)

        # Initialise simulation
        # -----------------------------------------------------------------------------------------------------
        # generate independent variable vector for simulation
        self.flown = pd.DataFrame(index=self.optimiser.plan.index, columns=self.optimiser.plan.columns)
        self.flown.config = self.optimiser.plan.config
        self.uniqConf = self.flown.config[~self.flown.index.duplicated(keep='last')]

        # set initial state
        self.x0 = self.optimiser.state

        # set initial time
        self.t0 = self.flown.index[0]

        # initialise counter
        self.i = 0

        return self

    def next(self):
        if self.i < self.flown.index.size - 1:
            # define simulation interval
            # -----------------------------------------------------------------------------------------------------
            tf = self.flown.index[self.i + 1]
            config = self.uniqConf.loc[self.t0]
            if self.i == 28:
                pass

            # log result of step
            # -----------------------------------------------------------------------------------------------------
            self.logger.info('\t'.join([str(self.i + 1), str(int(self.t0)), str(int(tf)),
                                        re.sub(' +', ' ', self.x0.to_string().replace('\n', ' '))]))

            # Controller action in this simulation interval based on the current state
            # -----------------------------------------------------------------------------------------------------
            U = self.guidance.react(t0=self.t0, x0=self.x0, xEst=self.estimator.x0, p0=self.estimator.p0, config=config)

            if U['replan']:
                if self.estimator is not None:
                    self.estimator.save(k=self.guidance.k)

            u = U['u'].values.T
            # Store observations for subsequent estimation
            # -----------------------------------------------------------------------------------------------------
            if self.estimator is not None:
                # observe
                self.estimator.observe(x=self.x0, u=u, p=self.p0)
                # estimate
                self.estimator.estimate()
                # predict
                try:
                    self.estimator.predict(delta=abs(tf - self.t0), config=config)
                except:
                    self.logger.warning("Prediction not performed")

            # Store current state and control
            # -----------------------------------------------------------------------------------------------------

            # check that control is within bounds
            umax = self.model.bounds['ub'](self.x0, u, self.p0)
            umin = self.model.bounds['lb'](self.x0, u, self.p0)

            for k, uk in enumerate(u):
                if uk > umax[k]:
                    u[k] = float(umax[k])
                if uk < umin[k]:
                    u[k] = float(umin[k])

            # self.flown.loc[self.t0].update(u.append(self.x0))
            self.flown.iloc[self.i].update(U['u'].append(self.x0))

            # Simulate interval with controller inputs
            # -----------------------------------------------------------------------------------------------------
            self.step(p=u, tf=tf, config=config)
            # except:
            #    pass
            self.i = self.i + 1

            if U['replan']:
                return self.flown, self.optimiser.plan
            else:
                return self.flown, None
        else:
            # include terminal cost (if any)
            self.q += self.model.phi(self.x0)
            # Store and log last state and control
            # -----------------------------------------------------------------------------------------------------
            self.flown.iloc[-1].update(self.optimiser.control.append(self.x0))
            self.logger.info('\t'.join([str(self.i), str(int(self.flown.index[-1])), str(int(self.flown.index[-1])),
                                        re.sub(' +', ' ', self.x0.to_string().replace('\n', ' '))]))
            self.logger.info('Total cost:' + str(self.q))

            raise StopIteration

    def step(self, p, tf, **kwargs):
        """Perform an integration of the equations of motion of the system from state x0 using the controls and parameters p

        :param p: vector of controls and parameters
        :param t0: initial independent variable
        :param tf: final time variable
        :param x0: initial state
        :type p: list
        :type t0: float
        :type tf: float
        :type x0: list
        """
        # get initial state, initial and final times
        delta = abs(tf - self.t0)

        self.opts['tf'] = delta
        self.x0 = kwargs.get('x0', self.x0)
        config = kwargs.get('config', None)

        # create cvodes integrator
        if type(self.model.ode) is dict:
            if config in self.model.ode.keys():
                f = integrator('f', 'cvodes', self.model.ode[config], self.opts)
            else:
                raise KeyError
        else:
            f = integrator('f', 'cvodes', self.model.ode, self.opts)

        # gather final state
        F = f(x0=self.x0, p=vertcat(p, self.p0))

        self.x0.update(pd.Series(np.array(F['xf']).flatten(), index=self.x0.keys()))
        self.x0.name = tf
        self.q += F['qf']
        self.t0 = tf
