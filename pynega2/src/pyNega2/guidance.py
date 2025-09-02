# -*- coding: utf-8 -*-
"""
pyNega
Control and guidance module
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
import logging
import pyNega2.utilities as utils
import pyNega2.optimisation as opt
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import copy
import pandas as pd

print("Guidance")


class Guidance(object):
    def __init__(self, **kwargs):
        self.optimiser = kwargs.get('optimiser', None)
        self.logger = logging.getLogger("pyNega." + __name__)
        self.format = kwargs.get('format', None)
        self.p0 = kwargs.get('p0', self.optimiser.model.p0)
        self.outdir = kwargs.get('out', None)
        self.k = 0

    def react(self, t0, x0, **kwargs):
        pass

    def saveStats(self, solver_stats):
        self.k += 1
        if self.outdir is not None:
            outFile = os.path.join(self.outdir, str(self.k).zfill(2) + '_stats.xml')
            with open(outFile, 'w') as f:
                f.write(parseString(dicttoxml(solver_stats, custom_root='solver_stats')).toprettyxml())


class Tactical(Guidance):
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)
        self.model = self.optimiser.model
        self.p0Real = kwargs.get('p0Real', DM([]))

    def react(self, t0, x0, **kwargs):
        config = kwargs.get('config', None)
        u = MX.sym('u', self.model.u.shape)

        la = self.optimiser.plan.iloc[1]
        cur = self.optimiser.plan.iloc[0]

        xf = la[self.model.x.keys()].values
        u0 = cur[self.model.u.keys()].values
        tf = self.optimiser.plan.index[1]
        u0 = self.optimiser.control.values
        x0 = x0.values

        delta = abs(tf - t0)

        # create cvodes integrator
        if type(self.model.ode) is dict:
            if config in self.model.ode.keys():
                f = integrator('f', 'cvodes', self.model.ode[config], dict(tf=delta))
            else:
                raise KeyError
        else:
            f = integrator('f', 'cvodes', self.model.ode, dict(tf=delta))

        # gather final state
        F = f(x0=x0, p=vertcat(u, self.p0Real))

        # collocation constraints
        deltaXState = F['xf'] - xf
        deltaX = vertcat(*self.model.g['deviation'](F['xf'])) - vertcat(*self.model.g['deviation'](xf))

        # cost        = 100.0*mtimes(mtimes(deltaX.T, diag(DM([1e3, 50, 1]))), deltaX) + F['qf']
        cost = 100 * mtimes(mtimes(deltaX.T, diag(DM([1, 100]))), deltaX) + mtimes(
            mtimes(deltaXState.T, diag(DM([0.0, 1, 1]))), deltaXState) + 1.0 * F['qf']

        g = MX([])
        lbg = DM([])
        ubg = DM([])
        arg = {}

        umax = vertcat(*self.model.bounds['ub'](x0, u, self.p0Real))
        umin = vertcat(*self.model.bounds['lb'](x0, u, self.p0Real))

        # ---------------- minimise deviation to the next energy and time while satisfying constraints ----------------
        # control constraints
        g = vertcat(g, umax - u)
        g = vertcat(g, u - umin)

        ubg = vertcat(ubg, DM_inf(2 * u.shape[0]))
        lbg = vertcat(lbg, GenDM_zeros(2 * u.shape[0]))

        nlp = {'x': u, 'f': cost, 'g': g}
        solver = nlpsol('solver', 'ipopt', nlp, {'ipopt.max_iter': 200})

        arg["x0"] = DM(u0)
        arg["lbg"] = lbg
        arg["ubg"] = ubg

        with opt.suppress_stdout_stderr():
            res = solver(**arg)

        errorOpt = res['f']

        # ------------------------- minimise cost while not exceeding minimum deviation ---------------------
        # g   = vertcat(g,   cost)
        # lbg = vertcat(lbg, -inf)
        # ubg = vertcat(ubg, 10)
        # cost = F['qf']

        # --------------------- minimise control deviation while not exceeding minimum deviation ---------------------
        # deltaU = u - u0
        # cost = mtimes(mtimes(deltaU.T, diag(DM([1000000.0, 1, 1000.0]))), deltaU)

        # nlp = {'x': u, 'f': cost, 'g': g}
        # solver = nlpsol('solver', 'ipopt', nlp,{'ipopt.max_iter':200})

        # arg["x0"]     = res['x']
        # arg['lam_x0'] = res['lam_x']
        # arg["lbg"]    = lbg
        # arg["ubg"]    = ubg

        # with opt.suppress_stdout_stderr():
        # res = solver(**arg)

        # solver_stats = solver.stats()

        F = f(x0=x0, p=vertcat(res['x'], self.p0Real))
        u = pd.Series(data=np.array(res['x']).squeeze(), index=self.model.u.keys())

        deltaX = vertcat(*self.model.g['deviation'](F['xf'])) - vertcat(*self.model.g['deviation'](xf))

        self.logger.info('Target state: ' + str(xf) + '. Predicted state: ' + str(F['xf']))
        self.logger.info('Es error: ' + str(deltaX[0] / 0.3048) + ' ft')
        self.logger.info('t error: ' + str(deltaX[1]) + ' s')

        self.optimiser.shrink()

        return {'u': u, 'replan': False}


class SbNMPC(Guidance, opt.Optimiser):
    # TODO
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)
        self.solvers = kwargs.get('solvers', ['sqpmethod'])

    def solve(self):
        pass

    def react(self, t0, x0, **kwargs):
        # get estimated parameters
        self.p0 = kwargs.get('p0', self.p0)
        replan = False

        if self.optimiser.N == self.optimiser.N0:
            # get optimal control at the current position
            u = self.optimiser.control

            # shrink horizon
            self.optimiser.shrink()
        elif t0 == self.optimiser.time and self.optimiser.N != 2:
            # x0 is the planned state at the current point
            # set planned state for the parameters! not the estimated, not the real!
            # when setting x0, the cost function and constraints on the shrinked problem
            # are automatically computed by the optimiser, therefore, f and g are evaluated at
            # the shrinked horizon!
            self.optimiser.setX0(x0=None)

            z0_cur = copy.copy(self.optimiser.z0)
            f0_cur = copy.copy(self.optimiser.f0)
            g0_cur = copy.copy(self.optimiser.g0)
            lam_g0_cur = copy.copy(self.optimiser.lam_g0)
            lam_z0_cur = copy.copy(self.optimiser.lam_z0)

            p = vertcat(x0, self.p0)

            # compute jacobian and hessian of the KKT system, with respect to parameters
            # but also with respect to the current state !
            fun = self.optimiser.sensitivity()

            # define optimal deviation variables
            deltaz = SX.sym('delta', self.optimiser.z.sparsity())

            arg = {}

            # bounds on g keep identical
            arg["lbg"] = self.optimiser.lbg
            arg["ubg"] = self.optimiser.ubg

            # initially, assume no perturbations
            arg["x0"] = GenDM_zeros(deltaz.sparsity())

            tau_opt = 5.0e-4
            tau_infs = 1.0e-10

            e_opt = 2 * tau_opt
            e_infs = 2 * tau_infs
            n = 1
            Nmax = 15

            update = False
            while (e_opt > tau_opt or e_infs > tau_infs) and n < Nmax:
                # calculate difference on parameters
                deltap = p - self.optimiser.p0

                # Get jacobian and hessian at the referece optimal solution to compute optimal sensitivities
                nlp_hess_l = fun['nlp_hess_l'](z=self.optimiser.z0, p=self.optimiser.p0, lam_g=self.optimiser.lam_g0)
                nlp_jac_f = fun['nlp_jac_f'](z=self.optimiser.z0, p=self.optimiser.p0)
                nlp_jac_g = fun['nlp_jac_g'](z=self.optimiser.z0, p=self.optimiser.p0)

                # define quadratic cost function
                f = 0.5 * mtimes(transpose(deltaz), mtimes(nlp_hess_l['L_zz'], deltaz)) + \
                    mtimes(transpose(deltap), mtimes(transpose(nlp_hess_l['L_zp']), deltaz)) + \
                    mtimes(nlp_jac_f['f_z'], deltaz)

                # linearise also bounds
                # g_l <= g_0 +  g_x \Delta g + g_p \Delta p <= g_u
                # lineare constraints to fulfill them up to first order
                g = nlp_jac_g["g"] + mtimes(nlp_jac_g["g_z"], deltaz) + mtimes(nlp_jac_g["g_p"], deltap)

                # define quatratic programming  problem
                qp = {'x': deltaz, 'f': f, 'g': g}

                with opt.suppress_stdout_stderr():
                    # solver = qpsol("solver", 'qpoases', qp)
                    solver = nlpsol("solver", 'snopt', qp)

                # -----------------------------------------------------------------------------------------

                # linearise also bounds
                # x_l <= x_0 +  1 \Delta x + 0 \Delta p <= x_u
                # x_l - x_0 <= \Delta x
                # \Delta x <= x_u - x_0
                arg["lbx"] = self.optimiser.lbz - self.optimiser.z0
                arg["ubx"] = self.optimiser.ubz - self.optimiser.z0

                arg['lam_x0'] = self.optimiser.lam_z0
                arg['lam_g0'] = self.optimiser.lam_g0

                try:
                    with opt.suppress_stdout_stderr():
                        res = solver(**arg)
                except:
                    solver_stats = dict(success=False)

                solver_stats = solver.stats()

                self.logger.info('Optimal sensitivity-based QP solution computed with exit code ' + str(
                    solver_stats['success']) + ' at q ' + str(n))

                if solver_stats['success']:
                    update = True

                    # update solution with the resulting trajectory
                    # evaluate constraints and cost function at the new solution
                    z0 = res['x'] + self.optimiser.z0
                    f0 = fun['nlp_jac_f'](z=z0, p=p)['f']
                    g0 = fun['nlp_jac_g'](z=z0, p=p)['g']
                    lam_g0 = res['lam_g']
                    lam_z0 = res['lam_x']
                    self.optimiser.update(z0=z0, f0=f0, g0=g0, lam_z0=lam_z0, lam_g0=lam_g0, p0=p)

                    # optimallity check
                    # evaluate optimallity error
                    L_z = fun['nlp_jac_l'](z=z0, p=p, lam_g=lam_g0)['L_z'].T + lam_z0
                    g = g0

                    L_z_norm = norm_inf(L_z)
                    g_max = mmax(g)

                    e_opt = L_z_norm / (norm_2(lam_g0) + 1.0)
                    e_infs = g_max / (norm_2(z0) + 1.0)

                    self.logger.info(
                        'Error in the Lagrange sensitivity ' + str(e_opt) + ' ( ' + str(L_z_norm) + ' ) <> ' + str(
                            tau_opt) + ' at q ' + str(n))
                    self.logger.info(
                        'Non-linear constraint infeasibility ' + str(e_infs) + ' ( ' + str(g_max) + ' ) <> ' + str(
                            tau_infs) + ' at q ' + str(n))

                    n += 1
                else:
                    update = False
                    break

            if n == Nmax:
                update = False

            if not update:
                self.logger.warning(
                    'Plan will not be updated based on sensitivities this time step. The trajectory will be recomputed')
                self.optimiser.update(z0=z0_cur, f0=f0_cur, g0=g0_cur, lam_z0=lam_z0_cur, lam_g0=lam_g0_cur, p0=p)

                try:
                    # solve open-loop OCP on the shrinked horizon
                    for solver in self.solvers:
                        solver_stats = self.optimiser.solve(silent=True, solver=solver)

                    if solver_stats['success']:
                        replan = True

                    self.logger.info('Re-plan exit code ' + str(solver_stats['success']))
                except Exception as e:
                    self.logger.error(str(e))
            else:
                replan = True

            if replan:
                self.saveStats(solver_stats)

            # get optimal control
            u = self.optimiser.control

            # shrink horizon
            self.optimiser.shrink()
        else:
            # get optimal control
            u = self.optimiser.control

        return {'u': u, 'replan': replan}


class OpenLoop(Guidance):
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)

    def react(self, t0, x0, **kwargs):
        # get optimal control
        u = self.optimiser.control
        if t0 == self.optimiser.time and self.optimiser.N != 2:
            self.optimiser.shrink()

        return {'u': u, 'replan': False}


class Strategic(Guidance):
    # TODO
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)
        # NOTE: if the trigger is not defined, the strategic guidance is equivalent to open-loop!
        self.trigger = kwargs.get('trigger', {
            'bounds': Function('bounds', [SX.sym('iv')], [inf]),
            'deviation': Function('deviation', [self.optimiser.model.x.cat], [0.0])
        })

    def react(self, t0, x0, **kwargs):
        replan = False
        # change new set point when t0 is the current IV of the plan
        if t0 == self.optimiser.getIV() and self.optimiser.N != 2:
            bounds = np.array(self.trigger['bounds'](t0))
            actual = np.array(self.trigger['deviation'](x0))
            plan = np.array(self.trigger['deviation'](self.optimiser.getX()))
            deviation = abs(actual - plan)

            self.logger.info(
                'DeltaX : ' + utils.formatArray(deviation, self.format) +
                '. Bound : ' + utils.formatArray(bounds, self.format))

            # check if a plan needs to be triggered
            if any(deviation > bounds):
                replan = True
                # set current state
                self.optimiser.setState(x0=x0)

                # solve open-loop OCP on the shrinked horizon
                # TODO solve in silent mode, always
                # TODO store solver results in a file for debugging (?) this should be done at solver level!!!!!!
                result = self.optimiser.solve(
                    opts={'start': 'warm'}, solver='snopt', silent=True, warmStart=True, rho=1.0e4)
                success = result['success']

                if success:
                    self.logger.info('Re-plan generated with exit code ' + str(success))
                else:
                    self.logger.error('Re-plan generated with exit code ' + str(success))

                # get optimal control
                u = self.optimiser.getU()

                # shrink horizon
                self.optimiser.shrinkHorizon(k=1)
            else:
                u = self.optimiser.getU()
                self.optimiser.shrinkHorizon(k=1)
        else:
            # get optimal control\
            u = self.optimiser.getU()

        return {'u': u, 'replan': replan}


class AsNMPC(Guidance, opt.Optimiser):
    # TODO
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)
        self.solvers = kwargs.get('solvers', ['sqpmethod'])

    def solve(self):
        pass

    def react(self, t0, x0, **kwargs):
        replan = False
        self.p0 = kwargs.get('p0', self.p0)
        xEst = kwargs.get('xEst', None)

        # compute forward sensitivities of the NLP solution
        # -----------------------------------------------------------------------------------------------------
        if self.optimiser.N == self.optimiser.N0:
            # get optimal control at the current position
            u = self.optimiser.control

            # shrink horizon
            self.optimiser.shrink()
        elif t0 == self.optimiser.time and self.optimiser.N != 2:
            # x0 is the planned state at the current point.
            # should be changed by the ESTIMATED state at the current point
            # this is equivalent to start the optimisation of the plan in the previous point
            # and the uodate the sensitivities on the observed state
            # x0 is the ESTIMATED state obtained from a MHE
            # in AsNMPC it is assumed that the plan is computed instantaneously when receiving a new measurement
            # when the new observation is available, the plan is updated based on sensitivities
            self.optimiser.setX0(x0=xEst)

            # re-plan using estimated parameters at the current point
            try:
                for solver in self.solvers:
                    solver_stats = self.optimiser.solve(solver=solver, silent=True)

                self.logger.info('Re-plan exit code ' + str(solver_stats['success']))

                # if the estimation was successful, update solution with the observed parameters
                if solver_stats['success']:
                    self.saveStats(solver_stats)
                    replan = True

                    # here x0 is the OBSERVED state
                    # p0 is the OBSERVED parameter.
                    p = vertcat(x0, self.p0)

                    # get deviation with respect to what was expected
                    deltap = p - self.optimiser.p0

                    # get sensitivities and update NLP primal and dual variables
                    fwd = self.optimiser.getForward2(fwd_p=deltap, nfwd=1)
                    finite = dict.fromkeys(fwd.keys())

                    for key, value in fwd.iteritems():
                        fwd[key] = np.array(value)
                        finite[key] = np.where(np.isfinite(fwd[key]))[0]

                    z0 = self.optimiser.z0
                    f0 = self.optimiser.f0
                    g0 = self.optimiser.g0
                    lam_z0 = self.optimiser.lam_z0
                    lam_g0 = self.optimiser.lam_g0
                    lam_p0 = self.optimiser.lam_p0

                    try:
                        e_infs = mmax(fwd['fwd_g'][finite['fwd_g']]) / (norm_2(fwd['fwd_x'][finite['fwd_x']]) + 1.0)
                        z0[finite['fwd_x']] += fwd['fwd_x'][finite['fwd_x']]
                        f0[finite['fwd_f']] += fwd['fwd_f'][finite['fwd_f']]
                        g0[finite['fwd_g']] += fwd['fwd_g'][finite['fwd_g']]
                        lam_z0[finite['fwd_lam_x']] += fwd['fwd_lam_x'][finite['fwd_lam_x']]
                        lam_g0[finite['fwd_lam_g']] += fwd['fwd_lam_g'][finite['fwd_lam_g']]
                        lam_p0[finite['fwd_lam_p']] += fwd['fwd_lam_p'][finite['fwd_lam_p']]
                        self.logger.info('Plan updated based on sensitivities. e_infs = ' + str(e_infs))
                    except:
                        self.logger.error(
                            'Error computing forward sensitivities. Plan will not be updated based on sensitivities')

                    # update optimiser solution with the corrected NLP primal and dual variables
                    self.optimiser.update(z0=z0, lam_z0=lam_z0, lam_g0=lam_g0, lam_p0=lam_p0, p0=p, f0=f0, g0=g0)
            except Exception as e:
                self.logger.error(str(e))

            # get optimal control at the current position
            u = self.optimiser.control

            self.optimiser.shrink()
        else:
            # get optimal control at the current position
            u = self.optimiser.control

        return {'u': u, 'replan': replan}


class INMPC(Guidance):
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)
        self.solvers = kwargs.get('solvers', ['sqpmethod'])

    def react(self, t0, x0, **kwargs):
        # update parameters from estimation, if provided
        self.p0 = kwargs.get('p0', self.p0)
        replan = False

        if self.optimiser.N == self.optimiser.N0:
            # get optimal control
            u = self.optimiser.control
            self.optimiser.shrink()
        elif t0 == self.optimiser.time and self.optimiser.N != 2:
            # set current state
            # x0 is the current state! OBSERVED, not estimated!
            # in INMPC it is assumed that the plan is computed instantaneously when receiving a new measurement
            self.optimiser.setX0(x0=x0)

            # p0 is the OBSERVED parameter when provided by the estimator
            self.optimiser.p0 = vertcat(x0, self.p0)

            try:
                # solve open-loop OCP on the shrinked horizon
                for solver in self.solvers:
                    solver_stats = self.optimiser.solve(silent=True, solver=solver)

                if solver_stats['success']:
                    self.saveStats(solver_stats)
                    replan = True

                self.logger.info('Re-plan exit code ' + str(solver_stats['success']))
            except Exception as e:
                self.logger.error(str(e))

            # get optimal control
            u = self.optimiser.control

            # shrink horizon
            self.optimiser.shrink()
        else:
            # get optimal control
            u = self.optimiser.control

        return {'u': u, 'replan': replan}


class SbNMPC2(Guidance, opt.Optimiser):
    # TODO
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)

    def solve(self):
        pass

    def react(self, t0, x0, **kwargs):
        # get estimated parameters
        self.p0 = kwargs.get('p0', self.p0)
        replan = False

        if self.optimiser.N == self.optimiser.N0:
            # get optimal control at the current position
            u = self.optimiser.control

            # shrink horizon
            self.optimiser.shrink()
        elif t0 == self.optimiser.time and self.optimiser.N != 2:
            # x0 is the planned state at the current point
            # set planned state for the parameters! not the estimated, not the real!
            # when setting x0, the cost function and constraints on the shrinked problem
            # are automatically computed by the optimiser, therefore, f and g are evaluated at
            # the shrinked horizon!
            self.optimiser.setX0(x0=None)
            self.optimiser.createSolver(solver='sqpmethod')

            p = vertcat(x0, self.p0)

            # get deviation with respect to what was expected
            deltap = p - self.optimiser.p0

            # get sensitivities and update NLP primal and dual variables
            fwd = self.optimiser.getForward2(fwd_p=deltap, nfwd=1)
            finite = dict.fromkeys(fwd.keys())
            replan = True

            for key, value in fwd.iteritems():
                fwd[key] = np.array(value)
                finite[key] = np.where(np.isfinite(fwd[key]))[0]

            z0 = self.optimiser.z0
            f0 = self.optimiser.f0
            g0 = self.optimiser.g0
            lam_z0 = self.optimiser.lam_z0
            lam_g0 = self.optimiser.lam_g0
            lam_p0 = self.optimiser.lam_p0

            try:
                z0[finite['fwd_x']] += fwd['fwd_x'][finite['fwd_x']]
                f0[finite['fwd_f']] += fwd['fwd_f'][finite['fwd_f']]
                g0[finite['fwd_g']] += fwd['fwd_g'][finite['fwd_g']]
                lam_z0[finite['fwd_lam_x']] += fwd['fwd_lam_x'][finite['fwd_lam_x']]
                lam_g0[finite['fwd_lam_g']] += fwd['fwd_lam_g'][finite['fwd_lam_g']]
                lam_p0[finite['fwd_lam_p']] += fwd['fwd_lam_p'][finite['fwd_lam_p']]

                # update optimiser solution with the corrected NLP primal and dual variables
                self.optimiser.update(z0=z0, lam_z0=lam_z0, lam_g0=lam_g0, lam_p0=lam_p0, p0=p, f0=f0, g0=g0)

                self.logger.info('Plan updated based on sensitivities')
            except:
                self.logger.error(
                    'Error computing forward sensitivities. Plan will not be updated based on sensitivities')

            # get optimal control
            u = self.optimiser.control

            # shrink horizon
            self.optimiser.shrink()
        else:
            # get optimal control
            u = self.optimiser.control

        return {'u': u, 'replan': replan}


class AsNMPC2(Guidance, opt.Optimiser):
    # TODO
    def __init__(self, **kwargs):
        Guidance.__init__(self, **kwargs)
        self.solvers = kwargs.get('solvers', ['sqpmethod'])

    def solve(self):
        pass

    def react(self, t0, x0, **kwargs):
        replan = False
        self.p0 = kwargs.get('p0', self.p0)
        xEst = kwargs.get('xEst', None)

        # compute forward sensitivities of the NLP solution
        # -----------------------------------------------------------------------------------------------------
        if self.optimiser.N == self.optimiser.N0:
            # get optimal control at the current position
            u = self.optimiser.control

            # shrink horizon
            self.optimiser.shrink()
        elif t0 == self.optimiser.time and self.optimiser.N != 2:
            # x0 is the planned state at the current point.
            # should be changed by the ESTIMATED state at the current point
            # this is equivalent to start the optimisation of the plan in the previous point
            # and the uodate the sensitivities on the observed state
            # x0 is the ESTIMATED state obtained from a MHE
            # in AsNMPC it is assumed that the plan is computed instantaneously when receiving a new measurement
            # when the new observation is available, the plan is updated based on sensitivities
            self.optimiser.setX0(x0=xEst)

            # re-plan using estimated parameters at the current point
            try:
                for solver in self.solvers:
                    solver_stats = self.optimiser.solve(solver=solver, silent=True)

                self.logger.info('Re-plan exit code ' + str(solver_stats['success']))

                # if the estimation was successful, update solution with the observed parameters
                if solver_stats['success']:
                    z0_cur = copy.copy(self.optimiser.z0)
                    f0_cur = copy.copy(self.optimiser.f0)
                    g0_cur = copy.copy(self.optimiser.g0)
                    lam_g0_cur = copy.copy(self.optimiser.lam_g0)
                    lam_z0_cur = copy.copy(self.optimiser.lam_z0)

                    # here x0 is the OBSERVED state
                    # p0 is the OBSERVED parameter.
                    p = vertcat(x0, self.p0)

                    # compute jacobian and hessian of the KKT system at the current solution with xest!,
                    # with respect to parameters but also with respect to the current state !
                    fun = self.optimiser.sensitivity()

                    # define optimal deviation variables
                    deltaz = SX.sym('delta', self.optimiser.z.sparsity())

                    arg = {}

                    # bounds on g keep identical
                    arg["lbg"] = self.optimiser.lbg
                    arg["ubg"] = self.optimiser.ubg

                    # initially, assume no perturbations
                    arg["x0"] = GenDM_zeros(deltaz.sparsity())

                    tau_opt = 5.0e-4
                    tau_infs = 1.0e-10

                    e_opt = 2 * tau_opt
                    e_infs = 2 * tau_infs
                    n = 1
                    Nmax = 15

                    update = False
                    while (e_opt > tau_opt or e_infs > tau_infs) and n < Nmax:
                        # calculate difference on parameters
                        deltap = p - self.optimiser.p0

                        # Get jacobian and hessian at the referece optimal solution to compute optimal sensitivities
                        nlp_hess_l = fun['nlp_hess_l'](z=self.optimiser.z0, p=self.optimiser.p0,
                                                       lam_g=self.optimiser.lam_g0)
                        nlp_jac_f = fun['nlp_jac_f'](z=self.optimiser.z0, p=self.optimiser.p0)
                        nlp_jac_g = fun['nlp_jac_g'](z=self.optimiser.z0, p=self.optimiser.p0)

                        # define quadratic cost function
                        f = 0.5 * mtimes(transpose(deltaz), mtimes(nlp_hess_l['L_zz'], deltaz)) + mtimes(
                            transpose(deltap), mtimes(
                                transpose(nlp_hess_l['L_zp']), deltaz)) + mtimes(nlp_jac_f['f_z'], deltaz)

                        # linearise also bounds
                        # g_l <= g_0 +  g_x \Delta g + g_p \Delta p <= g_u
                        # lineare constraints to fulfill them up to first order
                        g = nlp_jac_g["g"] + mtimes(nlp_jac_g["g_z"], deltaz) + mtimes(nlp_jac_g["g_p"], deltap)

                        # define quatratic programming  problem
                        qp = {'x': deltaz, 'f': f, 'g': g}

                        with opt.suppress_stdout_stderr():
                            # solver = qpsol("solver", 'qpoases', qp)
                            solver = nlpsol("solver", 'snopt', qp)

                        # -----------------------------------------------------------------------------------------

                        # linearise also bounds
                        # x_l <= x_0 +  1 \Delta x + 0 \Delta p <= x_u
                        # x_l - x_0 <= \Delta x
                        # \Delta x <= x_u - x_0
                        arg["lbx"] = self.optimiser.lbz - self.optimiser.z0
                        arg["ubx"] = self.optimiser.ubz - self.optimiser.z0

                        arg['lam_x0'] = self.optimiser.lam_z0
                        arg['lam_g0'] = self.optimiser.lam_g0

                        with opt.suppress_stdout_stderr():
                            res = solver(**arg)

                        solver_stats = solver.stats()

                        self.logger.info('Optimal sensitivity-based QP solution computed with exit code ' + str(
                            solver_stats['success']) + ' at q ' + str(n))

                        if solver_stats['success']:
                            update = True

                            # update solution with the resulting trajectory
                            # evaluate constraints and cost function at the new solution
                            z0 = res['x'] + self.optimiser.z0
                            f0 = fun['nlp_jac_f'](z=z0, p=p)['f']
                            g0 = fun['nlp_jac_g'](z=z0, p=p)['g']
                            lam_g0 = res['lam_g']
                            lam_z0 = res['lam_x']
                            self.optimiser.update(z0=z0, f0=f0, g0=g0, lam_z0=lam_z0, lam_g0=lam_g0, p0=p)

                            # optimallity check
                            # evaluate optimallity error
                            L_z = fun['nlp_jac_l'](z=z0, p=p, lam_g=lam_g0)['L_z'].T + lam_z0
                            g = g0

                            L_z_norm = norm_inf(L_z)
                            g_max = mmax(g)

                            e_opt = L_z_norm / (norm_2(lam_g0) + 1.0)
                            e_infs = g_max / (norm_2(z0) + 1.0)

                            self.logger.info('Error in the Lagrange sensitivity ' + str(e_opt) + ' ( ' + str(
                                L_z_norm) + ' ) <> ' + str(tau_opt) + ' at q ' + str(n))
                            self.logger.info('Non-linear constraint infeasibility ' + str(e_infs) + ' ( ' + str(
                                g_max) + ' ) <> ' + str(tau_infs) + ' at q ' + str(n))

                            n += 1
                        else:
                            update = False
                            break

                    if n == Nmax:
                        update = False

                    if update == False:
                        self.logger.warning(
                            'Plan will not be updated based on sensitivities this time step. '
                            'The trajectory will be recomputed')
                        self.optimiser.update(z0=z0_cur, f0=f0_cur, g0=g0_cur, lam_z0=lam_z0_cur, lam_g0=lam_g0_cur,
                                              p0=p)

                        try:
                            # solve open-loop OCP on the shrinked horizon
                            for solver in self.solvers:
                                solver_stats = self.optimiser.solve(silent=True, solver=solver)

                            if solver_stats['success']:
                                replan = True

                            self.logger.info('Re-plan exit code ' + str(solver_stats['success']))
                        except Exception as e:
                            self.logger.error(str(e))
                    else:
                        replan = True

                    if replan:
                        self.saveStats(solver_stats)

            except Exception as e:
                self.logger.error(str(e))

            # get optimal control at the current position
            u = self.optimiser.control

            self.optimiser.shrink()
        else:
            # get optimal control at the current position
            u = self.optimiser.control

        return {'u': u, 'replan': replan}


guidances = dict(OpenLoop=OpenLoop, Strategic=Strategic, INMPC=INMPC, AsNMPC=AsNMPC, SbNMPC=SbNMPC, SbNMPC2=SbNMPC2,
                 AsNMPC2=AsNMPC2, Tactical=Tactical)
