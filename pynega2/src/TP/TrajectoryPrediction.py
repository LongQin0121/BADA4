# -*- coding: utf-8 -*-

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"

import pyBada3.atmosphere as atm
import pyBada3.constants as const
import AircraftIntents as ACI
import pandas as pd
from fileOperations import *
from utilities_TP import *
import Phase as Ph
import Route as rt


# This class includes all the attributes and methods related with the prediction of the trajectory
class TrajectoryPrediction:

    # Initialize trajectory prediction variables, which will be overwritten during the code execution
    def __init__(self, default_aircraft, weather_route, ISA=True):
        # TODO: check which parameters should be moved to other Classes
        #  (like for instance temperatures and pressures to atmosphere?)
        # That would involve changing many pyBada classes
        self.aircraft = perf.bada4(default_aircraft)  # Aircraft object (BADA)
        self.delta = 0.0  # Normalized pressure
        self.theta = 0.0  # Normalized temperature
        self.sigma = 0.0  # Normalized air density
        self.Mach = 0.0  # Mach number
        self.v_CAS = 0.0  # Calibrated airspeed
        self.h_p = 0.0  # Pressure altitude
        self.x = 0  # State vector
        self.x_sym = 0  # State vector (symbolic)
        self.u = 0  # Control vector
        self.u_sym = 0  # Control vector (symbolic)
        self.s = 0.0  # Along track distance
        self.v = 0.0  # True airspeed
        self.h = 0.0  # Geometric altitude
        self.m = 0.0  # Mass
        self.sdot = 0.0  # Derivative of distance (ground speed)
        self.k_Mach = 0.0  # Energy share factor in constant Mach
        self.k_CAS = 0.0  # Energy share factor in constant CAS
        self.k_ESF = 0.0  # Energy share factor for acceleration and deceleration
        self.k_CAS_h = 0.0  # Intent given as a derivative of CAS w.r.t. to altitude
        self.speedBrakes = 0.0  # Speedbrakes deflection
        self.VS = 0.0  # Vertical speed
        self.gamma_a = 0.0  # Aerodynamic flight path angle
        self.h_dot = 0.0  # Derivative of the geometric altitude
        self.THR = 0.0  # Throttle
        self.T = 0.0  # Thrust
        self.Idle_Thrust = 0.0  # Idle thrust
        self.Max_Thrust = 0.0  # Maximum thrust
        self.CL = 0.0  # Lift coefficient
        self.CD = 0.0  # Drag coefficient
        self.L = 0.0  # Lift force
        self.D = 0.0  # Drag force
        self.LoadFactor = 0.0  # Load factor
        self.integration_type = "backwards"  # Integration direction: backwards or forwards
        self.integration_sign = -1  # Integration sign: -1 for backwards and 1 for forwards
        self.integration_current_time = 0.0  # Current time of the integration
        self.condition = 0.0  # Current value for the condition (depends on the phase) --> to be compared with EC
        self.end_condition_parameter = dict()  # Dictionary to know which parameter corresponds to the end condition
        self.throttle_f = self.throttle_thrust_mode  # Name of the function to return the throttle
        self.condition_f = self.end_condition_bigger  # Function to check if the condition has reached EC of phase
        self.T_intent = 0  # It indicates if one of the intents is throttle
        self.T_factor = False  # It indicates the factor by which the total thrust is multiplied
        self.rating = "MCMB"  # Current engine rating

        self.weather_route = weather_route  # Current route (with WeatherWaypoints in it)
        self.chig = 0  # Ground track
        self.chia = 0  # Aerodynamic track
        self.waypoint = []  # Waypoint, containing id, latitude, longitude and constraints
        self.weather_waypoint = rt.WeatherWaypoint()  # Weather waypoint: id, lat, lon, constraints and weather
        self.tau = 0  # Current temperature
        self.p = 0  # Current pressure
        self.tau_h = 0
        self.tau_s = 0
        self.p_h = 0
        self.p_s = 0
        self.Atmosphere = atm.Atmosphere(ISA)  # Instance of the Atmosphere Class in pyBada.atmosphere

        # Maybe we need to move these parameters to a winds class?
        self.ws = 0  # Along track wind
        self.wx = 0  # Cross wind
        self.wn = 0  # North wind
        self.we = 0  # East wind
        self.wn_h = 0  # North wind derivative with respect to altitude
        self.we_h = 0  # East wind derivative with respect to altitude
        self.wn_s = 0  # North wind derivative with respect to distance
        self.we_s = 0  # East wind derivative with respect to distance
        self.wn_dot = 0
        self.we_dot = 0
        self.wv_dot = 0

    # Initialize Aircraft Intents
    def initialize_AircraftIntents(self):

        # Initialize aircraft (by reading the corresponding BADA performance file)
        self.ACIG = ACI.AircraftIntents()
        self.ACIG.get_aircraft(self.aircraft)

        self.ACIMachT = ACI.MachT(self.aircraft)
        self.ACICAST = ACI.CAST(self.aircraft)
        self.ACIESFT = ACI.ESFT(self.aircraft)
        self.ACIVST = ACI.VST(self.aircraft)
        self.ACIFPAT = ACI.FPAT(self.aircraft)
        self.ACIALTT = ACI.ALTT(self.aircraft)
        self.ACIMachVS = ACI.MachVS(self.aircraft)
        self.ACICASVS = ACI.CASVS(self.aircraft)
        self.ACIESFVS = ACI.ESFVS(self.aircraft)
        self.ACIMachFPA = ACI.MachFPA(self.aircraft)
        self.ACICASFPA = ACI.CASFPA(self.aircraft)
        self.ACIESFFPA = ACI.ESFFPA(self.aircraft)
        self.ACIALTMACH = ACI.ALTMach(self.aircraft)
        self.ACIESFALT = ACI.ESFALT(self.aircraft)
        self.ACICAShFPA = ACI.CAShFPA(self.aircraft)
        self.ACICAShT = ACI.CAShT(self.aircraft)

        self.ACI_f = self.ACIFPAT  # Current set of intents, which change depending on the phase

    #################################################################################################################
    ## INITIALIZATION AND READING OPERATIONS: BADA INPUTS, CONDITIONS, AIRCRAFT INTENTS, PHASE INFO AND FIRST STATE##
    #################################################################################################################
    # Function to initialize parameters.
    # @profile: the profile loaded from XML.
    # TODO: add an error if the parameter is not part of the initial condition! --> stop execution
    def initialConditions(self, profile):
        # Initialize for the current example.
        params = {}
        for block in profile.get('blocks'):
            for param in profile.get('blocks')[block]['initialConditions']:
                params[param[0]] = float(param[1])

        CAS = None
        for param, value in params.items():
            if param == 'm':
                self.m = float(value)
            elif param == 's':
                self.s = float(value)
            elif param == 'h':
                self.h = float(value)
            elif param == 'TAS':
                self.v = float(value)
            elif param == 'CAS':
                CAS = float(value)
            # TODO: QNH
            elif param == 'alt_press':
                self.h_p = float(value)
                if self.Atmosphere.ISA:
                    self.h = self.h_p
                else:
                    # If the pressure is greater than the highest pressure h is equal to hp
                    delta_local = float(self.Atmosphere.delta(self.h_p))  # Obtain delta for a given pressure altitude
                    if (delta_local * const.p_0) > self.weather_route.h_spline.tck[0][-1]:
                        self.h = self.h_p
                    else:
                        self.h = self.weather_route.h_spline(delta_local * const.p_0, self.s)[
                            0]  # Obtain geometric altitude from spline

        self.getTauPressWnWe()  # Get weather for initial conditions

        if CAS is not None:
            theta = self.Atmosphere.theta(self.h, tau_real=self.tau)
            delta = self.Atmosphere.delta(self.h, p_real=self.p)
            sigma = delta / theta
            self.v = float(self.Atmosphere.cas2Tas(CAS, sigma, delta))

    # Depending on the phase intents, identify the corresponding functions to be used to compute the controls
    def findAircraftIntents(self):

        # TODO: error from IPA Class to say that the pair of intents is not considered
        if 'MACH' in self.current_phase.intent_types.keys() and 'THR' in self.current_phase.intent_types.keys():
            self.ACI_f = self.ACIMachT
            self.T_intent = 1
            self.THR = self.current_phase.intent_values["THR"]
            self.throttle_f = self.throttle_thrust_mode
        elif 'CAS' in self.current_phase.intent_types.keys() and 'THR' in self.current_phase.intent_types.keys():
            self.ACI_f = self.ACICAST
            self.T_intent = 1
            self.THR = self.current_phase.intent_values["THR"]
            self.throttle_f = self.throttle_thrust_mode
        elif 'ESF' in self.current_phase.intent_types.keys() and 'THR' in self.current_phase.intent_types.keys():
            self.ACI_f = self.ACIESFT
            self.T_intent = 1
            self.k_ESF = self.current_phase.intent_values["ESF"]
            self.THR = self.current_phase.intent_values["THR"]
            self.throttle_f = self.throttle_thrust_mode
        elif 'VS' in self.current_phase.intent_types.keys() and 'THR' in self.current_phase.intent_types.keys():
            self.VS = self.current_phase.intent_values["VS"]
            self.T_intent = 1
            self.ACI_f = self.ACIVST
            self.THR = self.current_phase.intent_values["THR"]
            self.throttle_f = self.throttle_thrust_mode
        elif 'FPA' in self.current_phase.intent_types.keys() and 'THR' in self.current_phase.intent_types.keys():
            self.gamma_a = self.current_phase.intent_values["FPA"]
            self.T_intent = 1
            self.ACI_f = self.ACIFPAT
            self.THR = self.current_phase.intent_values["THR"]
            self.throttle_f = self.throttle_thrust_mode
        elif ('alt' in self.current_phase.intent_types.keys() or 'alt_press' in self.current_phase.intent_types.keys()) \
                and 'THR' in self.current_phase.intent_types.keys():
            self.ACI_f = self.ACIALTT
            self.T_intent = 1
            self.THR = self.current_phase.intent_values["THR"]
            self.throttle_f = self.throttle_thrust_mode
        elif 'MACH' in self.current_phase.intent_types.keys() and 'VS' in self.current_phase.intent_types.keys():
            self.VS = self.current_phase.intent_values["VS"]
            self.T_intent = 0
            self.ACI_f = self.ACIMachVS
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'CAS' in self.current_phase.intent_types.keys() and 'VS' in self.current_phase.intent_types.keys():
            self.VS = self.current_phase.intent_values["VS"]
            self.T_intent = 0
            self.ACI_f = self.ACICASVS
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'ESF' in self.current_phase.intent_types.keys() and 'VS' in self.current_phase.intent_types.keys():
            self.VS = self.current_phase.intent_values["VS"]
            self.T_intent = 0
            self.ACI_f = self.ACIESFVS
            self.k_ESF = self.current_phase.intent_values["ESF"]
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'MACH' in self.current_phase.intent_types.keys() and 'FPA' in self.current_phase.intent_types.keys():
            self.gamma_a = self.current_phase.intent_values["FPA"]
            self.ACI_f = self.ACIMachFPA
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'CAS' in self.current_phase.intent_types.keys() and 'FPA' in self.current_phase.intent_types.keys():
            self.gamma_a = self.current_phase.intent_values["FPA"]
            self.ACI_f = self.ACICASFPA
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'ESF' in self.current_phase.intent_types.keys() and 'FPA' in self.current_phase.intent_types.keys():
            self.gamma_a = self.current_phase.intent_values["FPA"]
            self.T_intent = 0
            self.ACI_f = self.ACIESFFPA
            self.k_ESF = self.current_phase.intent_values["ESF"]
            self.throttle_f = self.throttle_no_thrust_mode
        elif ('alt' in self.current_phase.intent_types.keys() or 'alt_press' in self.current_phase.intent_types.keys()) \
                and 'MACH' in self.current_phase.intent_types.keys():
            self.ACI_f = self.ACIALTMACH
            self.T_intent = 0
            self.throttle_f = self.throttle_no_thrust_mode
        elif ('alt' in self.current_phase.intent_types.keys() or 'alt_press' in self.current_phase.intent_types.keys()) \
                and 'CAS' in self.current_phase.intent_types.keys():  # Same mode as ALT-MACH
            self.ACI_f = self.ACIALTMACH
            self.T_intent = 0
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'ACC_CAS_t' in self.current_phase.intent_types.keys() and (
                ('FPA' in self.current_phase.intent_types.keys() and self.current_phase.intent_values["FPA"] == 0) or
                ('alt' in self.current_phase.intent_types.keys() or 'alt_press' in self.current_phase.intent_types.keys())):
            self.gamma_a = 0.0
            self.ACI_f = self.ACIESFALT
            self.k_ESF = self.current_phase.intent_values["ACC_CAS_t"]
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'ACC_CAS_h' in self.current_phase.intent_types.keys() and 'FPA' in self.current_phase.intent_types.keys():
            self.gamma_a = self.current_phase.intent_values["FPA"]
            self.T_intent = 0
            self.ACI_f = self.ACICAShFPA
            self.k_CAS_h = self.current_phase.intent_values["ACC_CAS_h"]
            self.throttle_f = self.throttle_no_thrust_mode
        elif 'ACC_CAS_h' in self.current_phase.intent_types.keys() and 'THR' in self.current_phase.intent_types.keys():
            self.gamma_a = self.current_phase.intent_values["THR"]
            self.T_intent = 1
            self.ACI_f = self.ACICAShT
            self.k_CAS_h = self.current_phase.intent_values["ACC_CAS_h"]
            self.throttle_f = self.throttle_thrust_mode
        else:
            sys.exit(
                "ERROR: The combination of intents of phase %s is not modeled in this version of pyBada TP, "
                "stopping execution..." % self.current_phase.id)

    # Compute the first step of the profile and obtain the corresponding controls and auxiliary variables
    def compute_first_state(self):

        self.speedBrakes = 0
        self.THR = 0
        self.gamma_a = conv.deg2rad(-3)  # Initial guess for the gamma (i.e. flight path angle) refinement algorithm
        self.k_ESF = 0
        self.VS = 0

        self.findAircraftIntents()

        # Run gamma refinement algorithm to find right controls (Thrust and gamma)
        self.gamma_refinement()
        self.compute_auxiliary_variables_control_dependent()
        if self.T_intent != 1:
            self.Idle_Thrust = self.aircraft.TMin(delta=self.delta, theta=self.theta, M=self.Mach, h=self.h)
            self.Max_Thrust = self.aircraft.TMax(delta=self.delta, theta=self.theta, M=self.Mach, h=self.h,
                                                 rating=self.rating)
        elif self.THR == 0:
            self.Max_Thrust = self.aircraft.TMax(delta=self.delta, theta=self.theta, M=self.Mach, h=self.h,
                                                 rating=self.rating)
        self.throttle_f()

    #################################################
    ## END CONDITION CHECKS AND THROTTLE FUNCTIONS ##
    #################################################
    # Check if condition is higher or lower than the end condition
    def end_condition_check(self):

        if self.current_phase.endConditionValue > self.condition:
            self.condition_f = self.end_condition_bigger
        else:
            self.condition_f = self.end_condition_smaller

    # Check difference in conditions for end conditions higher than conditions
    def end_condition_bigger(self):
        diff_cond = self.current_phase.endConditionValue - self.condition
        return diff_cond

    # Check difference in conditions for end conditions lower than conditions
    def end_condition_smaller(self):
        diff_cond = self.condition - self.current_phase.endConditionValue
        return diff_cond

    # Update the condition value depending on the end condition type of the phase
    def update_condition(self):
        self.end_condition_parameter = {
            "MACH": self.Mach, "CAS": self.v_CAS, "s": self.s, "Delta_s": self.integration_direction_sign * self.s,
            "alt_press": self.h_p, "alt": self.h
        }
        self.condition = self.end_condition_parameter[self.current_phase.endConditionType]

    # Check if condition is converging towards the end condition;
    # stop the execution when this happens during "error_iterations" times
    # @diff_cond: current difference between condition and end condition
    # @prev_diff_cond: previous difference between condition and end condition
    # @current_errors: number of times the difference between condition and end condition has become bigger
    # @error_iterations: it defines the number of times that it is allowed to have a
    # diff_cond bigger than the prev_diff_cond
    def check_convergence(self, diff_cond, prev_diff_cond, current_errors, error_iterations):
        if diff_cond >= prev_diff_cond:
            current_errors += 1
            if current_errors == error_iterations:
                sys.exit("ERROR, End condition of phase %s is not met, stopping execution..." % self.current_phase.id)
        return current_errors

    # Check the integration direction and return the sign correspondingly
    # (used when calling the integrator --> the ode sign changes)
    def check_integration_direction(self):

        if self.integration_type == "backwards":
            self.integration_direction_sign = -1
        else:
            self.integration_direction_sign = 1

    # Do nothing is throttle is given as an intent
    def throttle_thrust_mode(self):
        return -1

    # Return throttle if throttle is not given as an intent
    def throttle_no_thrust_mode(self):
        self.THR = (self.T - self.Idle_Thrust) / (self.Max_Thrust - self.Idle_Thrust)

    ############################################################
    ## COMPUTATION OF STATES, CONTROLS AND AUXILIARY VARIABLES##
    ############################################################
    # Gamma refinement algorithm --> find gamma for the next iteration -->
    # initial gamma is the current value of gamma_g, which varies
    def gamma_refinement(self):

        gamma_tolerance = conv.deg2rad(0.05)
        gamma_error = 10000
        while gamma_error > gamma_tolerance:
            self.compute_auxiliary_variables_control_dependent()
            self.ACI_f.T(self)
            gamma_local = self.ACI_f.gamma(self)
            gamma_error = abs(gamma_local - self.gamma_a)
            self.gamma_a = gamma_local

    # Build ODE
    def buildODE(self):

        # Initialise symbolic State Vector u = [gamma , T]
        self.x_sym = struct_symSX(["s", "v", "h", "m"])
        s, v, h, m = self.x_sym[...]
        self.x_sym = vertcat(s, v, h, m)

        # Initialise symbolic Control Vector u = [gamma , T]
        self.u_sym = struct_symSX(["gamma", "T"])
        gamma, T = self.u_sym[...]
        self.u_sym = vertcat(gamma, T)

        # Compute derivatives
        FF = self.aircraft.ff(delta=self.delta, theta=self.theta, M=self.Mach, T=self.T, v=self.v, h=self.h)
        vdot = ((self.T - self.D) / m) - (const.g * sin(self.gamma_a))  # acceleration
        hdot = v * sin(self.gamma_a)  # vertical speed
        sdot = v * sqrt(pow(cos(self.gamma_a), 2) - pow(self.wx / v, 2)) + self.ws  # ground speed
        mdot = - FF  # fuel flow
        xdot = vertcat(sdot, vdot, hdot, mdot)
        ode = {'x': self.x_sym, 'p': self.u_sym, 'ode': self.integration_direction_sign * xdot}

        return ode

    # Update state vector
    def update_state_vector(self):

        # Compute controls: Thrust and Flight Path Angle (gamma)
        # TODO: BE CAREFUL!!! Maybe these two lines are not needed here?
        # self.ACI_f.T(self)
        # self.gamma_a = self.ACI_f.gamma(self)

        # State vector and control vector
        self.x = DM([self.s, self.v, self.h, self.m])
        self.u = [float(self.gamma_a), float(self.T)]

        # " -- ODE prediction -- solving a system of ODEs (controls given)"
        ode = self.buildODE()
        I = integrator('I', 'cvodes', ode, {'tf': self.current_phase.dt})
        res = I(p=self.u, x0=self.x)

        # STATE vector update
        self.x = res['xf']
        self.s = float(self.x[0])
        self.v = float(self.x[1])
        self.h = float(self.x[2])
        self.m = float(self.x[3])
        self.x = DM([self.s, self.v, self.h, self.m])  # State vector

    # Compute the auxiliary variables depending on the controls
    def compute_auxiliary_variables_control_dependent(self):
        # WARNING: turn dynamics are not modeled in this version of pyBada TP,
        # so the bank angle contribution to the Lift is not considered
        # self.h_dot      = self.Atmosphere.h_dot(self.v, self.gamma_a)
        self.sdot = self.v * sqrt(pow(cos(self.gamma_a), 2) - pow(self.wx / self.v, 2)) + self.ws
        if "VS" in self.current_phase.intent_types.keys():
            self.h_dot = self.Atmosphere.hp_dot2h_dot(self.VS, self.sdot, self.delta * const.p_0, self.p_h, self.p_s)
        else:
            self.h_dot = self.Atmosphere.h_dot(self.v, self.gamma_a)
            self.VS = self.Atmosphere.hp_dot(self.delta, self.h_dot, self.sdot, self.p_h, self.p_s)
        self.getWdot()  # Compute along track and cross track wind
        self.L = self.m * const.g * cos(self.gamma_a)
        self.LoadFactor = self.L / (self.m * const.g)
        self.CL = self.aircraft.CL(self.v, self.sigma, self.m, nz=self.LoadFactor)
        self.CD = self.aircraft.CD(C_L=self.CL, M=self.Mach, HLid=str(self.current_phase.config),
                                   LG=self.current_phase.gear)
        self.D = self.aircraft.D(self.v, self.sigma, self.CD)

    # Compute auxiliary variables depending on the states
    def compute_auxiliary_variables_state_dependent(self):

        self.getTauPressWnWe()  # Obtain temperature, pressure, north wind, east wind and chig
        self.getWsWx()  # Compute along track and cross track wind
        self.delta = self.Atmosphere.delta(self.h, p_real=self.p)  # Normalized pressure (updated at each time step)
        self.theta = self.Atmosphere.theta(self.h,
                                           tau_real=self.tau)  # Normalized temperature (updated at each time step)
        self.sigma = self.delta / self.theta  # Density (updated at each time step)
        self.Mach = self.Atmosphere.tas2Mach(self.v, self.theta)  # Mach
        self.v_CAS = self.Atmosphere.tas2Cas(self.v, self.theta, self.delta)  # Calibrated airspeed from True Airspeed
        # TODO: QNH!
        if self.Atmosphere.ISA:
            self.h_p = self.h
        else:
            # If the pressure is greater than the highest pressure h is equal to hp
            if (self.delta * const.p_0) > self.weather_route.h_spline.tck[0][-1]:
                self.h_p = self.h
            else:
                self.h_p = self.Atmosphere.hp(self.delta)  # Obtain pressure altitude

        # If throttle is 0 or one of the intents was not thrust we need to compute Idle thrust and maximum thrust values
        if self.T_intent != 1:
            self.Idle_Thrust = self.aircraft.TMin(delta=self.delta, theta=self.theta, M=self.Mach, h=self.h)
            self.Max_Thrust = self.aircraft.TMax(delta=self.delta, theta=self.theta, M=self.Mach, h=self.h,
                                                 rating=self.rating)
        elif self.THR == 0:
            self.Max_Thrust = self.aircraft.TMax(delta=self.delta, theta=self.theta, M=self.Mach, h=self.h,
                                                 rating=self.rating)

    # Obtain Temperature, Pressure, track and north and east wind for a given s value and altitude
    def getTauPressWnWe(self):
        if self.weather_route.waypoint_list:
            self.waypoint, self.chig = self.weather_route.getWaypointandTrackfromDistance(
                self.s)  # Get waypoint from the original route+track
            if self.weather_route.weatherpoint_list:
                self.weather_waypoint = self.weather_route.getWeatherWaypointfromDistance(
                    self.s)  # Get weatherwaypoint (it could be the same as waypoint)
                self.tau, self.p, self.tau_h, self.tau_s, self.tau_n, \
                    self.tau_e, self.p_h, self.p_s, self.p_n, self.p_e = \
                    self.Atmosphere.updateTauP(self.h, self.s, self.chig, self.weather_route.tau_spline,
                                               self.weather_route.p_spline)
                self.wn, self.we, self.wn_h, self.wn_s, self.we_h, self.we_s = \
                    self.Atmosphere.updateWind(self.h, self.s, self.weather_route.wn_spline,
                                               self.weather_route.we_spline)
                # Get temperature, pressure and wind from weather waypoint
                # self.tau, self.wn, self.we, self.p  = self.weather_waypoint.getWeatherdatafromalt(self.h)
            else:  # If ISA and existing route
                self.tau, self.p, self.tau_h, self.tau_s, self.tau_n, \
                    self.tau_e, self.p_h, self.p_s, self.p_n, self.p_e = \
                    self.Atmosphere.updateTauP(self.h)
        else:  # If ISA and no route
            self.tau, self.p, self.tau_h, self.tau_s, self.tau_n, \
                self.tau_e, self.p_h, self.p_s, self.p_n, self.p_e = \
                self.Atmosphere.updateTauP(self.h)

    # Compute along track and cross track wind
    def getWsWx(self):
        self.ws = self.wn * cos(self.chig) + self.we * sin(self.chig)
        self.wx = -self.wn * sin(self.chig) + self.we * cos(self.chig)
        wx_bar = self.wx / self.v
        self.chia = self.chig - asin(wx_bar / cos(self.gamma_a))

    def getWdot(self):
        self.wn_dot = self.wn_h * self.h_dot + self.wn_s * self.sdot
        self.we_dot = self.we_h * self.h_dot + self.we_s * self.sdot
        self.wv_dot = (self.wn_dot * cos(self.chia) + self.we_dot * sin(self.chia)) * cos(self.gamma_a)

    #################################################
    ## INTENTS MANAGEMENT AND DISCONTINUITY CHECKS ##
    #################################################

    # Update the values of h, M and CAS and look for discontinuities
    def update_h_M_CAS_with_intents_and_check_discontinuity(self):

        if "MACH" in self.current_phase.intent_types.keys():
            if abs((self.Mach - self.current_phase.intent_values["MACH"]) /
                   max(self.Mach, self.current_phase.intent_values["MACH"])) > 0.01:
                print(
                    "Warning!! end condition of phase %s already met, discontinuity! --> "
                    "Mach of the previous step: %.4f/ Mach intent value: %.4f",
                    (self.current_phase.id, self.Mach, self.current_phase.intent_values["MACH"]))
            self.Mach = self.current_phase.intent_values["MACH"]

        elif "CAS" in self.current_phase.intent_types.keys():
            if abs((self.v_CAS - self.current_phase.intent_values["CAS"]) /
                   max(self.v_CAS, self.current_phase.intent_values["CAS"])) > 0.01:
                print(
                    "Warning!! end condition of phase %s  already met, discontinuity! --> "
                    "CAS of the previous step: %.4f/ CAS intent value: %.4f",
                    (self.current_phase.id, self.v_CAS, self.current_phase.intent_values["CAS"]))
            self.v_CAS = self.current_phase.intent_values["CAS"]

        # TODO: problems here with h_spline (slightly different value of h is obtained (0.3m)!)-->not critical
        if "alt_press" in self.current_phase.intent_types.keys():
            if self.Atmosphere.ISA:
                new_h = self.current_phase.intent_values["alt_press"]
            else:
                # If the pressure is greater than the highest pressure h is equal to hp
                if (self.delta * const.p_0) > self.weather_route.h_spline.tck[0][-1]:
                    new_h = self.current_phase.intent_values["alt_press"]
                else:
                    # Obtain geometric altitude from spline
                    new_h = self.weather_route.h_spline(self.delta * const.p_0, self.s)[0]

            if abs((self.h_p - self.current_phase.intent_values["alt_press"]) / max(self.h, new_h)) > 0.01:
                print(
                    "Warning!! end condition of phase %s  already met, discontinuity! --> "
                    "Last altitude of the previous phase: %.4f/ First altitude of the current phase: %.4f",
                    (self.current_phase.id, self.h_p, self.current_phase.intent_values["alt_press"]))
            self.h = new_h
        elif "alt" in self.current_phase.intent_types.keys():
            if abs((self.h - self.current_phase.intent_values["alt"]) /
                   max(self.h, self.current_phase.intent_values["alt"])) > 0.01:
                print(
                    "Warning!! end condition of phase %s  already met, discontinuity! --> "
                    "Last altitude of the previous phase: %.4f/ First altitude of the current phase: %.4f",
                    (self.current_phase.id, self.h, self.current_phase.intent_values["alt"]))
            self.h = self.current_phase.intent_values["alt"]

    ######################################
    ## SMOOTH TRANSITIONS BETWEEN PHASES##
    ######################################
    # Generate the inputs needed for the smoothing process (interpolation when changing phase)
    # @posix_initial_time: initial time of the integration in POSIX format
    # @final_output: list containing the output in International System
    # @endcondition_identification: identify which is the parameter matching the end condition
    def generate_inputs_smoothing(self, posix_initial_time, final_output, endcondition_identification):

        current_time = posix_initial_time + self.integration_current_time
        current_datetime = sec2time(current_time).replace(microsecond=0)
        integration_step_output = self.generate_integration_step_results(current_datetime)
        prev_integration_results = final_output[len(final_output) - 1]
        current_integration_results = integration_step_output

        var_of_interest = self.current_phase.endConditionValue
        var_of_interest_pos = endcondition_identification[self.current_phase.endConditionType]

        return prev_integration_results, current_integration_results, var_of_interest, var_of_interest_pos,

    # Smooth the transition between phases by interpolating
    # @prev: previous results
    # @current: current results
    # @var: value of the parameter used for interpolation
    # @var_pos: position of the variable in the output list
    # @starting_interp_pos: output values in the output list from this position to the end will be
    # interpolated when smoothing the transition between phases
    def smooth_phase_transition(self, prev, current, var, var_pos, starting_interp_pos):

        if self.current_phase.endConditionType == "Delta_s":
            var = -1 * var

        var_max = current[var_pos]

        for i in range(starting_interp_pos, len(prev)):
            if np.all(np.diff([prev[var_pos], current[var_pos]]) > 0):
                interp_array = interp_gen(prev[var_pos], var_max, prev[i], current[i], var)
            else:
                interp_array = interp_gen(var_max, prev[var_pos], current[i], prev[i], var)
            interp_value = interp_array[1]
            current[i] = interp_value

        self.h = current[5]
        self.v = current[6]
        self.m = current[7]
        self.s = current[4]
        self.T = current[18]
        self.gamma_a = current[17]
        self.integration_current_time = current[3]

        ####################################################

    ## GENERATE LISTS WITH RESULTS OF THE INTEGRATION ##
    ####################################################
    # Generate a list with the results for one step in International System Units (SI)
    # @current_time: current time in the integration (datetime object)
    def generate_integration_step_results(self, current_time):

        output_list = [self.current_phase.id, str(current_time.date()), str(current_time.time()),
                       float(self.integration_current_time), self.s, self.h, self.v, self.m,
                       self.gamma_a, self.THR, self.speedBrakes, self.LoadFactor, self.h_p, self.VS, self.v_CAS,
                       self.Mach, self.sdot, self.gamma_a, self.T, self.Max_Thrust,
                       self.Idle_Thrust]

        return output_list

    # Main function to generate output of the integration step and to smooth transitions between phases
    # if end condition is already met
    # @diff_cond: difference between condition and end condition
    # @posix_initial_time: initial time in POSIX format
    # @final_output: output list in international system units
    # @starting_interp_pos: output values in the output list from this position to the end will be
    # interpolated when smoothing the transition between phases
    # @endcondition_identification: identify which is the parameter matching the end condition
    def generate_output_integration_step(self, diff_cond, posix_initial_time, final_output, starting_interp_pos,
                                         endcondition_identification):

        if diff_cond < 0:
            final_output = self.generate_output_phase_final_step(
                posix_initial_time, final_output, starting_interp_pos, endcondition_identification)
        else:
            final_output = self.generate_output_intermediate_steps(posix_initial_time, final_output)

        return final_output

    # Generate results for the last integration step of the current phase (with an interpolation)
    # @posix_initial_time: initial time in POSIX format
    # @final_output: output list in international system units
    # @starting_interp_pos: output values in the output list from this position to the end will be
    # interpolated when smoothing the transition between phases
    # @endcondition_identification: identify which is the parameter matching the end condition
    def generate_output_phase_final_step(self, posix_initial_time, final_output, starting_interp_pos,
                                         endcondition_identification):

        prev_integration_results, current_integration_results, var_of_interest, var_of_interest_pos = \
            self.generate_inputs_smoothing(posix_initial_time, final_output, endcondition_identification)
        self.smooth_phase_transition(prev_integration_results, current_integration_results, var_of_interest,
                                     var_of_interest_pos, starting_interp_pos)

        # Recompute the variables depending on the state and on the control
        # (in this case we recompute Tmax and Tmin again)
        self.T_intent = 0
        self.compute_auxiliary_variables_state_dependent()
        self.compute_auxiliary_variables_control_dependent()
        self.throttle_f()

        current_time = posix_initial_time + self.integration_current_time
        current_datetime = sec2time(current_time).replace(microsecond=0)
        integration_step_output = self.generate_integration_step_results(current_datetime)

        final_output.append(integration_step_output)

        return final_output

    # Generate output intermediate steps
    # @posix_initial_time: initial time in POSIX format
    # @final_output: output list in international system units
    def generate_output_intermediate_steps(self, posix_initial_time, final_output):

        current_time = posix_initial_time + self.integration_current_time
        current_datetime = sec2time(current_time).replace(microsecond=0)
        integration_step_output = self.generate_integration_step_results(current_datetime)

        final_output.append(integration_step_output)

        return final_output

    #########################
    ## UTILITIES FUNCTIONS ##
    #########################
    # Convert delta s into s; useful to do the comparison between condition and end condition
    def delta_s2s(self):
        self.current_phase.endConditionValue = self.current_phase.endConditionValue + \
                                               self.integration_direction_sign * self.s

    # Format the output dataframe to the desired float precision
    # @df: dataframe containing the resulting trajectory
    def format_df(self, df):

        df['t[s]'] = df['t[s]'].apply(lambda x: '%11.2f' % x)
        df['s[NM]'] = df['s[NM]'].apply(lambda x: '%5.2f' % conv.m2nm(x))
        df['h[ft]'] = df['h[ft]'].apply(lambda x: '%6.1f' % conv.m2ft(x))
        df['TAS[kt]'] = df['TAS[kt]'].apply(lambda x: '%4.1f' % conv.ms2kt(x))
        df['mass[kg]'] = df['mass[kg]'].apply(lambda x: '%7.1f' % x)
        df['FPA_a[º]'] = df['FPA_a[º]'].apply(lambda x: '%3.2f' % conv.rad2deg(x))
        df['THR[%]'] = df['THR[%]'].apply(lambda x: '%4.2f' % (x * 100))
        df['SB[%]'] = df['SB[%]'].apply(lambda x: '%4.2f' % (x * 100))
        df['nz[g]'] = df['nz[g]'].apply(lambda x: '%2.3f' % x)
        df['hp[ft]'] = df['hp[ft]'].apply(lambda x: '%6.1f' % conv.m2ft(x))
        df['VS[ft/min]'] = df['VS[ft/min]'].apply(lambda x: '%5.2f' % conv.ms2ftmin(x))
        df['CAS[kt]'] = df['CAS[kt]'].apply(lambda x: '%4.1f' % conv.ms2kt(x))
        df['mach[-]'] = df['mach[-]'].apply(lambda x: '%2.4f' % x)
        df['GS[kt]'] = df['GS[kt]'].apply(lambda x: '%4.1f' % conv.ms2kt(x))
        df['FPA_g[º]'] = df['FPA_g[º]'].apply(lambda x: '%3.2f' % conv.rad2deg(x))
        df['Thrust[daN]'] = df['Thrust[daN]'].apply(lambda x: '%7.1f' % (x / 10))
        df['Tidle[daN]'] = df['Tidle[daN]'].apply(lambda x: '%7.1f' % (x / 10))
        df['Tmax[daN]'] = df['Tmax[daN]'].apply(lambda x: '%7.1f' % (x / 10))

        return df

    ########################################################################
    ## IPA interface function; if constraint of next wp not met, call IPA ##
    ########################################################################

    # TODO: finish this function and call the IPA adapter!
    # TODO: route constraints met? if so, continue, if not, define new phase, i-1 in the for loop and break
    # TODO: modify profile with the right solution (i.e. phase) and at the end print the profile to a new XML
    # TODO: remove last integration results
    # TODO: reset all parameters to the original values when we started to integrate the conflictive phase!
    # TODO: when we break, decide to remove the previous results or keep them, and then add a new phase to profile
    # TODO: we only check one constraint (speed or altitude), as they won't happen at the same time
    # TODO: think about the constraint propagator at he beginning of the execution

    # TODO: IPA Class
    # TODO: hmin (too steep path)
    # TODO: hmax (geometric segment) -- FPA phase before problematic phase and then after the problematic phase
    # TODO: Vmax (keep previous results, add phase with constant speed and idle and end condition s of next waypoint)
    #       -- then add the part of the previous phase that
    # TODO: was not integrated

    # Check if the constraints of the next waypoints are met; if not, call the IPA module (IntentProcedureAdapter)
    def check_constraints_next_waypoint(self):

        if self.weather_route.waypoint_list:
            speed_constraints, altitude_constraints = self.weather_route.getNextWaypointConstraints(self.s)

            if self.v_CAS > speed_constraints[0]:
                print("Constraint not met")
            elif self.v_CAS < speed_constraints[1]:
                print("Constraint not met")
            elif self.h > altitude_constraints[0]:
                print("Constraint not met")
            elif self.h < altitude_constraints[1]:
                print("Constraint not met")

    ##############################
    ## MAIN PREDICTION FUNCTION ##
    ##############################
    # Function to predict the trajectory.
    # @profile:             profile loaded from XML
    # @pathFileCSV:         path to the output csv file
    # @ICAO:                aircraft model designator (e.g. A320)
    # @badaDir:             BADA directory where all the BADA folders (corresponding to each aircraft) are located
    # @BADA_folder_search:  it indicates whether the program looks for the designator "ICAO" in the BADA directory or,
    # instead, a default BADA file is used
    def predictTrajectory(self, profile, pathFileCSV, ICAO, badaDir, BADA_folder_search):

        startTime = time.time()  # Starting time of the program; used to show the execution time

        # Initial setup: starting time of the trajectory, BADA APM, AircraftIntents Class, output list and current
        # integration time
        traj_start_time = profile.get("start_time")
        if BADA_folder_search:
            badaFile = find_bada_file(ICAO, badaDir)
            self.aircraft = perf.bada4(badaFile)
        else:
            # By default, the BADA file in this directory is used as performance model --> A320-214.xml
            print("Default aircraft loaded")
        self.initialize_AircraftIntents()
        columnsTitleFinal = ['Phase', 'Day', 'UTC', 't[s]', 's[NM]', 'h[ft]', 'TAS[kt]', 'mass[kg]', 'FPA_a[º]',
                             'THR[%]', 'SB[%]', 'nz[g]', 'hp[ft]',
                             'VS[ft/min]', 'CAS[kt]', 'mach[-]', 'GS[kt]', 'FPA_g[º]', 'Thrust[daN]', 'Tmax[daN]',
                             'Tidle[daN]']
        final_output = []
        self.integration_current_time = 0

        # Values depending on the output order (to be changed if the output list order changes)
        # Output values in the output list from this position to the end will be interpolated
        # when smoothing the transition between phases
        starting_interp_pos = 3
        endcondition_identification = {
            "MACH": 15, "CAS": 14, "s": 4, "Delta_s": 4, "alt_press": 12, "alt": 5
        }  # Position of the variables that can be end condition in the output list

        # Iterate through the blocks inside the profile
        for block in profile.get('blocks'):

            # Read initial phase of the block
            for i in range(len(profile.get('blocks')[block]['phases'])):

                self.current_phase = Ph.Phase()
                phase = profile.get('blocks')[block]['phases'][str(i + 1)]
                self.current_phase.read_phase(phase, self)
                break

            self.integration_type = profile.get('blocks')[block][
                "integration"]  # Reading the integration type: forward or backward
            self.initialConditions(profile)  # Read initial state of the block
            self.compute_auxiliary_variables_state_dependent()  # Compute auxiliary variables depending on the scv

            # Put the first sample (initial condition) in the output data structure and save first state in the results
            self.compute_first_state()
            initial_time = traj_start_time.replace(microsecond=0)
            posix_initial_time = time2sec(initial_time)
            self.generate_output_intermediate_steps(posix_initial_time, final_output)

            # Iterate through the phases in the current block
            for i in range(len(profile.get('blocks')[block]['phases'])):

                # Read the phase and update the id, the set of intents, end condition, configuration and gear;
                # if phase is not active, go to next phase
                phase = profile.get('blocks')[block]['phases'][str(i + 1)]
                self.current_phase.read_phase(phase, self)
                if self.current_phase.active == "FALSE":
                    continue

                # Update h, M and CAS with the intent values and issue a warning if there is a discontinuity
                self.update_h_M_CAS_with_intents_and_check_discontinuity()
                self.findAircraftIntents()  # For each phase, check AircraftIntents function to be used and EC

                # Compute the difference between the condition and the EC; if negative, this phase should be skipped
                self.check_integration_direction()
                self.update_condition()
                if self.current_phase.endConditionType == "Delta_s":
                    self.delta_s2s()
                self.end_condition_check()
                diff_cond = self.condition_f()
                prev_diff_cond = diff_cond

                # Integration of the phase; integrate until the end condition is reached
                # If the current condition does not converge towards the endcondition after error_iterations,
                # issue an error message and stop execution
                error_iterations = 20
                current_errors = 0
                while diff_cond > 0:

                    # Increment integration current time
                    self.integration_current_time += self.current_phase.dt

                    # Build and solve the ODE and update the state vector (compute x_k (h, m, v, s) from u_k-1)
                    self.update_state_vector()

                    # Compute auxiliary variables depending on the current states (x_k)
                    self.compute_auxiliary_variables_state_dependent()

                    # Check if constraints of the next waypoint are met; if not, call IPA to adapt the profile
                    # self.check_constraints_next_waypoint()

                    # Compute the new control vector (u_k --> gamma, Thrust) --> this will be u_k-1 in the next
                    # iteration, used to compute x_k
                    self.gamma_refinement()  # gamma refinement iterative algorithm to find the controls

                    # Compute auxiliary variables depending on the current controls (u_k)
                    self.compute_auxiliary_variables_control_dependent()

                    # Compute the throttle or use directly the intent (if throttle is an intent in the current phase)
                    self.throttle_f()

                    # Update the condition value
                    self.update_condition()

                    # Compute the difference between condition and the end condition
                    diff_cond = self.condition_f()

                    # Generate the output of the current integration step
                    # If the end condition is already met, smooth the transition between phases;
                    # interpolate by using the end condition value
                    self.generate_output_integration_step(diff_cond, posix_initial_time, final_output,
                                                          starting_interp_pos, endcondition_identification)

                    # Check if the condition is converging towards the end condition,
                    # if this does not happen for the number of iterations stated in "error_iterations",
                    # stop the execution
                    current_errors = self.check_convergence(diff_cond, prev_diff_cond, current_errors, error_iterations)

            # Define float precision, convert units and print output into a csv
            dfFinal = pd.DataFrame(final_output, columns=columnsTitleFinal)
            traj_computation_time = time.time() - startTime
            print('Trajectory generated successfully in %.3f sec.' % traj_computation_time)
            startTime = time.time()
            dfFinal = self.format_df(dfFinal)
            dfFinal.to_csv(pathFileCSV, index=False)
            print('CSV file was created successfully in %.3f sec.' % (time.time() - startTime))

            return dfFinal, traj_computation_time
