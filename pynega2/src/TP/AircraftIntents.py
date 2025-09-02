# -*- coding: utf-8 -*-
"""
pyBada
AircraftIntents: The output of functions for each mode is the control vector u = [gamma,T].
Use the parameters vector p = [Pi,VS,gamma_a,k] corresponding to the flight intent to find the control vector.
...
Reference: DESCENT AircraftIntents MODE MODELS AND ASSOCIATED PARAMETERS (TABLE I);
"R. Dalmau;M. P ́erez-Batlle;X.Prats,“Real-time Identification of AircraftIntents Modes in Aircraft Descents Using
Surveillance Data,” in Proceedings of the 37th Digital Avionics Systems Conference (DASC), London, UK, 2018. IEEE/AIAA."
...
"""

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"

import math

from TrajectoryPrediction import *


# This class contains the attributes and methods related with aircraft intents and computation of the control vector
class AircraftIntents(object):

    def __init__(self, aircraft=None):
        self.aircraft = aircraft

    def get_aircraft(self, aircraft):
        self.aircraft = aircraft


# Thrust intent group

# Mach & Thrust
class MachT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        Av = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * tp.wn / tp.v + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * tp.we / tp.v
        Bv = tp.Atmosphere.Hv_h(tp.Mach, tp.tau_h)
        Cv = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * cos(tp.chig) + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * sin(tp.chig)
        wndot = 0
        wedot = 0
        gamma = asin(
            (tp.T - tp.D - tp.m * Av) / (tp.m * sqrt(
                pow(const.g + Bv, 2) + pow(wndot * cos(tp.chig) + wedot * sin(tp.chig) + Cv, 2)))
        ) - atan2(wndot * cos(tp.chig) + wedot * sin(tp.chig), const.g + Bv)

        return gamma


# CAS & Thrust
class CAST(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        Av = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * tp.wn / tp.v + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * tp.we / tp.v
        Bv = tp.Atmosphere.Fv_h(tp.h, tp.tau_h, tp.p_h, tp.v_CAS)
        Cv = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * cos(tp.chig) + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * sin(tp.chig)
        wndot = 0
        wedot = 0
        gamma = asin(
            (tp.T - tp.D - tp.m * Av) / (tp.m * sqrt(
                pow(const.g + Bv, 2) + pow(wndot * cos(tp.chig) + wedot * sin(tp.chig) + Cv, 2)))
        ) - atan2(wndot * cos(tp.chig) + wedot * sin(tp.chig), const.g + Bv)

        return gamma


# Energy share factor (for acceleration or deceleration) & Thrust
class ESFT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        Av = 0
        Cv = 0
        Bv = ((1 - tp.k_ESF) / tp.k_ESF) * const.g
        wndot = 0
        wedot = 0
        gamma = arcsin(
            (tp.T - tp.D - tp.m * Av) / (tp.m * sqrt(
                pow(const.g + Bv, 2) + pow(wndot * cos(tp.chig) + wedot * sin(tp.chig) + Cv, 2)))
        ) - atan2(wndot * cos(tp.chig) + wedot * sin(tp.chig), const.g + Bv)

        return gamma


# Vertical Speed & Thrust
class VST(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        return arcsin(tp.h_dot / tp.v)


# FPA & Thrust
class FPAT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        return tp.gamma_a


# Altitude & Thrust
class ALTT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        # dp = TP_object.h / TP_object.Atmosphere.dhp_dp(TP_object.delta*const.p_0) / TP_object.v
        # A = TP_object.p_n * TP_object.wn / TP_object.v + TP_object.p_e * TP_object.we / TP_object.v - dpp
        A = tp.p_n * tp.wn / tp.v + tp.p_e * tp.we / tp.v
        B = - tp.p_h
        C = - tp.p_n * cos(tp.chig) - tp.p_e * sin(tp.chig)

        gamma = asin(A / sqrt(B * B + C * C)) - atan2(C, B)
        return gamma


# Vertical Speed intent group

# Mach & Vertical Speed
class MachVS(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * tp.wn / tp.v + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * tp.we / tp.v
        Bv = tp.Atmosphere.Hv_h(tp.Mach, tp.tau_h)
        Cv = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * cos(tp.chia) + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * sin(tp.chia)
        wv_dot = 0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        return arcsin(tp.h_dot / tp.v)


# CAS & Vertical Speed
class CASVS(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * tp.wn / tp.v + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * tp.we / tp.v
        Bv = tp.Atmosphere.Fv_h(tp.h, tp.tau_h, tp.p_h, tp.v_CAS)
        Cv = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * cos(tp.chia) + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * sin(tp.chia)
        wv_dot = 0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        return arcsin(tp.h_dot / tp.v)


# Energy share factor (for acceleration or deceleration) & Vertical Speed
class ESFVS(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = 0.0
        Bv = ((1.0 - tp.k_ESF) / tp.k_ESF) * const.g
        Cv = 0.0
        wv_dot = 0.0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        return arcsin(tp.h_dot / tp.v)


# FPA intent group

# Mach & FPA
class MachFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * tp.wn / tp.v + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * tp.we / tp.v
        Bv = tp.Atmosphere.Hv_h(tp.Mach, tp.tau_h)
        Cv = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * cos(tp.chia) + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * sin(tp.chia)
        wv_dot = 0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        return tp.gamma_a


# CAS & FPA
class CASFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * tp.wn / tp.v + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * tp.we / tp.v
        Bv = tp.Atmosphere.Fv_h(tp.h, tp.tau_h, tp.p_h, tp.v_CAS)
        Cv = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * cos(tp.chia) + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * sin(tp.chia)
        wv_dot = 0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        return tp.gamma_a


# CAS & FPA_G
class CASFPAG(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * tp.wn / tp.v + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * tp.we / tp.v
        Bv = tp.Atmosphere.Fv_h(tp.h, tp.tau_h, tp.p_h, tp.v_CAS)
        Cv = tp.Atmosphere.Fv_n(tp.tau, tp.p, tp.tau_n, tp.p_n, tp.v_CAS) * cos(tp.chia) + \
             tp.Atmosphere.Fv_e(tp.tau, tp.p, tp.tau_e, tp.p_e, tp.v_CAS) * sin(tp.chia)
        wv_dot = 0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        # TODO: Convert gamma_g to gamma_a!
        gamma_g = tp.gamma_a
        wx_bar = tp.wx / tp.v
        ws_bar = tp.ws / tp.v
        tp.gamma_a = asin(
            (sin(gamma_g) * (sqrt(1 - wx_bar ** 2 - pow((ws_bar * sin(gamma_g)), 2)) + ws_bar * cos(gamma_g))))

        return tp.gamma_a


# Energy Share Factor (for acceleration or deceleration) & FPA
class ESFFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = 0.0
        Bv = ((1.0 - tp.k_ESF) / tp.k_ESF) * const.g
        Cv = 0.0
        wv_dot = 0.0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        return tp.gamma_a


# Misc

# Energy share factor (for acceleration or deceleration) & Altitude
# TODO: probably will be removed for another pair of guidance modes?
class ESFALT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        tp.T = tp.m * tp.k_ESF + tp.D

    def gamma(self, tp):
        return tp.gamma_a


#  Altitude & Mach
# TODO: do we need alt CAS too? Isn't it the same?
class ALTMach(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        Av = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * tp.wn / tp.v + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * tp.we / tp.v
        Bv = tp.Atmosphere.Hv_h(tp.Mach, tp.tau_h)
        Cv = tp.Atmosphere.Hv_n(tp.Mach, tp.tau_n) * cos(tp.chia) + tp.Atmosphere.Hv_e(tp.Mach, tp.tau_e) * sin(tp.chia)
        wv_dot = 0
        tp.T = tp.D + tp.m * (sin(tp.gamma_a) * (const.g + Bv) + Cv * cos(tp.gamma_a) + Av + wv_dot)

    def gamma(self, tp):
        # dp = TP_object.h / TP_object.Atmosphere.dhp_dp(TP_object.delta*const.p_0) / TP_object.v
        # A = TP_object.p_n * TP_object.wn / TP_object.v + TP_object.p_e * TP_object.we / TP_object.v - dp
        A = tp.p_n * tp.wn / tp.v + tp.p_e * tp.we / tp.v
        B = - tp.p_h
        C = - tp.p_n * cos(tp.chig) - tp.p_e * sin(tp.chig)

        gamma = asin(A / sqrt(B * B + C * C)) - atan2(C, B)
        return gamma


# SPECIAL AIRCRAFT INTENTS

# DEC_CAS-h and FPA
# TODO: do for FPA_a and FPA_g
class CAShFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        tp.T = tp.D + tp.m * sin(tp.gamma_a) * (const.g + tp.Atmosphere.Fv_h(tp.h, tp.tau_h, tp.p_h, tp.v_CAS) +
                                                tp.v_CAS * tp.Atmosphere.G_v(tp.v_CAS, tp.h) * tp.k_CAS_h)

    def gamma(self, tp):
        return tp.gamma_a


# DEC_CAS-h and T
# TODO: finish with the new Fh and G_vars
class CAShT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        tp.gamma_a = math.asin(
            (tp.T - tp.D) / (tp.m * (const.g + tp.Atmosphere.Fv_h(tp.h, tp.tau_h, tp.p_h, tp.v_CAS) +
                                     tp.v_CAS * tp.Atmosphere.G_v(tp.v_CAS, tp.h) * tp.k_CAS_h)))
        return tp.gamma_a
