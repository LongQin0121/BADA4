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


# This class contains all the attributes and methods related with the aircraft intents and the computation of the
# control vector
class AircraftIntents(object):

    def __init__(self, aircraft=None):
        self.aircraft = aircraft

    def get_aircraft(self, aircraft):
        self.aircraft = aircraft

    # Compute k_CAS (used when one of the aircraft intents is CAS)
    def compute_k_CAS(self, tp):

        # Eq (5) of Report (Energy Share Factor) to calculate k_Mach and k_CAS
        h_tropo = 11000  # [m]

        if tp.h < h_tropo:
            k_CAS = 1 / (1 + ((power((1 + (const.Amu * tp.v_CAS * tp.v_CAS * const.rho_0 / (2 * const.p_0))),
                                     (1 / const.Amu)) - 1) / (tp.delta)) * (power((((power(
                (1 + (const.Amu * tp.v_CAS * tp.v_CAS * const.rho_0 / (2 * const.p_0))),
                (1 / const.Amu)) - 1) / tp.delta) + 1), (const.Amu - 1))) - (
                                     const.R * const.tau_h / (const.g * const.Amu)) * (power((((power(
                (1 + (const.Amu * tp.v_CAS * tp.v_CAS * const.rho_0 / (2 * const.p_0))),
                (1 / const.Amu)) - 1) / tp.delta) + 1), const.Amu) - 1))
        else:
            k_CAS = 1 / (1 + ((power((1 + (const.Amu * tp.v_CAS * tp.v_CAS * const.rho_0 / (2 * const.p_0))),
                                     (1 / const.Amu)) - 1) / (tp.delta)) * (power((((power(
                (1 + (const.Amu * tp.v_CAS * tp.v_CAS * const.rho_0 / (2 * const.p_0))),
                (1 / const.Amu)) - 1) / tp.delta) + 1), (const.Amu - 1))))

        return k_CAS

    # Compute k_Mach (used when one of the intents is Mach)
    def compute_k_Mach(self, tp):

        # Eq (5) of Report (Energy Share Factor) to calculate k_Mach and k_CAS
        h_tropo = 11000  # [m]
        if tp.h < h_tropo:
            k_Mach = 1 / (1 - const.Agamma * const.tau_h * const.R * (tp.Mach ** 2) / (2 * const.g))
        else:
            k_Mach = 1

        return k_Mach


# Thrust intent group

# Mach & Thrust
class MachT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        tp.k_Mach = self.compute_k_Mach(tp)

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        return arcsin(((tp.T - tp.D) * tp.k_Mach) / (tp.m * const.g))

        # TODO: finish

        # Av = TP_object.aircraft.Hv_n(TP_object.aircraft.Mach) * TP_object.Wn / TP_object.v + TP_object.aircraft.Hv_e(TP_object.Mach) * TP_object.We / TP_object.v
        # Bv = TP_object.aircraft.Hv_h(TP_object.Mach)
        # Cv = TP_object.aircraft.Hv_n(TP_object.Mach) * cos(TP_object.chig) + TP_object.Hv_e(TP_object.Mach) * sin(TP_object.chig)
        # wndot = 0
        # wedot = 0
        # gamma = asin((TP_object.T - TP_object.D - TP_object.m * Av) / (TP_object.m * sqrt(pow(const.g + Bv, 2) + pow(wndot * cos(TP_object.chig) + wedot *
        #         sin(TP_object.chig) + Cv, 2))))- atan2(wndot * cos(TP_object.chig) + wedot * sin(TP_object.chig), const.g + Bv)
        #
        # return gamma


# CAS & Thrust
class CAST(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        tp.k_CAS = self.compute_k_CAS(tp)

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        return arcsin(((tp.T - tp.D) * tp.k_CAS) / (tp.m * const.g))
        # TODO: finish

        # Av = TP_object.aircraft.Fv_n(TP_object.aircraft.Mach) * TP_object.Wn / TP_object.v + TP_object.aircraft.Fv_e(TP_object.Mach) * TP_object.We / TP_object.v
        # Bv = TP_object.aircraft.Fv_h(TP_object.Mach)
        # Cv = TP_object.aircraft.Fv_n(TP_object.Mach) * cos(TP_object.chig) + TP_object.Fv_e(TP_object.Mach) * sin(TP_object.chig)
        # wndot = 0
        # wedot = 0
        # gamma = asin((TP_object.T - TP_object.D - TP_object.m * Av) / (TP_object.m * sqrt(pow(const.g + Bv, 2) + pow(wndot * cos(TP_object.chig) + wedot *
        #         sin(TP_object.chig) + Cv, 2))))- atan2(wndot * cos(TP_object.chig) + wedot * sin(TP_object.chig), const.g + Bv)
        #
        # return gamma


# Energy share factor (for acceleration or deceleration) & Thrust
class ESFT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        return arcsin(((tp.T - tp.D) * tp.k_ESF) / (tp.m * const.g))
        # TODO: finish

        # Av = 0
        # Cv = 0
        # Bv = ((1 - (TP_object.k_ESF)) / (TP_object.k_ESF)) * const.g
        # wndot = 0
        # wedot = 0
        # gamma = arcsin((TP_object.T - TP_object.D - TP_object.m * Av) / (TP_object.m * sqrt(pow(const.g + Bv, 2) + pow(wndot * cos(TP_object.chig) + wedot *
        #         sin(TP_object.chig) + Cv, 2))))- atan2(wndot * cos(TP_object.chig) + wedot * sin(TP_object.chig), const.g + Bv)
        #
        # return gamma


# Vertical Speed & Thrust
class VST(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        return arcsin(tp.VS / tp.v)


# FPA & Thrust
class FPAT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

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

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        return 0

        # TODO: Understand what is happening here
        # TODO: finish

        # #dp = (intent.type == INTENT_TYPE::ALT or intent.type == INTENT_TYPE::PRESS_ALT) ? 0: intent.value / atmosphere->dhp_dp() / scv.v
        #
        # # A = TP_object.aircraft.p_n * TP_object.Wn / TP_object.v + TP_object.aircraft.p_e * TP_object.We / TP_object.v - dp
        # A = TP_object.aircraft.p_n * TP_object.Wn / TP_object.v + TP_object.aircraft.p_e * TP_object.We / TP_object.v
        # B = - TP_object.P_h
        # C = - TP_object.p_n * cos(TP_object.chig) - TP_object.aircraft.p_e * sin(TP_object.chig)
        #
        # gamma = asin(A / sqrt(B * B + C * C)) - atan2(C, B)
        # return gamma


# Vertical Speed intent group

# Mach & Vertical Speed
class MachVS(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        tp.k_Mach = self.compute_k_Mach(tp)

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        tp.T = tp.D + ((tp.m * const.g * tp.VS) / (tp.k_Mach * tp.v))

    def gamma(self, tp):
        return arcsin(tp.VS / tp.v)


# CAS & Vertical Speed
class CASVS(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        tp.k_CAS = self.compute_k_CAS(tp)

    def T(self, tp):
        tp.T = tp.D + ((tp.m * const.g * tp.VS) / (tp.k_CAS * tp.v))

    def gamma(self, tp):
        return arcsin(tp.VS / tp.v)


# Energy share factor (for acceleration or deceleration) & Vertical Speed
class ESFVS(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        tp.T = tp.D + ((tp.m * const.g * tp.VS) / (tp.k_ESF * tp.v))

    def gamma(self, tp):
        return arcsin(tp.VS / tp.v)


# FPA intent group

# Mach & FPA
class MachFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        tp.k_Mach = self.compute_k_Mach(tp)

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        tp.T = tp.D + ((tp.m * const.g / tp.k_Mach) * sin(tp.gamma_a))

    def gamma(self, tp):
        return tp.gamma_a


# CAS & FPA
class CASFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        tp.k_CAS = self.compute_k_CAS(tp)

    def T(self, tp):
        tp.T = tp.D + ((tp.m * const.g / tp.k_CAS) * sin(tp.gamma_a))

    def gamma(self, tp):
        return tp.gamma_a


# CAS & FPA_G
class CASFPAG(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        tp.k_CAS = self.compute_k_CAS(tp)

    def T(self, tp):
        tp.T = tp.D + ((tp.m * const.g / tp.k_CAS) * sin(tp.gamma_a))

    def gamma(self, tp):
        # TODO: The value received on gamma_a is the gamma_g value to be later converted to gamma_a
        # TODO: finish
        gamma_g = tp.gamma_a
        wx_bar = tp.Wx / tp.v
        ws_bar = tp.Ws / tp.v
        tp.gamma_a = asin(
            (sin(gamma_g) * (sqrt(1 - wx_bar ** 2 - pow((ws_bar * sin(gamma_g)), 2)) + ws_bar * cos(gamma_g))))

        return tp.gamma_a


# Energy Share Factor (for acceleration or deceleration) & FPA
class ESFFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        tp.T = tp.D + ((tp.m * const.g / tp.k_ESF) * sin(tp.gamma_a))

    def gamma(self, tp):
        return tp.gamma_a


# Misc

# Energy share factor (for acceleration or deceleration) & Altitude
class ESFALT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        tp.T = tp.m * tp.k_ESF + tp.D

    def gamma(self, tp):
        return tp.gamma_a


#  Altitude & Mach
class ALTMach(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        tp.T = tp.D

    def gamma(self, tp):
        return 0

        # TODO: Finish
        # #dp = (intent.type == INTENT_TYPE::ALT or intent.type == INTENT_TYPE::PRESS_ALT) ? 0: intent.value / atmosphere->dhp_dp() / scv.v
        #
        # # A = TP_object.aircraft.p_n * TP_object.Wn / TP_object.v + TP_object.aircraft.p_e * TP_object.We / TP_object.v - dp
        # A = TP_object.aircraft.p_n * TP_object.Wn / TP_object.v + TP_object.aircraft.p_e * TP_object.We / TP_object.v
        # B = - TP_object.p_h
        # C = - TP_object.p_n * cos(TP_object.chig) - TP_object.aircraft.p_e * sin(TP_object.chig)
        #
        # gamma = asin(A / sqrt(B * B + C * C)) - atan2(C, B)
        # return gamma


# SPECIAL AIRCRAFT INTENTS

# DEC_CAS-h and FPA
class CAShFPA(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        tp.T = tp.D + tp.m * sin(tp.gamma_a) * (const.g + tp.aircraft.F_h_var(tp.v_CAS, tp.h) + tp.v_CAS *
                                                tp.aircraft.G_var(tp.v_CAS, tp.h) * tp.k_CAS_h)

    def gamma(self, tp):
        return tp.gamma_a


# DEC_CAS-h and T
# TODO: finish with the new Fh and G_vars
class CAShT(AircraftIntents):
    def __init__(self, aircraft):
        super().__init__(aircraft)

    def get_k_Mach(self, tp):
        return

    def get_k_CAS(self, tp):
        return

    def T(self, tp):
        if tp.THR == 0:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.T = tp.Idle_Thrust
        else:
            tp.Idle_Thrust = self.aircraft.TMin(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h)
            tp.Max_Thrust = self.aircraft.TMax(delta=tp.delta, theta=tp.theta, M=tp.Mach, h=tp.h, rating=tp.rating)
            tp.T = tp.Idle_Thrust + tp.THR * (tp.Max_Thrust - tp.Idle_Thrust)

    def gamma(self, tp):
        tp.gamma_a = math.asin((tp.T - tp.D) / (tp.m * (const.g + tp.aircraft.F_h_var(tp.v_CAS, tp.h) +
                                                        tp.v_CAS * tp.aircraft.G_var(tp.v_CAS, tp.h) * tp.k_CAS_h)))
        return tp.gamma_a
