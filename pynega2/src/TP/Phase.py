# -*- coding: utf-8 -*-
"""
pyBada
Phase module
2021
"""

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"

from fileOperations import *


# This class contains all the attributes and methods related with a phase of the profile
class Phase:

    def __init__(self):
        self.id = 0  # Phase ID
        self.dt = 0  # Time differential
        self.intent_values = []  # Values for the intents
        self.intent_types = 0  # Intent types
        self.config = 0  # Aircraft flaps/slats configuration
        self.gear = 0  # Aircraft gear
        self.T_factor = 1  # Thrust factor (the final maximum thrust will be multiplied by this value-->useful for TOGA rating)
        self.endConditionType = 0  # Type of end condition
        self.endConditionValue = 0  # Value of the end condition
        self.active = "TRUE"  # Phase active ("TRUE" or "FALSE")

    # Obtain all the phase info (intents, endcondition, gear, config, dt and id)
    def read_phase(self, phase, tp):

        # Intents
        intent1_type = phase['trajectoryPrediction']['intent1'][0]
        intent2_type = phase['trajectoryPrediction']['intent2'][0]
        self.intent_types = struct_symSX([intent1_type, intent2_type])
        self.intent_values = self.intent_types()

        intent_text = "intent1"
        self.readintent(intent_text, intent1_type, phase, tp)
        intent_text = "intent2"
        self.readintent(intent_text, intent2_type, phase, tp)

        # End condition
        self.endConditionType = phase['trajectoryPrediction']['endCondition'][0]
        endCondition_string = phase['trajectoryPrediction']['endCondition'][1]
        endCondition_factor = float(phase['trajectoryPrediction']['endCondition'][3])
        endCondition_delta = float(phase['trajectoryPrediction']['endCondition'][4])

        if endCondition_string == "V_GD" or endCondition_string == "GD":  # Compute green dot speed
            self.endConditionValue = endCondition_factor * (
                    tp.aircraft.SpeedCalculator.GreenDot(tp.m, tp.h, tp.aircraft.ICAO["designator"])
                    + endCondition_delta)
            # GD = opt.greenDot(TP_object.aircraft, TP_object.m, TP_object.delta, TP_object.theta)
            # self.endConditionValue = float(GD[0])
        elif endCondition_string in VF_available or endCondition_string == "V_S":  # Compute flap speed
            self.endConditionValue = endCondition_factor * (tp.aircraft.SpeedCalculator.VF(
                tp.aircraft.ICAO["designator"], self.config, self.gear, tp.m, tp.h, tp.aircraft) + endCondition_delta)
        else:
            self.endConditionValue = DM([(float(phase['trajectoryPrediction']['endCondition'][1]) + endCondition_delta)
                                         * endCondition_factor])

        # Configuration, gear, phase id, time step and phase activated
        self.config = phase['aircraft'][0]
        self.gear = phase['aircraft'][1]
        self.id = phase['idPhase']
        self.dt = float(phase['trajectoryPrediction']['phase_dt'])
        self.active = phase["active"]

    # Handle the use of intents set as the value of the previous phase
    def use_last_value_previous_phase(self, h1, M, v_CAS, intent, intent_factor, intent_delta):
        if intent == "prev_MACH":
            return intent_factor * (M + intent_delta)
        elif intent == "prev_ALT":
            return intent_factor * (h1 + intent_delta)
        elif intent == "prev_CAS":
            return intent_factor * (v_CAS + intent_delta)
        else:
            print("Warning: not a CAS, ALT or MACH intent!")

    def readintent(self, intent_text, intent_type, phase, tp):
        intent_value = phase['trajectoryPrediction'][intent_text][1]
        intent_factor = float(phase['trajectoryPrediction'][intent_text][3])
        intent_delta = float(phase['trajectoryPrediction'][intent_text][4])
        if intent_value in intents_keywords:
            converted_intent = self.use_last_value_previous_phase(
                tp.h, tp.Mach, tp.v_CAS, intent_value, intent_factor, intent_delta)
            self.intent_values[intent_type] = DM([float(converted_intent)])
        elif intent_value == "V_GD" or intent_value == "GD":  # Compute green dot speed
            self.intent_values[intent_type] = intent_factor * (tp.aircraft.SpeedCalculator.GreenDot(
                tp.m, tp.h, tp.aircraft.ICAO["designator"]) + intent_delta)
            # GD = opt.greenDot(TP_object.aircraft, TP_object.m, TP_object.delta, TP_object.theta)
            # self.intent_values[self.intent1] = float(GD[0])
        elif intent_value in VF_available or intent_value == "V_S":  # Compute flap speed
            self.intent_values[intent_type] = intent_factor * (tp.aircraft.SpeedCalculator.VF(
                tp.aircraft.ICAO["designator"], self.config, self.gear, tp.m, tp.h, tp.aircraft) + intent_delta)
        else:
            self.intent_values[intent_type] = DM(
                [(float(phase['trajectoryPrediction'][intent_text][1]) + intent_delta) * intent_factor])

        # If intent is THR save the factor
        if intent_type == "THR":
            tp.T_factor = phase['trajectoryPrediction'][intent_text][3]
            tp.rating = phase['trajectoryPrediction'][intent_text][5]
