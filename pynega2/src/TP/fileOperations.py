# -*- coding: utf-8 -*-
"""
Read the XML profile
"""

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"

from xml.etree import ElementTree
from casadi.tools import *
import time
import pyBada3.conversions as conv
import datetime
import string
from synonims import *
import xml.etree.ElementTree as ET
import pyBada3.performance as perf

intents_keywords = ["prev_MACH", "prev_ALT", "prev_CAS"]
v_symbolic = ["V_GD", "GD", "V_S"]
thrust_keywords = ["MCMB", "MCRZ", "IDLE", "MTFK", "TOGA"]

VF_conf_id = {"V_F1": "1", "V_F2": "3", "V_F3": "4", "V_FULL": "5"}
VF_available = ["V_F1", "V_F2", "V_F3", "V_FULL"]


# Supporting function for debugging.
# @array: the return of the importXML function.
def printXML(array):
    print('<profile id=' + array.get('id') + '>')
    for block in array.get('blocks'):
        print('  <block id=' + block + ' integration=' + array.get('blocks')[block]['integration'] + ' type=' +
              array.get('blocks')[block]['typeBlock'] + '>')
        print('    <initial-state>')
        for param in array.get('blocks')[block]['initialConditions']:
            print(
                '      <' + param[0] + ' units=' + str(param[2]) + ' factor=' + str(param[3]) + '>' + param[1] + '</' +
                param[0] + '>')
        print('    </initial-state>')
        for i in range(len(array.get('blocks')[block]['phases'])):
            phase = array.get('blocks')[block]['phases'][str(i + 1)]
            print('    <phase id=' + phase['idPhase'] + '>')
            print('      <nodes>' + str(phase['nodes']) + '</nodes>')
            print('      <optimise>' + phase['optimise'] + '</optimise>')
            print('      <aircraft>')
            print('        <config>' + phase['aircraft'][0] + '</config>')
            print('        <gear>' + phase['aircraft'][1] + '</gear>')
            print('      </aircraft>')
            print('      <trajectory-prediction>')
            print('        <intent1 type=' + phase['trajectoryPrediction']['intent1'][0] +
                  ' units=' + str(phase['trajectoryPrediction']['intent1'][2]) +
                  ' factor=' + str(phase['trajectoryPrediction']['intent1'][3]) +
                  '>' + str(phase['trajectoryPrediction']['intent1'][1]) + '</intent1>')
            print('        <intent2 type=' + phase['trajectoryPrediction']['intent2'][0] +
                  ' units=' + str(phase['trajectoryPrediction']['intent2'][2]) +
                  ' factor=' + str(phase['trajectoryPrediction']['intent2'][3]) +
                  '>' + str(phase['trajectoryPrediction']['intent2'][1]) + '</intent2>')
            print('        <end-condition variable=' + phase['trajectoryPrediction']['endCondition'][0] +
                  ' units=' + str(phase['trajectoryPrediction']['endCondition'][2]) +
                  '>' + str(phase['trajectoryPrediction']['endCondition'][1]) + '</end-condition>')
            print('      </trajectory-prediction>')
            if phase['optimise'] == 'true':
                print('      <trajectory-optimisation>')
                print('        <length>')
                print('          <upper units=' + str(phase['trajectoryOptimisation']['length'][0][1]) + '>' + str(
                    phase['trajectoryOptimisation']['length'][0][0]) + '</upper>')
                print('          <lower units=' + str(phase['trajectoryOptimisation']['length'][1][1]) + '>' + str(
                    phase['trajectoryOptimisation']['length'][1][0]) + '</lower>')
                print('        </length>')
                print('        <final-constraints>')
                print('          MISSING')
                print('        </final-constraints>')
                print('        <path-constraints>')
                print('          MISSING')
                print('        </path-constraints>')
                print('      </trajectory-optimisation>')


def import_scenario_XML(fileName):

    dom = ElementTree.parse(fileName)
    root = dom.getroot()

    scenario = {}
    try:
        for child in root.find('flight_parameters'):
            value = child.text
            units = child.attrib.get('units')
            scenario[child.tag] = [value, units]
    except:
        print("WARNING! flight-parameters tag in %s is empty", fileName)

    return scenario


# Main function to read XML file.
# @fileName: the path of the file want to read.
def import_profile_XML(fileName, BADA_folder_search, ac_model, badaDir, default_aircraft, scenario):

    startTime = time.time()
    dom = ElementTree.parse(fileName)
    root = dom.getroot()

    # BADA aircraft (needed for some conversions, like flaps/slats keywords to HLid values)
    if BADA_folder_search:
        badaFile = find_bada_file(ac_model, badaDir)
        aircraft = perf.bada4(badaFile)
    else:
        aircraft = perf.bada4(default_aircraft)  # Aircraft object (BADA)

    # <profile>
    idProfile = root.attrib['id']

    # Read runway elevation
    if root.find('runway_elevation') is not None:
        runway_elevation = root.find('runway_elevation').text
        runway_elevation_units = root.find('runway_elevation').attrib.get('units')
        runway_elevation = convert2SI(runway_elevation_units, runway_elevation)
    else:
        runway_elevation = 0

    # Read trajectory starting time
    if root.find('start_trajectory') is not None:
        if root.find('start_trajectory').find('date') is not None:
            start_date = root.find('start_trajectory').find('date').text
        else:
            start_date = "2020-12-20"
        if root.find('start_trajectory').find('time') is not None:
            start_time = root.find('start_trajectory').find('time').text
        else:
            start_time = "10:00:00"
    else:
        start_date = "2020-12-20"
        start_time = "10:00:00"

    datetime_str = start_date + " " + start_time
    datetime_start = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    # <block>
    blocks = {}
    for block in root.iter('block'):
        idBlock = block.get('id')
        typeBlock = block.get('type')

        # Read integration block
        for child in block.find('integration'):
            if child.tag == "direction":
                integration = child.text
            elif child.tag == "method":
                method = child.text
            elif child.tag == "step":
                dt = child.text
                if 'units' in child.attrib:
                    dt_units = child.attrib.get('units')
                else:
                    dt_units = "seconds"
                dt = convert2SI(dt_units, dt)
            else:
                print("Not a valid tag for integration tag group")

        # <initial-state>
        initialConditions = []
        for child in block.find('initial-state'):
            parameter = [child.tag, child.text, child.attrib.get('units'),
                         child.attrib.get('factor') if child.attrib.get('factor') is not None else 1.0]
            if not parameter[1].replace(".", "").replace(",", "").isdigit():
                try:
                    parameter[2] = scenario[parameter[1]][1]
                    parameter[1] = scenario[parameter[1]][0]
                except:
                    print("ERROR!! %s is not a valid value as initial condition. Stopping execution...", parameter[1])
                    sys.exit()
            elif parameter[1] == "V2":
                parameter[1] = aircraft.SpeedCalculator.V2(aircraft.ICAO["designator"])
            parameter[1] = convert2SI(parameter[2], parameter[1])
            if 'h_ref' in child.attrib:
                if child.attrib.get('h_ref') == "AAL":
                    parameter[1] = str(float(parameter[1]) + float(runway_elevation))

            initialConditions.append(parameter)

        # <phase>
        phases = {}
        i = 0
        for phase in block.findall('phase'):
            i = i + 1

            if 'id' in phase.attrib:
                idPhase = phase.attrib.get('id')
            else:
                idPhase = "unknown phase id"
            if 'description' in phase.attrib:
                description = phase.attrib.get('description')
            else:
                description = " "
            if 'active' in phase.attrib:
                active = phase.attrib.get('active')
            else:
                active = "TRUE"

            # <nodes> #Remove for eurocontrol release?
            if phase.find('nodes') is not None:
                nodes = phase.find('nodes').text
            else:
                nodes = 0
            # <optimise>
            if phase.find('optimise') is not None:
                optimise = phase.find('optimise').text
            else:
                optimise = False

            # <aircraft>
            if phase.find('aircraft') is not None:
                config = phase.find('aircraft').get('config') if phase.find('aircraft').get(
                    'config') is not None else "0"
                # Convert flaps/slats config (e.g. CONF1, flaps 15) into HLid (e.g. 1,2,3...)
                config = flapconf2HLid(config, aircraft, idPhase)
                gear = phase.find('aircraft').get('gear') if phase.find('aircraft').get('gear') is not None else "UP"
            else:
                config = "0"
                gear = "UP"

            #################################
            ##### Trajectory Prediction #####
            #################################
            if phase.find('trajectory-prediction').find('step') is not None:
                phase_dt = phase.find('trajectory-prediction').find('step').text
                if 'units' in phase.find('trajectory-prediction').find('step').attrib:
                    phase_dt_units = phase.find('trajectory-prediction').find('step').get('units')
                else:
                    phase_dt_units = "seconds"
                phase_dt = convert2SI(phase_dt_units, phase_dt)
            else:
                phase_dt = dt

            intent1 = [phase.find('trajectory-prediction').find('intent1').get('type'),
                       phase.find('trajectory-prediction').find('intent1').text,
                       phase.find('trajectory-prediction').find('intent1').get('units'),
                       phase.find('trajectory-prediction').find('intent1').get('factor') if phase.find(
                           'trajectory-prediction').find('intent1').get('factor') is not None else 1.0,
                       phase.find('trajectory-prediction').find('intent1').get('delta') if phase.find(
                           'trajectory-prediction').find('intent1').get('delta') is not None else 0]
            intent2 = [phase.find('trajectory-prediction').find('intent2').get('type'),
                       phase.find('trajectory-prediction').find('intent2').text,
                       phase.find('trajectory-prediction').find('intent2').get('units'),
                       phase.find('trajectory-prediction').find('intent2').get('factor') if phase.find(
                           'trajectory-prediction').find('intent2').get('factor') is not None else 1.0,
                       phase.find('trajectory-prediction').find('intent2').get('delta') if phase.find(
                           'trajectory-prediction').find('intent2').get('delta') is not None else 0]
            endCondition = [phase.find('trajectory-prediction').find('end-condition').get('variable'),
                            phase.find('trajectory-prediction').find('end-condition').text,
                            phase.find('trajectory-prediction').find('end-condition').get('units'),
                            phase.find('trajectory-prediction').find('end-condition').get('factor') if phase.find(
                                'trajectory-prediction').find('end-condition').get('factor') is not None else 1.0,
                            phase.find('trajectory-prediction').find('end-condition').get('delta') if phase.find(
                                'trajectory-prediction').find('end-condition').get('delta') is not None else 0]

            # Convert speed intents and/or speed end conditions defined with keywords (e.g. Vapp) to values
            intent1 = intentkeyword2intentvalue(intent1, aircraft, idPhase, config, gear, scenario)
            intent2 = intentkeyword2intentvalue(intent2, aircraft, idPhase, config, gear, scenario)
            endCondition = intentkeyword2intentvalue(endCondition, aircraft, idPhase, config, gear, scenario)

            # Check if intent is THR and a keyword --> "IDLE", "MCMB", "MCRZ", "MTFK" (i.e. "TOGA") and convert into throttle value
            rating = "MCMB"  # Rating is MCMB by default
            intent1, rating = thrustkeyword2throttlevalue(intent1, rating, idPhase)
            intent2, rating = thrustkeyword2throttlevalue(intent2, rating, idPhase)
            intent1.append(rating)
            intent2.append(rating)

            # Convert intent values to SI system if the intent/endcondition is not set as a function of the previous value or if it's green dot
            # Convert also delta values to SI
            intent1 = convertIntentEc2SI(intent1)
            intent2 = convertIntentEc2SI(intent2)
            endCondition = convertIntentEc2SI(endCondition)

            # Change altitude depending on the reference altitude
            intent1 = modify_altitude(phase, intent1, runway_elevation)
            intent2 = modify_altitude(phase, intent2, runway_elevation)
            endCondition = modify_altitude(phase, endCondition, runway_elevation)

            trajectoryPrediction = {
                'intent1': intent1, 'intent2': intent2, 'endCondition': endCondition, 'phase_dt': phase_dt
            }

            #################################
            #### Trajectory Optimisation ####
            #################################
            trajectoryOptimisation = {'optimisation': False}
            if phase.find('trajectory-optimisation') is not None:
                # <length>
                if phase.find('trajectory-optimisation').find('length').find('upper') is not None and phase.find(
                        'trajectory-optimisation').find('length').find('lower') is not None:
                    length = [[phase.find('trajectory-optimisation').find('length').find('upper').text,
                               phase.find('trajectory-optimisation').find('length').find('upper').get('units')],
                              [phase.find('trajectory-optimisation').find('length').find('lower').text,
                               phase.find('trajectory-optimisation').find('length').find('lower').get('units')]]
                    length[0] = convert2SI(length[1], length[0])
                    length[2] = convert2SI(length[3], length[2])
                elif phase.find('trajectory-optimisation').find('length').find('upper') is not None:
                    length = [[phase.find('trajectory-optimisation').find('length').find('upper').text,
                               phase.find('trajectory-optimisation').find('length').find('upper').get('units')],
                              [None, None]]
                    length[0] = convert2SI(length[1], length[0])
                elif phase.find('trajectory-optimisation').find('length').find('lower') is not None:
                    length = [[None, None],
                              [phase.find('trajectory-optimisation').find('length').find('lower').text,
                               phase.find('trajectory-optimisation').find('length').find('lower').get('units')]]
                    length[2] = convert2SI(length[3], length[2])
                else:
                    length = [[None, None], [None, None]]

                # <final-constraints>
                constraintsFinal = []
                for constraint in phase.find('trajectory-optimisation').find('final-constraints').findall('constraint'):
                    if constraint.find('upper') is not None and constraint.find('lower') is not None:
                        constraintsFinal = [[constraint.find('upper').text, constraint.find('upper').get('units'),
                                             constraint.find('upper').get('factor') if constraint.find('upper').get(
                                                 'factor') is not None else 1.0],
                                            [constraint.find('lower').text, constraint.find('lower').get('units'),
                                             constraint.find('lower').get('factor') if constraint.find('lower').get(
                                                 'factor') is not None else 1.0]]
                        constraintsFinal[0] = convert2SI(constraintsFinal[1], constraintsFinal[0])
                        constraintsFinal[2] = convert2SI(constraintsFinal[3], constraintsFinal[2])
                    elif constraint.find('upper') is not None:
                        constraintsFinal = [[constraint.find('upper').text, constraint.find('upper').get('units'),
                                             constraint.find('upper').get('factor') if constraint.find('upper').get(
                                                 'factor') is not None else 1.0],
                                            [None, None, None]]
                        constraintsFinal[0] = convert2SI(constraintsFinal[1], constraintsFinal[0])
                    elif constraint.find('lower') is not None:
                        constraintsFinal = [[None, None, None],
                                            [constraint.find('upper').text, constraint.find('upper').get('units'),
                                             constraint.find('lower').get('factor') if constraint.find('lower').get(
                                                 'factor') is not None else 1.0]]
                        constraintsFinal[2] = convert2SI(constraintsFinal[3], constraintsFinal[2])
                    else:
                        constraintsFinal = [[None, None, None], [None, None, None]]
                # <path-constraints>
                constraintsPath = []
                for constraint in phase.find('trajectory-optimisation').find('path-constraints').findall('constraint'):
                    if constraint.find('upper') is not None and constraint.find('lower') is not None:
                        constraintsPath = [[constraint.find('upper').text, constraint.find('upper').get('units'),
                                            constraint.find('upper').get('factor') if constraint.find('upper').get(
                                                'factor') is not None else 1.0],
                                           [constraint.find('lower').text, constraint.find('lower').get('units'),
                                            constraint.find('lower').get('factor') if constraint.find('lower').get(
                                                'factor') is not None else 1.0]]
                        constraintsPath[0] = convert2SI(constraintsPath[1], constraintsPath[0])
                        constraintsPath[2] = convert2SI(constraintsPath[3], constraintsPath[2])
                    elif constraint.find('upper') is not None:
                        constraintsPath = [[constraint.find('upper').text, constraint.find('upper').get('units'),
                                            constraint.find('upper').get('factor') if constraint.find('upper').get(
                                                'factor') is not None else 1.0],
                                           [None, None, None]]
                        constraintsPath[0] = convert2SI(constraintsPath[1], constraintsPath[0])
                    elif constraint.find('lower') is not None:
                        constraintsPath = [[None, None, None],
                                           [constraint.find('upper').text, constraint.find('upper').get('units'),
                                            constraint.find('lower').get('factor') if constraint.find('lower').get(
                                                'factor') is not None else 1.0]]
                        constraintsPath[2] = convert2SI(constraintsPath[3], constraintsPath[2])
                    else:
                        constraintsPath = [[None, None, None], [None, None, None]]
                trajectoryOptimisation = {
                    'length': length, 'constraintsFinal': constraintsFinal, 'constraintsPath': constraintsPath
                }
            # store phases in a dictionary.
            phases[str(i)] = {
                'idPhase': idPhase, 'description': description, 'active': active, 'nodes': nodes, 'optimise': optimise,
                'aircraft': [config, gear], 'trajectoryPrediction': trajectoryPrediction,
                'trajectoryOptimisation': trajectoryOptimisation
            }
        # store blocks in a dictionary.
        blocks[idBlock] = {
            'integration': integration, 'typeBlock': typeBlock, 'dt': dt, 'method': method,
            'initialConditions': initialConditions, 'phases': phases
        }
    # returns the profile id and the blocks (which contain all the lower-level information).

    print('XML file was imported successfully in %.3f sec.' % (time.time() - startTime))
    return {'id': idProfile, 'blocks': blocks, 'start_time': datetime_start}


##########################
### Auxiliar functions ###
##########################

# Find the corresponding BADA xml file depending on the ICAO designator
# @ICAO: aircraft designator (e.g. "A320")
# @badaDir: BADA directory (where all the BADA folders are located)
def find_bada_file(ICAO, badaDir):
    Icao2Bada = {}
    folders = os.listdir(badaDir)
    for aircraft1 in folders:
        xml = os.path.join(badaDir, aircraft1, aircraft1 + '.xml')
        if os.path.isfile(xml):
            tree = ET.parse(xml)
            root = tree.getroot()
            ICAO = root.find('ICAO').find('designator').text
            Icao2Bada[ICAO] = aircraft1

    if ICAO in Icao2Bada.keys():
        ICAO = ICAO
    else:
        ICAO = synonims[ICAO]

    badaFile = os.path.join(badaDir, Icao2Bada[ICAO], Icao2Bada[ICAO] + '.xml')

    return badaFile


# Convert intent/endcondition to international system
# @intent_ec: intent or end condition
def convertIntentEc2SI(intent_ec):
    # Convert intent value and endcondition to SI
    if intent_ec[1] not in intents_keywords and intent_ec[1] not in v_symbolic:
        intent_ec[1] = convert2SI(intent_ec[2], intent_ec[1])

    # Convert delta to SI
    if float(intent_ec[4]) != 0.0:
        intent_ec[4] = convert2SI(intent_ec[2], intent_ec[4])

    return intent_ec


# Convert units to international system
# @original_units: original units type
# @original_value: original units value
def convert2SI(original_units, original_value):

    SI_units = ["kg", "m", "ms", "rad", "ms2", "seconds"]
    converted_value = original_value

    if original_units not in SI_units:
        if original_units == "lbs":
            converted_value = str(conv.lb2kg(float(original_value)))
        elif original_units == "t":
            converted_value = str(float(original_value) * 1000)
        elif original_units == "NM":
            converted_value = str(conv.nm2m(float(original_value)))
        elif original_units == "km":
            converted_value = str(float(original_value) * 1000)
        elif original_units == "ft":
            converted_value = str(conv.ft2m(float(original_value)))
        elif original_units == "km":
            converted_value = str(float(original_value) * 1000)
        elif original_units == "kt":
            converted_value = str(conv.kt2ms(float(original_value)))
        elif original_units == "kmh":
            converted_value = str(float(original_value) * 1000 / 3600)
        elif original_units == "ftmin":
            converted_value = str(conv.ftmin2ms(float(original_value)))
        elif original_units == "deg":
            converted_value = str(conv.deg2rad(float(original_value)))
        elif original_units == "minutes":
            converted_value = float(original_value) * 60
        elif original_units == "hours":
            converted_value = float(original_value) * 3600

    return converted_value


# Modify altitude depending on the reference altitude
# @phase: phase data
# @intent_ec: intent or endcondition
# @runway_elevation: airport runway elevation
def modify_altitude(phase, intent_ec, runway_elevation):

    if 'h_ref' in phase.find('trajectory-prediction').find('intent1').attrib:
        h_ref = phase.find('trajectory-prediction').find('intent1').attrib.get('h_ref')
        intent_ec.append(h_ref)
        if h_ref == "AAL":
            intent_ec[1] = str(float(intent_ec[1]) + float(runway_elevation))
    else:
        intent_ec.append("AMSL")

    return intent_ec


# Obtain the corresponding HLid value (flaps/slats configuration) -->
#   convert configurations of the vertical profile defined with keywords to HLid
# @flap_conf: flap configuration as read from the vertical profile
# @aircraft: BADA aircraft
# @phase_id: phase id
def flapconf2HLid(flap_conf, aircraft, phase_id):

    HLid = "0"
    if flap_conf.isdigit():
        HLid = flap_conf
    else:
        found = False
        # Iterate over all BADA configurations for the given aircraft and find the one that matches the configuration for the current phase
        current_conf = string.lower(flap_conf).replace(" ", "")
        for HLid, bada_conf in aircraft.name.iteritems():
            bada_conf = string.lower(bada_conf).replace(" ", "")  # Remove spaces and convert bada conf into lower case
            if current_conf == bada_conf:
                found = True
                break
        if not found:
            print(
                "\nWARNING!\nThe flaps and slats configuration specified for phase %s (%s) does not match any of the "
                "available configurations in the BADA performance file, which are the following:",
                (phase_id, current_conf))
            print(
                "- Available keywords: %s\n- Available HLid values: %s\nConfiguration for phase %s automatically "
                "will be set to CLEAN (HLid=0) and the execution will resume. You may also stop the current execution, "
                "modify the profile accordingly and restart the program.\n",
                (aircraft.name.values(), aircraft.name.keys(),
                 phase_id))
            HLid = "0"

    return HLid


# Obtain the corresponding speed value depending on the speed keyword
# @speedkeyword: speed keyword (e.g. MMO)
# @aircraft: BADA aircraft
# @phase_id: phase id
def intentkeyword2intentvalue(intent_ec, aircraft, phase_id, config, gear, scenario):

    if not intent_ec[1].replace("-", "").replace(".", "").replace(",", "").isdigit() \
            and intent_ec[1] not in intents_keywords:
        if intent_ec[0] == "alt" or intent_ec[0] == "alt_press":
            try:
                intent_ec[2] = scenario[intent_ec[1]][1]
                intent_ec[1] = scenario[intent_ec[1]][0]
            except:
                print("ERROR!! %s is not a valid altitude intent or end condition for phase %s. Stopping execution...",
                      (intent_ec[1], phase_id))
                sys.exit()
        elif intent_ec[0] == "Delta_s" or intent_ec[0] == "s":
            try:
                intent_ec[2] = scenario[intent_ec[1]][1]
                intent_ec[1] = scenario[intent_ec[1]][0]
            except:
                print("ERROR!! %s is not a valid distance end condition for phase %s. Stopping execution...",
                      (intent_ec[1], phase_id))
                sys.exit()
        elif intent_ec[0] == "FPA":
            try:
                intent_ec[2] = scenario[intent_ec[1]][1]
                intent_ec[1] = scenario[intent_ec[1]][0]
            except:
                print("ERROR!! %s is not a valid FPA intent for phase %s. Stopping execution...",
                      (intent_ec[1], phase_id))
                sys.exit()
        elif intent_ec[0] == "CAS" or intent_ec[0] == "MACH":
            if not intent_ec[1].replace(".", "").replace(",", "").isdigit() or intent_ec[1] not in intents_keywords:
                if intent_ec[1] == "VMO":
                    intent_ec[2] = "kt"
                    intent_ec[1] = aircraft.vmo
                elif intent_ec[1] == "MMO":
                    intent_ec[2] = "kt"
                    intent_ec[1] = aircraft.mmo
                elif intent_ec[1] == "V_GD" or intent_ec[
                    1] == "GD":  # Value computed while predicting the trajectory, as we need values of altitude and mass
                    intent_ec[2] = "ms"
                elif intent_ec[1] == "V_des" or intent_ec[1] == "M_des":
                    intent_ec[2] = scenario[intent_ec[1]][1]
                    intent_ec[1] = scenario[intent_ec[1]][0]
                elif intent_ec[1] == "V2":
                    intent_ec[2] = "ms"
                    intent_ec[1] = aircraft.SpeedCalculator.V2(aircraft.ICAO["designator"])
                elif intent_ec[1] == "V_app":
                    intent_ec[2] = "ms"
                    intent_ec[1] = aircraft.SpeedCalculator.Vapp(aircraft.ICAO["designator"])
                elif intent_ec[
                    1] in VF_available:  # Value computed while predicting the trajectory, as we need the value of altitude and mass
                    if VF_conf_id[intent_ec[1]] == config:
                        intent_ec[2] = "ms"
                    else:
                        print("ERROR!! VF speed and flap configuration do not match. Stopping execution...")  # TODO: is this necessary?
                        # sys.exit()
                # In take-off, it is the minimum speed at which the slats may be retracted;
                # On approach, S speed is the target speed when the aircraft is in CONF1 1.1 VS
                # in clean configuration Limited to VFE CONF1-2kt
                elif intent_ec[1] == "V_S":
                    if config == "1":
                        intent_ec[2] = "ms"
                    else:
                        print("ERROR!! S target speed is used for CONF1 only. Stopping execution...")
                        # sys.exit() #TODO: is this necessary?
                else:
                    print("ERROR!! %s is not a valid speed intent or end condition for phase %s. Stopping execution...",
                          (intent_ec[1], phase_id))
                    sys.exit()

    return intent_ec


# Convert a throttle/thrust intent given as a keyword (e.g. TOGA, IDLE,...) into a throttle value [0,1]
# @intent: intent
# @aircraft: BADA aircraft model
# @rating: phase engine rating
def thrustkeyword2throttlevalue(intent, rating, phase_id):

    if intent[0] == "THR":
        if intent[1] in thrust_keywords:
            rating = intent[1]
            if intent[1] == "IDLE":
                intent[1] = 0.0
                rating = "MCMB"
            else:
                intent[1] = 1.0 * float(intent[3])
                if float(intent[3]) > 1.0:
                    print("ERROR!! Factor value (i.e. throttle) for phase %s too high", phase_id)
                    sys.exit()
                if rating == "TOGA":
                    rating = "MTKF"
                    print("Rating %s is not considered in this BADA model; "
                          "Thrust equal to 1.3*MCMB will be considered instead", rating)
        else:
            print("This intent is not valid, please set the intent values to one of these values: %s. "
                  "Stopping execution...", thrust_keywords)
            sys.exit()

    return intent, rating
