# coding=utf-8
"""
pyBada Trajectory Prediction Main
"""

import fileOperations as fO
import TrajectoryPrediction as TP
import plotResults as pr
import Route as rt

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"


ac_model = "A320"  # Aircraft model designator
badaDir = '/home/raulsaez21/Documents/PhD_home/Thesis/Data/BADA_files'  # Path to BADA directory where all the aircraft folders are contained
BADA_folder_search = False  # If True, the trajectory predictor will look for the aircraft model designator in "badaDir"; if False, an aircraft will be used by default
default_aircraft = "./input/A320-214.xml"
# default_aircraft    = "./input/B738W26.xml"

scenarioXML = './input/scenario.xml'
routeXML = './input/route.xml'

profileXML = './input/A320_example-descent-profile_v2.0.xml'  # Vertical profile file
# profileXML         = './input/A320_EDDF_EMPAX1B_fromADNIS_OFP.xml'       # Vertical profile file
# profileXML         = './input/A320_example-climb-profile_v1.0.xml'       # Vertical profile file
# profileXML          = './input/VT1_A320_Descent.xml'                    # Homeyra profile 1
# profileXML          = './input/VT2_A320_Descent.xml'                    # Homeyra profile 2

filePathCSV = './output/output_trajectory.csv'  # Trajectory (.csv) output file
filePathPlot = './output/trajectory_plot.png'  # Plot output file

# Weather directories
weather_directory = './input/gribs'
date = "160728"
time = "120100"
label = "gfsanl"
pressure_file = './pressure_levels.csv'

# Used to control if the weather is computed as a function of h and s or only as a function of h!
# (replicating the weather of the 1st waypoint for all)
function_h = False

# These could be knobs in a separate file
# TODO: if both true, there is an error!
ISA = False  # TODO: right now, this variable controls the use of winds and ISA, all together, we need to separate it!
use_route = True  # TODO: put this to true and check

route = rt.WeatherRoute(ISA, use_route, weather_dir=weather_directory, date=date, time=time, label=label,
                        pressure_file=pressure_file, route_filename=routeXML,
                        function_h=function_h)

scenario = fO.import_scenario_XML(scenarioXML)

profile = fO.import_profile_XML(profileXML, BADA_folder_search, ac_model, badaDir, default_aircraft, scenario)

tp = TP.TrajectoryPrediction(default_aircraft, route, ISA)
df, computation_time = tp.predictTrajectory(profile, filePathCSV, ac_model, badaDir, BADA_folder_search)

pr.plot(df, filePathPlot, 1)
