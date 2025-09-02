# -*- coding: utf-8 -*-
"""
pyBada
Route module
2021
"""

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"

from fileOperations import *
from grib import *
import pandas as pd
import pyBada3.conversions as conv
import math
from utilities_TP import *
from geographiclib.geodesic import Geodesic


# This class defines a waypoint
class Waypoint:
    def __init__(self):
        self.id = "waypoint"  # Waypoint id
        self.latitude = 0.0  # Latitude
        self.longitude = 0.0  # Longitude
        self.speed_constraint = []  # Speed constraints [upper, lower]
        self.altitude_constraint = []  # Altitude constraints [upper, lower]


# SubClass of Waypoint Class; it includes also the weather of the waypoint
class WeatherWaypoint(Waypoint, object):

    def __init__(self):
        super(WeatherWaypoint, self).__init__()
        self.weather = pd.DataFrame()  # Dataframe with temperature, pressure, north and east winds

    # Obtain temperature, pressure, north wind and east wind for a given altitude
    # @h: altitude
    def getWeatherdatafromalt(self, h):
        index = self.weather[self.weather['geopot_alt'].gt(h)].index[-1]
        Tau = interp_gen(self.weather['geopot_alt'][index], self.weather['geopot_alt'][index - 1],
                         self.weather['T'][index], self.weather['T'][index - 1], h)
        W_N = interp_gen(self.weather['geopot_alt'][index], self.weather['geopot_alt'][index - 1],
                         self.weather['W_N'][index], self.weather['W_N'][index - 1], h)
        W_E = interp_gen(self.weather['geopot_alt'][index], self.weather['geopot_alt'][index - 1],
                         self.weather['W_E'][index], self.weather['W_E'][index - 1], h)
        Pressure = interp_gen(self.weather['geopot_alt'][index], self.weather['geopot_alt'][index - 1],
                              self.weather['Pressure'][index], self.weather['Pressure'][index - 1], h)

        return Tau[1], W_N[1], W_E[1], Pressure[1]

    # Obtain temperature, pressure, north wind and east wind for a given altitude
    # @h: altitude
    def getAltfromPress(self, press):
        try:
            index = self.weather[self.weather['Pressure'].gt(press)].index[-1]
        except:
            index = len(self.weather['Pressure']) - 1
        Alt = interp_gen(self.weather['Pressure'][index], self.weather['Pressure'][index - 1],
                         self.weather['geopot_alt'][index], self.weather['geopot_alt'][index - 1], press)

        return Alt[1]


# This class contains all the attributes and methods related with the route,
# which contains a list of waypoints and the tracks and distances between them
class Route:

    def __init__(self, use_route, route_filename=0):
        self.waypoint_list = []  # List of waypoints defining the route
        self.track_distance_list = []  # List of tracks[0] and distances[1] between waypoints
        if use_route:
            self.import_route_XML(route_filename)  # Read the route from "fileName"
            self.propagate_constraints()  # Expands the constraints to all the waypoints
            self.computeRouteTracksDists()  # Compute the tracks and distances between waypoints

    # Read XML containing the route
    def import_route_XML(self, fileName):

        dom = ElementTree.parse(fileName)
        root = dom.getroot()

        try:
            waypoints = root.findall("waypoint")
            # for wp in root.find('waypoint'):
            for wp in waypoints:
                waypoint = Waypoint()
                waypoint.id = wp.find('name').text
                waypoint.latitude = conv.deg2rad(float(wp.find('latitude').text))
                waypoint.longitude = conv.deg2rad(float(wp.find('longitude').text))

                # Checking speed constraints
                speed_units = "ms"
                speed_lower = "0"
                speed_upper = "100000"
                if wp.find('speed_constraint') is not None:
                    speed_units = wp.find('speed_constraint').get("units") if wp.find('speed_constraint').get(
                        'units') is not None else "ms"
                    speed_upper = wp.find('speed_constraint').get("upper") if wp.find('speed_constraint').get(
                        'upper') is not None else "100000"
                    speed_lower = wp.find('speed_constraint').get("lower") if wp.find('speed_constraint').get(
                        'lower') is not None else "0"

                speed_upper = float(convert2SI(speed_units, speed_upper))
                speed_lower = float(convert2SI(speed_units, speed_lower))
                waypoint.speed_constraint = [speed_upper, speed_lower]

                # Checking altitude constraints
                alt_units = "m"
                alt_lower = "0"
                alt_upper = "100000"
                if wp.find('altitude_constraint') is not None:
                    alt_units = wp.find('altitude_constraint').attrib.get("units") if wp.find(
                        'altitude_constraint').get('units') is not None else "m"
                    alt_upper = wp.find('altitude_constraint').attrib.get("upper") if wp.find(
                        'altitude_constraint').get('upper') is not None else "100000"
                    alt_lower = wp.find('altitude_constraint').attrib.get("lower") if wp.find(
                        'altitude_constraint').get('lower') is not None else "0"

                alt_upper = float(convert2SI(alt_units, alt_upper))
                alt_lower = float(convert2SI(alt_units, alt_lower))
                waypoint.altitude_constraint = [alt_upper, alt_lower]

                self.waypoint_list.append(waypoint)
        except:
            print("ERROR! waypoint-list tag in %s is empty", fileName)

    # Return the waypoint and the current track depending on the s value
    # (it returns the first waypoint; i.e., the origin of the segment)
    # @s: distance
    def getWaypointandTrackfromDistance(self, s):
        s_cum = 0
        for i in range(0, len(self.track_distance_list)):
            s_cum += self.track_distance_list[i][1]
            if s_cum > s:
                break

        return self.waypoint_list[i], self.track_distance_list[i][0]

    # Return the next waypoint speed and altitude constraints
    # @s: distance
    def getNextWaypointConstraints(self, s):
        s_cum = 0
        for i in range(0, len(self.track_distance_list)):
            s_cum += self.track_distance_list[i][1]
            if s_cum > s:
                break

        return self.waypoint_list[i + 1].speed_constraint, self.waypoint_list[i + 1].altitude_constraint

    # Propagate all constraints between waypoints (part of the IntentProcedureAdapter -- IPA)
    def propagate_constraints(self):
        none_alt_upper, none_speed_upper = 0, 0
        # Propagation constraints (assuming backwards and descent config)
        for i in range(0, len(self.waypoint_list)):
            # Getting the current element
            wpt = self.waypoint_list[i]

            # Upper Altitude
            if wpt.altitude_constraint[0] is not None:
                for j in range(i - none_alt_upper, i):
                    self.waypoint_list[j].altitude_constraint[0] = wpt.altitude_constraint[0]
            else:
                none_alt_upper += 1

            # Lower Altitude
            if wpt.altitude_constraint[1] is None:
                j = 0
                # Find the first value (not None)
                while (j + i < len(self.waypoint_list) - 1) and (
                        self.waypoint_list[i + j].altitude_constraint[1] is None):
                    j += 1
                # Copy the value to all the ones empty that have been previously checked
                for k in range(i, j):
                    self.waypoint_list[k].altitude_constraint[1] = self.waypoint_list[j].altitude_constraint[1]

            # Upper Speed
            if wpt.speed_constraint[0] is not None:
                for j in range(i - none_speed_upper, i):
                    self.waypoint_list[j].speed_constraint[0] = wpt.speed_constraint[0]
            else:
                none_speed_upper += 1

            # Lower Speed
            if wpt.speed_constraint[1] is None:
                j = 0
                # Find the first value (not None)
                while (j + i < len(self.waypoint_list) - 1) and (self.waypoint_list[i + j].speed_constraint[1] is None):
                    j += 1
                # Copy the value to all the ones empty that have been previously checked
                for k in range(i, j):
                    self.waypoint_list[k].speed_constraint[1] = self.waypoint_list[j].speed_constraint[1]

    # Compute tracks and distances between the waypoints
    def computeRouteTracksDists(self):
        for x in range(1, len(self.waypoint_list)):
            track = self.computeTrack(self.waypoint_list[x - 1], self.waypoint_list[x])
            dist = self.computeDistance(self.waypoint_list[x - 1], self.waypoint_list[x])
            self.track_distance_list.append([track, dist])

    # Compute track between two waypoints (lat, lon)
    # @w1: waypoint 1 (lat, lon)
    # @w2: waypoint 2 (lat, lon)
    def computeTrack(self, w1, w2):
        delta_lon = w2.longitude - w1.longitude
        X = cos(w2.latitude) * sin(delta_lon)
        Y = cos(w1.latitude) * sin(w2.latitude) - sin(w1.latitude) * cos(w2.latitude) * cos(delta_lon)
        track = math.atan2(X, Y)
        while track < 0:
            track += 2 * math.pi
        while track >= (2 * math.pi):
            track -= 2 * math.pi

        return track

    # Compute distance between 2 waypoints (lat, lon)
    # @w1: waypoint 1 (lat, lon)
    # @w2: waypoint 2 (lat, lon)
    def computeDistance(self, w1, w2):
        return Geodesic.WGS84.Inverse(conv.rad2deg(w1.latitude), conv.rad2deg(w1.longitude), conv.rad2deg(w2.latitude),
                                      conv.rad2deg(w2.longitude))['s12']


# Subclass of Route Class; it contains a list with weather points and obtains the weather for a given point in the route
class WeatherRoute(Route, object):
    # TODO: add a function to add more points to the weatherpoint_list,
    #  a part from the ones already in the waypoint_list (the constraints in this case won't be propagated)
    # TODO: maybe we should separate ISA and winds (we could do ISA with and without winds)-->to be implemented

    def __init__(self, ISA, use_route, weather_dir=0, date=0, time=0, label=0, pressure_file=0, route_filename=0,
                 function_h=False):
        super(WeatherRoute, self).__init__(use_route, route_filename=route_filename)
        self.weatherpoint_list = []  # List of weather points
        self.track_distance_list_weatherpoint = []  # List of track and distances between the weather points
        self.tau_spline = 0  # Temperature spline
        self.p_spline = 0  # Pressure spline
        self.wn_spline = 0  # North wind spline
        self.we_spline = 0  # East wind spline
        self.h_spline = 0  # Geometric (geopotenital) altitude spline

        # TODO: so far, hardcoded weather to accelerate the computations
        test_weatherpoint_list_pickle = './input/weatherpoints_test.pkl'
        test_weatherpoint_dist_pickle = './input/weatherpoints_dist_test.pkl'

        # TODO: ISA is controlling this too!
        # Just read route and fill the weather if we are not in ISA conditions
        if not ISA:
            if os.path.isfile(test_weatherpoint_list_pickle):
                self.weatherpoint_list = load_obj(test_weatherpoint_list_pickle)
                self.track_distance_list_weatherpoint = load_obj(test_weatherpoint_dist_pickle)
            else:
                self.import_weather_route_XML(
                    route_filename)  # Read the route from "fileName", waypoints saved are weather waypoints
                self.computeRouteTracksDistsWeather()
                # Get the weather of the route (weather for each waypoint)--> this includes winds!
                self.getRouteWeather(weather_dir, date, time, label, pressure_file, function_h)
                save_obj(self.weatherpoint_list, test_weatherpoint_list_pickle)
                save_obj(self.track_distance_list_weatherpoint, test_weatherpoint_dist_pickle)

            self.CreateWeatherSplines(ISA)

    # Read XML containing the route
    def import_weather_route_XML(self, fileName):

        dom = ElementTree.parse(fileName)
        root = dom.getroot()

        try:
            waypoints = root.findall("waypoint")
            # for wp in root.find('waypoint'):
            for wp in waypoints:
                weather_waypoint = WeatherWaypoint()
                weather_waypoint.id = wp.find('name').text
                weather_waypoint.latitude = conv.deg2rad(float(wp.find('latitude').text))
                weather_waypoint.longitude = conv.deg2rad(float(wp.find('longitude').text))

                # Checking speed constraints
                speed_units = "ms"
                speed_lower = "0"
                speed_upper = "100000"
                if wp.find('speed_constraint') is not None:
                    speed_units = wp.find('speed_constraint').get("units") if wp.find('speed_constraint').get(
                        'units') is not None else "ms"
                    speed_upper = wp.find('speed_constraint').get("upper") if wp.find('speed_constraint').get(
                        'upper') is not None else "100000"
                    speed_lower = wp.find('speed_constraint').get("lower") if wp.find('speed_constraint').get(
                        'lower') is not None else "0"

                speed_upper = float(convert2SI(speed_units, speed_upper))
                speed_lower = float(convert2SI(speed_units, speed_lower))
                weather_waypoint.speed_constraint = [speed_upper, speed_lower]

                # Checking altitude constraints
                alt_units = "m"
                alt_lower = "0"
                alt_upper = "100000"
                if wp.find('altitude_constraint') is not None:
                    alt_units = wp.find('altitude_constraint').attrib.get("units") if wp.find(
                        'altitude_constraint').get('units') is not None else "m"
                    alt_upper = wp.find('altitude_constraint').attrib.get("upper") if wp.find(
                        'altitude_constraint').get('upper') is not None else "100000"
                    alt_lower = wp.find('altitude_constraint').attrib.get("lower") if wp.find(
                        'altitude_constraint').get('lower') is not None else "0"

                alt_upper = float(convert2SI(alt_units, alt_upper))
                alt_lower = float(convert2SI(alt_units, alt_lower))
                weather_waypoint.altitude_constraint = [alt_upper, alt_lower]

                self.weatherpoint_list.append(weather_waypoint)
        except:
            print("ERROR! waypoint-list tag in %s is empty", fileName)

    # Get information of weather for each waypoint
    def getRouteWeather(self, weather_dir, date, time, label, pressure_file, function_h):

        # Get the closest file but before and read grib
        grib_model = Grib(weather_dir, date, time, label)

        # Weather as a function of h and s, or just as a function of h
        # (replicating the weather of the first waypoint for all waypoints)
        if function_h:
            weather = grib_model.getWaypointWeather(
                (self.weatherpoint_list[0].latitude, self.weatherpoint_list[0].longitude))
            wind_df = pd.read_table(pressure_file, header=None, sep=",", names=["Pressure"])
            wind_df["W_N"] = weather[10:31, 0]
            wind_df["W_E"] = weather[10:31, 1]
            wind_df["T"] = weather[10:31, 2]
            wind_df["geopot_alt"] = weather[10:31, 3]
            self.weatherpoint_list[0].weather = wind_df

            for weatherpoint in self.weatherpoint_list:
                weatherpoint.weather = wind_df
        else:
            # The components of the wind are obtained for each pressure level from 100 Pa up to 100000 Pa
            # (we select the ones from 10000 up to 100000)
            for weatherpoint in self.weatherpoint_list:
                weather = grib_model.getWaypointWeather((weatherpoint.latitude, weatherpoint.longitude))
                wind_df = pd.read_table(pressure_file, header=None, sep=",", names=["Pressure"])
                wind_df["W_N"] = weather[10:31, 0]
                wind_df["W_E"] = weather[10:31, 1]
                wind_df["T"] = weather[10:31, 2]
                wind_df["geopot_alt"] = weather[10:31, 3]
                weatherpoint.weather = wind_df

        print("Weather obtained")

    # TODO. where do we save these splines? Do we save them in the route object? Atmosphere object?
    def CreateWeatherSplines(self, ISA):
        atmosphere_object = atm.Atmosphere(ISA)
        self.tau_spline, self.p_spline = atmosphere_object.computeTauPressSpline(self)
        self.wn_spline, self.we_spline = atmosphere_object.computeWindSpline(self)
        self.h_spline = atmosphere_object.computeAltSpline(self)
        print("Splines created")

    # Compute tracks and distances between the weather waypoints
    def computeRouteTracksDistsWeather(self):
        for x in range(1, len(self.weatherpoint_list)):
            track = self.computeTrack(self.weatherpoint_list[x - 1], self.weatherpoint_list[x])
            dist = self.computeDistance(self.weatherpoint_list[x - 1], self.weatherpoint_list[x])
            self.track_distance_list_weatherpoint.append([track, dist])

    # Return the waypoint and the current track depending on the s value
    # @s: distance
    def getWeatherWaypointfromDistance(self, s):
        s_cum = 0
        for i in range(0, len(self.track_distance_list_weatherpoint)):
            s_cum += self.track_distance_list_weatherpoint[i][1]
            if s_cum > s:
                break

        return self.weatherpoint_list[i]
