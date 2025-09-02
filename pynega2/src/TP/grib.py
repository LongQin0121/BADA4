# -*- coding: utf-8 -*-
"""
GRIB Class
Developed @ICARUS by Ramon Dalmau, M.Sc
2017
"""

__author__ = 'Ramon Dalmau'
__copyright__ = "Copyright 2017, ICARUS"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ramon Dalmau"
__email__ = "ramon.dalmau@upc.edu"
__status__ = "Development"

import pyBada3.constants as const
import pyBada3.atmosphere as atm
from scipy.interpolate import interp1d
import numpy as np
from pyproj import Proj
import pygrib
import glob
import os
import datetime


# Class to read and manage Grib files (it uses the pygrib library!)
class Grib:
    def __init__(self, weather_directory, date, time, label, proj=None):
        filename = self.getClosestFile(weather_directory, date, time, label)
        self.Parse(filename)
        if proj is None:
            self.myProj = Proj("+proj=utm +zone=32, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        else:
            self.myProj = proj

    def Parse(self, filename):
        gr = pygrib.open(filename)
        self.data = {}
        self.data['U component of wind'] = gr.select(name='U component of wind', typeOfLevel='isobaricInhPa')
        self.data['V component of wind'] = gr.select(name='V component of wind', typeOfLevel='isobaricInhPa')
        self.data['Geopotential Height'] = gr.select(name='Geopotential Height',
                                                     typeOfLevel='isobaricInhPa')  # TODO: There should be a field with geometric height! --> read() to see parameters
        self.data['Temperature'] = gr.select(name='Temperature', typeOfLevel='isobaricInhPa')
        self.lats, self.lons = self.data['Temperature'][0].latlons()
        self.lats = np.unique(self.lats)
        self.lons = np.unique(self.lons)
        self.deltaLat = (self.lats.max() - self.lats.min()) / (self.lats.size - 1)
        self.deltaLon = (self.lons.max() - self.lons.min()) / (self.lons.size - 1)
        self.levels = self.getLevels('Temperature')

    def getIndex(self, waypoint):
        lat, lon = waypoint

        ila = self.find_nearest_above(self.lats, lat)
        ilo = self.find_nearest_above(self.lons, lon)

        if ilo is None:
            ilo = 0

        return ila, ilo

    def getLevelsIndex(self, p):
        # find nearest levels
        iAb = self.find_nearest_above(self.levels, p)

        if iAb is None:
            iAb = -1

        iBl = iAb - 1

        return iAb, iBl

    def getValues(self, field, waypoint, indexLateral, level=None):
        if level is None:
            msg = self.data[field]
        else:
            msg = [self.data[field][level]]

        ila, ilo = indexLateral
        lat, lon = waypoint

        # special case, last point acheieved
        if ilo == 0:
            p = [[self.lats[ila - 1], self.lons[ilo - 1], None], [self.lats[ila], self.lons[ilo - 1], None],
                 [self.lats[ila - 1], 0.0, None], [self.lats[ila], 0.0, None]]

            # p = [[self.lats[ila - 1], self.lons[ilo - 1], None], [self.lats[ila], self.lons[ilo - 1], None],
            #     [self.lats[ila - 1], self.lons[ilo - 1] + self.deltaLon, None], [self.lats[ila], self.lons[ilo - 1] + self.deltaLon, None]]
        else:
            p = [[self.lats[ila - 1], self.lons[ilo - 1], None], [self.lats[ila], self.lons[ilo - 1], None],
                 [self.lats[ila - 1], self.lons[ilo], None], [self.lats[ila], self.lons[ilo], None]]

        res = np.empty(0)
        for m in msg:
            p[0][2] = m.data(lat1=p[0][0], lon1=p[0][1], lat2=p[0][0], lon2=p[0][1])[0]  # m.values[ila - 1, ilo - 1]
            p[1][2] = m.data(lat1=p[1][0], lon1=p[1][1], lat2=p[1][0], lon2=p[1][1])[0]  # m.values[ila, ilo - 1]
            p[2][2] = m.data(lat1=p[2][0], lon1=p[2][1], lat2=p[2][0], lon2=p[2][1])[0]  # m.values[ila - 1, ilo]
            p[3][2] = m.data(lat1=p[3][0], lon1=p[3][1], lat2=p[3][0], lon2=p[3][1])[0]  # m.values[ila, ilo]
            if ilo == 0:
                p[2][1] = 360.0
                p[3][1] = 360.0

            res = np.append(res, [self.bilinear_interpolation(lat, lon, p)])

        return res

    def getMagitude(self, magnitude, waypoint, h, indexLateral=None, indexVertical=None):
        # get pressure at the current pressure altitude
        p = atm.hp2delta(h) * const.p_0 / 100.0

        if indexLateral is None:
            indexLateral = self.getIndex(waypoint)

        if indexVertical is None:
            indexVertical = self.getLevelsIndex(p)

        # get index of the closest vertical levels
        iAb, iBl = indexVertical

        # find temperatures above and below the current pressure level
        tAb = self.getValues(magnitude, waypoint, indexLateral, iAb)[0]
        tBl = self.getValues(magnitude, waypoint, indexLateral, iBl)[0]

        x = np.array([self.levels[iBl], self.levels[iAb]], dtype=np.float64)
        y = np.array([tBl, tAb], dtype=np.float64)

        # create interpolator
        f = interp1d(x, y, fill_value='extrapolate')

        return f(p)

    def getLevels(self, field):
        msg = self.data[field]
        res = np.empty(0)
        for m in msg:
            res = np.append(res, [m['level']])

        return res

    def getWaypointWeather(self, waypoint, h=None):
        indexLateral = self.getIndex(waypoint)

        if h is None:
            Wn = self.getValues('V component of wind', waypoint, indexLateral)
            We = self.getValues('U component of wind', waypoint, indexLateral)
            tau = self.getValues('Temperature', waypoint, indexLateral)
            geopot_alt = self.getValues('Geopotential Height', waypoint, indexLateral)

            weather = np.empty((tau.size, 0))
            weather = np.column_stack((weather, Wn))
            weather = np.column_stack((weather, We))
            weather = np.column_stack((weather, tau))
            weather = np.column_stack((weather, geopot_alt))

        else:
            Wn = self.getMagitude('V component of wind', waypoint, h, indexLateral)
            We = self.getMagitude('U component of wind', waypoint, h, indexLateral)
            tau = self.getMagitude('Temperature', waypoint, h, indexLateral)
            geopot_alt = self.getValues('Geopotential Height', waypoint, indexLateral)
            weather = (Wn, We, tau, geopot_alt)

        return weather

    def getWeather(self, lat, lon, h):
        waypoint = (lat, lon)
        indexLateral = self.getIndex(waypoint)

        Wn = self.getMagitude('V component of wind', waypoint, h, indexLateral)
        We = self.getMagitude('U component of wind', waypoint, h, indexLateral)
        tau = self.getMagitude('Temperature', waypoint, h, indexLateral)
        weather = (Wn, We, tau)

        return weather

    def getData(self, type, lat, lon, h):
        waypoint = (lat, lon)
        indexLateral = self.getIndex(waypoint)

        return self.getMagitude(type, waypoint, h, indexLateral)

    def getRouteWeather(self, route):
        weather = []
        for w in route:
            weather += [self.getWaypointWeather(w)]

        return weather

    def delta(self, h, lat, lon):
        return atm.hp2delta(h)

    def theta(self, h, lat, lon):
        return self.getMagitude(h, lat, lon, 'Temperature') / const.tau_0

    def wn(self, h, lat, lon):
        return self.getMagitude(h, lat, lon, 'V component of wind')

    def we(self, h, lat, lon):
        return self.getMagitude(h, lat, lon, 'U component of wind')

    def PlotInAx(self, ax1, level=0):
        LATS, LONS = self.data['U component of wind'][level].latlons()

        U = np.asmatrix(self.data['U component of wind'][level].values)
        V = np.asmatrix(self.data['V component of wind'][level].values)

        X, Y = self.myProj(LONS, LATS)
        X = np.asmatrix(X)
        Y = np.asmatrix(Y)

        return ax1.barbs(X / 1852.0, Y / 1852.0, U, V, length=4, pivot='middle')

    def getClosestFile(self, path, date, time, label):
        targetDate = datetime.datetime.strptime(str(date) + ' ' + str(time), '%y%m%d %H%M%S')
        availFolders = os.listdir(path)

        if len(availFolders) == 0:
            return None

        availDates = [datetime.datetime.strptime(x, '%Y%m%d') for x in availFolders]
        before = self.nearestDateBefore(availDates, targetDate)

        if before is not None:
            dateFolder = os.path.join(path, availFolders[availDates.index(before)])
        else:
            dateFolder = os.path.join(path, availFolders[availDates.index(min(availDates))])

        # Label is what it goes before the _4_ in the weather files (and path the path to the weather files)
        files = glob.glob(os.path.join(dateFolder, label + '_4_*'))
        if len(files) == 0:
            return None

        availDates = [file.split('_4_')[1].split('.')[0].split('_') for file in files]
        availDates = [
            datetime.datetime.strptime(' '.join(date[:2]), '%Y%m%d %H%M') + datetime.timedelta(hours=int(date[-1])) for
            date in availDates]
        before = self.nearestDateBefore(availDates, targetDate)

        if before is not None:
            file = files[availDates.index(before)]
        else:
            file = files[availDates.index(min(availDates))]

        return file

    def nearestDateBefore(self, items, pivot):
        items = [x for x in items if x < pivot]
        if len(items) > 0:
            return min(items, key=lambda x: abs(x - pivot))
        else:
            return None

    def find_nearest_above(self, my_array, target):
        diff = my_array - target
        mask = np.ma.less_equal(diff, 0)
        # We need to mask the negative differences and zero
        # since we are looking for values above
        if np.all(mask):
            return None  # returns None if target is greater than any value

        masked_diff = np.ma.masked_array(diff, mask)
        return masked_diff.argmin()

    def bilinear_interpolation(self, x, y, points):
        """Interpolate (x,y) from values associated with four points.

        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.

            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0

        """
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

        points = sorted(points)  # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
                ) / ((x2 - x1) * (y2 - y1) + 0.0)
