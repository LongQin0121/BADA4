import contextlib
import datetime
import errno
import io
import math
import os
import pandas as pd
import pygrib
import requests
import shlex
import signal
import stat
import subprocess
import sys
import urllib
from functools import wraps


start_date = "20180415"
stop_date = "20180601"

INV = './get_inv.pl'
GRIB = './get_grib.pl'
GREP = 'grep'
SERVER_GRIB = 'https://nomads.ncdc.noaa.gov/data/rap130/'
SERVER_INV = 'https://nomads.ncdc.noaa.gov/data/rap130/'
SERVERANL_INV = 'https://nomads.ncdc.noaa.gov/data/rucanl/inventory/'
SERVERANL_GRIB = 'https://nomads.ncdc.noaa.gov/data/rucanl/'
WGRIB2 = 'wgrib2'
first = True

FIELDS = '|'.join(['VGRD', 'UGRD'])
LEVELS = '|'.join([str(lev) for lev in range(200, 1025, 25)])
CHIG = 304.0

# Coordinates to analyse
LAT = 39.859562
LON = 255.351938
NAMES = ['lat', 'lon', 'level', 'label']

global COORDS
global NAMES
COORDS = None  # dict(lat1=LAT, lat2=LAT, lon1=LON, lon2=LON)

TMPGRIB = 'out.grb2'
TMPGRIB2 = 'out2.grb2'
RESULT = 'result.csv'
PROFILES = 'profiles'


# ------------------------------------- FUNCTIONS -----------------------
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def cosd(deg):
    return math.cos(deg * math.pi / 180.0)


def sind(deg):
    return math.sin(deg * math.pi / 180.0)


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def curGrib(matching=False):
    if matching:
        cWgrib = ' '.join([WGRIB2, TMPGRIB, '-match', '\":(' + FIELDS + '):(' + LEVELS + ') mb:\"', '-grib', TMPGRIB2])
        with nostdout():
            pWgrib = subprocess.Popen(shlex.split(cWgrib), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            pWgrib.communicate()
            pWgrib.wait()
            os.rename(TMPGRIB2, TMPGRIB)

    cWgrib = ' '.join([WGRIB2, TMPGRIB, '-lola', str(LON) + ':1:1', str(LAT) + ':1:1', TMPGRIB2, 'grib'])
    with nostdout():
        pWgrib = subprocess.Popen(shlex.split(cWgrib), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        pWgrib.communicate()
        pWgrib.wait()
        return pWgrib.returncode


class TimeoutError(Exception):
    pass


def timeout(seconds=30, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def exists(path):
    r = requests.head(path)
    return r.status_code == 200


flatten = lambda z: [x for y in z for x in y]


def executable(filepath):
    st = os.stat(filepath)
    return bool(st.st_mode & stat.S_IEXEC)


@timeout(160, os.strerror(errno.ETIMEDOUT))
def downloadGrib(urlGrib, urlInv, download=False):
    print("Downloading " + urlGrib)
    if not download:
        if not exists(urlInv) or not exists(urlGrib):
            return 1

        cInv = './get_inv.pl ' + urlInv
        cGrep = 'grep -E \":(' + FIELDS + '):(' + LEVELS + ') mb:\"'
        cGrib = './get_grib.pl ' + urlGrib + ' out.grb2'

        pInv = subprocess.Popen(shlex.split(cInv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        pGrep = subprocess.Popen(shlex.split(cGrep), stdin=pInv.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pGrib = subprocess.Popen(shlex.split(cGrib), stdin=pGrep.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # blocks until execution is done
        pGrib.communicate()
        pGrib.wait()
        return pGrib.returncode
    else:
        if not exists(urlGrib):
            return 1

        urllib.urlretrieve(urlGrib, TMPGRIB)
        return 0


def changePermissions():
    st = os.stat(TMPGRIB)
    os.chmod(TMPGRIB, st.st_mode | stat.S_IEXEC)


def windProfile(msg):
    global COORDS
    global NAMES
    dataV = []
    levelV = []
    labelV = []
    level = None
    for m in msg:
        # get latitude and longitudes for this message
        data = m.data()
        level = m['level']
        name = m['shortName']
        value = data[0]

        dataV.append(value)
        levelV.append(level)
        labelV.append(name)

        if name == 'u':
            u = value
        elif name == 'v':
            v = value
            value = v * cosd(CHIG) + u * sind(CHIG)
            dataV.append(value)
            levelV.append(level)
            labelV.append('s')

    df = pd.DataFrame.from_dict(dict(lat=LAT, data=dataV, lon=LON, level=levelV, label=labelV)).set_index(NAMES)
    return df


def readGrib():
    changePermissions()
    gr = pygrib.open(TMPGRIB2)
    gr.seek(0)
    res = windProfile(gr)
    return res


# ------------------------------------- MAIN ROUTINE -----------------------
start = datetime.datetime.strptime(start_date, "%Y%m%d")
stop = datetime.datetime.strptime(stop_date, "%Y%m%d")
forecastTimes = [1, 3, 6]

makedir(PROFILES)

if os.path.isfile(RESULT):
    os.remove(RESULT)

while start < stop:
    # iterate for each time
    start = start + datetime.timedelta(hours=6)
    month = start.date().strftime('%Y%m')
    day = start.date().strftime('%Y%m%d')
    hour = start.time().strftime('%H%M')
    monthdir = os.path.join(PROFILES, str(month))
    daydir = os.path.join(monthdir, str(day))
    makedir(monthdir)
    makedir(daydir)
    for f in forecastTimes:
        # iterate for each forecast!
        path = month + '/' + day + '/rap_130_' + day + '_' + hour + '_' + str(f).zfill(3)
        urlGrib = SERVER_GRIB + path + '.grb2'
        urlInv = SERVER_INV + path + '.inv'
        if downloadGrib(urlGrib=urlGrib, urlInv=urlInv) == 0:
            changePermissions()
            if curGrib() == 0:
                forecast = readGrib()
                forecast.to_csv(os.path.join(daydir, str(hour) + '_forecast_' + str(f) + '.csv'), sep=',')
                dateAnl = start + datetime.timedelta(hours=f)
                month_ = dateAnl.date().strftime('%Y%m')
                day_ = dateAnl.date().strftime('%Y%m%d')
                hour_ = dateAnl.time().strftime('%H%M')
                # look for the corresponding analysis!
                path = month_ + '/' + day_ + '/rap_130_' + day_ + '_' + hour_ + '_000'
                urlGrib = SERVERANL_GRIB + path + '.grb2'
                urlInv = SERVERANL_INV + path + '.inv'
                if downloadGrib(urlGrib=urlGrib, urlInv=urlInv, download=True) == 0:
                    changePermissions()
                    if curGrib(matching=True) == 0:
                        analysis = readGrib()
                        analysis.to_csv(os.path.join(daydir, str(hour) + '_analysis_' + str(f) + '.csv'), sep=',')
                        # Compute metrics
                        error = (analysis - forecast).abs()
                        error = error.reset_index()
                        error = error[error['label'] == 's']
                        stats = error.describe()
                        # stats.loc['rmse_w'] = math.sqrt( ( error['v'] ** 2 + error['u'] ** 2) / len(error.index) )
                        stats.loc['rmse_s'] = math.sqrt((error['data'] ** 2).sum() / len(error.index))
                        stats.loc['datetime'] = start
                        stats.loc['forecast'] = str(f).zfill(3)
                        stats = stats['data'].to_frame()
                        stats = stats.T
                        stats.reset_index(inplace=True, drop=True)
                        stats.set_index(['datetime', 'forecast'], inplace=True)
                        stats.to_csv(RESULT, mode='a', header=first)

                        first = False
