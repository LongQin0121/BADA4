#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyNega
batch script for pyNega demonstration with a CDO
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

import shutil
import argparse
import os
import sys
from mpi4py import MPI
import subprocess
import shlex
import datetime
import calendar
import time
import xmltodict
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import tarfile
import logging
import xml.etree.ElementTree as ET
from scipy.stats import truncnorm
import numpy as np
import pandas as pd
from functools import wraps
import errno
import signal

np.random.seed(0)


class TimeoutError(Exception):
    pass


def timeout(seconds=30, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
            # print error_message

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


def setNumericalConstraints(profile):
    tree = ET.parse(profile)
    root = tree.getroot()

    types = ['gamma', 'v', 'h', 't']
    ub = ['1', '600', '37000', '2000']
    lb = ['-9', '200', '6000', '0']
    units = ['deg', 'kt', 'ft', None]

    for phase in root.findall('phase'):
        path = phase.find('path')
        if path is not None:
            for k, t in enumerate(types):
                constraint = ET.SubElement(parent=path, tag='constraint', attrib=dict(type=t))
                if units[k] is None:
                    ET.SubElement(parent=constraint, tag='lb').text = str(lb[k])
                    ET.SubElement(parent=constraint, tag='ub').text = str(ub[k])
                else:
                    ET.SubElement(parent=constraint, tag='lb', attrib=dict(units=units[k])).text = str(lb[k])
                    ET.SubElement(parent=constraint, tag='ub', attrib=dict(units=units[k])).text = str(ub[k])

    tree.write(profile, encoding='utf-8', xml_declaration=True)


def setSbBounds(profile):
    tree = ET.parse(profile)
    root = tree.getroot()
    for phase in root.findall('phase'):
        path = phase.find('path')
        beta = path.find(".//constraint[@type='beta']")
        if beta is not None:
            ub = beta.find('ub')
            if ub is None:
                ub = ET.SubElement(parent=beta, tag='ub')
            ub.text = '1.0'

    tree.write(profile, encoding='utf-8', xml_declaration=True)


def setTOD(profile, distance):
    tree = ET.parse(profile)
    root = tree.getroot()

    # distances of fixed phases must not be considered!
    phases = root.findall('phase')
    if len(phases) > 1:
        for phase in phases[1:]:
            distance -= float(phase.find('d').text)

    # first phase has flexible distance
    phase = phases[0]

    d = phase.find('d')
    for child in d[:]:
        d.remove(child)
    d.set('units', 'nm')
    d.text = str(distance)

    tree.write(profile, encoding='utf-8', xml_declaration=True)


def setTOD2(profile, distance):
    tree = ET.parse(profile)
    root = tree.getroot()

    # distances of fixed phases must not be considered!
    phases = root.findall('phase')
    if len(phases) > 1:
        for phase in phases[1:]:
            distance -= float(phase.find('d').text)

    # first phase has flexible distance
    phase = phases[0]

    d = phase.find('d')
    for child in d[:]:
        d.remove(child)
    d.set('units', 'nm')
    d.text = str(distance)

    tree.write(profile, encoding='utf-8', xml_declaration=True)


def allowThrust(profile):
    tree = ET.parse(profile)
    root = tree.getroot()

    # Allow thrust all along the descent
    for phase in root.findall('phase'):
        path = phase.find('path')
        Tmin = path.find(".//constraint[@type='Tmin']")
        if Tmin is not None:
            if Tmin.find('ub') is not None:
                Tmin.remove(Tmin.find('ub'))

    tree.write(profile, encoding='utf-8', xml_declaration=True)


def setCTA(profile, CTA):
    tree = ET.parse(profile)
    root = tree.getroot()

    phase = root.findall('phase')[-1]
    final = phase.find('final')
    constraint = final.find(".//constraint[@type='t']")
    constraint.find('lb').text = str(CTA)
    constraint.find('ub').text = str(CTA)

    tree.write(profile, encoding='utf-8', xml_declaration=True)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class MPILogFile(object):
    def __init__(self, comm, filename, mode):
        self.file_handle = MPI.File.Open(comm, filename, mode)
        self.file_handle.Set_atomicity(True)
        self.buffer = bytearray

    def write(self, msg):
        b = bytearray()
        b.extend(map(ord, msg))
        self.file_handle.Write_shared(b)

    def close(self):
        self.file_handle.Sync()
        self.file_handle.Close()


class MPIFileHandler(logging.FileHandler):
    def __init__(self, filename, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND, encoding=None, delay=0,
                 comm=MPI.COMM_WORLD):
        encoding = None
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        if delay:
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPILogFile(self.comm, self.baseFilename, self.mode)
        return stream

    def close(self):
        if self.stream:
            self.stream.close()
            self.stream = None


def createMPILogHandler(job_name, log_file, comm):
    logger = logging.getLogger(job_name + "[%i]" % comm.rank)
    logger.setLevel(logging.DEBUG)
    # create a shared file handler
    mh = MPIFileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    mh.setFormatter(formatter)
    logger.addHandler(mh)
    # create a console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    return logger


def header(i, date, forecast):
    return ' '.join([str(i), str(date), str(forecast), '- '])


def makeTarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def createXML(root, array, output):
    xml = parseString(dicttoxml(array, custom_root=root, attr_type=False).encode('utf-8')).toprettyxml()
    with open(output, 'w') as file:
        file.write(xml)


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("config", help="Configuration file.", type=str)

    # Positional optional arguments
    parser.add_argument("-v", "--verbose", help="verbosa mode", type=int, default=1)
    parser.add_argument("-ci", "--ci", help="cost idnex", type=float, default=15)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    return args


@timeout(500, os.strerror(errno.ETIMEDOUT))
def execute(call):
    Process = subprocess.Popen(shlex.split(call), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = Process.communicate()  # blocks until execution is done
    Process.wait()
    returncode = Process.returncode
    return returncode, stdout, stderr


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    args = parseArguments()
    fconf = args.config  # farm run configuration file
    verbose = args.verbose  # silent activated/deactivated (TODO)
    ci = args.ci  # cost index for the initial trajectory

    try:
        with open(fconf) as fd:
            conf = xmltodict.parse(fd.read())['config']
    except Exception as e:
        print('Error parsing profile. Error details: ' + str(e))

    resDir = conf['resDir']
    outDir = os.path.join(resDir, 'output')
    inDir = os.path.join(resDir, 'input')
    profileDir = conf['profileDir']
    originalConf = conf['configFile']
    print(originalConf)
    log = os.path.join(resDir, conf['log'])

    try:
        with open(originalConf) as fd:
            originalDict = xmltodict.parse(fd.read())['config']
    except Exception as e:
        print('Error parsing profile. Error details: ' + str(e))

    if rank == 0:
        if os.path.exists(log):
            os.remove(log)

        for dir in [resDir, outDir, inDir]:
            makedir(dir)

    comm.Barrier()

    logger = createMPILogHandler('pyNegaBatch', log, comm)

    tc = conf['testCases']['tc']
    if type(tc) is not list:
        tc = [tc]

    N = len(tc)  # total number of test cases to simulate

    # Parallelizable region (MPI)
    parallelLaunchs = (N / size) + 1

    simStartTimeFromEpoch = calendar.timegm(time.gmtime())
    simStartTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # block parallel launch
    comm.Barrier()

    if rank == 0:
        logger.info("# simulations:     " + str(N))
        logger.info("# processes:       " + str(size))
        logger.info("# parallel launch: " + str(parallelLaunchs))

    comm.Barrier()

    for round in range(parallelLaunchs):
        i = rank + round * size
        if i < len(tc):
            tci = tc[i]
            datei = datetime.datetime.strptime(str(tci['datetime']), '%Y-%m-%d %H:%M:%S')
            forecasti = tci['forecast']

            H = header(i, datei, forecasti)
            logger.info(H + 'Calculating target time and time window')

            month = datei.date().strftime('%Y%m')
            day = datei.date().strftime('%Y%m%d')
            hour = datei.time().strftime('%H%M')

            # set paths based on date time
            # -----------------------------------------------------------------------------------------------------
            pathi = os.path.join(str(month), str(day))
            tag = str(hour) + '_' + str(forecasti).zfill(2)

            inDiri = os.path.join(inDir, pathi, tag)
            outDiri = os.path.join(outDir, pathi, tag)

            # remove duplicated results
            # -----------------------------------------------------------------------------------------------------
            if os.path.isdir(outDiri):
                shutil.rmtree(outDiri)

            if os.path.isdir(inDiri):
                shutil.rmtree(inDiri)

            for dir in [outDiri, inDiri]:
                makedir(dir)

            profileForecast = os.path.join(profileDir, pathi, str(hour) + '_forecast_' + str(forecasti) + '.csv')
            profileAnalysis = os.path.join(profileDir, pathi, str(hour) + '_analysis_' + str(forecasti) + '.csv')
            profilei = os.path.join(inDiri, 'profile.xml')
            confi = os.path.join(inDiri, 'config.xml')

            dicti = originalDict.copy()
            dicti['forecast'] = profileForecast
            dicti['analysis'] = profileAnalysis
            dicti['out'] = outDiri

            # create files
            # -----------------------------------------------------------------------------------------------------
            shutil.copyfile(dicti['model'], profilei)

            dicti['model'] = profilei
            createXML('config', dicti, confi)

            # get minimum cost trajectory with flexible TOD. This gives the optimal TOD and the optimal time of arrival
            # -----------------------------------------------------------------------------------------------------
            call = 'pyNega-cdo.py ' + confi + ' -p 0 -v 0 -s 0 -c fuel -crz 1 -ci ' + str(
                ci) + ' -o target.csv -nlp ipopt'
            returncode, stdout, stderr = execute(call)

            if returncode:
                logger.error(H + 'Computing minimum cost trajectory')
                continue

            targetDf = pd.read_csv(os.path.join(outDiri, 'target.csv'), sep=',').set_index('s')
            target = targetDf.iloc[-1]['t']  # optimal time
            distance = targetDf.index[0] / 1852.0  # optimal distance

            # modify profile xml to fix the distance from TOD to metering fix
            # -----------------------------------------------------------------------------------------------------
            setTOD(profilei, distance)

            # get earliest trajectory with fixed TOD
            # -----------------------------------------------------------------------------------------------------
            # call = 'pyNega-cdo.py ' + confi + ' -p 0 -v 0 -s 0 -c earliest -o earliest.csv -nlp ipopt'
            # returncode, stdout, stderr = execute(call)

            # if returncode:
            #   logger.error(H + 'computing earliest')
            #   continue

            # earliestDf = pd.read_csv(os.path.join(outDiri,'earliest.csv'), sep=',').set_index('s')
            # earliest   = earliestDf.iloc[-1]['t'] # earliest time of arrival
            earliestDf = pd.read_csv(os.path.join(outDiri, 'target.csv'), sep=',').set_index('s')
            earliest = earliestDf.iloc[-1]['t']  # earliest time of arrival

            # get latest trajectory with fixed TOD
            # -----------------------------------------------------------------------------------------------------
            call = 'pyNega-cdo.py ' + confi + ' -p 0 -v 0 -s 0 -c latest -o latest.csv -nlp ipopt'
            returncode, stdout, stderr = execute(call)

            if returncode:
                logger.error(H + 'Computing latest')
                continue

            latestDf = pd.read_csv(os.path.join(outDiri, 'latest.csv'), sep=',').set_index('s')
            latest = latestDf.iloc[-1]['t']  # latest time of arrival

            # compute random CTA in between the feasible time window
            # -----------------------------------------------------------------------------------------------------
            if abs(latest - earliest) < 10.0:
                logger.error(H + 'Time window too small')
                continue

            # generate truncated normal distribution centered in the middle of the time window
            # -----------------------------------------------------------------------------------------------------
            X = get_truncated_normal(mean=abs(latest + earliest) / 2.0, sd=0.2 * abs(latest - earliest), low=earliest,
                                     upp=latest)

            CTA = X.rvs()
            logger.info(H + 'Time window [' + str(int(float(earliest))) + ',' + str(int(float(CTA))) + ',' + str(
                int(float(latest))) + ']')

            # modify profile with the time constraint and allow for thrust!
            # -----------------------------------------------------------------------------------------------------
            allowThrust(profilei)
            setSbBounds(profilei)
            setNumericalConstraints(profilei)
            setCTA(profilei, CTA)

            logger.info(H + 'Simulating ...')

            # simulate
            # -----------------------------------------------------------------------------------------------------
            call = 'pyNega-cdo.py ' + confi + ' -p 0 -v 0 -s 1 -c fuel'
            try:
                returncode, stdout, stderr = execute(call)
            except:
                returncode = 1

            if returncode:
                logger.error(H + 'Simulation failed')

    comm.Barrier()

sys.exit(0)
