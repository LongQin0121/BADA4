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
__version__ = '1.0'

import argparse
import os
import shlex
import subprocess
import xml.etree.ElementTree as ET

__config__ = '/home/niavisitor/Desktop/pyNega/config.tmp.xml'
__batch__ = '/home/niavisitor/Desktop/pyNega/configBatch.tmp.xml.s'
# __ci__         = 30
# __guidances__  = ['SbNMPC','AsNMPC2','INMPC','OpenLoop']
__resdir__ = '/home/niavisitor/Desktop/pyNega/results/'
# __execute__    = 'python batch.py configBatch.xml -ci ' + str(__ci__)
__cwd__ = os.getcwd()


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("guidance", help="Guidance strategy.", type=str)

    # Positional optional arguments
    parser.add_argument("-l", "--lam", help="Poisson parameter", type=float, default=0.5)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parseArguments()
    guidance = args.guidance
    __lam__ = args.lam


    def execute(call):
        print(shlex.split(call))
        Process = subprocess.Popen(shlex.split(call), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = Process.communicate()  # blocks until execution is done
        Process.wait()
        returncode = Process.returncode
        return returncode, stdout, stderr


    def setGuidance(profile, guidance, out):
        tree = ET.parse(profile)
        root = tree.getroot()
        g = root.find('guidance')
        if g is not None:
            g.text = guidance
        g = root.find('lam')
        if g is not None:
            g.text = str(__lam__)
        tree.write(out, encoding='utf-8', xml_declaration=True)


    def setPath(profile, path, out):
        tree = ET.parse(profile)
        root = tree.getroot()
        g = root.find('resDir')
        if g is not None:
            g.text = path
        g = root.find('configFile')
        if g is not None:
            g.text = os.path.join(__cwd__, 'config.xml')
        tree.write(out, encoding='utf-8', xml_declaration=True)


    setGuidance(__config__, guidance, os.path.join(__cwd__, 'config.xml'))
    setPath(__batch__, os.path.join(__resdir__, guidance), os.path.join(__cwd__, 'configBatch.xml'))
    # returncode, stdout, stderr = execute(__execute__)
    # os.remove(os.path.join(__cwd__, 'config.xml'))
    # os.remove(os.path.join(__cwd__, 'configBatch.xml'))
