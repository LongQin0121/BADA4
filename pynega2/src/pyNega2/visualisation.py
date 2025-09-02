# -*- coding: utf-8 -*-
"""
pyNega
Visualisation module
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

import matplotlib.pyplot as plt
from casadi import *


class Plotter(object):
    """pyNega plotter class

    This class is intended to plot executed and planned state and controls in real-time

    :ivar model: pyNega model class
    :ivar nu: number of controls
    :ivar nx: number of states
    :ivar ny: number of auxiliary variables
    :ivar keys: names of the variables to plot
    :ivar ax: matplotlib axes where the data will be plotted
    :ivar line: matplotlib lines where the data will be plotted
    :ivar fig: matplotlib figure
    """

    def __init__(self, model, auxiliar=None):

        # Default inputs
        if auxiliar is None:
            auxiliar = []

        # set model
        self.nx = model.x.shape[0]  # number of states
        self.nu = model.u.shape[0]  # number of controls
        self.ny = len(auxiliar)  # number of auxiliary variables

        # get keys of the variables from the model
        self.keys = [str(i) for i in vertsplit(model.u)] + [str(i) for i in vertsplit(model.x)] + auxiliar

        # initialise axes to plot data
        self.ax = dict.fromkeys(self.keys)

        # initialise plan and execution line for each axes
        self.line = {'plan': dict.fromkeys(self.keys), 'execution': dict.fromkeys(self.keys)}
        self.scat = {'plan': dict.fromkeys(self.keys), 'execution': dict.fromkeys(self.keys)}

        # create matplotlib figure
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.subplots_adjust(hspace=0.2, wspace=0.05)

        # associate each variable to a different ax
        N = max(self.nx, self.nu, self.ny)
        for i in range(self.nu):
            self.ax[self.keys[i]] = self.fig.add_subplot(3, N, i + 1)
        for i in range(self.nx):
            self.ax[self.keys[self.nu + i]] = self.fig.add_subplot(3, N, N + i + 1)
        for i in range(self.ny):
            self.ax[self.keys[self.nx + self.nu + i]] = self.fig.add_subplot(3, N, 2 * N + i + 1)

        # set parameters for the different axes (title, ticks, grid...) and initialise line
        for v, ax in self.ax.iteritems():
            ax.set_title(v)
            ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                           labelbottom='off', labelleft='off', labelright='off')
            ax.grid(True)

            for key in ['plan', 'execution']:
                self.line[key][v], = ax.plot([], [], linewidth=2)
                self.scat[key][v], = ax.plot([], [], marker="o", ls="")

    def plot(self, data, which, type='line', **kwargs):
        """Plot data. This function plots the data but not removes previous plots

        :param which: plan or execution
        :param data: data to plot
        :param type: type of plot (scatter or line)
        :type which: str
        :type type: str
        :type data: pandas DataFrame
        """
        if data is None:
            return
        if which in self.line.keys():
            for v in data.columns:
                if type == 'line':
                    if v in self.line[which].keys():
                        self.line[which][v], = self.ax[v].plot(data.index, data[v], **kwargs)
                        self.ax[v].relim()
                        self.ax[v].autoscale_view(True, True, True)
                        # raise Exception("Column " + v + " has not an associated line for plotting.
                        # Available options are: " + str(self.line[which].keys()))
                elif type == 'scatter':
                    if v in self.scat[which].keys():
                        self.scat[which][v], = self.ax[v].plot(data.index, data[v], marker="o", ls="", **kwargs)
                        self.ax[v].relim()
                        self.ax[v].autoscale_view(True, True, True)
                        # raise Exception("Column " + v + " has not an associated pathcolletion for plotting.
                        # Available options are: " + str(self.scat[which].keys()))
                else:
                    raise Exception("<type> argument not valid " + type + ". Available options are: <line, scatter>")
        else:
            raise Exception("<which> argument not valid " + which + ". Available options are: <execution, plan>")

        self.fig.canvas.draw()

    def update(self, data, which, type='line', **kwargs):
        """Update data. This function plots the data overwriting previous plots

        :param which: plan or execution
        :param data: data to plot
        :param type: type of plot (scatter or line)
        :type which: str
        :type type: str
        :type data: pandas DataFrame
        """
        if data is None:
            return
        if which in self.line.keys():
            for v in data.columns:
                if type == 'line':
                    if v in self.line[which].keys():
                        self.line[which][v].set_data(data.index, data[v])
                        self.ax[v].relim()
                        self.ax[v].autoscale_view(True, True, True)
                        # raise Exception("Column " + v + " has not an associated line for plotting.
                        # Available options are: " + str(self.line[which].keys()))
                elif type == 'scatter':
                    if v in self.scat[which].keys():
                        self.scat[which][v].set_data(data.index, data[v])
                        self.ax[v].relim()
                        self.ax[v].autoscale_view(True, True, True)
                        # raise Exception("Column " + v + " has not an associated pathcolletion for plotting.
                        # Available options are: " + str(self.scat[which].keys()))
                else:
                    raise Exception("<type> argument not valid " + type + ". Available options are: <line, scatter>")
        else:
            raise Exception("<which> argument not valid " + which + ". Available options are: <execution, plan>")

        self.fig.canvas.draw()
