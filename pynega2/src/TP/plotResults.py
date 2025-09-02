# -*- coding: utf-8 -*-
"""
Plot generation module
"""

__author__ = "Technical University of Catalonia - BarcelonaTech (UPC)"

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import time


# Plotting function.
# @filePath: the .csv input file.
# @plotPath: the path to save the plot.
# @option: [0, 1, 2] -> works only for 0 in example #01.
def plot(df, plotPath, fig_num, option=0):

    startTime = time.time()
    # 0 for TAS, CAS, Mach and Altitude vs. Distance to go
    # 1 for Tmax, Tidle, throttle and speedbreakes vs. Distance to go
    # 2 for gamma (flight path angle), nz (vertical load factor), and VS (vertical speed) vs. Distance to go
    plot_selector = option

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    dist = list(df["s[NM]"].tolist())  # Distance to go [NM]
    dist = [float(element) for element in dist]
    if plot_selector == 0:
        alt = list(df["h[ft]"].tolist())  # Altitude [ft]
        alt = [float(element) / 100 for element in alt]  # Altitude [FL]
        TAS = list(df["TAS[kt]"].tolist())  # True Airspeed [kt]
        CAS = list(df["CAS[kt]"].tolist())  # Calibrated Airspeed [kt]
        M = list(df["mach[-]"].tolist())  # Mach [-]
        TAS = [float(element) for element in TAS]
        CAS = [float(element) for element in CAS]
        M = [float(element) for element in M]

        # Create a new Figure
        fig1 = plt.figure(fig_num)

        # Add ax1
        ax1 = fig1.add_subplot(111)
        line1 = ax1.plot(dist, alt, '-', linewidth=3.0, color=colors["blue"], label="$h_{p}$")
        line2 = ax1.plot(dist, TAS, '-', linewidth=3.0, color=colors["green"], label="$v_{TAS}$")
        line3 = ax1.plot(dist, CAS, '-', linewidth=3.0, color=colors["black"], label="$v_{CAS}$")
        ax1.axhline(y=350, linestyle="--", linewidth=1.0, color=colors["black"])  # VMO
        plt.ylabel("Altitude [FL] / Airspeed [kt]", size=14)
        ax1.tick_params(axis='both', which='major', labelsize=13)

        # Add ax2, that shares the x-axis with ax1
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        line4 = ax2.plot(dist, M, '-', linewidth=3.0, color=colors["red"], label="M")
        ax2.axhline(y=0.82, linestyle="--", linewidth=1.0, color=colors["red"])  # MMO
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("Mach [-]", size=14)
        plt.xlabel("Distance to go [NM]", size=14)
        ax2.tick_params(axis='both', which='major', labelsize=13)

        lns = line1 + line2 + line4 + line3
        labs = [l.get_label() for l in lns]

        # Change "loc" value to change the legend location
        ax1.legend(lns, labs, loc=3, fontsize=16)
        plt.grid(alpha=0.65, linestyle='dotted')

        # Placing the text for the VMO and MMO lines
        # (the first 2 numbers correspond to X and Y coordinated in the plot; change them accordingly)
        plt.text(128, 0.832, 'MMO', fontsize=12.0, color=colors["red"])
        plt.text(128, 0.63, 'VMO', fontsize=12.0, color=colors["black"])

        # Set the limits in the y-axis, change if required or remove
        ax1.set_ylim(0, 510)
        ax2.set_ylim(0, 0.9)

        plt.savefig(plotPath, format='png')

        print('Plot was created successfully in %.3f sec.' % (time.time() - startTime))

    elif plot_selector == 1:

        T = list(df["Thrust[daN]"].tolist())  # Thrust [daN]
        T_idle = list(df["Tidle[daN]"].tolist())  # Idle Thrust[daN]
        T_max = list(df["Tmax[daN]"].tolist())  # Maximum Thrust[kt]
        throttle = list(df["THR[%]"].tolist())  # Throttle [-]
        spdbrk = list(df["sb[%]"].tolist())  # Speedbrakes [-]
        T = [float(element) for element in T]
        T_idle = [float(element) for element in T_idle]
        T_max = [float(element) for element in T_max]
        throttle = [float(element) for element in throttle]
        spdbrk = [float(element) for element in spdbrk]

        # Create a new Figure
        fig1 = plt.figure(fig_num)

        # Add ax1
        ax1 = fig1.add_subplot(111)
        line1 = ax1.plot(dist, T, '-', linewidth=3.0, color=colors["blue"], label="$T$")
        line2 = ax1.plot(dist, T_idle, '-', linewidth=3.0, color=colors["green"], label="$T_{idle}$")
        line3 = ax1.plot(dist, T_max, '-', linewidth=3.0, color=colors["black"], label="$T_{max}$")
        plt.ylabel("Thrust [daN]", size=14)
        ax1.tick_params(axis='both', which='major', labelsize=13)

        # Add ax2, that shares the x-axis with ax1
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        line4 = ax2.plot(dist, throttle, '-', linewidth=3.0, color=colors["red"], label="THR")
        line5 = ax2.plot(dist, spdbrk, '-', linewidth=3.0, color=colors["red"], label="Sb")

        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("Speedbrakes [%] / Throttle [%]", size=14)
        plt.xlabel("Distance to go [NM]", size=14)
        ax2.tick_params(axis='both', which='major', labelsize=13)

        lns = line1 + line2 + line3 + line4 + line5
        labs = [l.get_label() for l in lns]

        # Change "loc" value to change the legend location
        ax1.legend(lns, labs, loc=3, fontsize=16)
        plt.grid(alpha=0.65, linestyle='dotted')

        # Set the limits in the y-axis, change if required or remove
        ax1.set_ylim(0, 200)
        ax2.set_ylim(0, 1.1)

        plt.savefig(plotPath, format='png')

    else:
        VS = list(df["VS[ft/min]"].tolist())  # Vertical Speed [ft/min]
        gamma = list(df["FPA_a[\N{DEGREE SIGN}]"].tolist())  # Aero. flight path angle [degrees]
        nz = list(df["nz[g]"].tolist())  # Vertical load factor [g]
        VS = [float(element) for element in VS]
        gamma = [float(element) for element in gamma]
        nz = [float(element) for element in nz]

        # Create a new Figure
        fig1 = plt.figure(fig_num)

        # Add ax1
        ax1 = fig1.add_subplot(111)
        line1 = ax1.plot(dist, VS, '-', linewidth=3.0, color=colors["blue"], label="$V_{S}$")
        plt.ylabel("Vertical speed [ft/min]", size=14)
        ax1.tick_params(axis='both', which='major', labelsize=13)

        # Add ax2, that shares the x-axis with ax1
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        line2 = ax2.plot(dist, gamma, '-', linewidth=3.0, color=colors["red"], label="$\gamma_{a}$")
        line3 = ax2.plot(dist, nz, '-', linewidth=3.0, color=colors["red"], label="$n_{z}$")

        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("Flight path angle [deg.] / Vertical load factor [g*10]", size=14)
        plt.xlabel("Distance to go [NM]", size=14)
        ax2.tick_params(axis='both', which='major', labelsize=13)

        lns = line1 + line2 + line3
        labs = [l.get_label() for l in lns]

        # Change "loc" value to change the legend location
        ax1.legend(lns, labs, loc=3, fontsize=16)
        plt.grid(alpha=0.65, linestyle='dotted')

        # Set the limits in the y-axis, change if required or remove
        ax1.set_ylim(-10000, 10000)
        ax2.set_ylim(-10, 10)

        plt.savefig(plotPath, format='png')
