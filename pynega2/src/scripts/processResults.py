import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyBada3.atmosphere as atmosphere
import pyBada3.performance as perf
from scipy import integrate

__version__ = '1.0'
states = ['t', 'v', 'h', 'energy']

aircraft = perf.bada4('A320-214.xml')
atm = atmosphere.Atmosphere(ISA=True)


def calculateMetrics(df):
    df['f'] = np.nan
    df['eD'] = np.nan
    df['eT'] = np.nan

    for index, row in df.iterrows():
        # ------ calculate atmospheric conditions ------
        delta = atm.delta(h=row['h'])
        theta = atm.theta(h=row['h'])
        sigma = delta / theta

        M = atm.tas2Mach(v=row['v'], theta=theta)

        # ------ calculate minimum thrust ------
        Tmin = aircraft.TMin(delta=delta, theta=theta, M=M, h=row['h'])

        # ------ calculate fuel flow ---------
        df.loc[index, 'f'] = aircraft.ff(delta=delta, theta=theta, M=M, T=row['T'], v=row['v'], h=row['h'])

        # ------ calculate energy added with thrust -----
        TExtra = max(row['T'] - Tmin, 0.0) / (aircraft.MREF * 9.80665)
        df.loc[index, 'eT'] = TExtra * row['v']

        # ------ calculate energy removed with speed-brakes ------
        CD = row['beta'] * 0.005
        DExtra = max(aircraft.D(row['v'], sigma, CD), 0.0) / (aircraft.MREF * 9.80665)
        df.loc[index, 'eD'] = DExtra * row['v']

    return df


def energy(v, h):
    return 0.5 * v * v / 9.80665 + h


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("results", help="Results root directory.", type=str)

    # Positional optional arguments
    parser.add_argument("-o", "--output", help="Output file", type=str, default='results.csv')
    parser.add_argument("-p", "--plot", help="Plot results", type=int, default=0)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    for asd in ['000']:
        __resdir__ = './results/'
        # __resdir__  = './0_results/bosss-two/results/' + asd

        for g in [d for d in os.listdir(__resdir__) if os.path.isdir(os.path.join(__resdir__, d))]:
            gDir = os.path.join(__resdir__, g)
            __resfile__ = os.path.join(__resdir__, g + '.csv')

            if os.path.exists(__resfile__):
                os.remove(__resfile__)

            outDir = os.path.join(gDir, 'output')
            months = os.listdir(outDir)
            first = True
            for month in months:
                print("> Processing " + str(month) + '...')
                monthDir = os.path.join(outDir, month)
                days = os.listdir(monthDir)
                for day in days:
                    print("> > Processing " + str(day) + '...')
                    dayDir = os.path.join(monthDir, day)
                    hours = os.listdir(dayDir)
                    for hour in hours:
                        print("> > > Processing " + str(day) + ' ' + str(hour) + '...')
                        plan = None
                        flown = None

                        hourDir = os.path.join(dayDir, hour)
                        planFile = os.path.join(hourDir, '00_plan.csv')
                        flownFile = os.path.join(hourDir, 'flown.csv')
                        logFile = os.path.join(hourDir, 'pyNega.log')

                        hour, forecast = hour.split('_')
                        date = pd.to_datetime(day + hour, format='%Y%m%d%H%M')

                        # ----------------- read INITIAL plan file -----------------------------

                        if os.path.exists(planFile):
                            plan = pd.read_csv(planFile, sep=',').set_index('s').select_dtypes(include=np.number)
                            plan['energy'] = energy(plan['v'], plan['h'])
                        else:
                            continue

                        # ----------------- read flown file -----------------------------
                        if os.path.exists(flownFile):
                            flown = pd.read_csv(flownFile, sep=',').set_index('s').select_dtypes(include=np.number)
                            flown['energy'] = energy(flown['v'], flown['h'])
                        else:
                            continue

                        # ----------------- read log file -----------------------------
                        if os.path.exists(logFile):
                            log = pd.read_csv(logFile, sep='-',
                                              names=['datetime', 'module', 'status', 'message']).set_index('datetime')
                        else:
                            continue

                        # ----------------- read log file -----------------------------
                        err = (flown[states] - plan[states]).abs()

                        # ----------------- calculate cost -----------------------------
                        flown = calculateMetrics(flown)
                        plan = calculateMetrics(plan)

                        fuel = integrate.trapz(flown['f'], flown['t'])
                        fuelPlan = integrate.trapz(plan['f'], plan['t'])
                        eTPlan = integrate.trapz(plan['eT'], plan['t'])  # - integrate.trapz(plan['eT'], plan['t'])
                        eT = integrate.trapz(flown['eT'], flown['t'])  # - integrate.trapz(plan['eT'], plan['t'])
                        eD = integrate.trapz(flown['eD'], flown['t'])  # - integrate.trapz(plan['eD'], plan['t'])

                        errStats = err.describe()
                        errStats.loc['final_error'] = err.loc[0]
                        errStats.loc['datetime'] = str(date)
                        errStats.loc['fuel'] = str(fuel)
                        errStats.loc['ExtraFuel'] = str(fuel - fuelPlan)
                        errStats.loc['ExtraFuel%'] = str((fuel - fuelPlan) * 100.0 / fuelPlan)
                        errStats.loc['eT'] = str(eT)
                        errStats.loc['ExtraET'] = str(eT - eTPlan)
                        errStats.loc['eD'] = str(eD)
                        errStats.loc['ExtraED'] = str(eD)
                        errStats.loc['forecast'] = str(forecast)
                        errStats = errStats.transpose()

                        errStats.reset_index(inplace=True, drop=False)
                        errStats.set_index(['datetime', 'forecast', 'index'], inplace=True)
                        errStats.to_csv(__resfile__, mode='a', header=first)
                        print("> > > ...OK")
                        first = False

    sys.exit(0)
