"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import StateAbsolutePlotter
from .reader import StateType

import typeguard
import tqdm

import matplotlib.pyplot as plt
import numpy as np

@typeguard.typechecked
class ChargeStateAbsolutePlotter(StateAbsolutePlotter):
    def plot(self) -> None:
        """plot charge states relative abundance on logarithmic scale"""

        colorChargeStates = self.getChargeStateColors()

        data = self.readData(StateType.CHARGE_STATE)

        # prepare plot
        figure = plt.figure(dpi=300)
        axes = figure.add_subplot(111)
        axes.set_title("ChargeState population Data: " + self.plotName)
        axes.set_xlabel("time[s]")
        axes.set_ylabel("relative abundance")

        if self.axisScale != "":
            axes.set_yscale(self.axisScale)

        axes.set_ylim((self.minimumRelativeAbundance, 1.))

        maxTime = 0
        print("plotting charge states absolute ...")

        for entry, readerListEntry, linestyle in zip(data, self.readerList, self.plotLineStyles):
            if isinstance(readerListEntry, list):
                reader = readerListEntry[0]
                for i in readerListEntry[1:]:
                    if i.dataSetName != reader.dataSetName:
                        raise ValueError("inconsistent data set Name in group of readers")
            else:
                reader = readerListEntry

            print(f"\t plotting {reader.dataSetName}")

            mean, stdDev, axisDict, timeSteps, chargeStates = entry

            maxTime = max(maxTime, np.max(timeSteps))

            #get bar width
            widthBars = np.empty_like(timeSteps)
            widthBars[:-1] = timeSteps[1:] - timeSteps[:-1]
            widthBars[-1] = widthBars[-2]

            chargeStateMask = np.fromiter(
                map(lambda chargeState: chargeState in self.chargeStatesToPlot, chargeStates), dtype=np.bool_)

            aboveMinimumAbundanceMask = np.any(
                (mean + stdDev) > self.minimumRelativeAbundance,
                axis=axisDict["timeStep"])

            plottingMask = np.logical_and(chargeStateMask, aboveMinimumAbundanceMask)
            del aboveMinimumAbundanceMask
            del chargeStateMask

            for chargeState, plotMask in tqdm.tqdm(zip(chargeStates, plottingMask)):
                stateAxis = axisDict["chargeState"]

                if plotMask:
                    # plot mean
                    axes.plot(
                        timeSteps,
                        np.take(mean, chargeState, axis=stateAxis),
                        linewidth=1,
                        alpha=0.5,
                        color=colorChargeStates[chargeState],
                        linestyle = linestyle,
                        label=f"[{reader.dataSetName}] chargeState {chargeState}")

                    # plot standard deviation
                    axes.bar(
                        timeSteps,
                        2 * np.take(stdDev, chargeState, axis=stateAxis),
                        width=widthBars,
                        bottom = (
                            np.take(mean, chargeState, axis=stateAxis)
                            - np.take(stdDev, chargeState, axis=stateAxis)),
                        align='center',
                        color=colorChargeStates[chargeState],
                        alpha=0.2)

        axes.set_xlim((0,maxTime))
        handles, labels = axes.get_legend_handles_labels()
        uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

        print("\t saving...")
        plt.savefig(self.figureStoragePath + "AbsoluteAbundance_ChargeStates_" + self.plotName,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(figure)