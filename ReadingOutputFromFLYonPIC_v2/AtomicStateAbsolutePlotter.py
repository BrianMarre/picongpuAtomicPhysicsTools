"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import AtomicStatePlotter
from .SCFlyTools import AtomicConfigNumberConversion as conv

import typeguard
import tqdm
import matplotlib.pyplot as plt
import numpy as np

@typeguard.typechecked
class AtomicStateAbsolutePlotter(AtomicStatePlotter):
    axisScale : str = ""

    def plot(self) -> None:
        """plot atomic populations on logarithmic scale"""
        colorChargeStates = self.getChargeStateColors()

        data = self.readData()

        # prepare plot
        figure = plt.figure(dpi=300)
        axes = figure.add_subplot(111)
        axes.set_title("AtomicPopulation Data: " + self.plotName)
        axes.set_xlabel("time[s]")
        axes.set_ylabel("relative abundance")

        if self.axisScale != "":
            axes.set_yscale(self.axisScale)

        axes.set_ylim((self.minimumRelativeAbundance, 1.))

        maxTime = 0

        for entry, readerListEntry, speciesDescriptor, linestyle in zip(data, self.readerList, self.speciesDescriptorList, self.plotLineStyles):
            if isinstance(readerListEntry, list):
                reader = readerListEntry[0]
                for i in readerListEntry[1:]:
                    if i.dataSetName != reader.dataSetName:
                        raise ValueError("inconsistent data set Name in group of readers")
            else:
                reader = readerListEntry

            print(f"plotting {reader.dataSetName} ...")

            mean, stdDev, axisDict, timeSteps, atomicStates = entry

            numberAtomicStates = np.shape(mean)[axisDict['atomicState']]
            maxTime = max(maxTime, np.max(timeSteps))

            #get bar width
            widthBars = np.empty_like(timeSteps)
            widthBars[:-1] = timeSteps[1:] - timeSteps[:-1]
            widthBars[-1] = widthBars[-2]

            chargeStates = np.fromiter(map(
                    lambda atomicStateCollectionIndex: conv.getChargeState(
                        atomicStateCollectionIndex,
                        speciesDescriptor.atomicNumber,
                        speciesDescriptor.numberLevels),
                    atomicStates), dtype="u1")

            chargeStateMask = np.fromiter(map(lambda chargeState: chargeState in self.chargeStatesToPlot, chargeStates), dtype=np.bool_)

            aboveMinimumAbundanceMask = np.any(
                (mean + stdDev) > self.minimumRelativeAbundance,
                axis=axisDict["timeStep"])

            atomicStatePlottingMask = np.logical_and(chargeStateMask, aboveMinimumAbundanceMask)
            del aboveMinimumAbundanceMask
            del chargeStateMask

            for collectionIndex, tuple_ in tqdm.tqdm(enumerate(zip(chargeStates, atomicStatePlottingMask))):
                chargeState, plotMask = tuple_
                if plotMask:
                    stateAxis = axisDict["atomicState"]

                    # plot mean
                    axes.plot(
                        timeSteps,
                        np.take(mean, collectionIndex, axis=stateAxis),
                        linewidth=1,
                        alpha=0.5,
                        color=colorChargeStates[chargeState],
                        linestyle = linestyle,
                        label=f"[{reader.dataSetName}] chargeState {chargeState}")

                    # plot standard deviation
                    axes.bar(
                        timeSteps,
                        2 * np.take(stdDev, collectionIndex, axis=stateAxis),
                        width=widthBars,
                        bottom = (
                            np.take(mean, collectionIndex, axis=stateAxis)
                            - np.take(stdDev, collectionIndex, axis=stateAxis)),
                        align='center',
                        color=colorChargeStates[chargeState],
                        alpha=0.2)

        axes.set_xlim((0,maxTime))
        handles, labels = axes.get_legend_handles_labels()
        uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

        print("saving...")
        plt.savefig(f"{self.figureStoragePath}/AbsoluteAbundance_{self.plotName}",
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(figure)
