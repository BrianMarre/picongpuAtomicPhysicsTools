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

@typeguard.typechecked
class AtomicStateAbsolutePlotter(AtomicStatePlotter):

    def plot_absolute(self, data : list) -> None:
        """plot atomic populations on logarithmic scale"""
        colorChargeStates = self.getChargeStateColors()

        data = self.readData()

        # prepare plot
        figure = plt.figure(dpi=300)
        axes = figure.add_subplot(111)
        axes.set_title("AtomicPopulation Data: " + self.dataName)
        axes.set_xlabel("time[s]")
        axes.set_ylabel("relative abundance")
        axes.set_yscale('log')
        axes.set_ylim((1e-7,1))

        maxTime = 0

        for entry, speciesDescriptor in tqdm.tqdm(zip(data, self.speciesDescriptorList)):
            mean, stdDev, axisDict, timeSteps, atomicStates = entry

            numberAtomicStates = np.shape(mean)[axisDict['atomicState']]
            maxTime = max(maxTime, np.max(timeSteps))

            #get bar width
            widthBars = np.empty_like(timeSteps)
            widthBars[:-1] = timeSteps[1:] - timeSteps[:-1]
            widthBars[-1] = widthBars[-2]

            chargeStateMask = np.fromiter(map(
                    lambda collectionIndex: return (conv.getChargeState(
                            atomicStates[collectionIndex],
                            speciesDescriptor.atomicNumber,
                            speciesDescriptor.numLevels) in self.chargeStatesToPlot)), range(numberAtomicStates))

            aboveMinimum()

            for collectionIndex in tqdm.tqdm(range(numberAtomicStates)):
                chargeState =

                if chargeState in self.chargeStatesToPlot:
                    ### plot mean value
                    axes.plot(timeSteps_FLYonPIC, mean[:, collectionIndex], linewidth=1, alpha=0.5,
                            color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState))

                    ### plot standard deviation
                    axes.bar(timeSteps_FLYonPIC, 2 * stdDev[:, collectionIndex], width=widthBars,
                        bottom = mean[:, collectionIndex] - stdDev[:, collectionIndex],
                        align='center', color=colorChargeStates[chargeState], alpha=0.2)

        haveSCFLYData = ((type(atomicPopulationData) == np.ndarray)
        and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
        and (type(timeSteps_SCFLY) == np.ndarray)
        and (axisDict_SCFLY != None))

        if haveSCFLYData:

            maxTime = max(maxTime, np.max(timeSteps_SCFLY))

            assert(axisDict_SCFLY['atomicState'] == 1), "wrong axis ordering in SCFLY data"
            assert(axisDict_SCFLY['timeStep'] == 0), "wrong axis ordering in SCFLY data"

            # number Iterations
            numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

            print("plotting SCFLY absolute ...")

            # for each atomic state
            for i, configNumber in enumerate(atomicConfigNumbers_SCFLY):
                chargeState = conv.getChargeState(configNumber, config.openPMDReaderConfig.atomicNumber, config.openPMDReaderConfig.numLevels)

                if chargeState in config.chargeStatesToPlot:
                    axes.plot(timeSteps_SCFLY, atomicPopulationData[:, i], linewidth=1, alpha=0.5, linestyle="--",
                            color=colorChargeStates[chargeState], label="[SCFLY] chargeState " + str(int(chargeState)))

        axes.set_xlim((0,maxTime))
        handles, labels = axes.get_legend_handles_labels()
        uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

        print("saving...")
        plt.savefig(config.figureStoragePath + "AtomicPopulationData_absolute_" + config.dataName,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(figure)

    # number of states to plot
    numberStatesToPlot : int
