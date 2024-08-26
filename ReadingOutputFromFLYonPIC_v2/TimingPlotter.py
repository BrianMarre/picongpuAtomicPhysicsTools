"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import reader
from . import Plotter

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import numpy.typing as npt
import copy

import typeguard

@typeguard.typechecked
class TimingPlotter(Plotter):
    # overwrite to ensure only TimingDataReader
    readerList : list[reader.TimingDataReader]

    # whether to also plot init and clean up as a separate steps
    plotCleanUpAndInit : bool

    # only show each periodStepTicks'
    numberXTicks : int

    # number y ticks
    numberYTicksSimulationTime : int
    numberYTicksStepTime : int

    def readAll(self) -> list[npt.NDArray]:
        """call read on all readers in the readerList"""
        readerDataSets = []
        for readerInList in self.readerList:
            readerDataSets.append(readerInList.read())
        return readerDataSets

    def plotTotalSimulationTime(self, axes_timeTotal, readerDataSets) -> Legend:
        maxStep = 0
        minStep = 0
        maxTime = 0
        for timingData, reader_ in zip(readerDataSets, self.readerList):
            if not self.plotCleanUpAndInit:
                timingData = timingData[1:-1]

            axes_timeTotal.plot(
                timingData['step'],
                timingData['time_total[msec]'] / 1000 / 60,
                ".-",
                label=reader_.dataSetName)

            maxStep = max(maxStep, np.max(timingData['step']))
            minStep = min(minStep, np.min(timingData['step']))
            maxTime = max(maxTime, np.max(timingData['time_total[msec]'] / 1000 / 60))

        axes_timeTotal.set_xlabel("PIC step")
        axes_timeTotal.set_ylabel("total time [min]")
        axes_timeTotal.set_ylim(bottom=0)

        xticks = list(range(minStep, maxStep, int(np.ceil((maxStep - minStep) / self.numberXTicks))))
        xticks[0] = minStep
        xticks[-1] = maxStep

        labels = copy.copy(xticks)

        if self.plotCleanUpAndInit:
            # set special labels for init cleanup
            labels[0] = "init"
            labels[-1] = "cleanup"

        axes_timeTotal.set_xticks(xticks, labels)
        yticks = np.linspace(0., maxTime, num=self.numberYTicksSimulationTime)
        axes_timeTotal.set_yticks(yticks)

        handles, labels = axes_timeTotal.get_legend_handles_labels()
        lgd_timeTotal= axes_timeTotal.legend(
            handles,
            labels,
            loc='upper right',
            bbox_to_anchor=(1.21, 1.05),
            fontsize='small')

        return lgd_timeTotal

    def plotStepTimes(self, axes_timeSteps, readerDataSets) -> Legend:
        maxStep = 0
        minStep = 0
        maxTime = 0

        for timingData, reader_ in zip(readerDataSets, self.readerList):
            if not self.plotCleanUpAndInit:
                timingData = timingData[1:-1]

            axes_timeSteps.plot(
                timingData['step'],
                timingData['time_step[msec]'] / 1000,
                ".-",
                label=reader_.dataSetName)

            maxStep = max(maxStep, np.max(timingData['step']))
            minStep = min(minStep, np.min(timingData['step']))
            maxTime = max(maxTime, np.max(timingData['time_step[msec]'] / 1000))

        axes_timeSteps.set_xlabel("PIC step")
        axes_timeSteps.set_ylabel("calculation step time [s]")
        axes_timeSteps.set_ylim(bottom=0)

        xticks = list(range(minStep, maxStep+1, int(np.ceil((maxStep - minStep) / self.numberXTicks))))
        xticks[0] = minStep
        xticks[-1] = maxStep
        labels = copy.copy(xticks)

        if self.plotCleanUpAndInit:
            labels[0] = "init"
            labels[-1] = "cleanup"

        axes_timeSteps.set_xticks(xticks, labels)
        yticks = np.linspace(0., maxTime, num=self.numberYTicksStepTime)
        axes_timeSteps.set_yticks(yticks)

        handles, labels = axes_timeSteps.get_legend_handles_labels()
        lgd_timeSteps = axes_timeSteps.legend(
            handles,
            labels,
            loc='upper right',
            bbox_to_anchor=(1.21, 1.05),
            fontsize='small')

        return lgd_timeSteps

    def plot(self) -> None:
        """plot step time and accumulated simulation time over time step"""

        figure, (axes_timeTotal, axes_timeSteps) = plt.subplots(2,1, dpi=200, layout="constrained")
        title = plt.suptitle("FLYonPIC Timing Data")

        readerDataSets = self.readAll()

        print(f"plotting Timing Data {self.plotName} ...")

        lgd_timeTotal = self.plotTotalSimulationTime(axes_timeTotal, readerDataSets)
        lgd_timeSteps = self.plotStepTimes(axes_timeSteps, readerDataSets)

        print("\t saving ...")

        plt.savefig(
            self.figureStoragePath + f"/TimingData_{self.plotName}",
            bbox_extra_artists=(lgd_timeSteps,lgd_timeTotal, title),
            bbox_inches='tight')
