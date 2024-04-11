"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 PIConGPU contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import matplotlib.pyplot as plt
import numpy as np
import copy

import Reader
import Config as cfg

import typeguard

@typeguard.typechecked
def plot(config : cfg.TimingDataPlot.TimingDataPlot):
    casesStepTimes = []
    for i, basePath in enumerate(config.caseBasePaths):
        sampleStepTimes = []
        for fileName in config.caseFileNames[i]:
            timingData = Reader.TimingData.readTimingData(basePath + fileName)
            sampleStepTimes.append(timingData)
        casesStepTimes.append(sampleStepTimes)


    figure, (axes_timeTotal, axes_timeSteps) = plt.subplots(2,1, dpi=200, layout="constrained")
    title = plt.suptitle("FLYonPIC Timing Data")

    # plot total run time
    maxStep = 0
    minStep = 0
    maxTime = 0
    for i, sampleStepTimes in enumerate(casesStepTimes):
        for sampleIdx, timingData in enumerate(sampleStepTimes):
            if not config.plotCleanUpAndInit:
                timingData = timingData[1:-1]

            axes_timeTotal.plot(
                timingData['step'],
                timingData['time_total[msec]'] / 1000 / 60,
                ".-",
                label=(config.caseDataNames[i] + ", "+ str(sampleIdx)))

            maxStep = max(maxStep, np.max(timingData['step']))
            minStep = min(minStep, np.min(timingData['step']))
            maxTime = max(maxTime, np.max(timingData['time_total[msec]'] / 1000 / 60))

    axes_timeTotal.set_xlabel("PIC step")
    axes_timeTotal.set_ylabel("total time [min]")
    axes_timeTotal.set_ylim(bottom=0)

    xticks = list(range(minStep, maxStep+1, int(np.ceil(maxStep / 10))))
    xticks[0] = minStep
    xticks[-1] = maxStep

    labels = copy.copy(xticks)

    if config.plotCleanUpAndInit:
        # set special labels for init cleanup
        labels[0] = "init"
        labels[-1] = "cleanup"

    axes_timeTotal.set_xticks(xticks, labels)
    yticks = np.linspace(0., maxTime, num=12)
    axes_timeTotal.set_yticks(yticks)

    handles, labels = axes_timeTotal.get_legend_handles_labels()
    lgd_timeTotal= axes_timeTotal.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.21, 1.05), fontsize='small')

    # plot per step calculation time
    maxStep = 0
    minStep = 0
    maxTime = 0
    for i, sampleStepTimes in enumerate(casesStepTimes):
        for sampleIdx, timingData in enumerate(sampleStepTimes):
            if not config.plotCleanUpAndInit:
                timingData = timingData[1:-1]

            axes_timeSteps.plot(
                timingData['step'],
                timingData['time_step[msec]'] / 1000,
                ".-",
                label=(config.caseDataNames[i] + ", " + str(sampleIdx)))
            maxStep = max(maxStep, np.max(timingData['step']))
            minStep = min(minStep, np.min(timingData['step']))
            maxTime = max(maxTime, np.max(timingData['time_step[msec]'] / 1000))

    axes_timeSteps.set_xlabel("PIC step")
    axes_timeSteps.set_ylabel("step calculation time [s]")
    axes_timeSteps.set_ylim(bottom=0)

    xticks = list(range(minStep, maxStep+1, int(np.ceil(maxStep / 10))))
    xticks[0] = minStep
    xticks[-1] = maxStep
    labels = copy.copy(xticks)

    if config.plotCleanUpAndInit:
        labels[0] = "init"
        labels[-1] = "cleanup"

    axes_timeSteps.set_xticks(xticks, labels)
    yticks = np.linspace(0., maxTime, num=13)
    axes_timeSteps.set_yticks(yticks)

    handles, labels = axes_timeSteps.get_legend_handles_labels()
    lgd_timeSteps = axes_timeSteps.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.21, 1.05), fontsize='small')

    plt.savefig(config.fileName, bbox_extra_artists=(lgd_timeSteps,lgd_timeTotal, title), bbox_inches='tight')
