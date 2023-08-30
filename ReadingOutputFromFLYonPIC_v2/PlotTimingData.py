import TimingData_Reader as reader
import matplotlib.pyplot as plt
import numpy as np
import copy

class Config_TimingData:
    def __init__(self, caseBasePaths, caseFileNames, caseDataNames, fileName):
        self.caseBasePaths = caseBasePaths
        self.caseFileNames = caseFileNames
        self.caseDataNames = caseDataNames
        self.fileName = fileName

def plot(config):
    casesStepTimes = []
    for i, basePath in enumerate(config.caseBasePaths):
        sampleStepTimes = []
        for fileName in config.caseFileNames[i]:
            timingData = reader.readTimingData(basePath + fileName)
            sampleStepTimes.append(timingData)
        casesStepTimes.append(sampleStepTimes)

    figure, (axes_timeTotal, axes_timeSteps) = plt.subplots(2,1, dpi=200, layout="constrained")
    title = plt.suptitle("FLYonPIC Timing Data")

    maxStep = 0
    maxTime = 0
    for i, sampleStepTimes in enumerate(casesStepTimes):
        for sampleIdx, timingData in enumerate(sampleStepTimes):
            axes_timeTotal.plot(timingData['step'], timingData['time_total[msec]'] / 1000 / 60, ".-", label=(config.caseDataNames[i] + ", "+ str(sampleIdx)))
            maxStep = max(maxStep, np.max(timingData['step']))
            maxTime = max(maxTime, np.max(timingData['time_total[msec]'] / 1000 / 60))

    axes_timeTotal.set_xlabel("PIC step")
    axes_timeTotal.set_ylabel("total time [min]")
    axes_timeTotal.set_ylim(bottom=0)
    xticks = list(range(-1, maxStep+1, int(np.ceil(maxStep / 10))))
    xticks[0] = -1
    xticks[-1] = maxStep
    labels = copy.copy(xticks)
    labels[0] = "init"
    labels[-1] = "cleanup"
    axes_timeTotal.set_xticks(xticks, labels)
    yticks = list(range(0, int(maxTime), int(np.ceil(maxTime / 12))))
    yticks.append(maxTime)
    axes_timeTotal.set_yticks(yticks)

    handles, labels = axes_timeTotal.get_legend_handles_labels()
    lgd_timeTotal= axes_timeTotal.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.21, 1.05), fontsize='small')

    maxStep = 0
    maxTime = 0
    for i, sampleStepTimes in enumerate(casesStepTimes):
        for sampleIdx, timingData in enumerate(sampleStepTimes):
            axes_timeSteps.plot(timingData['step'], timingData['time_step[msec]'] / 1000, ".-", label=(config.caseDataNames[i] + ", " + str(sampleIdx)))
            maxStep = max(maxStep, np.max(timingData['step']))
            maxTime = max(maxTime, np.max(timingData['time_step[msec]'] / 1000))

    axes_timeSteps.set_xlabel("PIC step")
    axes_timeSteps.set_ylabel("step time [s]")
    axes_timeSteps.set_ylim(bottom=0)

    xticks = list(range(-1, maxStep+1, int(np.ceil(maxStep / 10))))
    xticks[0] = -1
    xticks[-1] = maxStep
    labels = copy.copy(xticks)
    labels[0] = "init"
    labels[-1] = "cleanup"
    axes_timeSteps.set_xticks(xticks, labels)
    yticks = list(range(0, int(maxTime), int(np.ceil(maxTime / 13))))
    yticks[-1] = maxTime
    axes_timeSteps.set_yticks(yticks)

    handles, labels = axes_timeSteps.get_legend_handles_labels()
    lgd_timeSteps = axes_timeSteps.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.21, 1.05), fontsize='small')

    plt.savefig(config.fileName, bbox_extra_artists=(lgd_timeSteps,lgd_timeTotal, title), bbox_inches='tight')

if __name__ == "__main__":
    basePath = "/home/marre55/picInputs/testSCFlyComparison_Ar/"
    files_30ppc = ["output_compare_30ppc_2.result", "output_compare_30ppc_3.result", "output_compare_30ppc_4.result"]
    files_60ppc = ["output_compare_60ppc_1.result", "output_compare_60ppc_2.result", "output_compare_60ppc_3.result", "output_compare_60ppc_4.result"]

    basePath_Li = "/home/marre55/picInputs/testSCFlyComparison_Li/"
    files_Li = ["output_compare_30ppc_1.result"]

    config_Argon = Config_TimingData([basePath, basePath], [files_30ppc, files_60ppc], ["30ppc", "60ppc"], "TimingData_Ar")
    config_Lithium = Config_TimingData([basePath_Li], [files_Li], ["30ppc"], "TimingData_Li")

    plot(config_Argon)
    plot(config_Lithium)