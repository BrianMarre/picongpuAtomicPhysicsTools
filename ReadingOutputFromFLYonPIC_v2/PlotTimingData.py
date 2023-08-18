import TimingData_Reader as reader
import matplotlib.pyplot as plt
import numpy as np
import copy

basePath = "/home/marre55/picInputs/testSCFlyComparison/"
files_30ppc = ["output_compare_30ppc_2.result", "output_compare_30ppc_3.result", "output_compare_30ppc_4.result"]
files_60ppc = ["output_compare_60ppc_1.result", "output_compare_60ppc_2.result", "output_compare_60ppc_3.result", "output_compare_60ppc_4.result"]

stepTimes_30ppc = []
for fileName in files_30ppc:
    timingData = reader.readTimingData(basePath + fileName)
    stepTimes_30ppc.append(timingData)

stepTimes_60ppc = []
for fileName in files_60ppc:
    timingData = reader.readTimingData(basePath + fileName)
    stepTimes_60ppc.append(timingData)

figure, (axes_timeTotal, axes_timeSteps) = plt.subplots(2,1, dpi=200)
plt.suptitle("FLYonPIC Timing Data")

maxStep = 0
maxTime = 0
for i, entry in enumerate(stepTimes_30ppc):
    axes_timeTotal.plot(entry['step'], entry['time_total[msec]'] / 1000 / 60, ".-", label=("30ppc, "+str(i)))
    maxStep = max(maxStep, np.max(entry['step']))
    maxTime = max(maxTime, np.max(entry['time_total[msec]'] / 1000 / 60))
for i, entry in enumerate(stepTimes_60ppc):
    axes_timeTotal.plot(entry['step'], entry['time_total[msec]'] / 1000 / 60, ".-", label=("60ppc, "+str(i)))
    maxStep = max(maxStep, np.max(entry['step']))
    maxTime = max(maxTime, np.max(entry['time_total[msec]'] / 1000 / 60))

axes_timeTotal.set_xlabel("PIC step")
axes_timeTotal.set_ylabel("total time [min]")
axes_timeTotal.set_ylim(bottom=0)
xticks = list(range(0, maxStep+1, 10))
xticks[0] = -1
xticks[-1] = maxStep
labels = copy.copy(xticks)
labels[0] = "init"
labels[-1] = "cleanup"
axes_timeTotal.set_xticks(xticks, labels)
yticks = list(range(0, int(maxTime), 60))
yticks.append(maxTime)
axes_timeTotal.set_yticks(yticks)

axes_timeTotal.legend()

maxStep = 0
maxTime = 0
for i, entry in enumerate(stepTimes_30ppc):
    axes_timeSteps.plot(entry['step'], entry['time_step[msec]'] / 1000, ".-", label=("30ppc, "+str(i)))
    maxStep = max(maxStep, np.max(entry['step']))
    maxTime = max(maxTime, np.max(entry['time_step[msec]'] / 1000))
for i, entry in enumerate(stepTimes_60ppc):
    axes_timeSteps.plot(entry['step'], entry['time_step[msec]'] / 1000, ".-", label=("60ppc, "+str(i)))
    maxStep = max(maxStep, np.max(entry['step']))
    maxTime = max(maxTime, np.max(entry['time_step[msec]'] / 1000))

axes_timeSteps.set_xlabel("PIC step")
axes_timeSteps.set_ylabel("step time [s]")
axes_timeSteps.set_ylim(bottom=0)

xticks = list(range(0, maxStep+1, 10))
xticks[0] = -1
xticks[-1] = maxStep
labels = copy.copy(xticks)
labels[0] = "init"
labels[-1] = "cleanup"
axes_timeSteps.set_xticks(xticks, labels)
yticks = list(range(0, int(maxTime), 30))
yticks[-1] = maxTime
axes_timeSteps.set_yticks(yticks)

handles, labels = axes.get_legend_handles_labels()
lgd = axes_timeSteps.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')
axes_timeSteps.legend()

plt.tight_layout(pad=1.5)

plt.savefig("TimingData")
