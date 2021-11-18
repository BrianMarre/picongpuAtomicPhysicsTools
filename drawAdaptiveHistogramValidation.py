import numpy as np
import openpmd_api as io
import matplotlib.pyplot as plt

import AdaptiveHistogram_Reader as reader

filepath = "/home/marre55/picongpuOutput/testAdaptiveHistogramOutput/simOutput/C_adaptiveHistogramPerSuperCell/"
filename = filepath + "adaptiveHistogramPerSuperCell_%T.bp"


series = io.Series(filename, io.Access.read_only)
step = series.iterations[0]
meshname = "adaptiveHistogram"
openPmdMesh = step.meshes[meshname]

gridExtent, numBins, leftBoundaryBins, widthBins, weightBins = (
    reader.readAdaptiveHistogramFromMesh(series, openPmdMesh))

print(numBins)

# read data from file
#timeSteps, histograms = reader.readAccumulatedAdaptiveHistogram(filename)

"""
# output histograms (as combined waterfall?)
# entry [0]

iteration = timeSteps[0]
histogram = histograms[0]

fig = plt.figure(dpi=400)
plt.title("adaptiveHistogram, iteration" + str(iteration))

plt.xlabel("energy electron[AU]")
plt.ylabel("macro particle weight")

leftBoundaries = np.empty(len(histogram))
weights = np.empty(len(histogram))
widths = np.empty(len(histogram))

i = 0
for k,v in histogram.items():
    leftBoundaries[i] = k
    weights[i] = v[0]
    widths[i] = v[1]

plt.bar(
    leftBoundaries,
    weights,
    widths,
    align = 'edge')

plt.savefig(str(iteration) + "_adaptiveHistogramOutput.png")
"""