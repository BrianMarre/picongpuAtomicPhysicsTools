import numpy as np
import openpmd_api as io
import matplotlib.pyplot as plt

import AdaptiveHistogram_Reader as reader

filepath = "/home/marre55/picongpuOutput/testAdaptiveHistogramOutput/simOutput/C_adaptiveHistogramPerSuperCell/"
filename = filepath + "adaptiveHistogramPerSuperCell_%T.bp"
meshname = "adaptiveHistogram"

 # open series
accumulatedAdaptiveHistogram = {}

    gridExtent, numBins, leftBoundaryBins, widthBins, weightBins, argumentUNIT = (
        reader.readAdaptiveHistogram(filename, meshname, step))
    # accumulate over all grid points
    # create one single histogram(weight(energy)) for each time step
    # output histograms (as combined waterfall)

    print(gridExtent)

    #bin = accumulatedAdaptiveHistogram.get(leftBoundaryBins)


    # exists => add weight
    #if bin:
    #    accumulatedAdaptiveHistogram[leftBoundaryBins].append(weightBins)
    # does not exist create new key with weight as initial value
    #else:
    #    states[configIndex] = [dataWeightings[i]]


    #fig = plt.figure(dpi=400)
    #plt.title("adaptiveHistogram")

    #plt.xlabel("energy electron[AU]")
    #plt.ylabel("macro particle weight")


    #plt.bar(
    #    leftBoundaryBins,
    #    )
    #plt.savefig(str(i) + "_adaptiveHistogramOutput.png")