import openPMD_Reader as openPMD
import SCFly_Reader as SCFly
import openpmd_api as io

import plottingHelper as util

def DrawStateDistributionSCFly():
    fileNameOccuppationData = "/home/marre55/scflyInput/testCase/xout_converted.dat"
    fileNameStateNaming = "/home/marre55/scflyInput/testCase/atomicStateNaming.dat"

    filenamePIConGPU_output

    atomicNumber = 18
    numLevels = 10
    numTimeSteps = 100

    title = "FLYonPIC vs SCFly"

    resultsPIConGPU = openPMD.getAtomicStateData(filenamePIConGPU_output, speciesName, atomicNumber)
    resultSCFly = SCFly.readSCFlyPreProcessed(fileNameOccuppationData, fileNameStateNaming, numTimeSteps, atomicNumber, numLevels)

    # open series for iteration time data
    series = io.Series(filenamePIConGPU_output, io.Access.read_only)

    # create plot
    fig = plt.figure(dpi=400)
    ax = fig.add_axes((0.1,0.25,0.8,0.7))
    plt.title(title)