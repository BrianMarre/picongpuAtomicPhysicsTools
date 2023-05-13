import numpy as np
import ConfigNumberConversion as cn_conv

def readPreProcessesScFlyOutputFile(fileName, numLevels, numTimeSteps):
    """read in preprocessed scfly output file

        preprocess = separate state description and reverse column splitting
    """

    timeStepCollums = []
    for i in range(numTimeSteps):
        timeStepCollums.append((str(i), 'f8'))
    print(timeStepCollums)

    dataOccupation = np.loadtxt(fileName, dtype=([("state", 'U8')]).append(timeStepCollums), skiprows=1)

    print(([("state", 'U8')]).append(timeStepCollums))


fileNameOccuppationData = "/home/marre55/scflyInput/testCase/xout_converted.dat"
fileNameStateNaming = "/home/marre55/scflyInput/testCase/atomicStatenaming.dat"