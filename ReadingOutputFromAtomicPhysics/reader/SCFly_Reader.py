import numpy as np
import ConfigNumberConversion as cn_conv

def readPreProcessedScFlyOutputFile(fileName, numTimePoints):
    """read in preprocessed scfly output file

        preprocess = separate state description and reverse column splitting

        @returns numpy structured array dtype = [("state", 'U8'), ("i", 'f8')]
    """

    timeStepCollums = []
    for i in range(numTimePoints):
        timeStepCollums.append((str(i), 'f8'))

    dtype = [("state", 'U8')]
    dtype.extend(timeStepCollums)

    dataOccupation = np.loadtxt(fileName, dtype=dtype, skiprows=1)

    return dataOccupation

def readNameTable(fileName, numLevels, Z):
    """read SCFly atomic state name table
     @returns dict of {SCFly_Name : FLYonPIC_Name}
    """

    occupationNumberCollums = []
    ids = []
    for i in range(numLevels):
        occupationNumberCollums.append((str(i), 'u1'))
        ids.append(str(i))

    dtype = [("state", 'U8')]
    dtype.extend(occupationNumberCollums)

    conversionTable = np.loadtxt(fileName, dtype=dtype)

    configNumbers = []
    for levelVector in conversionTable[ids]:
         configNumbers.append(cn_conv.getConfigNumber(levelVector, Z))

    result = dict(zip(conversionTable["state"], configNumbers))

    # add completely ionized state by hand
    result["18+"] = 0

    return result


def readSCFlyPreProcessed(fileNameOutput, fileNameNameTable, numTimeSteps, Z, numLevels):
    """ read pre-processed SCFly output

        @returns structured array [("configNumber", 'u8'), ("i",'f8')]
            timeStep index : number density 1/cm^3 of atomic state
    """ 
    conversionDict = readNameTable(fileNameNameTable, numLevels, Z)

    resultSCFly = readPreProcessedScFlyOutputFile(fileNameOutput, numTimeSteps+1)

    dtype = [("state", 'u8')]
    ids = []
    for i in range(numTimeSteps+1):
        dtype.append((str(i), 'f8'))
        ids.append(str(i))

    result = np.empty(len(conversionDict.keys()), dtype=dtype)

    for i, entry in enumerate(resultSCFly):
        if entry["state"] in conversionDict:
            result[i]["state"] = conversionDict[entry["state"]]
            result[ids] = entry[ids]
    return result
