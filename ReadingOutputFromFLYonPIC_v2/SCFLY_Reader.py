import numpy as np
import ConfigNumberConversion as conv

def readSCFLYNames(fileName, Z, numLevels):
    """read SCFly atomic state names from file and convert them to FLYonPIC atomicConfigNumbers

    @attention states must be uniquely described by their shell occupation number vector!

    @param numLevels number of shells included in the data set
    @param fileName path of file describing SCFLY atomic states,
        file must give in each line name of one state and occupation number for each shell for state

        Exp.: h_10001 1 0 0 0 0 0 0 0 0 0\n ...
    @param Z atomic number of the element for which atomic states are contained

     @returns dict of {SCFLYname : FLYonPIC_atomicConfigNumber}
    """

    # prepare result datatype description
    occupationNumberCollums = []
    ids = []
    for i in range(numLevels):
        occupationNumberCollums.append((str(i), 'u1'))
        ids.append(str(i))

    dtype = [("state", 'U10')]
    dtype.extend(occupationNumberCollums)

    # load file content
    conversionTable = np.loadtxt(fileName, dtype=dtype)

    # create translation dictionary
    configNumbers = []
    for levelVector in conversionTable[ids]:
         configNumbers.append(conv.getConfigNumber(levelVector, Z))

    SCFLYStateName_to_atomicConfigNumber = dict(zip(conversionTable["state"], configNumbers))
    atomicConfigNumber_to_StateName = dict(zip(configNumbers, conversionTable["state"]))

    # add completely ionized state by hand
    SCFLYStateName_to_atomicConfigNumber[str(Z)+"+"] = 0

    return SCFLYStateName_to_atomicConfigNumber, atomicConfigNumber_to_StateName


def getSCFLY_Data(fileName, SCFLY_to_FLYonPIC):
    """load SCLFY main output file and extract population data

    @param fileName main output file of SCFLY
    @param SCFLY_to_FLYonPIC dictionary converting SCFLY state names to FLYonPIC atomiConfigNumbers

    @returns populationData[numberStates, numerTimeSteps],
    """
    stateNames = SCFLY_to_FLYonPIC.keys()

    # create regex matching every scfly stateName
    stateName_regex= ""
    for stateName in stateNames:
        stateName_regex += stateName + "|"
    stateName_regex = stateName_regex[:-2] + "\+"

    numberNamedStates = len(stateNames)
    del stateNames

    # get all unique blockSizes
    regex_blockShapeDescriptions = r"\s+(\d+)\s+(\d+)\s+(\d+)\n (" + stateName_regex + ")(\s+\d.\d+E[+-]\d+)+\n"
    result_BlockShapeDescriptions = [('startIndex', 'u4'), ('endIndex', 'u4'), ('numStates', 'u4'), ('state', 'U10'), ('rubbish', 'f4')]

    blockShapeDescriptions = np.fromregex(fileName, regex_blockShapeDescriptions, result_BlockShapeDescriptions)
    del regex_blockShapeDescriptions
    del result_BlockShapeDescriptions

    if np.any(blockShapeDescriptions['numStates'] != numberNamedStates):
        raise RuntimeError("inconsistent number of atomic states")

    blockSizes = blockShapeDescriptions['endIndex'] - blockShapeDescriptions['startIndex'] + 1

    # there will only ever be two unique block sizes, full and maybe one partial for the last block
    # [(partial block size), full block size]
    uniqueBlockSizes = np.unique(blockSizes)

    totalNumberSteps = np.sum(blockSizes)
    del blockSizes

    # read in data for each unique Blocksize
    populationDataByBlockSize = {}
    timeDataByBlockSize = {}
    blockIdsByBlockSize = {}
    for blockSize in uniqueBlockSizes:
        # create regex
        regex_StateEntries = "(" + stateName_regex + ")"
        regex_TimeEntry = "Time"
        blockIds = []
        blockIdCollums = []

        for i in range(blockSize):
            regex_StateEntries += "\s+(\d.\d+E[+-]\d+)"
            regex_TimeEntry += "\s+(\d.\d+E[+-]\d+)"
            blockIds.append(str(i))
            blockIdCollums.append((str(i), 'f8'))
        regex_StateEntries += "\n"
        regex_TimeEntry += "\n"

        # @attention, must be separated due to python
        dtype_StateEntries = [('state', 'U10')]
        dtype_StateEntries.extend(blockIdCollums)

        dtype_TimeEntry = blockIdCollums

        # load all blocks for blocksize
        populationDataByBlockSize[blockSize] = np.fromregex(fileName, regex_StateEntries, dtype_StateEntries)
        timeDataByBlockSize[blockSize] = np.fromregex(fileName, regex_TimeEntry, dtype_TimeEntry)

        # store block ids for later
        blockIdsByBlockSize[blockSize] = blockIds

    # get atomicConfigNumber for each state from first blockSize(partial or full(in this case will be only blockSize))
    atomicConfigNumbers = np.fromiter(
        map(lambda stateName: SCFLY_to_FLYonPIC[stateName],
            populationDataByBlockSize[uniqueBlockSizes[0]]['state'][0:numberNamedStates]), dtype='u8')

    atomicStatePopulationData = np.empty((totalNumberSteps, numberNamedStates), dtype='f8')
    timeData = np.zeros(totalNumberSteps, dtype='f8')

    # get list of state names from reference block
    referenceListStates = populationDataByBlockSize[uniqueBlockSizes[0]]['state'][0:numberNamedStates]
    # for each block
    blockIndex = {}
    for entry in blockShapeDescriptions:
        start = entry['startIndex'] - 1
        end = entry['endIndex'] - 1
        blockSize = end - start + 1

        # update blockIndex
        if(blockIndex.get(blockSize) is None):
            blockIndex[blockSize] = 0
        else:
            blockIndex[blockSize] = blockIndex[blockSize] + 1

        # check that states in current block are ordered consistent to reference block
        if np.any(populationDataByBlockSize[blockSize]
                  [blockIndex[blockSize]*numberNamedStates:(blockIndex[blockSize]+1)*numberNamedStates]
                  ['state'] != referenceListStates):
            raise RuntimeError("inconsistent state ordering/naming between blocks")

        # store block
        for j, entry in enumerate(blockIdsByBlockSize[blockSize]):
            atomicStatePopulationData[start + j, :] = (
                populationDataByBlockSize[blockSize][
                    blockIndex[blockSize]*numberNamedStates:(blockIndex[blockSize]+1)*numberNamedStates][entry])
            timeData[start + j] = timeDataByBlockSize[blockSize][blockIndex[blockSize]][entry]

    return atomicStatePopulationData, {'timeStep':0, 'atomicState':1}, atomicConfigNumbers, timeData

if __name__ == "__main__":
    numLevels = 10
    Z = 18

    SCFLY_output = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC/xout"
    SCFLY_stateNames = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC/atomicStateNaming.input"

    # load in state names
    SCFLY_to_FLYonPIC, = readSCFLYNames(SCFLY_stateNames, Z, numLevels)
    # get all unique BlockSizes
    atomicPopulationData, atomicConfigNumbers, timeData = getSCFLY_Data(SCFLY_output, SCFLY_to_FLYonPIC)
