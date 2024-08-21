"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import StateDistributionReader
from ..SCFlyTools import AtomicConfigNumberConversion as conv

import numpy as np
import numpy.typing as npt

import typeguard

@typeguard.typechecked
class SCFLYReader(StateDistributionReader):
    """read atomic state populations from scfly output"""

    SCFLYOutputFileName : str
    # path of SCFLY output file

    SCFLYatomicStateNamingFile : str
    """
    path to atomicStateNaming input file, contains for each SCFLY state its corresponding occupation number vector

    file must give in each line name of one state and occupation number for each shell for state

    Exp.: h_10001 1 0 0 0 0 0 0 0 0 0\n ...
    """

    atomicNumber : int
    # atomic number of ion species
    numberLevels : int
    # maximum principal quantum number used, i.e. number of shells included in the data set

    def readSCFLYNames(self) -> tuple[dict[str, np.uint64], dict[np.uint64, str]]:
        """
        read SCFly atomic state names from file and convert them to FLYonPIC atomicConfigNumbers

        @attention states must be uniquely described by their shell occupation number vector!

        @returns dict of {SCFLYname : FLYonPIC_atomicConfigNumber}
        """

        # prepare result datatype description
        occupationNumberCollums = []
        ids = []
        for i in range(self.numberLevels):
            occupationNumberCollums.append((str(i), 'u1'))
            ids.append(str(i))

        dtype = [("state", 'U10')]
        dtype.extend(occupationNumberCollums)

        # load file content
        conversionTable = np.loadtxt(self.SCFLYatomicStateNamingFile, dtype=dtype)

        # create translation dictionary
        configNumbers = []
        for levelVector in conversionTable[ids]:
            configNumbers.append(np.uint64(conv.getConfigNumber(levelVector, self.atomicNumber)))

        SCFLYStateName_to_atomicConfigNumber = dict(zip(conversionTable["state"], configNumbers))
        atomicConfigNumber_to_StateName = dict(zip(configNumbers, conversionTable["state"]))

        # add completely ionized state by hand
        SCFLYStateName_to_atomicConfigNumber[str(self.atomicNumber)+"+"] = np.uint64(0.)
        atomicConfigNumber_to_StateName[np.uint64(0.)] = str(self.atomicNumber)+"+"

        return SCFLYStateName_to_atomicConfigNumber, atomicConfigNumber_to_StateName

    def check(self) -> None:
        """check internal state"""
        if(self.SCFLYatomicStateNamingFile == ""):
            raise ValueError("must provide a file list SCFLY state names and correspondig level vectors")
        if(self.SCFLYOutputFileName == ""):
            raise ValueError("must provide SCFLY output file to read")
        if self.atomicNumber <= 0:
            raise ValueError("atomic Number must be > 0")
        if self.numberLevels <= 0:
            raise ValueError("numberLevels must be > 0")

    def getSCFLYStateNameRegex(self, SCFLY_to_FLYonPIC : dict[str, np.uint64]) -> tuple[str, int]:
        """get regex matching all SCFLY state names"""
        stateNames = SCFLY_to_FLYonPIC.keys()

        # create regex matching every scfly stateName
        stateName_regex= ""
        for stateName in stateNames:
            stateName_regex += stateName + "|"
        stateName_regex = stateName_regex[:-2] + "\+"

        numberNamedStates = len(stateNames)
        del stateNames

        return stateName_regex, numberNamedStates

    def getAllUniqueBlockSizes(self, stateName_regex : str, numberNamedStates : int) -> tuple[
            npt.NDArray[np.uint32],
            int,
            npt.NDArray[np.dtype(
                [('startIndex', 'u4'), ('endIndex', 'u4'), ('numStates', 'u4'), ('state', 'U10'), ('rubbish', 'f4')])]]:
        """get size of all blocks in SCFLY output and number of blocks total"""
        regex_blockShapeDescriptions = r"\s+(\d+)\s+(\d+)\s+(\d+)\n (" + stateName_regex + ")(\s+\d.\d+E[+-]\d+)+\n"
        result_BlockShapeDescriptions = [('startIndex', 'u4'), ('endIndex', 'u4'), ('numStates', 'u4'), ('state', 'U10'), ('rubbish', 'f4')]

        blockShapeDescriptions = np.fromregex(self.SCFLYOutputFileName, regex_blockShapeDescriptions, result_BlockShapeDescriptions)
        del regex_blockShapeDescriptions
        del result_BlockShapeDescriptions

        if np.any(blockShapeDescriptions['numStates'] != numberNamedStates):
            raise RuntimeError("inconsistent number of atomic states")

        blockSizes = blockShapeDescriptions['endIndex'] - blockShapeDescriptions['startIndex'] + 1

        # there will only ever be two unique block sizes, full and maybe one partial for the last block
        # [(partial block size), full block size]
        uniqueBlockSizes = np.unique(blockSizes)

        totalNumberSteps = int(np.sum(blockSizes))
        del blockSizes

        return uniqueBlockSizes, totalNumberSteps, blockShapeDescriptions

    def getSCFLYOutputLineRegexAndDtype(self, stateName_regex : str, blockSize : np.uint32) -> tuple[
            str,
            str,
            list[tuple[str, str]],
            list[tuple[str, str]],
            list[str]]:
        """
            create regexes and structured array dtypes for time and state lines for a given blockSize

            to be used by np.fromregex to read in an SCFLY ouptut file block of given block size
        """

        # create regex
        regex_StateEntries = "(" + stateName_regex + ")"
        regex_TimeEntry = "Time"
        blockIds = []
        blockIdCollums = []

        for i in range(blockSize):
            # number in scientific notation prepended with spaces
            regex_StateEntries += "\s+(\d.\d+E[+-]\d+)"
            regex_TimeEntry += "\s+(\d.\d+E[+-]\d+)"
            blockIds.append(str(i))
            blockIdCollums.append((str(i), 'f8'))
        regex_StateEntries += "\n"
        regex_TimeEntry += "\n"

        # @attention, must be two separate statements due to python returning an unnamed object from constant constructor
        dtype_StateEntries = [('state', 'U10')]
        dtype_StateEntries.extend(blockIdCollums)

        dtype_TimeEntry = blockIdCollums

        return regex_StateEntries, regex_TimeEntry, dtype_StateEntries, dtype_TimeEntry, blockIds

    def readInDataForEachUniqueBlockSize(self, uniqueBlockSizes : npt.NDArray[np.uint32], stateName_regex : str) -> \
        tuple[
            dict[np.uint32, npt.NDArray],
            dict[np.uint32, npt.NDArray],
            dict[np.uint32, list[str]]]:
        """read in SCFLY output file for each unique block size"""

        populationDataByBlockSize = {}
        timeDataByBlockSize = {}
        blockIdsByBlockSize = {}
        for blockSize in uniqueBlockSizes:
            regex_StateEntries, regex_TimeEntry, dtype_StateEntries, dtype_TimeEntry, blockIds = (
                self.getSCFLYOutputLineRegexAndDtype(stateName_regex, blockSize))

            # read all blocks for current blocksize
            populationDataByBlockSize[blockSize] = np.fromregex(self.SCFLYOutputFileName, regex_StateEntries, dtype_StateEntries)
            timeDataByBlockSize[blockSize] = np.fromregex(self.SCFLYOutputFileName, regex_TimeEntry, dtype_TimeEntry)

            # store block ids for later
            blockIdsByBlockSize[blockSize] = blockIds

        return populationDataByBlockSize, timeDataByBlockSize, blockIdsByBlockSize

    def read(self) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.uint64]], dict[str, int], tuple[np.float64]]:
        """
        load SCLFY main output file and extract population data

        an SCFLY output file contains a header describing simulation settings, a first section containing the rate
          matrix of the first time step and second section containing the actual output as one line for every state
          and output quantity in blocks with each block having a header column and afterwards continuing the row from
          the previous block.

        @returns
                numberDensity(numberStates, numberTimeSteps) [1/cm^3],
                (list of times points, list of atomic state config numbers),
                axisDict ... map of atomicState and timeStep physics axis to axis index in the data table : {"atomicState" : int, "timeStep": int},
                empty tuple
        """
        self.check()

        SCFLY_to_FLYonPIC, other = self.readSCFLYNames()
        del other

        stateName_regex, numberNamedStates = self.getSCFLYStateNameRegex(SCFLY_to_FLYonPIC)
        uniqueBlockSizes, totalNumberSteps, blockShapeDescriptions = self.getAllUniqueBlockSizes(stateName_regex, numberNamedStates)
        populationDataByBlockSize, timeDataByBlockSize, blockIdsByBlockSize = self.readInDataForEachUniqueBlockSize(
            uniqueBlockSizes, stateName_regex)
        del stateName_regex

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

        # sort data the be equivalent to FLYonPIC sorting
        chargeStates = np.fromiter(map(
            lambda atomicConfigNumber : conv.getChargeState(
                atomicConfigNumber,
                self.atomicNumber,
                self.numberLevels),
            atomicConfigNumbers), dtype = 'u1')
        sortedIndices = np.lexsort((atomicConfigNumbers, chargeStates))
        del chargeStates

        atomicConfigNumbersSorted = atomicConfigNumbers[sortedIndices]
        atomicStatePopulationDataSorted = atomicStatePopulationData[:, sortedIndices]
        del atomicConfigNumbers
        del atomicStatePopulationData

        return atomicStatePopulationDataSorted, (timeData, atomicConfigNumbersSorted), {'timeStep':0, 'atomicState':1}, (np.float64(1.0),)

    # @todo
    @typeguard.typechecked
    def getSCFLY_FirstStepRates(fileName : str, SCFLY_to_FLYonPIC : dict[str, int]):
        """
        load SCLFY atomic.data output file and extract population data

        @param fileName atomic.data output file of SCFLY
        @param SCFLY_to_FLYonPIC dictionary converting SCFLY state names to FLYonPIC atomiConfigNumbers

        @returns rateData[numberStates, numerTimeSteps],

        noChange = 0
        spontaneousDeexcitation = 1
        electronicExcitation = 2
        electronicDeexcitation = 3
        electronicIonization = 4
        autonomousIonization = 5
        """
        raise NotImplementedError
