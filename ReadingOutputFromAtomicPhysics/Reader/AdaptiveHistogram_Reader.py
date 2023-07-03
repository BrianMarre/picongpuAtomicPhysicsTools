import openpmd_api as io
import numpy as np
import struct
import math

# read adaptivehsitogram data from given mesh
def readAdaptiveHistogramFromMesh(series, openPmdMesh):
    """ read an adaptive histogram from a given openPMD mesh and organize the data into
        a more convenient form

        an adaptive histogram contains the following data:
            numBins:uint32                      ... number of occupied Bins, <= maxNumBins
            leftBoundaryBins:float32[ma xNumBins] ... left Boundary of bins, only occupied bins are initialized
            widthBins:float32[maxNumBins]        ... width of bins, -||-
            weightBins:float32[maxNumBins]      ... weight of macro particles in bin, -||-

        This data is stored in order consecutively in linear memory, eg.
        memoryIndex |0       1                   ... maxNumBins                      maxNumBins+1 ...
        data        |numBins  leftBoundary bin #1 ... leftBoundary bin #maxNumBins    width bin #1 ...
        This is a representation is also used in the output formating.
        For a Cartesian grid with n_x sample points in the x dimension, and one
        adaptive histogram for each grid point, the output is a scalar field with
        n_x * (1+3*maxNumBins) sample points in the x-dimension.
        The additional points encode the actual histogram values by simple reproducing
        the memory layout described above, eg. the x dimension contains n_x adaptive
        histograms blocks, each block reproducing the memory layout behind each other.

        returns: gridExtent, numBins(gridExtent), leftBoundaryBins((gridExtent, maxNumBins)),
            widthBins((gridExtent, maxNumBins)), weightBins((gridExtent, maxNumBins))

        Note: In the notation <name>(<extent>) <extent> describes the shape of the output

        numBins ... number of occupied Bins
        leftBoundaryBins ... left boundary of bins
        widthBins ... width of bins
        weightBins ... weight of particles in bin, equivalent number of physical particles
    """

    data = openPmdMesh[io.Mesh_Record_Component.SCALAR] # data is stored in artificially increased extent

    # get histogram attributes
    numValuesPerBin = openPmdMesh.get_attribute("numberValuesPerBin")
    maxNumBins = openPmdMesh.get_attribute("maxNumBins")

    adaptiveHistogramData = data.load_chunk() # (... , y, x) indexation, artificial extent increase in x-direction
    series.flush()

    # get different extents
    dataExtent = np.shape(adaptiveHistogramData)
    numDimension = len(dataExtent)

    # extent of data array is artificially increased in x direction, and indexation is (...,y,x)
    gridExtent = np.copy(dataExtent)
    gridExtent[numDimension - 1] = gridExtent[numDimension - 1] / (numValuesPerBin * maxNumBins + 1)
    binExtent = np.append(gridExtent, maxNumBins)

    #extract data
    numBins = np.empty(gridExtent, dtype=np.uint16)

    leftBoundaryBins = np.empty(binExtent)
    widthBins = np.empty(binExtent)
    weightBins = np.empty(binExtent)

    #iterate over grid to extract data
    numEntriesAdaptiveHistogram = numValuesPerBin * maxNumBins + 1

    # debug only

    currentGridIndex = np.zeros(numDimension, dtype=np.uint64)
    for n in range(np.prod(gridExtent)): # for each gridPoint
        startIndexCurrentSuperCell = int(currentGridIndex[-1] * numEntriesAdaptiveHistogram)

        # numberBins is always stored as first entry for super cell
        numBins[tuple(currentGridIndex)] = np.uint16(
            struct.unpack(
                'i',
                adaptiveHistogramData[tuple(currentGridIndex[:-1])][startIndexCurrentSuperCell]
                )[0]
            )

        # select first all but the last dimension, last dim is extended, and choose correct slice from last dim x
        startIndexValues = startIndexCurrentSuperCell+1

        leftBoundaryBins[tuple(currentGridIndex)] = adaptiveHistogramData[
            tuple(currentGridIndex[:-1])][ # super cell Index for non extended dims(... ,y)
            startIndexValues:(startIndexValues + maxNumBins)] # x = (0:maxNumBins)+startIndex, end is exclusive

        # adavance offset for next value
        startIndexValues += maxNumBins

        widthBins[tuple(currentGridIndex)] = adaptiveHistogramData[
            tuple(currentGridIndex[:-1])][
            startIndexValues:(startIndexValues + maxNumBins)] # x = (0:maxNumBins)+startIndex, end is exclusive

        # adavance offset for next value
        startIndexValues += maxNumBins

        weightBins[tuple(currentGridIndex)] = adaptiveHistogramData[
            tuple(currentGridIndex[:-1])][
            startIndexValues:(startIndexValues + maxNumBins)] # x = (0:maxNumBins)+startIndex, end is exclusive

        # update to next gridIndex
        for d in range(numDimension):
            if ( currentGridIndex[d] < (gridExtent[d] - 1) ):
                currentGridIndex[d] += 1
                break
            else:
                currentGridIndex[d] = 0

    return gridExtent, numBins, leftBoundaryBins, widthBins, weightBins

def testReadFromMesh(filename):
    """ short test function for readAdaptiveHistogramFromMesh.

    should be called on test data, store index in each data point
    compares entries against with correct index, thereby proving correct recomposition
    """

    series = io.Series(filename, io.Access.read_only)
    for iteration in series.iterations:
        step = series.iterations[iteration]
        meshname = "adaptiveHistogram"
        openPmdMesh = step.meshes[meshname]

        gridExtent, numBins, leftBoundaryBins, widthBins, weightBins = (
            readAdaptiveHistogramFromMesh(series, openPmdMesh))

        maxNumBins = openPmdMesh.get_attribute("maxNumBins")
        numValuesPerBin = openPmdMesh.get_attribute("numberValuesPerBin")
        dataEntryLength = (maxNumBins * numValuesPerBin + 1)

        # numBins
        # create correct result
        correctResult = np.reshape(
            np.arange(np.prod(gridExtent)) * dataEntryLength,
            gridExtent) # always at the beginning of the histogram memory
        # compare
        numBinsCorrect = np.all(correctResult == numBins) # == soll ?

        # leftBoundaryBins
        # create correct result
        binExtent = np.append(gridExtent, maxNumBins)
        numDimension = len(gridExtent)
        correctResult = np.empty(binExtent)
        currentIndex = np.zeros(numDimension, dtype=np.uint64)
        for i in range(np.prod(gridExtent)):
            correctResult[tuple(currentIndex)] = np.arange(maxNumBins) + i * dataEntryLength + 1

            # update to next gridIndex
            for d in range(numDimension-1, -1, -1):
                if ( currentIndex[d] < (gridExtent[d] - 1) ):
                    currentIndex[d] += 1
                    break
                else:
                    currentIndex[d] = 0
        # compare
        leftBoundaryBinsCorrect = np.all(correctResult == leftBoundaryBins) # == soll ?

        # widthBins
        correctResult += 12
        widthBinsCorrect = np.all(correctResult == widthBins) # == soll ?

        # weightBins
        correctResult += 12
        weightBinsCorrect = np.all(correctResult == weightBins) # == soll ?

        if not (numBinsCorrect and leftBoundaryBinsCorrect and widthBinsCorrect and weightBinsCorrect):
            print(iteration)
            return False

    return True

# create accumulated adaptiveHistogram from file for each time step
def readAccumulatedAdaptiveHistogram(filename):
    """ reads the adaptive histogram openPMD output from given files and
        returns accumulated histograms over all gridpoint for each time step

        return: timeSteps, histograms

        timeSteps ... list if iterations for which a histogram output exists
        histograms ... list of accumulated histograms, each histogram being a
            dictionary(key:leftBoundaryBin, value:(weight, widthBin))
    """
    meshname = "adaptiveHistogram"

    histograms = []
    timeSteps = []

    series = io.Series(filename, io.Access.read_only)
    for iteration in series.iterations:
        step = series.iterations[iteration]

        openPmdMesh = step.meshes[meshname]

        argumentUNIT = openPmdMesh.get_attribute("ATOMIC_UNIT_ENERGY")

        gridExtent, numBins, leftBoundaryBins, widthBins, weightBins = (
            readAdaptiveHistogramFromMesh(series, openPmdMesh))

        numDimension = len(gridExtent)

        accumulatedWeights = {} # dictionary(binLeftBoundary:[weight])
        accumulatedWidths = {} # dictionary(binLeftBoundary:width)
        currentGridIndex = np.zeros(numDimension, dtype=np.uint64)
        for n in range(np.prod(gridExtent)): # for each gridPoint

            # for each occupied bin in adaptive histogram
            for i in range(numBins[tuple(currentGridIndex)]):
                bin = accumulatedWeights.get(leftBoundaryBins[tuple(currentGridIndex)][i])

                # exists => add weight
                if bin:
                    (accumulatedWeights[leftBoundaryBins[tuple(currentGridIndex)][i]]
                        ).append(weightBins[tuple(currentGridIndex)][i])

                # does not exist create new key with weight as initial value
                else:
                    accumulatedWidths[leftBoundaryBins[tuple(currentGridIndex)][i]] = (
                        widthBins[tuple(currentGridIndex)][i])
                    accumulatedWeights[leftBoundaryBins[tuple(currentGridIndex)][i]] = (
                        [weightBins[tuple(currentGridIndex)][i]])

            # update to next gridIndex
            for d in range(numDimension-1, -1, -1):
                if ( currentGridIndex[d] < (gridExtent[d] - 1) ):
                    currentGridIndex[d] += 1
                    break
                else:
                    currentGridIndex[d] = 0

        # add together bin weights numerically stable
        accumulatedAdaptiveHistogram = {}
        for k in accumulatedWeights.keys():
            totalWeight = math.fsum(accumulatedWeights[k])
            accumulatedAdaptiveHistogram[k] = (totalWeight, accumulatedWidths[k])

        # append result of current time step to output
        histograms.append(accumulatedAdaptiveHistogram)
        timeSteps.append(iteration)

    del series
    return timeSteps, histograms, argumentUNIT