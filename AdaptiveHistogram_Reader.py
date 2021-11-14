import openpmd_api as io
import numpy as np

filepath = "/home/marre55/picongpuOutput/testAdaptiveHistogramOutput/simOutput/C_adaptiveHistogramPerSuperCell/"
filename = filepath + "adaptiveHistogramPerSuperCell_%T.bp"
meshname = "adaptiveHistogram"

def readAdaptiveHistogram(filename, meshname):
    """ read an adaptive histogram from openPMD output file/s given and organize the data

        an adaptive histogram contains the following data:
            numBins:uint32                      ... number of occupied Bins, <= maxNumBins
            leftBoundaryBins:float32[ma xNumBins] ... left Boundary of bins, only occupied bins are initialized
            widthBins:float32[maxNumBins]        ... width of bins, -||-
            weightBins:float32[maxNumBins]      ... weight of macro particles in bin, -||-

        This data is stored in order consecutively in linear memory, eg.
        memoryIndex |0       1                   ... maxNumBins                      maxNumBins+1 ...
        data        |numBin  leftBoundary bin #1 ... leftBoundary bin #maxNumBins    width bin #1 ...
        This is a representation is also used in the output formating.
        For a cartesian grid with n_x sample points in the x dimension, and one
        adaptive histogram for each grid point, the output is a scalar field with
        n_x * (1+3*maxNumBins) sample points in the x-dimension.
        The additional points encode the actual histogram values by simple reproducing
        the memory layout described above, eg. the x dimension contains n_x adaptive
        histograms blocks, each block reproducing the memory layout behind each other.

        returns: numBin(gridExtent), leftBoundaryBins(gridExtent, maxNumBins),
            widthBins(gridExtent, maxNumBins), weightBins(gridExtent, maxNumBins), argumentUNIT

        numBin ... number of occupied Bins
        leftBoundaryBins ... left boundary of bins
        widthBins ... width of bins
        weightBins ... weight of paritcles in bin, equivalent number of physical particles
        argumentUNIT ... unit of argument axis
    """
    series = io.Series(filename, io.Access.read_only) # open series

    for i in series.iterations: # for each time step output
        step = series.iterations[i]
        openPmdMesh = step.meshes[meshname]

        data = openPmdMesh[io.Mesh_Record_Component.SCALAR] # data is stored in artificially increased extent

        # get histogram attributes
        numValuesPerBin = openPmdMesh.get_attribute("numberValuesPerBin")
        maxNumBins = openPmdMesh.get_attribute("maxNumBins")
        argumentUNIT = openPmdMesh.get_attribute("ATOMIC_UNIT_ENERGY")

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

        currentGridIndex = np.zeros(numDimension, dtype=np.uint64)
        for n in range(np.prod(gridExtent)): # for each gridPoint
            startIndexCurrentSuperCell = int(currentGridIndex[-1] * numEntriesAdaptiveHistogram)

            # numberBins is always stored as first entry for super cell
            numBins[tuple(currentGridIndex)] = np.uint16(adaptiveHistogramData[tuple(currentGridIndex[:-1])][startIndexCurrentSuperCell])

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

            # update next gridIndex
            for d in range(numDimension):
                if ( currentGridIndex[d] < (gridExtent[d] - 1) ):
                    currentGridIndex[d] += 1
                    break
                else:
                    currentGridIndex[d] = 0

    del series
    return numBins, leftBoundaryBins, widthBins, weightBins, argumentUNIT

numBins, leftBoundaryBins, widthBins, weightBins, argumentUNIT = readAdaptiveHistogram(filename, meshname)
