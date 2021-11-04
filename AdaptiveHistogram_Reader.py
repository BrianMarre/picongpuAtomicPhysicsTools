import openpmd_api as io
import numpy as np

filepath = "/home/marre55/picongpuOutput/testAdaptiveHistogramOutput/simOutput/C_adaptiveHistogramPerSuperCell/"
filename = filepath + "adaptiveHistogramPerSuperCell_%T.bp"
meshname = "adaptiveHistogram"

def readAdaptiveHistogram(filename, meshname)
    """ read an adaptive histogram from openPMD output file/s given and organize the data

        an adaptive histogram contains the following data:
            numBins

        returns: numBin(gridExtent), leftBoundaryBin(gridExtent, maxNumBins),
            widthBin(gridExtent, maxNumBins), weightBin(gridExtent, maxNumBins), argumentUNIT

        numBin ... number of occupied Bins
        leftBoundaryBin ... left boundary of bins
        widthBin ... width of bins
        weightBin ... weight of paritcles in bin, actual numer of pyhsical particles
        argumentUNIT ... unit of argument axis
    """
    series = io.Series(filename, io.Access.read_only) # open series

    for i in series.iterations: # for each output time step
        step = series.iterations[i]
        openPmdMesh = step.meshes[meshname]

        data = openPmdMesh[io.Mesh_Record_Component.SCALAR] # data is stored in artifically increased extent

        # get histogram attributes
        numValuesPerBin = openPmdMesh.get_attribute("numberValuesPerBin")
        maxNumBins = openPmdMesh.get_attribute("maxNumBins")
        ArgumentUnit = openPmdMesh.get_attribute("ATOMIC_UNIT_ENERGY")

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

        leftBoundaryBin = np.empty(binExtent)
        widthBin = np.empty(binExtent)
        weightBin = np.empty(binExtent)

        #iterate over grid to extract data
        numEntriesAdaptiveHistogram = numValuesPerBin * maxNumBins + 1

        currentGridIndex = np.zeros(numDimension, dtype=np.uint64)
        for n in range(np.prod(gridExtent)): # for each gridPoint
            startIndexCurrentSuperCell = int(currentGridIndex[-1] * numEntriesAdaptiveHistogram)

            # numberBins is always stored as first entry for super cell
            numBins[tuple(currentGridIndex)] = np.uint16(adaptiveHistogramData[tuple(currentGridIndex[:-1])][startIndexCurrentSuperCell])

            # select first all but the last dimension, last dim is extended, and choose correct slice from last dim x
            startIndexValues = startIndexCurrentSuperCell+1

            leftBoundaryBin[tuple(currentGridIndex)] = adaptiveHistogramData[
                tuple(currentGridIndex[:-1])][ # super cell Index for non extended dims(... ,y)
                startIndexValues:(startIndexValues + maxNumBins)] # x = (0:maxNumBins)+startIndex, end is exclusive

            # adavance offset for next value
            startIndexValues += maxNumBins

            widthBin[tuple(currentGridIndex)] = adaptiveHistogramData[
                tuple(currentGridIndex[:-1])][
                startIndexValues:(startIndexValues + maxNumBins)] # x = (0:maxNumBins)+startIndex, end is exclusive

            # adavance offset for next value
            startIndexValues += maxNumBins

            weightBin[tuple(currentGridIndex)] = adaptiveHistogramData[
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

