"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""
import openpmd_api as opmd

import numba
import numpy as np
from tqdm import tqdm

import typeguard

@numba.jit(fastmath=True, nopython=True, cache=True)
def fastHistogram(weights, propertyIndices, typicalWeight, numberPropertyIndexValues):
    """ kernel for accumulating weight of entries according to their propertyIndex

        @param weights array of weight values
        @param propertyIndices array of property index values corresponding to weight entries
        @param typicalWeight normalization factor for weight accumulation
        @param numberPropertyIndexValues number of unique property index values in the output

        @attention 0 <= propertyIndex value < numberPropertyIndexValues

        @returns accumulatedWeight[propertyIndex]
    """

    # number of possible atomicStateCollectionIndice Values
    result = np.zeros(numberPropertyIndexValues)

    for entryIdx in range(propertyIndices.shape[0]):
        state = int(propertyIndices[entryIdx])
        result[state] += weights[entryIdx] / typicalWeight

    return result

@numba.jit(cache=True, parallel=True)
def callFastHistogramInParallel(weights, propertyIndex, typicalWeight, numberPropertyIndexValues, numberWorkers, chunkSize):
    """ stage for parallel call of fast histogram kernel

        @param weights array of weight values
        @param propertyIndex array of property index values corresponding to weights,
        @param typicalWeight normalization factor for weight accumulation
        @param numberIndexValues number of unique property index values in the output
        @param numberWorkers number of parallel worker to use
        @param chunkSize number of entries to attempt to pass to each worker each iteration

        @attention 0 <= propertyIndex < numberPropertyIndexValues

        @returns accumulatedWeight[numberPropertyIndexValues]
    """

    numberChunks = int(np.ceil(weights.shape[0] / chunkSize))
    numberIterations = int(np.ceil(numberChunks / numberWorkers))

    result = np.zeros((numberWorkers + 1, numberPropertyIndexValues), dtype=np.float32)

    for iterationIdx in range(numberIterations):
        for workerIdx in numba.prange(numberWorkers):
            if (numberChunks > (iterationIdx * numberWorkers + workerIdx)):
                # get chunks of allocated memory, will return empty views if no data remaining
                w_ = weights[(iterationIdx * numberWorkers + workerIdx) * chunkSize:(iterationIdx * numberWorkers + workerIdx + 1) * chunkSize]
                p_ = propertyIndex[(iterationIdx * numberWorkers + workerIdx) * chunkSize:(iterationIdx * numberWorkers + workerIdx + 1) * chunkSize]

                # store intermediate result in workers own memory
                result[workerIdx + 1] += fastHistogram(w_, p_, typicalWeight, numberPropertyIndexValues)

        # sum worker results, to final accumulation
        result[0,:] = np.sum(result[1:], axis=0)
        # reset worker memory to avoid double counting
        result[1:, :] = 0

    # return final values of summation column
    return result[0, :]

@typeguard.typechecked
def getPropertyIndexHistogram(
        fileName : str,
        speciesName : str,
        propertyName : str,
        numberPropertyIndexValues : int,
        numberWorkers : int,
        chunkSize : int):
    """ returns the atomic state data of an openPMD particle output of a simulation

        @param fileName absolute/relative path of output
        @param speciesName string identifier of species
        @param numberAtomicStates number of unique atomic states in output
        @param numberWorkers number of independent threads to use
        @param chunkSize number of particles to try to to pass each worker in a chunk

        @returns np.array((numberTimeSteps, numberAtomicStates))= accumulatedWeight/scalingFactor, np.array(numberIterations)= time, scalingFactor
    """
    series = opmd.Series(fileName, opmd.Access.read_only)

    numberIterations = np.shape(series.iterations)[0]

    accumulatedWeight = np.empty((numberIterations, numberPropertyIndexValues))
    timeSteps = np.empty(numberIterations, dtype='f8')

    print(fileName + " iteration:")
    for i, stepIdx in tqdm(enumerate(series.iterations)):

        # get subGroup for simulation iteration
        step = series.iterations[stepIdx]

        # get subGroup of specific species
        species = step.particles[speciesName]

        # get recordComponents
        weightingRecordComponent = species["weighting"][opmd.Mesh_Record_Component.SCALAR]
        propertyIndexRecordComponent = species[propertyName][opmd.Mesh_Record_Component.SCALAR]

        # mark to be loaded, @todo give option to load chunkwise
        weights = weightingRecordComponent.load_chunk()
        propertyIndices = propertyIndexRecordComponent.load_chunk()

        # load data
        series.flush()

        typicalWeight = np.mean(weights[0:100])

        accumulatedWeight[i] = callFastHistogramInParallel(
            weights,
            propertyIndices,
            typicalWeight,
            numberPropertyIndexValues,
            numberWorkers,
            chunkSize)
        timeSteps[i] = step.get_attribute("time") * step.get_attribute("timeUnitSI")

        del weights
        del propertyIndices

    # delete series
    del series

    return accumulatedWeight, timeSteps, typicalWeight

@typeguard.typechecked
def getAtomicPopulationData(
        fileName : str,
        speciesName : str,
        numberAtomicStates : int,
        numberWorkers : int,
        chunkSize : int):
    """ returns the atomic state data of an openPMD particle output of a simulation

        @param fileName absolute/relative path of output
        @param speciesName string identifier of species
        @param numberAtomicStates number of unique atomic states in output
        @param numberWorkers number of independent threads to use
        @param chunkSize number of particles to try to to pass each worker in a chunk

        @returns np.array((numberTimeSteps, numberAtomicStates))= accumulatedWeight/scalingFactor, np.array(numberIterations)= time, scalingFactor
    """

    return getPropertyIndexHistogram(
        fileName,
        speciesName,
        "atomicStateCollectionIndex",
        numberAtomicStates,
        numberWorkers,
        chunkSize)

@typeguard.typechecked
def getBoundElectronsData(
    fileName : str,
    speciesName : str,
    numberChargeStates : int,
    numberWorkers : int,
    chunkSize : int):
    """
        returns the boundElectrons data of an atomic particle output of a simulation

        @param fileName absolute/relative path of output
        @param speciesName string identifier of species
        @param numberAtomicStates number of unique charge states in output
        @param numberWorkers number of independent threads to use
        @param chunkSize number of particles to try to to pass each worker in a chunk

        @returns np.array((numberTimeSteps, numberChargeStates))= accumulatedWeight/scalingFactor, np.array(numberIterations)= time, scalingFactor
    """
    return getPropertyIndexHistogram(
        fileName,
        speciesName,
        "boundElectrons",
        numberChargeStates,
        numberWorkers,
        chunkSize)