"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""
import openpmd_api as io

import numba
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import typeguard

@numba.jit(fastmath=True, nopython=True, cache=True)
def fastHistogram(weights, atomicStateCollectionIndices, typicalWeight, numberAtomicStates):
    """ kernel for accumulating weight of entries according to their atomicStateCollectionIndex

        @param weights array of weight values
        @param atomicStateCollectionIndices array of atomicStateCollectionIndices corresponding to weights
        @param typicalWeight normalization factor for weight accumulation
        @param numberAtomicStates number of atomic states in the input,
            all entries in atomicStateCollectionIndices must be < numberAtomicStates
    """

    # number of possible atomicStateCollectionIndice Values
    result = np.zeros(numberAtomicStates)

    for entryIdx in range(atomicStateCollectionIndices.shape[0]):
        state = atomicStateCollectionIndices[entryIdx]
        result[state] += weights[entryIdx] / typicalWeight

    return result

@numba.jit(cache=True, parallel=True)
def callFastHistogramInParallel(weights, atomicStates, typicalWeight, numberAtomicStates, numberWorkers, chunkSize):
    """ stage for parallel call of fast histogram kernel

        @param weights array of weight values
        @param atomicStates array of atomicStateCollectionIndices corresponding to weights
        @param typicalWeight normalization factor for weight accumulation
        @param numberAtomicStates number of atomic states in the input,
            all entries in atomicStateCollectionIndices must be < numberAtomicStates
        @param numberWorkers number of parallel worker to use
        @param chunkSize number of entries to attempt to pass to each worker each iteration
    """

    numberChunks = int(np.ceil(weights.shape[0] / chunkSize))
    numberIterations = int(np.ceil(numberChunks / numberWorkers))

    result = np.zeros((numberWorkers + 1, numberAtomicStates), dtype=np.float32)

    for iterationIdx in range(numberIterations):
        for workerIdx in numba.prange(numberWorkers):
            if (numberChunks > (iterationIdx * numberWorkers + workerIdx)):
                # get chunks of allocated memory, will return empty views if no data remaining
                w_ = weights[(iterationIdx * numberWorkers + workerIdx) * chunkSize:(iterationIdx * numberWorkers + workerIdx + 1) * chunkSize]
                s_ = atomicStates[(iterationIdx * numberWorkers + workerIdx) * chunkSize:(iterationIdx * numberWorkers + workerIdx + 1) * chunkSize]

                # store intermediate result in workers own memory
                result[workerIdx + 1] += fastHistogram(w_, s_, typicalWeight, numberAtomicStates)

        # sum worker results, to final accumulation
        result[0,:] = np.sum(result[1:], axis=0)
        # reset worker memory to avoid double counting
        result[1:, :] = 0

    # return final values of summation column
    return result[0, :]

@typeguard.typechecked
def getAtomicPopulationData(
        filename : str,
        speciesName : str,
        numberAtomicStates : int,
        numberWorkers : int,
        chunkSize : int):
    """ returns the atomic state data of an openPMD particle output of a simulation

        Paramters:
            filename(string-like) ... filenames of data input
            speciesName(string-like) ... string identifier of species
            collectionIndex_to_atomicConfigNumber .. numpy.array[collectionIndex] = atomicConfigNumber as used in
                FYLonPIC input

        @returns np.array(numberIterations)= time and np.array((numberTimeSteps, numberAtomicStates))= accumulatedWeight
    """
    series = io.Series(filename, io.Access.read_only)

    numberIterations = np.shape(series.iterations)[0]

    accumulatedWeight = np.empty((numberIterations, numberAtomicStates))
    timeSteps = np.empty(numberIterations, dtype='f8')

    print(filename + " iteration:")
    for i, stepIdx in tqdm(enumerate(series.iterations)):

        # get subGroup for simulation iteration
        step = series.iterations[stepIdx]

        # get subGroup of specific species
        species = step.particles[speciesName]

        # get recordComponents
        weightingRecordComponent = species["weighting"][io.Mesh_Record_Component.SCALAR]
        atomicStateCollectionIndexRecordComponent = species["atomicStateCollectionIndex"][io.Mesh_Record_Component.SCALAR]

        # mark to be loaded, @todo give option to load chunkwise
        weights = weightingRecordComponent.load_chunk()
        atomicStateCollectionIndices = atomicStateCollectionIndexRecordComponent.load_chunk()

        # load data
        series.flush()

        typicalWeight = np.mean(weights[0:100])

        accumulatedWeight[i] = callFastHistogramInParallel(
            weights,
            atomicStateCollectionIndices,
            typicalWeight,
            numberAtomicStates,
            numberWorkers,
            chunkSize)
        timeSteps[i] = step.get_attribute("time") * step.get_attribute("timeUnitSI")

        del weights
        del atomicStateCollectionIndices

    # delete series
    del series

    return accumulatedWeight, timeSteps, typicalWeight
