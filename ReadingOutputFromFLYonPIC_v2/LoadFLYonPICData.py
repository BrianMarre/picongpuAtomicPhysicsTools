"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import typeguard

import numpy as np
import math

from . import Reader
from . import Config as cfg

@typeguard.typechecked
def loadFLYonPICData(config : cfg.AtomicPopulationPlot.PlotConfig):
    if(config.FLYonPICAtomicStateInputDataFile == ""):
        return None, None, None, None, None
    if(len(config.FLYonPICOutputFileNames) == 0):
        return None, None, None, None, None

    # load atomic input Data to get conversion atomicStateCollectionIndex to atomicConfigNumber
    atomicConfigNumbers = np.loadtxt(
        config.FLYonPICAtomicStateInputDataFile,
        dtype=[('atomicConfigNumber', 'u8'), ('excitationEnergy', 'f4')])['atomicConfigNumber']

    numberSamples = len(config.FLYonPICOutputFileNames)
    numberAtomicStates = np.shape(atomicConfigNumbers)[0]

    # load in FLYonPIC data
    sampleListAtomicPopulationData = []
    sampleListTimeSteps = []
    for fileName in config.FLYonPICOutputFileNames:
        sampleAccumulatedWeights, sampleTimeSteps, typicalWeight = Reader.openPMD.getAtomicPopulationData(
            config.FLYonPICBasePath + fileName,
            config.speciesName,
            numberAtomicStates,
            config.numberWorkers,
            config.chunkSize)

        sampleListAtomicPopulationData.append(sampleAccumulatedWeights)
        sampleListTimeSteps.append(sampleTimeSteps)
        del typicalWeight

    numberIterations = np.shape(sampleListTimeSteps[0])[0]

    # check for common time steps
    for sampleTimeSteps in sampleListTimeSteps[1:]:
        if np.any(sampleTimeSteps != sampleListTimeSteps[0]):
            raise RuntimeError("inconsistent time steps in samples")

    timeSteps = sampleListTimeSteps[0]
    del sampleListTimeSteps

    # throw data into common to array, must be done here since
    data = np.empty((numberSamples, numberIterations, numberAtomicStates), dtype='f8')
    for i, sample in enumerate(sampleListAtomicPopulationData):
        data[i] = sample
    del sampleListAtomicPopulationData

    # calculate total density
    totalDensity = np.empty((numberSamples, numberIterations), dtype='f8')
    for i in range(numberSamples):
        for j in range(numberIterations):
            totalDensity[i,j] = math.fsum(data[i, j])

    # convert to relative abundances
    data = data / totalDensity[:, :, np.newaxis]

    # calculate mean abundance and standard deviation
    mean = np.mean(data, axis = 0)

    if (numberSamples > 1):
        stdDev = np.std(data, axis = 0, ddof = 1)
    else:
        stdDev = np.zeros((numberIterations, numberAtomicStates))

    axisDict = {'timeStep':0, 'atomicState':1}
    return mean, stdDev, axisDict, atomicConfigNumbers, timeSteps
