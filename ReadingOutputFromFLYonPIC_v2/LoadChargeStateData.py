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
def loadChargeStateData(config : cfg.ChargeStatePlot.PlotConfig):

    numberSamples = len(config.outputFileNames)
    numberChargeStates = config.atomicNumber + 1

    # load in FLYonPIC data
    sampleListAtomicPopulationData = []
    sampleListTimeSteps = []
    sampleListTypicalWeights = []
    for fileName in config.outputFileNames:
        sampleAccumulatedWeights, sampleTimeSteps, typicalWeight = Reader.openPMD.getBoundElectronsData(
            config.basePath + fileName,
            config.speciesName,
            numberChargeStates,
            config.numberWorkers,
            config.chunkSize)

        sampleListAtomicPopulationData.append(sampleAccumulatedWeights)
        sampleListTimeSteps.append(sampleTimeSteps)
        sampleListTypicalWeights.append(typicalWeight)

    numberIterations = np.shape(sampleListTimeSteps[0])[0]

    # check for common time steps
    for sampleTimeSteps in sampleListTimeSteps[1:]:
        if np.any(sampleTimeSteps != sampleListTimeSteps[0]):
            raise RuntimeError("inconsistent time steps in samples")

    timeSteps = sampleListTimeSteps[0]
    del sampleListTimeSteps

    # throw data into common to array, with common scaling, must be done here since number of time steps not known before
    data = np.empty((numberSamples, numberIterations, numberChargeStates), dtype='f8')
    typicalWeight = np.mean(sampleListTypicalWeights)
    for i, sample in enumerate(sampleListAtomicPopulationData):
        data[i] = sample * sampleListTypicalWeights[i] / typicalWeight
    del sampleListAtomicPopulationData
    del sampleListTypicalWeights

    # calculate total density, in units of typical weighting
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

    axisDict = {'timeStep':0, 'chargeState':1}
    return mean, stdDev, axisDict, timeSteps
