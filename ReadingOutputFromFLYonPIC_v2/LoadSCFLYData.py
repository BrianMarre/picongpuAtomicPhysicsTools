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
from .SCFlyTools import AtomicConfigNumberConversion as conv
from . import Config as cfg

@typeguard.typechecked
def loadSCFLYdata(config : cfg.AtomicPopulationPlot.PlotConfig):
    if(config.SCFLYatomicStateNamingFile == ""):
        return None, None, None, None
    if(config.SCFLYOutputFileName == ""):
        return None, None, None, None

    # load data
    atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps = Reader.SCFLY.getSCFLY_PopulationData(
        config.SCFLYOutputFileName,
        Reader.SCFLY.readSCFLYNames(config.SCFLYatomicStateNamingFile, config.atomicNumber, config.numLevels)[0])

    # calculate total densities
    assert((len(np.shape(atomicPopulationData)) == 2) and (axisDict['timeStep'] == 0))
    totalDensity = np.fromiter(map(lambda timeStep: math.fsum(timeStep) , atomicPopulationData), dtype='f8')
    # calculate relative abundances
    atomicPopulationData = atomicPopulationData / totalDensity[:, np.newaxis]

    # sort data according to FLYonPIC sorting
    chargeStates = np.fromiter(map(
        lambda atomicConfigNumber : conv.getChargeState(atomicConfigNumber, config.atomicNumber, config.numLevels),
        atomicConfigNumbers), dtype = 'u1')
    sortedIndices = np.lexsort((atomicConfigNumbers, chargeStates))
    del chargeStates

    atomicConfigNumbersSorted = atomicConfigNumbers[sortedIndices]
    atomicPopulationDataSorted = atomicPopulationData[:, sortedIndices]
    del atomicConfigNumbers
    del atomicPopulationData

    return atomicPopulationDataSorted, axisDict, atomicConfigNumbersSorted, timeSteps
