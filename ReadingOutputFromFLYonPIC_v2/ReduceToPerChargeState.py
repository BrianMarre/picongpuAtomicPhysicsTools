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

from .SCFlyTools import AtomicConfigNumberConversion as conv
from . import Config as cfg

@typeguard.typechecked
def reduceToPerChargeState(config : cfg.AtomicPopulationPlot.PlotConfig, populationData, axisDict, atomicConfigNumbers):
    shape = np.shape(populationData)
    numberTimeSteps = shape[axisDict['timeStep']]
    numberAtomicStates = shape[axisDict['atomicState']]
    del shape

    assert(numberAtomicStates == np.shape(atomicConfigNumbers)[0]), "shape of populationData not consistent with atomicConfigNumbers"
    assert(axisDict['timeStep'] == 0)
    assert(axisDict['atomicState'] == 1)

    # reduce to per charge state
    chargeStateData = np.zeros((numberTimeSteps, config.atomicNumber + 1))
    for i in range(numberAtomicStates):
        atomicConfigNumber = atomicConfigNumbers[i]
        chargeState = conv.getChargeState(atomicConfigNumber, config.atomicNumber, config.numLevels)
        chargeStateData[:, int(chargeState)] += populationData[:, i]

    axisDict = {'timeStep' : 0, 'chargeState' : 1}
    return chargeStateData, axisDict
