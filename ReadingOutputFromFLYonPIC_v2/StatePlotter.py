"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import Plotter
from . import reader
from .SpeciesDescriptor import SpeciesDescriptor
from .SCFlyTools import AtomicConfigNumberConversion as conv

import numpy as np
import numpy.typing as npt

import math
import typeguard
import typing

@typeguard.typechecked
class StatePlotter(Plotter):
    # single reader will be plotted directly, list of readers will be plotted as mean and standard deviation of reader data
    readerList : list[reader.StateDistributionReader | list[reader.StateDistributionReader]]

    # colormap to use
    colorMap : typing.Any = None

    # number of colors in colormap
    numberColorsInColorMap : int = 0

    # one species descriptor for each readerList entry
    speciesDescriptorList : list[SpeciesDescriptor]

    lineWidth : float = 1.0

    def checkSamplesConsistent(self, sampleList : list) -> typing.Any:
        # check all samples element wise equal
        if len(sampleList) > 1:
            for sample in sampleList[1:]:
                if np.any(sample != sampleList[0]):
                    raise RuntimeError("samples inconsistent")
            return sample
        return sampleList[0]

    def getChargeStateColors(self, additionalIndices : list[int] = []):
        """@return dictionary assigning one color to each charge state"""

        if (self.colorMap is None) or (self.numberColorsInColorMap < 1):
            raise ValueError(f"{self} must specify a colorMap")

        colors = iter([self.colorMap(i) for i in range(self.numberColorsInColorMap)])

        if self.numberColorsInColorMap < (len(additionalIndices) + len(self.chargeStatesToPlot)):
            print("Warning: number of colors is less than requested unique lines colors, some colors may repeat!")

        ## assign all chargeStates a color
        colorChargeStates = {}
        for z in self.chargeStatesToPlot:
            try:
                colorChargeStates[z] = next(colors)
            except StopIteration:
                colors = iter([self.colorMap(i) for i in range(self.numberColorsInColorMap)])
                colorChargeStates[z] = next(colors)

        for index in additionalIndices:
            try:
                colorChargeStates[index] = next(colors)
            except StopIteration:
                colors = iter([self.colorMap(i) for i in range(self.numberColorsInColorMap)])
                colorChargeStates[index] = next(colors)

        return colorChargeStates

    def reduceToPerChargeState(self, atomicStatePopulationData, axisDict, atomicConfigNumbers, speciesDescriptor):
        shape = np.shape(atomicStatePopulationData)
        numberTimeSteps = shape[axisDict['timeStep']]
        numberAtomicStates = shape[axisDict['atomicState']]
        del shape

        numberChargeStates = speciesDescriptor.atomicNumber + 1

        # reduce to per charge state
        chargeStatePopulationData = np.zeros((numberTimeSteps, numberChargeStates))
        for i, atomicConfigNumber in enumerate(atomicConfigNumbers):
            chargeState = conv.getChargeState(
                atomicConfigNumber,
                speciesDescriptor.atomicNumber,
                speciesDescriptor.numberLevels)
            chargeStatePopulationData[:, int(chargeState)] += np.take(
                atomicStatePopulationData, i, axisDict["atomicState"])

        axisDict = {'timeStep' : 0, 'chargeState' : 1}

        chargeStates = np.arange(numberChargeStates)

        return chargeStatePopulationData, axisDict, chargeStates

    def _readSamples(
            self,
            readerListEntry : list[reader.StateDistributionReader],
            targetStateType : int ,
            speciesDescriptor : SpeciesDescriptor):

        # load  data
        populationDataSamples : list[npt.NDArray[np.float64]] = []
        timeStepsSamples : list[npt.NDArray[np.float64]] = []
        statesSamples : list[npt.NDArray[typing.Any]] = []
        axisDictSamples : list[dict[str, int]] = []
        scalingFactorSamples : list[np.float64] = []

        for readerInList in readerListEntry:
            tuple_sample = readerInList.read()

            if readerInList.RETURN_STATE_TYPE != targetStateType:
                atomicPopulationData, axisValuesTuple, axisDict, scalingFactorTuple = tuple_sample

                chargeStateData, axisDict, chargeStates = self.reduceToPerChargeState(
                    atomicPopulationData, axisDict, axisValuesTuple[1], speciesDescriptor)

                tuple_sample = (chargeStateData, (axisValuesTuple[0], chargeStates), axisDict, scalingFactorTuple)

            populationDataSamples.append(tuple_sample[0])
            timeStepsSamples.append(tuple_sample[1][0])
            statesSamples.append(tuple_sample[1][1])
            axisDictSamples.append(tuple_sample[2])
            scalingFactorSamples.append(tuple_sample[3][0])

        return populationDataSamples, timeStepsSamples, statesSamples, axisDictSamples, scalingFactorSamples

    def calculateMeanAndStdDevAbundance(
            self,
            speciesDescriptor : SpeciesDescriptor,
            readerList : list[reader.StateDistributionReader],
            targetStateType : int,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, int], npt.NDArray[np.float64], npt.NDArray[typing.Any]]:
        populationDataSamples, timeStepsSamples, statesSamples, axisDictSamples, scalingFactorSamples \
            = self._readSamples(readerList, targetStateType, speciesDescriptor)

        timeSteps = self.checkSamplesConsistent(timeStepsSamples)
        states = self.checkSamplesConsistent(statesSamples)
        axisDict = self.checkSamplesConsistent(axisDictSamples)
        del timeStepsSamples
        del statesSamples
        del axisDictSamples

        numberSamples = len(readerList)
        numberStates = np.shape(states)[0]
        numberIterations = np.shape(timeSteps)[0]

        # throw data into common to array, must be done here since number of samples not previously known
        data = np.empty((numberSamples, numberIterations, numberStates), dtype='f8')
        for i, sampleData in enumerate(populationDataSamples):
            data[i] = sampleData
        del sampleData

        # calculate total density
        totalDensity = np.empty((numberSamples, numberIterations), dtype='f8')
        for sample in range(numberSamples):
            for iteration in range(numberIterations):
                totalDensity[sample,iteration] = math.fsum(data[sample, iteration])

        # convert to relative abundances -> may disregard scaling factor
        data = data / totalDensity[:, :, np.newaxis]

        # calculate mean abundance and standard deviation
        mean = np.mean(data, axis = 0)

        if numberSamples > 1 :
            stdDev = np.std(data, axis = 0, ddof = 1)
        else:
            stdDev = np.zeros((numberIterations, numberStates))

        return mean, stdDev, axisDict, timeSteps, states

    def getCommonReturnStateType(
            self,
            readerList : list[reader.StateDistributionReader],
            targetStateType : int = reader.StateType.ATOMIC_STATE) -> int:
        """
        will return reader.StateType.ATOMIC_STATE if for all readers
            .RETURN_STATE_TYPE == targetStateType == reader.StateType.ATOMIC_STATE, otherwise will return
            reader.StateType.CHARGE_STATE
        """
        return_state_type = targetStateType

        for i in readerList:
            if i.RETURN_STATE_TYPE != return_state_type:
                return_state_type = reader.StateType.CHARGE_STATE

        return return_state_type

    def readData(self, forcedTargetStateType : int = reader.StateType.ATOMIC_STATE) -> list[tuple]:
        """
        @returns mean, stdDev, axisDict, timeSteps, states[, chargeState]
        """

        if len(self.readerList) <= 0:
            raise ValueError("need at least one reader to be able to plot something")

        data = []

        for readerListEntry, speciesDescriptor in zip(self.readerList, self.speciesDescriptorList):
            if isinstance(readerListEntry, list):
                targetStateType = self.getCommonReturnStateType(readerListEntry, forcedTargetStateType)

                data.append(self.calculateMeanAndStdDevAbundance(speciesDescriptor, readerListEntry, forcedTargetStateType))
            else:
                targetStateType = self.getCommonReturnStateType([readerListEntry], forcedTargetStateType)
                data.append(self.calculateMeanAndStdDevAbundance(
                    speciesDescriptor,
                    [readerListEntry],
                    targetStateType))

        return data
