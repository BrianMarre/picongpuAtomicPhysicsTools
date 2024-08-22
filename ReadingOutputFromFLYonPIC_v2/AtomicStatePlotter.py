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
import pydantic
import typing

@typeguard.typechecked
class AtomicStatePlotter(Plotter):
    # single reader will be plotted directly, list of readers will be plotted as mean and standard deviation of reader data
    readerList : list[reader.StateDistributionReader | list[reader.StateDistributionReader]]

    # one species descriptor for each readerList entry
    speciesDescriptorList : list[SpeciesDescriptor]

    # colormap to use
    colorMap : typing.Any
    # number of colors in colormap
    numberColorsInColorMap : int

    # path for storing plots
    figureStoragePath : str

    # descriptive name of data set, used for plot labeling and storage naming, must be unique
    plotName : str

    def checkSamplesConsistent(self, sampleList : list) -> typing.Any:
        # check all samples element wise equal
        if len(sampleList) > 1:
            for sample in sampleList[1:]:
                if np.any(sample != sampleList[0]):
                    raise RuntimeError("samples inconsistent")
            return sample
        return sampleList[0]

    def _readSamples(self, readerListEntry : list[reader.StateDistributionReader]):
        # load  data
        populationDataSamples : list[npt.NDArray[np.float64]] = []
        timeStepsSamples : list[npt.NDArray[np.float64]] = []
        atomicStatesSamples : list[npt.NDArray[np.uint64]] = []
        axisDictSamples : list[dict[str, int]] = []
        scalingFactorSamples : list[np.float64] = []

        for readerInList in readerListEntry:
            tuple_sample = readerInList.read()
            populationDataSamples.append(tuple_sample[0])
            timeStepsSamples.append(tuple_sample[1][0])
            atomicStatesSamples.append(tuple_sample[1][1])
            axisDictSamples.append(tuple_sample[2])
            scalingFactorSamples.append(tuple_sample[3][0])

        return populationDataSamples, timeStepsSamples, atomicStatesSamples, axisDictSamples, scalingFactorSamples

    def calculateMeanAndStdDevAbundance(
            self,
            speciesDescriptor : SpeciesDescriptor,
            readerList : list[reader.StateDistributionReader]
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, int], npt.NDArray[np.float64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
        populationDataSamples, timeStepsSamples, atomicStatesSamples, axisDictSamples, scalingFactorSamples = self._readSamples(readerList)

        timeSteps = self.checkSamplesConsistent(timeStepsSamples)
        atomicStates = self.checkSamplesConsistent(atomicStatesSamples)
        axisDict = self.checkSamplesConsistent(axisDictSamples)
        del timeStepsSamples
        del atomicStatesSamples
        del axisDictSamples

        numberSamples = len(readerList)
        numberAtomicStates = np.shape(atomicStates)[0]
        numberIterations = np.shape(timeSteps)[0]

        # throw data into common to array, must be done here since number of samples not previously known
        data = np.empty((numberSamples, numberIterations, numberAtomicStates), dtype='f8')
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
            stdDev = np.zeros((numberIterations, numberAtomicStates))

        chargeStates = np.fromiter(map(
                    lambda atomicStateCollectionIndex: conv.getChargeState(
                        atomicStateCollectionIndex,
                        speciesDescriptor.atomicNumber,
                        speciesDescriptor.numberLevels),
                    atomicStates), dtype="u1")

        return mean, stdDev, axisDict, timeSteps, atomicStates, chargeStates

    def readData(self) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, int], npt.NDArray[np.float64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]]:

        if len(self.readerList) <= 0:
            raise ValueError("need at least one reader to be able to plot something")

        data = []

        for readerListEntry, speciesDescriptor in zip(self.readerList, self.speciesDescriptorList):
            if isinstance(readerListEntry, list):
                data.append(self.calculateMeanAndStdDevAbundance(speciesDescriptor, readerListEntry))
            else:
                data.append(self.calculateMeanAndStdDevAbundance(speciesDescriptor, [readerListEntry]))

        return data

    def getChargeStateColors(self, additionalIndices : list[int] = []):
        """@return dictionary assigning one color to each charge state"""
        colors = iter([self.colorMap(i) for i in range(self.numberColorsInColorMap)])

        if self.numberColorsInColorMap < (len(additionalIndices) + len(self.chargeStatesToPlot)):
            print("Warning: number of colors is less than requested unique lines colors, some colors will repeat!")

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

    def plot(self) -> None:
        """plot atomic state population over time"""
        raise NotImplementedError("need to be implemented by daughter classes")

            # @todo rework
        self.plotChargeStates(data)
        self.plotRelativeAbundanceOverall(data)
