"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import json

from . import ChargeStateColors

from . import ReduceToPerChargeState

import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.scale as scale

from labellines import labelLines

@typeguard.typechecked
def plot_all(
    tasks_general : list[cfg.AtomicPopulationPlot.PlotConfig],
    tasks_diff : list[cfg.AtomicPopulationPlot.PlotConfig],
    tasks_recombination : list[cfg.AtomicPopulationPlot.PlotConfig],
    FLYonPICInitialChargeState : int = 0):

from . import Plotter
from . import reader

import numpy as np
import numyp.typing as npt

import math
import logging
import typeguard
import pydantic

class SpeciesDescriptors(pydantic.BaseModel):
    # atomic number of ion species
    atomicNumber : int
    # maximum principal quantum number used
    numLevels : int

@typeguard.typechecked
class AtomicStatePlotter(Plotter):
    # single reader will be plotted directly, list readers will be plotted as mean and standard deviation of reader results
    readerList : list[reader.StateDistributionReader | list[reader.StateDistributionReader]]

    speciesDescriptorList : list[SpeciesDescriptors]

    # chargeStates to plot
    chargeStatesToPlot : list[int]

    # colormap to use
    colorMap : typing.Any
    # number of colors in colormap
    numColorsInColorMap : int

    # path for storing plots
    figureStoragePath : str

    # descriptive name of data set, used for plot labeling and storage naming, must be unique
    dataName : str

    # minimum population for inclusion in plot
    minimumPopulation : float

    def checkSamplesConsitent[T](self, sampleList : list[T]) -> T:
        # check all samples element wise equal
        if len(smapleList) > 1:
            for sample in sampleList[1:]:
                if np.any(sample != timeStepsSamples[0]):
                    raise RuntimeError("samples inconsistent")
            return sample
        return sampleList[0]

    def _readSamples(self, readerList : list[reader.StateDistributionReader]):
        # load  data
        populationDataSamples : list[npt.NDArray[np.float64]] = []
        timeStepsSamples : list[npt.NDArray[np.float64]] = []
        atomicStatesSamples : list[npt.NDArray[np.uint64]] = []
        axisDictSamples : list[dict[str, int]] = []
        scalingFactorSamples : list[np.float64] = []

        for reader in readerListEntry:
            tuple_sample = reader.read()
            populationDataSamples.append(tuple_sample[0])
            timeStepsSamples.append(tuple_sample[1][0])
            atomicStatesSamples.append(tuple_sample[1][1])
            axisDictSamples.append(tuple_sample[2])
            scalingFactorSamples.append(tuple_sample[3][0])

        return populationDataSamples, timeStepsSamples, atomicStatesSamples, axisDictSamples, scalingFactorSamples

    def calulateMeanAndStdDev(self, readerList : list[reader.StateDistributionReader]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, int], npt.NDArray[np.float64], npt.NDArray[np.uint64]]:
        populationDataSamples, timeStepsSamples, atomicStatesSamples, axisDictSamples, scalingFactorSamples = self._readSamples(readerList)

        timeSteps = self.checkSamplesConsitent(timeStepsSamples)
        atomicStates = self.checkSamplesConsitent(atomicStatesSamples)
        axisDict = self.checkSamplesConsitent(axisDictSamples)
        del timeStepsSamples
        del atomicStatesSamples
        del axisDictSamples

        numberSamples = len(readerList)
        numberAtomicStates = np.shape(atomicState)[0]
        numberIterations = np.shape[timeSteps][0]

        # throw data into common to array, must be done here since number of samples not previously known
        data = np.empty((numberSamples, numberIterations, numberAtomicStates), dtype='f8')
        for i, sample in enumerate(populationDataSamples):
            data[i] = sample
        del sampleListAtomicPopulationData

        # calculate total density
        totalDensity = np.empty((numberSamples, numberIterations), dtype='f8')
        for sample in range(numberSamples):
            for iteration in range(numberIterations):
                totalDensity[sample,iteration] = math.fsum(data[i, j])

        # convert to relative abundances -> may disregard scaling factor
        data = data / totalDensity[:, :, np.newaxis]

        # calculate mean abundance and standard deviation
        mean = np.mean(data, axis = 0)

        if numberSamples > 1 :
            stdDev = np.std(data, axis = 0, ddof = 1)
        else
            stdDev = np.zeros((numberIterations, numberAtomicStates))

        return mean, stdDev, axisDict, timeSteps, atomicStates

    def readData(self) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, int], npt.NDArray[np.float64], npt.NDArray[np.uint64]]]:
        if len(self.readerList) <= 0:
            raise ValueError("need at least one reader to be able to plot something")

        data = []

        for readerListEntry in self.readerList:
            if isinstance(readerListEntry, list):
                data.append(self.calculateMeanAndStdDev(readerListEntry))
            else:
                data.append([readerListEntry])

        return data

    def getChargeStateColors(self, additionalIndices : list[int] = []):
        """@return dictionary assigning one color to each charge state"""
        colors = iter([self.colorMap(i) for i in range(self.numColorsInColorMap)])

        if self.numColorsInColorMap < (len(additionalIndices) + len(self.chargeStatesToPlot)):
            print("Warning: number of colors is less than requested unique lines colors, some colors will repeat!")

        ## assign all chargeStates a color
        colorChargeStates = {}
        for z in self.chargeStatesToPlot:
            try:
                colorChargeStates[z] = next(colors)
            except StopIteration:
                colors = iter([self.colorMap(i) for i in range(self.numColorsInColorMap)])
                colorChargeStates[z] = next(colors)

        for index in additionalIndices:
            try:
                colorChargeStates[index] = next(colors)
            except StopIteration:
                colors = iter([self.colorMap(i) for i in range(self.numColorsInColorMap)])
                colorChargeStates[index] = next(colors)

        return colorChargeStates

    def plot(self) -> None:
        """plot absolute atomic state population over time"""
        raise NotImplementedError("need to be implemented by daughter classes")

        # @todo rework
        self.plotAbsolute(data)
        self.plotChargeStates(data)
        self.plotRelativeAbundanceOverall(data)
