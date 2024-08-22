"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import reader
from .SpeciesDescriptor import SpeciesDescriptor
from .SCFlyTools import AtomicConfigNumberConversion as conv
from .AtomicStatePlotter import AtomicStatePlotter

import copy
import typeguard
import numpy as np
import numpy.typing as npt

@typeguard.typechecked
class AtomicStateDiffPlotter(AtomicStatePlotter):
    # data source for reference to compare to
    referenceReaders : list[reader.StateDistributionReader]
    referenceSpeciesDescriptor : SpeciesDescriptor

    def check(self) -> None:
        if len(self.readerList) != 1:
            raise ValueError("may only create diff plot between two data sets")

    def readData(self) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, int], npt.NDArray[np.float64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]]:
        """appends referenceReaders results as last element of data"""

        self.check()

        data = super().readData()
        data.append(self.calculateMeanAndStdDevAbundance(self.referenceSpeciesDescriptor, self.referenceReaders))

        return data

    def getDataSetNames(self) -> tuple[str, str]:
        self.check()

        dataSetNamesReferenceReaders = []
        for reader in self.referenceReaders:
            dataSetNamesReferenceReaders.append(reader.dataSetName)

        dataSetNameReference = self.checkSamplesConsistent(dataSetNamesReferenceReaders)

        dataSetNamesSamples = []
        for reader in self.readerList[0]:
            dataSetNamesSamples.append(reader.dataSetName)

        dataSetNameSample = self.checkSamplesConsistent(dataSetNamesSamples)

        return dataSetNameSample, dataSetNameReference

    def getSpeciesDescriptor(self) -> SpeciesDescriptor:
        speciesDescriptorListCombined = copy.deepcopy(self.speciesDescriptorList)
        speciesDescriptorListCombined.append(self.referenceSpeciesDescriptor)

        return self.checkSamplesConsistent(speciesDescriptorListCombined)

    def checkAproximatelyEqual(self, a, b, relativeErrorLimit : float = 1.e-7):
        if not np.all(np.fromiter(map(
                lambda value_a, value_b: \
                    ((np.abs(value_a - value_b) / value_b) <= relativeErrorLimit) if value_b > 0 else value_b == 0,
                a, b),
                dtype=np.bool_)):
            raise ValueError("not within relativeErrorLimit equal")
        return a

    def getDiff(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint64], dict[str, int], str, str, SpeciesDescriptor]:
        data = self.readData()

        mean_sample, stdDev_sample, axisDict_sample, timeSteps_sample, atomicStates_sample, chargeStates_sample = data[0]
        mean_reference, stdDev_reference, axisDict_reference, timeSteps_reference, atomicStates_reference, chargeStates_reference = data[1]
        del chargeStates_sample, chargeStates_reference
        del data

        atomicStates = self.checkSamplesConsistent([atomicStates_sample, atomicStates_reference])
        del atomicStates_sample, atomicStates_reference

        axisDict = self.checkSamplesConsistent([axisDict_sample, axisDict_reference])
        del axisDict_sample, axisDict_reference

        # may come from different sources, and may differ slightly due to different rounding
        timeSteps = self.checkAproximatelyEqual(timeSteps_sample, timeSteps_reference)
        del timeSteps_sample, timeSteps_reference

        dataSetNameSample, dataSetNameReference = self.getDataSetNames()
        speciesDescriptor = self.getSpeciesDescriptor()

        diff = mean_sample - mean_reference

        return diff, timeSteps, atomicStates, axisDict, dataSetNameSample, dataSetNameReference, speciesDescriptor