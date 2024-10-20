"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from .SCFlyTools import AtomicConfigNumberConversion as conv
from . import reader
from .StatePlotter import StatePlotter
from .SpeciesDescriptor import SpeciesDescriptor

import numpy as np
import numpy.typing as npt

import typeguard

@typeguard.typechecked
class AtomicStatePlotter(StatePlotter):

    def checkAllReaderReturnAtomicStates(self, readerList : list[reader.StateDistributionReader]):
        if self.getCommonReturnStateType(readerList) != reader.StateType.ATOMIC_STATE:
            raise ValueError(f"at least one reader in: {readerList}\n does not return atomic state population data")

    def calculateMeanAndStdDevAbundance(
            self,
            speciesDescriptor : SpeciesDescriptor,
            readerList : list[reader.StateDistributionReader],
            targetStateType : int
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, int], npt.NDArray[np.float64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
        """append chargeStates to parent class output"""

        if targetStateType != reader.StateType.ATOMIC_STATE:
            raise ValueError("targetStateType must be reader.StateType.ATOMIC_STATE")

        self.checkAllReaderReturnAtomicStates(readerList)
        mean, stdDev, axisDict, timeSteps, atomicStates = super().calculateMeanAndStdDevAbundance(
            speciesDescriptor, readerList, reader.StateType.ATOMIC_STATE)

        chargeStates = np.fromiter(map(
                    lambda atomicStateCollectionIndex: conv.getChargeState(
                        atomicStateCollectionIndex,
                        speciesDescriptor.atomicNumber,
                        speciesDescriptor.numberLevels),
                    atomicStates), dtype="u1")

        return mean, stdDev, axisDict, timeSteps, atomicStates, chargeStates

    def plot(self) -> None:
        """plot atomic state population over time"""
        raise NotImplementedError("need to be implemented by daughter classes")

        # @todo rework
        self.plotRelativeAbundanceOverall(data)
