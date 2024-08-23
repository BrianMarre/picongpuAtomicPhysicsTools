"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from .OpenPMDParticleReader import OpenPMDParticleReader

import numpy as np
import numpy.typing as npt

import typeguard

@typeguard.typechecked
class OpenPMDParticleReader_ChargeState(OpenPMDParticleReader):
    """read in charge state distribution from openPMD particle dump"""

    # atomic number of species
    atomicNumber : int

    def check(self) -> None:
        super().check()

        if self.atomicNumber <= 0:
            raise ValueError("atomicNumber must be >= 0")

    def read(self) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.uint64]], dict[str, int], tuple[np.float64]]:
        """see AtomicStateDistributionReader.py for documentation"""
        self.check()

        numberChargeStates = self.atomicNumber

        accumulatedWeights, timeSteps, typicalWeight = self.getPropertyIndexHistogram(
            self.FLYonPICOpenPMDOutputFileName,
            self.speciesName,
            "boundElectrons",
            numberChargeStates,
            self.numberWorkers,
            self.chunkSize)

        return accumulatedWeight, (timeSteps, np.arange(numberChargeStates, dtype="u8")), {"timeStep" : 0, "chargeState" : 1}, (typicalWeight,)
