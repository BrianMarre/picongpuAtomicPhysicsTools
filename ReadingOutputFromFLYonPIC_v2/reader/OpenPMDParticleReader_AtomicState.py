"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from .OpenPMDParticleReader import OpenPMDParticleReader
from . import StateType

import numpy as np
import numpy.typing as npt

import typeguard

@typeguard.typechecked
class OpenPMDParticleReader_AtomicState(OpenPMDParticleReader):
    """read in atomic state distribution from openPMD particle dump"""

    # path of file FLYonPIC atomic state input data file, used for converting collection Index to configNumber
    FLYonPICAtomicStatesInputFileName : str

    RETURN_STATE_TYPE : int = StateType.ATOMIC_STATE

    def check(self) -> None:
        super().check()

        if(self.FLYonPICAtomicStatesInputFileName == ""):
            # no FLYonPIC atomic input data file
            raise ValueError("FLYonPIC atomic data input file required")

    def loadAtomicInputData(self) -> npt.NDArray[np.uint64]:
        return np.loadtxt(
            self.FLYonPICAtomicStatesInputFileName,
            dtype=[('atomicConfigNumber', 'u8'), ('excitationEnergy', 'f4')])['atomicConfigNumber']

    def read(self) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.uint64]], dict[str, int], tuple[np.float64]]:
        """see AtomicStateDistributionReader.py for documentation"""
        self.check()

        atomicConfigNumbers = self.loadAtomicInputData()
        numberAtomicStates = np.shape(atomicConfigNumbers)[0]

        accumulatedWeights, timeSteps, typicalWeight = self.getPropertyIndexHistogram(
            self.FLYonPICOpenPMDOutputFileName,
            self.speciesName,
            "atomicStateCollectionIndex",
            numberAtomicStates,
            self.numberWorkers,
            self.chunkSize)

        return accumulatedWeight, (timeSteps, atomicConfigNumbers), {"timeStep" : 0, "atomicState" : 1}, (typicalWeight,)
