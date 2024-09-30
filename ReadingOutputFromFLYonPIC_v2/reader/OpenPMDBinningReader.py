"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import StateDistributionReader
from . import StateType

import numpy as np
import numpy.typing as npt
import typeguard
import openpmd_api as opmd

@typeguard.typechecked
class OpenPMDBinningReader(StateDistributionReader):
    """read atomic state distribution PIConGPU binning output"""

    binnerOutputFileName : str
    # openPMD output file name, a regex describing openPMD naming of openPMD output files, see openPMD-api for details

    FLYonPICAtomicStatesInputDataFileName : str
    # path of file FLYonPIC atomic state input data file, used for converting collection Index to configNumber

    RETURN_STATE_TYPE : int = StateType.ATOMIC_STATE

    def loadAtomicStateData(self) -> npt.NDArray[np.uint64]:
        # load atomic input Data to get conversion atomicStateCollectionIndex to atomicConfigNumber
        atomicConfigNumbers = np.loadtxt(
            self.FLYonPICAtomicStatesInputDataFileName,
            dtype=[('atomicConfigNumber', 'u8'), ('excitationEnergy', 'f4')])['atomicConfigNumber']
        return atomicConfigNumbers

    def read(self) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.uint64]], dict[str, int], tuple[np.float64]]:
        """
        see AtomicStateDistributionReader.py for documentation

        @attention assumes first axis to be time axis and second axis to be atomic state axis
        """
        atomicConfigNumbers = self.loadAtomicStateData()

        series = opmd.Series(self.binnerOutputFileName, opmd.Access.read_only)
        listIterations = list(series.iterations)

        numberIterations = len(listIterations)
        if numberIterations == 0:
            raise ValueError("output: " + self.binnerOutputFileName + " contains 0 iterations")

        # get number of time steps in each dump
        numberTimeStepsPerDump, numberAtomicStatesInDump = np.shape(
            series.iterations[listIterations[0]].meshes["Binning"][opmd.Mesh_Record_Component.SCALAR])

        # check number of atomic state is consistent
        numberAtomicStates = np.shape(atomicConfigNumbers)[0]
        if numberAtomicStatesInDump != numberAtomicStates:
            raise ValueError("number of atomic states in binning output does not match the number of atomic states in the input file")
        del numberAtomicStatesInDump

        # allocate global array, for continuous storage of data
        accumulatedWeights = np.empty((numberIterations * numberTimeStepsPerDump, numberAtomicStates), dtype = "f8")
        timeSteps = np.empty(numberIterations * numberTimeStepsPerDump, dtype="f8")

        # iterations
        for i, stepIdx in enumerate(listIterations):
            step = series.iterations[stepIdx]

            binningMesh = step.meshes["Binning"]
            binningRecordComponent = binningMesh[opmd.Mesh_Record_Component.SCALAR]

            # @todo implement chunking, BrianMarre, 2024
            binning_data_step = binningRecordComponent.load_chunk()
            series.flush()

            accumulatedWeights[i*numberTimeStepsPerDump:(i+1)*numberTimeStepsPerDump] = binning_data_step

            timeDumpStep = step.get_attribute("time")
            diffTimeBinToDumpStep = -(numberTimeStepsPerDump - np.array(binningMesh.get_attribute("time_axis_bin_edges")[1:]))

            timeSteps[i*numberTimeStepsPerDump:(i+1)*numberTimeStepsPerDump] = (
                (timeDumpStep + diffTimeBinToDumpStep) * step.get_attribute("timeUnitSI"))


        scalingFactor = binningMesh.get_attribute("weightScalingFactor")

        return accumulatedWeights, (timeSteps, atomicConfigNumbers), {"timeStep":0, "atomicState":1}, (np.float64(1.0),)
