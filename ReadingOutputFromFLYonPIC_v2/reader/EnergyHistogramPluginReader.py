"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import Reader

import numpy as np
import numpy.typing as npt
import re

import typeguard

@typeguard.typechecked
class EnergyHistogramPluginReader(Reader):
    """read output from PIConGPU energy histogram"""

    # output file name, may be path absolute or relative to execution location
    outputFileName : str

    def readBins(self) -> tuple[npt.NDArray[np.float64], np.float64]:
        #read in energy values
        outputFile = open(self.outputFileName)

        with open(self.outputFileName) as fp:
            for i, line in enumerate(fp):
                headerline = line
                break

        regexHeaderBinEntry = r"( \d+\.*\d*)"
        energyBins = np.array(re.findall(regexHeaderBinEntry, headerline), dtype="f8")
        regexMinimum = "<(\d+\.*\d*)"
        minimumEnergy = np.float64(re.findall(regexMinimum, headerline)[0])

        return energyBins, minimumEnergy

    def read(self) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], dict[str, int], tuple[np.float64, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]]:

        energyBins, minimumEnergy = self.readBins()
        data = np.loadtxt(self.outputFileName, dtype="f8")

        # first collumn is the stepIndex
        timeSteps = data[:, 0]
        # second collumn is the underflow
        underflow = data [:, 1]
        # second last is overflow
        overflow = data[:,-2]
        # in between is actual data
        numberDensity = data[:,2:-2]

        return numberDensity, (timeSteps, energyBins), {"timeStep" : 0, "energy" : 1}, (minimumEnergy, (underflow, overflow))
