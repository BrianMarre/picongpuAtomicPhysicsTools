"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
import typing

class ScanConfig(pydantic.BaseModel):
    atomicNumber : int
    SCFLYatomicStateNamingFile : str
    atomicDataInputFile : str

    # eV
    electronTemperatures : list[float]
    # 1/cm^3
    ionDensities : list[float]
    # s
    timePoints : list[float]
    # initial state occupation number vector
    initialStateLevelVector : tuple[int, ...]

    outputBasePath : str
    outputFileName : str
    SCFLYBinaryPath : str
    dataSeriesName : str

    numberStatesToPlot : int
    colorMap : typing.Any
    numColorsInColorMap : int

    processedDataStoragePath : str
    figureStoragePath : str

    runSCFLY : bool
    plotSummary : bool
