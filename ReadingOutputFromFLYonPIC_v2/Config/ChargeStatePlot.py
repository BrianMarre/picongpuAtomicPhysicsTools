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
import numpy as np

class PlotConfig(pydantic.BaseModel):
    """Config object for plotting atomic populations"""
    # name of ion species in openPMD output
    speciesName : str
    # atomic number of ion species
    atomicNumber : int

    # number of threads to use for loading FLYonPIC data
    numberWorkers : int
    # number of particles to pass to each thread
    chunkSize : int

    #chargeStates to plot, empty means all
    chargeStatesToPlot : list[int]

    # base path for openPMD output fileNames
    basePath : str
    # FLYonPIC output fileNames, each a regex describing openPMD naming of openPMD output files
    outputFileNames : list[str]

    # colormap to use
    colorMap : typing.Any
    # number of colors in colormap
    numColorsInColorMap : int

    # path for storing processed input data
    processedDataStoragePath : str
    # path for storing plots
    figureStoragePath : str
    # descriptive name of data set, used for plot labeling and storage naming, must be unique
    dataName : str
    # True: read form raw simulation output, False: load previously processed data from processedDataStoragePath
    loadRaw : bool
