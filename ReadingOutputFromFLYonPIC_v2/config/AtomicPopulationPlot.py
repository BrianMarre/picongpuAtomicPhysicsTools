"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from. import OpenPMDReader

import pydantic
import typing



class PlotConfig(pydantic.BaseModel):
    """Config object for plotting atomic populations"""

    #openPMD reader config
    openPMDReaderConfig : OpenPMDReader.ReaderConfig

    #chargeStates to plot, empty means all
    chargeStatesToPlot : list[int]

    # name of states to plot
    numberStatesToPlot : int
    # colormap to use
    colorMap : typing.Any
    # number of colors in colormap
    numColorsInColorMap : int

    # path for storing plots
    figureStoragePath : str

    # descriptive name of data set, used for plot labeling and storage naming, must be unique
    dataName : str

    # True: read form raw simulation output, False: load previously processed data from processedDataStoragePath
    loadRaw : bool
