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

    # True: read form raw simulation output, False: load previously processed data from processedDataStoragePath
    loadRaw : bool
