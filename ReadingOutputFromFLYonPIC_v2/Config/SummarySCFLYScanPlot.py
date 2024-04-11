"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic

class PlotConfig(pydantic.BaseModel):
    """plotting config for summary plot"""
    loadRawEachSCLFYSim : bool
    """switch for loading each SCFLY simulation results from raw data"""
    loadRawSummaryData : bool
    """switch for loading summary plot data results from raw data"""
    dataSetName : str
    """name to use for data set"""
