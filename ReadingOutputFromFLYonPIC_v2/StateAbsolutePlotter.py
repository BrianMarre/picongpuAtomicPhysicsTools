"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from .StatePlotter import StatePlotter

class StateAbsolutePlotter(StatePlotter):
    # scale to use on relative abundance axis, set to empty string to get linear scale
    axisScale : str = "log"

    # description of linestyle descriptors to use in plots for each reader
    plotLineStyles : list[str]

    # minimum relative abundance
    minimumRelativeAbundance : float = 1.e-5

    # max time to include in the plot
    maxTime : float = -1

    # chargeStates to plot
    chargeStatesToPlot : list[int]