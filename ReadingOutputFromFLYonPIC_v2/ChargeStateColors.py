"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import typeguard
from . import Config as cfg

@typeguard.typechecked
def getChargeStateColors(config : cfg.AtomicPopulationPlot.PlotConfig, additionalIndices : list[int] = []):
    """@return dictionary assigning one color to each charge state"""
    colors = iter([config.colorMap(i) for i in range(config.numColorsInColorMap)])

    ## assign all chargeStates a color
    colorChargeStates = {}
    for z in config.chargeStatesToPlot:
        try:
            colorChargeStates[z] = next(colors)
        except StopIteration:
            colors = iter([config.colorMap(i) for i in range(config.numColorsInColorMap)])
            colorChargeStates[z] = next(colors)

    for index in additionalIndices:
        try:
            colorChargeStates[index] = next(colors)
        except StopIteration:
            colors = iter([config.colorMap(i) for i in range(config.numColorsInColorMap)])
            colorChargeStates[index] = next(colors)

    return colorChargeStates
