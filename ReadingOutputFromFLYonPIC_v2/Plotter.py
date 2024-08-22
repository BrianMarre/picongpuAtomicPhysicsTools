"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
import typeguard

from . import reader

@typeguard.typechecked
class Plotter(pydantic.BaseModel):
    # collection of readers each representing one data source
    readerList : list

    # descriptive name of data set, used for plot labeling and storage naming, must be unique
    plotName : str

    # path for storing plots
    figureStoragePath : str

    def plot(self) -> None:
        """create plot from read by entries of readerList"""
        raise NotImplementedError("abstract interface only")