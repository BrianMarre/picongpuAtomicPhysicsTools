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
import typing

import numpy.typing as npt

@typeguard.typechecked
class Reader(pydantic.BaseModel):
    dataSetName : str

    def read(self) -> tuple[npt.NDArray, list[npt.NDArray], dict[str, int], tuple[typing.Any]]:
        """
        read in data from source

        @returns
            data table,
            index values corresponding to each axis of data table,
            axisDict map of name to data table axis describing the physical meaning of an axis,
            tuple of additional values necessary for interpretation of data table values like for example scaling factors
        """
        raise NotImplementedError("abstract interface only")
