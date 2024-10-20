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

import typeguard
import typing

@typeguard.typechecked
class StateDistributionReader(Reader):

    # kind of state read from data
    RETURN_STATE_TYPE : int

    def read(self) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[typing.Any]], dict[str, int], tuple[np.float64]]:
        """
        read in state distribution over time from source

        @returns
            data table ... accumulated weight of macro particles for atomic state and time step : np.ndarray((numberTimeSteps, numberAtomicStates)),
            (list of times points, list of states),
            axisDict ... map of state and timeStep physics axis to axis index in the data table : {"atomicState/chargeState" : int, "timeStep": int},
            (scaling factor,)
        """
        raise NotImplementedError("abstract interface only")
