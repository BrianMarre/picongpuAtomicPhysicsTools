"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 PIConGPU contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import typeguard

@typeguard.typechecked
class TimingDataPlot:
    def __init__(
            self,
            caseBasePaths : list[str],
            caseFileNames : list[list[str]],
            caseDataNames : list[str],
            fileName : str,
            plotCleanUpAndInit : bool):
        self.caseBasePaths = caseBasePaths
        self.caseFileNames = caseFileNames
        self.caseDataNames = caseDataNames
        self.fileName = fileName
        self.plotCleanUpAndInit = plotCleanUpAndInit
