"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from .SCFlyTools import AtomicConfigNumberConversion as conv
from .AtomicStateDiffPlotter import AtomicStateDiffPlotter

import typeguard

@typeguard.typechecked
class AtomicStateDiffOverviewPlotter(AtomicStateDiffPlotter):
    # show every periodOfTimeTicks' on the time axis
    periodOfTimeTicks : int = 5

    def plot(self) -> None:
        """plot difference (sample - reference) for each atomic state and each step"""

        diff, timeSteps, atomicStates, timeSteps, axisDict, dataSetNameSample, dataSetNameReference, speciesDescriptor = self.getDiff()

        print(f"plotting overview of {dataSetNameSample} vs {dataSetNameReference}...")

        numberAtomicStates = np.shape(atomicStates)[0]
        numberTimeSteps = np.shape(timeSteps)[0]

        Y, X = np.meshgrid(np.arange(0,numberAtomicStates), np.arange(0,numberTimeSteps))

        # prepare plot
        figure = plt.figure(dpi=200, figsize=(20,30))
        axes = figure.add_subplot(111)
        title = axes.set_title(
            f"Difference of relative abundances of atomic states {dataSetNameSample} vs {dataSetNameReference}: "
            + self.plotName)
        axes.set_xlabel("PIC step")
        axes.set_ylabel("atomic states")

        yticks = np.arange(0, numberAtomicStates)
        ylabels = list(map(
            lambda atomicConfigNumber: str(conv.getLevelVector(
                atomicConfigNumber, speciesDescriptor.atomicNumber, speciesDescriptor.numLevels)),
            atomicStates))
        axes.set_yticks(yticks, ylabels)
        axes.yaxis.set_tick_params(labelsize=2)

        xticks = np.arange(0, numberTimeSteps)[::self.periodOfTimeTicks]
        axes.set_xticks(xticks, xticks)

        plt.pcolormesh(X, Y, diff, cmap=self.colorMap, norm=color.SymLogNorm(linthresh=1e-8),)
        plt.colorbar()
        figure.tight_layout()
        plt.savefig(f"{self.figureStoragePath}/AtomicPopulation_diff_{self.plotName}")
        plt.close(figure)
