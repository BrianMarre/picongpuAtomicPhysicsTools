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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as scale

@typeguard.typechecked
class AtomicStateDiffLineoutPlotter(AtomicStateDiffPlotter):
    # indices of steps to plot line out for
    lineoutSteps : list[int]

    # figure width and height in inches
    figsize : tuple[float, float]

    # linearThreshhold
    linthresh : float = 1.e-8

    def plot(self) -> None:
        """plot difference lineout of (sample - reference) for all atomic state for a set of time steps"""
        diff, timeSteps, atomicStates, axisDict, dataSetNameSample, dataSetNameReference, speciesDescriptor = self.getDiff()

        print(f"plotting lineouts of {dataSetNameSample} vs {dataSetNameReference}...")

        numberAtomicStates = np.shape(atomicStates)[0]

        numberFigures = len(self.lineoutSteps)

        figure, axes = plt.subplots(numberFigures, 1, dpi=200, figsize=self.figsize)
        title = figure.suptitle(f" {dataSetNameSample} vs {dataSetNameReference} relative abundances: " + self.plotName)

        maxAbsDiff = np.max(np.abs(np.take(diff, self.lineoutSteps, axisDict["atomicState"]))) * 1.1

        atomicStateCollectionIndices = np.arange(numberAtomicStates)

        # plot all time steps
        for i, stepIdx in enumerate(self.lineoutSteps):
            axePair = axes[i]
            axePair.set_title("step: " + str(stepIdx))
            axePair.plot(atomicStateCollectionIndices, np.take(diff, stepIdx, axisDict["timeStep"]), linestyle="-", marker="x")
            axePair.set_xticks([])
            axePair.set_xlim((0, numberAtomicStates))
            axePair.set_ylim((-maxAbsDiff, maxAbsDiff))
            axePair.set_yscale(scale.SymmetricalLogScale(axePair.yaxis, linthresh=self.linthresh))

        xlabels = np.fromiter(
            map(lambda atomicConfigNumber: str(conv.getLevelVector(atomicConfigNumber,
                                                                speciesDescriptor.atomicNumber,
                                                                speciesDescriptor.numberLevels)),
            atomicStates), dtype='U20')
        axePair.set_xticks(atomicStateCollectionIndices, xlabels)
        axePair.set_ylabel(f"({dataSetNameSample} - {dataSetNameReference}) relative abundance")
        axePair.set_xlabel("atomic states")
        plt.setp(axePair.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        axePair.xaxis.set_tick_params(labelsize=2)

        print("\t saving ...")
        plt.tight_layout()
        plt.savefig(self.figureStoragePath + "AtomicState_DiffLineout_" + self.plotName, bbox_extra_artists=(title,))
        plt.close(figure)