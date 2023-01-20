"""atomicPhysics rate calculation script
This file is part of the PIConGPU.
Copyright 2023 PIConGPU contributors
Authors: Brian Marre, Axel Huebl
License: GPLv3+
"""

import numpy as np
import scipy.constants.physical_constants as const
import scipy.special as scipy

class BoundBoundTransitions:
    def _gaunt(U, cxin1, cxin2, cxin3, cxin4, cxin5):
        """ calculate gaunt factor

            @param U     float (energy interacting electron)/(delta Energy Transition)
            @param cxin1 float gaunt approximation coefficient 1
            @param cxin2 float gaunt approximation coefficient 2
            @param cxin3 float gaunt approximation coefficient 3
            @param cxin4 float gaunt approximation coefficient 4
            @param cxin5 float gaunt approximation coefficient 5
        """

        if (U < 1.):
            return 0.
        else:
            return cxin1 * np.log(U) + cxin2 + cxin3 / (U + cxin5) + cxin4/(U + cxin5)**2

    def _multiplicity(levelVector)
        """ get degeneracy of atomicState with given levelVector
        """
        result = 1.
        for i,n_i in enumerate(levelVector):
            result *= scipy.comb(2*(i+1)**2, n_i)
        return result

    def collsionalBoundBoundCrossSection(self, excitation, energyElectron, energyDiffLowerUpper,
        collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5)):
        """ cross section for de-/excitation transition due to interaction with free electron

            @param excitation bool true =^= excitation, false =^= deexcitation
            @param energyElectron float energy of interacting free electron, [eV]
            @param energyDiffLowerUpper upperStateEnergy - lowerStateEnergy, [eV]
            @param collisionalOscillatorStrength of the transition, unitless
            @param cxin1 float gaunt approximation coefficient 1
            @param cxin2 float gaunt approximation coefficient 2
            @param cxin3 float gaunt approximation coefficient 3
            @param cxin4 float gaunt approximation coefficient 4
            @param cxin5 float gaunt approximation coefficient 5

            @return unit 1e6b
        """
        U = energyElectron/energyDiffLowerUpper # unitless
        E_Rydberg = const.value("Rydberg constant times hc in eV") # eV

        a0 = const.value("Bohr radius") # m
        c0 = 8 * (np.pi * a0)**2 / np.sqrt(3.) # m^2

        crossSection = (c0 * (E_Rydberg / energyDiffLowerUpper)**2 * collisionalOscillatorStrength
            * (energyDiffLowerUpper / energyElectron)
            * self._gaunt(U, cxin1, cxin2, cxin3, cxin4, cxin5)) / 1e22 # 1e6b

        if excitation:
            return crossSection # 1e6
        else:
            statisticalRatio = (
                self._multiplicity(lowerStateLevelVector)/self._multiplicity(upperStateLevelVector)
                * (energyElectron + energyDiffLowerUpper) / energyElectron) # unitless
            return statisticalRatio * crossSection # 1e6

    def rateCollisonalBoundBoundTransition(excitation, )