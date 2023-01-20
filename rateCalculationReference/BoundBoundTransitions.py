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

    def collisionalBoundBoundCrossSection(self, excitation, energyElectron, energyDiffLowerUpper,
        collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5):
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

    def rateCollisonalBoundBoundTransition(
        self, excitation, UNIT_LENGTH,
        energyElectron, energyElectronBinWidth, densityElectrons,
        energyDiffLowerUpper, collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5)
        """ rate of collisional de-/excitation

            @param excitation bool true =^= excitation, false =^= deexcitation
            @param UNIT_LENGTH conversion factor internal -> SI for length

            @param energyElectron float central energy of electron bin, [eV]
            @param energyElectronBinWidth width of energy bin, [eV]
            @param densityElectrons number density of physical electrons in bin, 1/(UNIT_LENGTH^3 * eV)

            @param energyDiffLowerUpper upperStateEnergy - lowerStateEnergy, [eV]
            @param collisionalOscillatorStrength of the transition, unitless
            @param cxin1 float gaunt approximation coefficient 1
            @param cxin2 float gaunt approximation coefficient 2
            @param cxin3 float gaunt approximation coefficient 3
            @param cxin4 float gaunt approximation coefficient 4
            @param cxin5 float gaunt approximation coefficient 5

            @return unit 1/s
        """

        sigma = self.collisionalBoundBoundCrossSection(self, excitation,
            energyElectron, energyDiffLowerUpper,
            collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5) # 1e6b

        electronRestMassEnergy = const.value("electron mass energy equivalent in MeV") * 1e6 # eV

        # eV * 1e6b * m^2/(1e6b) * 1/(UNIT_LENGTH^3 * eV) * (1/(m/UNIT_LENGTH))**3 * m/s * unitless
        # = eV/(eV) * m^2 * 1/m^3 * m/s = 1/s
        return (energyElectronBinWidth * sigma * 1e22 * densityElectrons * (1./UNIT_LENGTH)**3
            * const.value("speed of light in vacuum")
            * np.sqrt(1. - 1./ (1 . + energyElectron / electronRestMassEnergy)**2)) # 1/s

    def rateSpontaneousDeexcitation(self, absorptionOscillatorStrength, frequencyPhoton,
        lowerStateLevelVector, upperStateLevelVector)
        """ rate of spontaneous deexcitation under photon emission

            @param frequencyPhoton [1/s]
            @param absorptionOscillatorStrength unitless
        """
        # (As)^2 * N/A^2 * / (kg * m/s ) = A^2 * s^2 kg*m/s^2 * 1/A^2 /(kg *m/s )
        # = A^2/A^2 * s^2/s^2 * (kg*m)/(kg*m) * 1/(1/s) = s
        scalingConstant = (2 * np.pi * const.value("elementary charge")**2
            * const.value("vacuum mag. permeability") / (
            const.value("electron mass") * const.value("speed of light in vacuum") )) # s

        ratio = self._multiplicity(lowerStateLevelVector)/self._multiplicity(upperStateLevelVector)
        return scalingConstant * frequencyPhoton**2 * ratio * absorptionOscillatorStrength