import BoundBoundTransitions as boundbound
import BoundFreeTransitions as boundfree

import scipy.constants as const

if __name__ == "__main__":
    # electron histogram info
    energyElectron = 1000. # eV
    energyElectronBinWidth = 10. # eV
    densityElectrons = 1e28 #eV

    # bound-free transition data
    ionizationEnergy = 100. # eV
    excitationEnergyDifference = 5. # eV
    screenedCharge = 5. # e
    lowerStateLevelVectorBoundFree = (1,1,0,0,0,0,1,0,1,0)
    upperStateLevelVectorBoundFree = (1,1,0,0,0,0,1,0,0,0)

    # bound-bound transition data
    energyDiffLowerUpper = 5. # eV
    cxin1 = 1.
    cxin2 = 2.
    cxin3 = 3.
    cxin4 = 4.
    cxin5 = 5.
    collisionalOscillatorStrength = 1.
    absorptionOscillatorStrength = 1.e-1
    lowerStateLevelVectorBoundBound = (1,1,1,0,0,0,1,0,0,0)
    upperStateLevelVectorBoundBound = (1,1,0,0,0,0,1,0,1,0)

    print("bound-free:")
    print("\t collisional ionization rate[1/s]: "
        + str(boundfree.collisionalIonizationRate(
            energyElectron, energyElectronBinWidth, densityElectrons,
            ionizationEnergy, excitationEnergyDifference, screenedCharge,
            lowerStateLevelVectorBoundFree, upperStateLevelVectorBoundFree)))

    print("bound-bound:")
    print("\t collisional excitation rate [1/s]: "
        + str(boundbound.rateCollisionalBoundBoundTransition(
            energyElectron, energyElectronBinWidth, densityElectrons,
            energyDiffLowerUpper, collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound,
            excitation=true)))
    print("\t collisional deexcitation rate [1/s]: "
        + str(boundbound.rateCollisionalBoundBoundTransition(
            energyElectron, energyElectronBinWidth, densityElectrons,
            energyDiffLowerUpper, collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound,
            excitation=false)))

    print("\t spontaneous radiative deexcitation rate [1/s]: "
        + str(boundbound.rateSpontaneousDeexcitation(
            absorptionOscillatorStrength, frequencyPhoton,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound)))