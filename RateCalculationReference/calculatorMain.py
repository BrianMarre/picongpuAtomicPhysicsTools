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
    frequencyPhoton = energyDiffLowerUpper / const.physical_constants["Planck constant in eV/Hz"][0] # 1/s

    print("cross sections:")
    print("- bound-free")
    print("\t collisional ionization cross section: \t  {0:.12e} 1e6*barn".format(
        boundfree.BoundFreeTransitions.collisionalIonizationCrossSection(
            energyElectron, ionizationEnergy, excitationEnergyDifference, screenedCharge,
            lowerStateLevelVectorBoundFree, upperStateLevelVectorBoundFree)))

    print("- bound-bound")
    print("\t collisional excitation cross section: \t  {0:.12e} 1e6*barn".format(
        boundbound.BoundBoundTransitions.collisionalBoundBoundCrossSection(
            energyElectron,
            energyDiffLowerUpper, collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound,
            excitation=True)))
    print("\t collisional deexcitation cross section:  {0:.12e} 1e6*barn".format(
        boundbound.BoundBoundTransitions.collisionalBoundBoundCrossSection(
            energyElectron,
            energyDiffLowerUpper, collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound,
            excitation=False)))

    print("rates:")
    print("- bound-free")
    print("\t collisional ionization rate:  \t\t  {0:.12e} 1/s".format(
        boundfree.BoundFreeTransitions.rateCollisionalIonization(
            energyElectron, energyElectronBinWidth, densityElectrons,
            ionizationEnergy, excitationEnergyDifference, screenedCharge,
            lowerStateLevelVectorBoundFree, upperStateLevelVectorBoundFree)))

    print("- bound-bound")
    print("\t collisional excitation rate:  \t\t  {0:.12e} 1/s".format(
        boundbound.BoundBoundTransitions.rateCollisionalBoundBoundTransition(
            energyElectron, energyElectronBinWidth, densityElectrons,
            energyDiffLowerUpper, collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound,
            excitation=True)))
    print("\t collisional deexcitation rate:  \t  {0:.12e} 1/s".format(
        boundbound.BoundBoundTransitions.rateCollisionalBoundBoundTransition(
            energyElectron, energyElectronBinWidth, densityElectrons,
            energyDiffLowerUpper, collisionalOscillatorStrength, cxin1, cxin2, cxin3, cxin4, cxin5,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound,
            excitation=False)))
    print("\t spontaneous radiative deexcitation rate: {0:.12e} 1/s".format(
        boundbound.BoundBoundTransitions.rateSpontaneousDeexcitation(
            absorptionOscillatorStrength, frequencyPhoton,
            lowerStateLevelVectorBoundBound, upperStateLevelVectorBoundBound)))
