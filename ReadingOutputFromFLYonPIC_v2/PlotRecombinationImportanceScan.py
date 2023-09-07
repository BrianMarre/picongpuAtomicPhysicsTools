import typeguard
import pydantic

import PlotAtomicPopulations
import Config as cfg
import ScanSCFly

@typeguard.typechecked
def processScanData(
    scanConfig : cfg.Scan.ScanConfig,
    plotConfigs : list[cfg.AtomicPopulationPlot.AtomicPopulationPlotConfig]):
    """process entire scan data"""

    initialChargeState = scanConfig.atomicNumber - np.sum(scanConfig.initialChargeState)

    numberElectronTemperatures = len(scanConfig.electronTemperatures)
    numberIonDensities = len(scanConfig.ionDensities)

    RecombinationPerInitialState= np.empty((numberIonDensities, numberElectronTemperatures), dtype='f4')
    IonizationPerInitial= np.empty((numberIonDensities, numberElectronTemperatures), dtype='f4')

    maxRecombination
    maxOther

    for config in plotConfigs:
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = PlotAtomicPopulations.preProcess(config)
            del mean
            del stdDev
            del axisDict_FLYonPIC
            del atomicConfigNumbers_FLYonPIC
            del timeSteps_FLYonPIC
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = PlotAtomicPopulations.loadProcessed(config)
            del mean
            del stdDev
            del axisDict_FLYonPIC
            del atomicConfigNumbers_FLYonPIC
            del timeSteps_FLYonPIC

        # reduce to per charge state
        chargeStateData, axisDict_ChargeState = reduceToPerChargeState(
            config,
            atomicPopulationData,
            axisDict_SCFLY,
            atomicConfigNumbers_SCFLY)

        # sum over all recombination states ... charge states below init charge state of FLYonPIC
        sumRecombinationStates = 0
        for i in range(initialChargeState):
            sumRecombinationStates += chargeStateData[:,i]

        # get max relative abundance over time
        np.max(sumRecombinationStates)

        sumRecombinationStates = 0
        for i in range(initialChargeState):
            sumRecombinationStates += chargeStateData[:,i]

        # compare to relative abundance in initial charge state



@typeguard.typechecked
def plotRecombinationImportanceMap(tasks : list[cfg.AtomicPopulationPlotConfig]):


if __name__ == "__main__":
    pass
