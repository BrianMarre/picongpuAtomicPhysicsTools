import typeguard

import json
from tqdm import tqdm
import numpy as np

import PlotAtomicPopulations
import ScanSCFLY as scan
import Config as cfg

import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.scale as scale

@typeguard.typechecked
def processScanData(scanConfig : cfg.Scan.ScanConfig, ZeroCutoffLimit : float):
    """extract summary from data of entire scan"""

    baseConfigs, conditions, axisDictConditions = scan.generateBaseConfigs(scanConfig)
    plottingConfigs = scan.generatePlottingConfigs(scanConfig, baseConfigs)

    initialChargeState = scanConfig.atomicNumber - np.sum(scanConfig.initialStateLevelVector)

    numberElectronTemperatures = len(scanConfig.electronTemperatures)
    numberIonDensities = len(scanConfig.ionDensities)

    maxSurplusRecombination = np.empty((numberElectronTemperatures, numberIonDensities), dtype='f4')
    maxSurplusIonization = np.empty((numberElectronTemperatures, numberIonDensities), dtype='f4')

    maxPartition = np.empty((numberElectronTemperatures, numberIonDensities), dtype='f4')
    maxSurplusIonizationToInitialState = np.empty((numberElectronTemperatures, numberIonDensities), dtype='f4')

    assert (axisDictConditions['electronTemperature'] == 0)
    assert (axisDictConditions['ionDensity'] == 1)

    for i, config in enumerate(plottingConfigs):
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = PlotAtomicPopulations.preProcess(config)

            # SCFLY scan will not contain FLYonPIC data
            del mean
            del stdDev
            del axisDict_FLYonPIC
            del atomicConfigNumbers_FLYonPIC
            del timeSteps_FLYonPIC
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = PlotAtomicPopulations.loadProcessed(config)

            # SCFLY scan will not contain FLYonPIC data
            del mean
            del stdDev
            del axisDict_FLYonPIC
            del atomicConfigNumbers_FLYonPIC
            del timeSteps_FLYonPIC

        # reduce to per charge state
        chargeStateData, axisDict_ChargeState = PlotAtomicPopulations.reduceToPerChargeState(
            config,
            atomicPopulationData,
            axisDict_SCFLY,
            atomicConfigNumbers_SCFLY)

        # initial charge State
        initialChargeState = chargeStateData[:,initialChargeState]

        # sum over all recombination states ... charge states below init charge state of FLYonPIC
        sumRecombinationStates = np.zeros_like(chargeStateData[:,0])
        for i in range(initialChargeState):
            sumRecombinationStates += chargeStateData[:,i]

        # sum over all (above initial charge state) states
        sumAboveInitial = np.zeros_like(chargeStateData[:,0])
        for i in range(initialChargeState + 1, config.atomicNumber + 1):
            sumAboveInitial += chargeStateData[:,i]

        # remove below FLYonPIC resolution to avoid nans
        initialChargeState_nonZero = np.where(
            initialChargeState <= ZeroCutoffLimit,
            ZeroCutoffLimit,
            initialChargeState)

        # compare to relative abundance in initial charge state
        maxRecombinationToInitial[conditions[i]] = np.max(
            sumRecombinationStates / initialChargeState_nonZero)
        maxIonizationToInitial[conditions[i]] = np.max(
            sumAboveInitial / initialChargeState_nonZero)

    axisDict = {'electronTemperature':0, "ionDensity":1}
    return maxRecombinationToInitial, maxIonizationToInitial, axisDict

@typeguard.typechecked
def loadScanData(scanConfig : cfg.Scan.ScanConfig, ZeroCutoffLimit : float,
                 additonalDataName : str = ""):
    """load scan data either from file or from raw"""

    if additonalDataName != "":
        addName = "_" + additonalDataName
    else:
        addName = ""

    if scanConfig.loadRaw:
        # do processing of scanData
        maxRecombinationToInitial, maxIonizationToInitial, axisDict \
            = processScanData(scanConfig, ZeroCutoffLimit)

        # write pre-processed scan summary data to file
        np.savetxt(scanConfig.processedDataStoragePath
                   + "maxRecombinationToInitial_"
                   + scanConfig.dataSeriesName + addName + ".data",
                   maxRecombinationToInitial)
        np.savetxt(scanConfig.processedDataStoragePath
                   + "maxIonizationToInitial_"
                   + scanConfig.dataSeriesName + addName + ".data",
                   maxIonizationToInitial)
        with open(config.processedDataStoragePath + "axisDict_ScanSummary_"
                  + scanConfig.dataSeriesName + addName + ".dict", 'w') as File:
            json.dump(axisDict, File)
    else:
        # load previously processed data from file
        maxRecombinationToInitial = np.loadtxt(
            scanConfig.processedDataStoragePath + "maxRecombinationToInitial_"
            + scanConfig.dataSeriesName + addName + ".data")
        maxIonizationToInitial = np.loadtxt(
            scanConfig.processedDataStoragePath + "maxIonizationToInitial_"
            + scanConfig.dataSeriesName + addName + ".data")
        with open(config.processedDataStoragePath + "axisDict_ScanSummary_"
                  + scanConfig.dataSeriesName + addName + ".dict", 'r') as File:
            axisDict = json.load(File)

    return maxRecombinationToInitial, maxIonizationToInitial, axisDict

@typeguard.typechecked
def checkScanConfigsCanBeStitched(scanConfigs : list[cfg.Scan.ScanConfig]):
    """
    check that given list of scanConfigs can be stitched together

    passes silently if ok
    """
    first = scanConfigs[0]

    atomicNumber = first.atomicNumber
    namingFile = first.SCFLYatomicStateNamingFile
    timePoints = first.timePoints
    initialState = first.initialStateLevelVector
    SCFLYBinaryPath = first.SCFLYBinaryPath
    figureStoragePath = first.figureStoragePath

    for i, scanConfig in enumerate(scanConfigs):
        if (scanConfig.atomicNumber != atomicNumber):
            raise RuntimeError("unable to stitch scans together["
                + str(i) + ": atomic number different!]")
        if (scanConfig.SCFLYatomicStateNamingFile != namingFile):
            raise RuntimeError("unable to stitch scans together["
                + str(i) + ": naming file different!]")
        if np.any(scanConfig.timePoints != timePoints):
            raise RuntimeError("unable to stitch scans together["
                + str(i) + ": timePoints different!]")
        if np.any(scanConfig.initialStateLevelVector != initialState):
            raise RuntimeError("unable to stitch scans together["
                + str(i) + ": initial state different!]")
        if (scanConfig.SCFLYBinaryPath != SCFLYBinaryPath):
            raise RuntimeError("unable to stitch scans together["
                + str(i) + ": different SCFLY binary!]")
        if (figureStoragePath != scanConfig.figureStoragePath):
            print("Warning!: inconsistent figure storage paths, using first one only")

@typeguard.typechecked
def plotSummary(scanConfigs : list[cfg.Scan.ScanConfig],
                zeroCutoffLimit : float, seriesName : str):
    """plot summary plot of stitched together scan"""

    # sanity check before we stitch
    checkScanConfigsCanBeStitched(scanConfigs)

    # prepare plot
    figure, axes = plt.subplots(ncols=2, nrows=1, dpi=300)
    axes[0].set_title(
        "population in recombination vs initial charge state: "
         + seriesName)
    axes[1].set_title(
        "population in ionization vs initial charge state: " + seriesName)

    axes[0].set_xlabel("electron temperature")
    axes[0].set_ylabel("ion density")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[1].set_xlabel("electron temperature")
    axes[1].set_ylabel("ion density")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")


    # plot data for each part
    for scanConfig in tqdm(scanConfigs):
        # get data
        maxRecombinationToInitial, maxIonizationToInitial, axisDict \
            = loadScanData(scanConfig, zeroCutoffLimit)

        assert ((axisDict['electronTemperature'] == 0)
            and (axisDict['ionDensities'] == 1))

        temperatures, densities = np.meshgrid(scanConfig.electronTemperatures,
                             scanConfig.ionDensities)

        axes[0].pcolormesh(temperatures, densities, maxRecombinationToInitial,
                           cmap=plt.cm.cividis, norm=color.LogNorm(vmin=1.e-7))

        axes[1].pcolormesh(temperatures, densities, maxIonizationToInitial,
                           cmap=plt.cm.cividis, norm=color.LogNorm(vmin=1.e-7))

    plt.colorbar()
    figure.tight_layout()
    plt.savefig(scanConfigs[0].figureStoragePath + "AtomicPopulation_diff_" + config.dataName)
    plt.close(figure)


if __name__ == "__main__":
    pass
