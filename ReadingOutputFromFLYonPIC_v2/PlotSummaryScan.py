import typeguard

import json
from tqdm import tqdm
import numpy as np

import PlotAtomicPopulations
import ScanSCFLY as scan
import SCFlyTools
import Config as cfg

import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.scale as scale

@typeguard.typechecked
def processScanData(scanConfig : cfg.SCFLYScan.ScanConfig,
                    summaryConfig : cfg.SummaryScanPlot.PlotConfig,
                    tasks):
    """extract summary from data of entire scan"""

    baseConfigs, conditions, axisDictConditions = tasks
    plottingConfigs = scan.generatePlottingConfigs(
        scanConfig, baseConfigs, summaryConfig.loadRawEachSCLFYSim)

    initialChargeState = scanConfig.atomicNumber - np.sum(scanConfig.initialStateLevelVector)

    numberElectronTemperatures = len(scanConfig.electronTemperatures)
    numberIonDensities = len(scanConfig.ionDensities)

    maxRecombinationToInitial = np.empty(
        (numberElectronTemperatures, numberIonDensities), dtype='f8')
    maxIonizationToInitial = np.empty(
        (numberElectronTemperatures, numberIonDensities), dtype='f8')

    assert (axisDictConditions['electronTemperature'] == 0)
    assert (axisDictConditions['ionDensity'] == 1)

    print()
    print("loading scan data for summary plot...")
    for i, config in enumerate(tqdm(plottingConfigs)):
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY \
                    = PlotAtomicPopulations.preProcess(config)

            # SCFLY scan will never contain FLYonPIC data
            del mean
            del stdDev
            del axisDict_FLYonPIC
            del atomicConfigNumbers_FLYonPIC
            del timeSteps_FLYonPIC
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY \
                    = PlotAtomicPopulations.loadProcessed(config)

            # SCFLY scan will never contain FLYonPIC data
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
        initialChargeStateAbundance = chargeStateData[:, initialChargeState]

        # sum over all recombination states ... charge states below init charge state of FLYonPIC
        sumRecombinationStates = np.zeros_like(chargeStateData[:,0])
        for z in range(initialChargeState):
            sumRecombinationStates += chargeStateData[:,z]

        # sum over all (above initial charge state) states
        sumAboveInitial = np.zeros_like(chargeStateData[:,0])
        for z in range(initialChargeState + 1, config.atomicNumber + 1):
            sumAboveInitial += chargeStateData[:,z]

        # compare to relative abundance in initial charge state
        maxRecombinationToInitial[conditions[i]] = np.max(
            sumRecombinationStates / initialChargeStateAbundance)
        maxIonizationToInitial[conditions[i]] = np.max(
            sumAboveInitial / initialChargeStateAbundance)

    axisDict = {'electronTemperature':0, "ionDensity":1}
    return maxRecombinationToInitial, maxIonizationToInitial, axisDict

@typeguard.typechecked
def loadScanData(scanConfig : cfg.SCFLYScan.ScanConfig,
                 summaryConfig : cfg.SummaryScanPlot.PlotConfig,
                 tasks):
    """load scan data either from file or from raw"""

    if summaryConfig.additionalDataName != "":
        addName = "_" + summaryConfig.additionalDataName
    else:
        addName = ""

    if summaryConfig.loadRawSummaryData:
        # do processing of scanData
        maxRecombinationToInitial, maxIonizationToInitial, axisDict \
            = processScanData(scanConfig, summaryConfig, tasks)

        # write pre-processed scan summary data to file
        np.savetxt(scanConfig.processedDataStoragePath
                   + "maxRecombinationToInitial_"
                   + scanConfig.dataSeriesName + addName + ".data",
                   maxRecombinationToInitial)
        np.savetxt(scanConfig.processedDataStoragePath
                   + "maxIonizationToInitial_"
                   + scanConfig.dataSeriesName + addName + ".data",
                   maxIonizationToInitial)
        with open(scanConfig.processedDataStoragePath + "axisDict_ScanSummary_"
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
        with open(scanConfig.processedDataStoragePath + "axisDict_ScanSummary_"
                  + scanConfig.dataSeriesName + addName + ".dict", 'r') as File:
            axisDict = json.load(File)

    return maxRecombinationToInitial, maxIonizationToInitial, axisDict

@typeguard.typechecked
def checkScanConfigsCanBeStitched(scanConfigs : list[cfg.SCFLYScan.ScanConfig]):
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
def plotSummary(scanConfigs : list[cfg.SCFLYScan.ScanConfig],
                tasks,
                summaryConfig : cfg.SummaryScanPlot.PlotConfig):
    """plot summary plot of stitched together scan"""

    # sanity check before we stitch
    checkScanConfigsCanBeStitched(scanConfigs)

    # prepare plot
    figure, axes = plt.subplots(ncols=2, nrows=1, dpi=300, figsize=(10,5))
    figure.suptitle("Relative Abundance Quotients: "
                    + summaryConfig.seriesName)

    axes[0].set_title("recombination vs initial charge state")
    axes[1].set_title("ionized states vs initial charge state")

    axes[0].set_xlabel("electron temperature [eV]")
    axes[0].set_ylabel("ion density [1/cm^3]")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")

    axes[1].set_xlabel("electron temperature [eV]")
    axes[1].set_ylabel("ion density [1/cm^3]")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")

    # plot data for each part
    print("plotting summary of scan...")
    for i, scanConfig in enumerate(scanConfigs):
        # get data
        maxRecombinationToInitial, maxIonizationToInitial, axisDict \
            = loadScanData(scanConfig, summaryConfig, tasks[i])

        assert ((axisDict['electronTemperature'] == 0)
            and (axisDict['ionDensity'] == 1))

        temperatures, densities = np.meshgrid(scanConfig.electronTemperatures,
                             scanConfig.ionDensities)

        left = axes[0].pcolormesh(
            temperatures, densities, maxRecombinationToInitial,
            cmap=plt.cm.viridis, norm=color.LogNorm(vmin=1.e-7))

        right = axes[1].pcolormesh(
            temperatures, densities, maxIonizationToInitial,
            cmap=plt.cm.viridis, norm=color.LogNorm(vmin=1.e-2, vmax=1.e2))

    figure.colorbar(left, ax = axes[0])
    figure.colorbar(right, ax = axes[1])
    figure.tight_layout()
    plt.savefig(scanConfigs[0].figureStoragePath + "SummaryPlot_"
                + scanConfig.dataSeriesName)
    plt.close(figure)


if __name__ == "__main__":
    processedDataStoragePath = "preProcessedData/"

    scanConfig_Cu = cfg.SCFLYScan.ScanConfig(
        atomicNumber = 29,
        SCFLYatomicStateNamingFile = "/home/marre55/scflyInput/29_atomicStateNaming.input",
        atomicDataInputFile = "/home/marre55/scfly/atomicdata/FLYCHK_input_files/atomic.inp.29",
        electronTemperatures = np.concatenate([np.arange(1,10)*1e2, (np.arange(10)+1)*1e3]), # eV
        ionDensities = np.concatenate([np.arange(1,10)*1e21, (np.arange(10)+1)*1e22]), # 1/cm^3
        timePoints = np.arange(101) * 3.3e-17, # s
        initialStateLevelVector = (2, 8, 17, 0, 0, 0, 0, 0, 0, 0),
        outputBasePath = "/home/marre55/scflyInput/",
        SCFLYBinaryPath = "/home/marre55/scfly/code/exe/scfly",
        outputFileName = "xout",
        dataSeriesName ="Cu_recombination_IPD",
        numberStatesToPlot = 870,
        colorMap = plt.cm.tab20b,
        numColorsInColorMap = 20,
        processedDataStoragePath = processedDataStoragePath,
        figureStoragePath = "SCFLY_Cu_Recombination_IPD_ScanImages/",
        runSCFLY = False,
        plotEachSim = False,
        plotSummary = True)

    summaryConfig_Cu = cfg.SummaryScanPlot.PlotConfig(
        loadRawEachSCLFYSim = False,
        loadRawSummaryData = True,
        additionalDataName = "",
        seriesName = "Cu Initial: 2+")

    plotSummary([scanConfig_Cu], summaryConfig_Cu)