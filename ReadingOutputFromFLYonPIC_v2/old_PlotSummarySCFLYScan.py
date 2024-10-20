"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import typeguard

import json
from tqdm import tqdm
import numpy as np

from . import PlotAtomicPopulations
from . import ScanSCFLY as scan
from . import SCFlyTools
from . import Config as cfg

import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.scale as scale

@typeguard.typechecked
def processScanData(scanConfig : cfg.SCFLYScan.ScanConfig,
                    summaryConfig : cfg.SummarySCFLYScanPlot.PlotConfig,
                    tasks : tuple[
                    list[SCFlyTools.BaseConfig.BaseConfig],
                    list[tuple[int,int]],
                    dict[str, int]]):
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
                 summaryConfig : cfg.SummarySCFLYScanPlot.PlotConfig,
                 tasks : tuple[
                    list[SCFlyTools.BaseConfig.BaseConfig],
                    list[tuple[int,int]],
                    dict[str, int]]):
    """load scan data either from file or from raw"""

    if summaryConfig.loadRawSummaryData:
        # do processing of scanData
        maxRecombinationToInitial, maxIonizationToInitial, axisDict \
            = processScanData(scanConfig, summaryConfig, tasks)

        # write pre-processed scan summary data to file
        np.savetxt(scanConfig.processedDataStoragePath
                   + "maxRecombinationToInitial_"
                   + scanConfig.dataSeriesName + ".data",
                   maxRecombinationToInitial)
        np.savetxt(scanConfig.processedDataStoragePath
                   + "maxIonizationToInitial_"
                   + scanConfig.dataSeriesName + ".data",
                   maxIonizationToInitial)
        with open(scanConfig.processedDataStoragePath + "axisDict_ScanSummary_"
                  + scanConfig.dataSeriesName + ".dict", 'w') as File:
            json.dump(axisDict, File)
    else:
        # load previously processed data from file
        maxRecombinationToInitial = np.loadtxt(
            scanConfig.processedDataStoragePath + "maxRecombinationToInitial_"
            + scanConfig.dataSeriesName + ".data")
        maxIonizationToInitial = np.loadtxt(
            scanConfig.processedDataStoragePath + "maxIonizationToInitial_"
            + scanConfig.dataSeriesName + ".data")
        with open(scanConfig.processedDataStoragePath + "axisDict_ScanSummary_"
                  + scanConfig.dataSeriesName + ".dict", 'r') as File:
            axisDict = json.load(File)

    return maxRecombinationToInitial, maxIonizationToInitial, axisDict


@typeguard.typechecked
def plotSummary(scanConfigs : list[cfg.SCFLYScan.ScanConfig],
                tasksList : list[tuple[
                    list[SCFlyTools.BaseConfig.BaseConfig],
                    list[tuple[int,int]],
                    dict[str, int]]],
                summaryConfigList : list[cfg.SummarySCFLYScanPlot.PlotConfig]):
    """plot summary plot for each scanConfig into combined figure"""

    # check for consistent storage paths
    firstFigureStoragePath = scanConfigs[0].figureStoragePath
    for scanConfig in scanConfigs[1:]:
        if scanConfig.figureStoragePath != firstFigureStoragePath:
            print("Warning: inconsistent figure storage paths!\n"
                  "Summary plot will be stored at " + firstFigureStoragePath)
            break

    numberScans = len(scanConfigs)
    figure, axes = plt.subplots(ncols=2, nrows=numberScans,
                                dpi=200, figsize=(8,4*numberScans))
    figure.suptitle("Relative Abundance Quotients:")

    for i, scanConfig in enumerate(tqdm(scanConfigs)):
        summaryConfig = summaryConfigList[i]
        tasks = tasksList[i]

        # dimensionality depends on nrows
        if numberScans > 1:
            axePairLeft = axes[i, 0]
            axePairRight = axes[i, 1]
        else:
            axePairLeft = axes[0]
            axePairRight = axes[1]

        if i == 0:
            axePairLeft.set_title("recombination vs initial charge state:\n"
                                 + summaryConfig.dataSetName)
            axePairRight.set_title("ionized states vs initial charge state:\n"
                                 + summaryConfig.dataSetName)
        else:
            axePairLeft.set_title(summaryConfig.dataSetName)
            axePairRight.set_title(summaryConfig.dataSetName)

        # prepare plots
        axePairLeft.set_xlabel("electron temperature [eV]")
        axePairLeft.set_ylabel("ion density [1/cm^3]")
        axePairLeft.set_yscale("log")
        axePairLeft.set_xscale("log")

        axePairRight.set_xlabel("electron temperature [eV]")
        axePairRight.set_ylabel("ion density [1/cm^3]")
        axePairRight.set_yscale("log")
        axePairRight.set_xscale("log")

        # plot data for each part
        # get data
        maxRecombinationToInitial, maxIonizationToInitial, axisDict \
            = loadScanData(scanConfig, summaryConfig, tasks)

        assert ((axisDict['electronTemperature'] == 0)
            and (axisDict['ionDensity'] == 1))

        densities, temperatures = np.meshgrid(
            scanConfig.ionDensities, scanConfig.electronTemperatures)

        left = axePairLeft.pcolormesh(
            temperatures, densities, maxRecombinationToInitial,
            cmap=plt.cm.viridis, norm=color.LogNorm(vmin=1.e-7))

        right = axePairRight.pcolormesh(
            temperatures, densities, maxIonizationToInitial,
            cmap=plt.cm.viridis, norm=color.LogNorm(vmin=1.e-2, vmax=1.e2))

        figure.colorbar(left, ax = axePairLeft)
        figure.colorbar(right, ax = axePairRight)

    figure.tight_layout()

    fileName = scanConfigs[0].figureStoragePath + "SummaryPlot"
    for scanConfig in scanConfigs:
        fileName = fileName + "_" + scanConfig.dataSeriesName
    plt.savefig(fileName)

    plt.close(figure)
