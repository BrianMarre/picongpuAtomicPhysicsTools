"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
import typeguard
import typing
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess

from . import SCFlyTools
from . import PlotAtomicPopulations as plotter
from . import PlotSummarySCFLYScan as summary
from . import Config as cfg


@typeguard.typechecked
def generateBaseConfigs(scanConfig : cfg.SCFLYScan.ScanConfig):
    baseConfigs = []
    conditions = []

    print("generating BaseConfigs...")
    for i, electronTemperature in enumerate(tqdm(scanConfig.electronTemperatures)):
        for j, ionDensity in enumerate(tqdm(scanConfig.ionDensities, leave=False)):
            # create config for case
            comparisonFLYonPIC_Ar = SCFlyTools.Config.FLYonPICComparison.FLYonPICComparison(
                atomicNumber = scanConfig.atomicNumber,
                electronTemperature = electronTemperature, # eV
                ionDensity = ionDensity, # 1/cm^3
                timePoints = scanConfig.timePoints, # s
                initialStateLevelVector = scanConfig.initialStateLevelVector,
                SCFLYatomicStateNamingFile = scanConfig.SCFLYatomicStateNamingFile,
                atomicDataInputFile = scanConfig.atomicDataInputFile,
                outputFileName = scanConfig.outputFileName,
                basePath = scanConfig.outputBasePath,
                folderName = (scanConfig.dataSeriesName
                    + "_" + str(i) + "_Temp_" + str(j) + "_Density"))

            # generate BaseConfig
            baseConfig = comparisonFLYonPIC_Ar.get()

            # store SCFLY BaseConfig
            baseConfigs.append(baseConfig)
            conditions.append((i,j))

    axisDictConditions = {"electronTemperature":0, "ionDensity":1}
    return baseConfigs, conditions, axisDictConditions

@typeguard.typechecked
def generateSetups(tasks : list[SCFlyTools.BaseConfig.BaseConfig]):
    print("generating setups...")
    for setup in tqdm(tasks):
        # generate setup and execute SCFLY
        setup.generateSCFLYSetup()

@typeguard.typechecked
def runSingleSCFLYScan(scanConfig : cfg.SCFLYScan.ScanConfig,
                 baseConfigs : list[SCFlyTools.BaseConfig.BaseConfig],
                 chunkSize : int = 1):
    chunk = []
    numberProcess = 0

    for baseConfig in baseConfigs:
        if (numberProcess < chunkSize):
            # current slot is empty -> start new SCFLY run
            chunk.append(baseConfig.execute(scanConfig.SCFLYBinaryPath))
            numberProcess += 1
        else:
            unassigned = True
            while unassigned:
                for i, slot in enumerate(chunk):
                    if slot.poll() is not None:
                        # current slot is finished -> start new SCFLY run
                        slot.wait()
                        chunk.pop(i)
                        chunk.append(baseConfig.execute(scanConfig.SCFLYBinaryPath))
                        unassigned = False
                        break

# all started, wait for them to finish
    for slot in chunk:
        slot.wait()


@typeguard.typechecked
def generatePlottingConfigs(
        config : cfg.SCFLYScan.ScanConfig,
        tasks : list[SCFlyTools.BaseConfig.BaseConfig_TimeDependent],
        loadRaw : bool = True) -> list[cfg.AtomicPopulationPlot.PlotConfig]:
    # list of PlottingConfig instances
    plotConfigs = []

    for setup in tasks:
        # create plotting config
        plotConfig = cfg.AtomicPopulationPlot.PlotConfig(
            FLYonPICAtomicStateInputDataFile =  "",
            SCFLYatomicStateNamingFile =        config.SCFLYatomicStateNamingFile,
            FLYonPICOutputFileNames =           [],
            FLYonPICBasePath =                  "",
            SCFLYOutputFileName =               setup.basePath + setup.folderName + "/"+ setup.outputFileName,
            numberStatesToPlot =                config.numberStatesToPlot,
            colorMap =                          config.colorMap,
            numColorsInColorMap =               config.numColorsInColorMap,
            speciesName =                       "",
            atomicNumber=                       setup.atomicNumber,
            numLevels =                         len(config.initialStateLevelVector),
            processedDataStoragePath =          config.processedDataStoragePath,
            figureStoragePath =                 config.figureStoragePath,
            dataName =                          "SCFLY_" + setup.folderName,
            loadRaw =                           loadRaw)

        # store plotting config
        plotConfigs.append(plotConfig)

    return plotConfigs

@typeguard.typechecked
def plotEachSCFLYScan(tasks : list[cfg.AtomicPopulationPlot.PlotConfig], FLYonPICInitialChargeState : int):
    for plotConfig in tasks:
        # plot SCFLY data
        plotter.plot_all([plotConfig], [], [plotConfig], FLYonPICInitialChargeState)

@typeguard.typechecked
def runScanList(scanConfigs : list[cfg.SCFLYScan.ScanConfig],
                chunkSize : int = 1,
                plotCombined : bool = False,
                summaryPlotConfigs : list[cfg.SummarySCFLYScanPlot.PlotConfig] = None):

    tasksList = []
    generatedSummaryPlotConfigs = []

    for scanConfig in tqdm(scanConfigs):
        # create scan baseConfigs
        baseConfigs, conditions, axisDict_conditions = generateBaseConfigs(scanConfig)

        # optionally run SCFLY
        if scanConfig.runSCFLY:
            generateSetups(baseConfigs)
            runSingleSCFLYScan(scanConfig, baseConfigs, chunkSize)

        # create storage storage directory if it does not exist
        os.makedirs(scanConfig.figureStoragePath, exist_ok=True)

        # optionally plot results
        if scanConfig.plotEachSim:
            plotConfigs = generatePlottingConfigs(scanConfig, baseConfigs)
            initialChargeState = int(scanConfig.atomicNumber - np.sum(scanConfig.initialStateLevelVector))
            plotEachSCFLYScan(plotConfigs, initialChargeState)

        if ((not plotCombined) and scanConfig.plotSummary):
            # plot for each scan separately
            summary.plotSummary(
                [scanConfig],
                [(baseConfigs, conditions, axisDict_conditions)],
                [cfg.SummarySCFLYScanPlot.PlotConfig(
                    loadRawEachSCLFYSim = True,
                    loadRawSummaryData = True,
                    dataSetName = scanConfig.dataSeriesName)])
        else:
            tasksList.append((baseConfigs, conditions, axisDict_conditions))
            if summaryPlotConfigs is None:
                generatedSummaryPlotConfigs.append(
                    cfg.SummarySCFLYScanPlot.PlotConfig(
                        loadRawEachSCLFYSim = True,
                        loadRawSummaryData = True,
                        dataSetName = scanConfig.dataSeriesName))

        # save temperatures and ionDensities for reference
        np.savetxt(scanConfig.figureStoragePath + "electronTemperatures_"
                   + scanConfig.dataSeriesName + ".txt",
                   scanConfig.electronTemperatures)
        np.savetxt(scanConfig.figureStoragePath + "ionDensities_"
                   + scanConfig.dataSeriesName + ".txt", scanConfig.ionDensities)

    # optional, do combined plot
    if (plotCombined and (summaryPlotConfigs is None)):
        # no summaryPlotConfigs provided -> use generated ones
        summary.plotSummary(
            scanConfigs,
            tasksList,
            generatedSummaryPlotConfigs)
    elif plotCombined and (summaryPlotConfigs is not None):
        # use provided configs
        summary.plotSummary(
            scanConfigs,
            tasksList,
            summaryPlotConfigs)
