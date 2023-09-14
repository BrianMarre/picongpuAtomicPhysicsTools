import pydantic
import typeguard
import typing
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess

import SCFlyTools
import PlotAtomicPopulations as plotter
import PlotSummaryScan as summary
import Config as cfg


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
def runScanList(scanConfigs : list[cfg.SCFLYScan.ScanConfig], chunkSize : int = 1):
    for scanConfig in tqdm(scans):
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

        if scanConfig.plotSummary:
            summary.plotSummary(
                [scanConfig],
                [(baseConfigs, conditions, axisDict_conditions)],
                [cfg.SummaryScanPlot.PlotConfig(
                    loadRawEachSCLFYSim = False, # @todo change back, Brian Marre
                    loadRawSummaryData = True,
                    additionalDataName = "",
                    seriesName = scanConfig.dataSeriesName)])

        # save temperatures and ionDensities for reference
        np.savetxt(scanConfig.figureStoragePath + "electronTemperatures.txt", scanConfig.electronTemperatures)
        np.savetxt(scanConfig.figureStoragePath + "ionDensities.txt", scanConfig.ionDensities)


if __name__=="__main__":
    processedDataStoragePath = "preProcessedData/"
    chunkSize = 24

    scanConfig_Ar = cfg.SCFLYScan.ScanConfig(
        atomicNumber = 18,
        SCFLYatomicStateNamingFile = "/home/marre55/scflyInput/18_atomicStateNaming.input",
        atomicDataInputFile = "/home/marre55/scfly/atomicdata/FLYCHK_input_files/atomic.inp.18",
        electronTemperatures = np.concatenate([np.arange(1,10)*1e2, (np.arange(10)+1)*1e3]), # eV
        ionDensities = np.concatenate([np.arange(1,10)*1e21, (np.arange(10)+1)*1e22]), # 1/cm^3
        timePoints = np.arange(101) * 3.3e-17, # s
        initialStateLevelVector = (2, 8, 6, 0, 0, 0, 0, 0, 0, 0),
        outputBasePath = "/home/marre55/scflyInput/",
        SCFLYBinaryPath = "/home/marre55/scfly/code/exe/scfly",
        outputFileName = "xout",
        dataSeriesName ="Ar_",
        numberStatesToPlot = 470,
        colorMap = plt.cm.tab10,
        numColorsInColorMap = 10,
        processedDataStoragePath = processedDataStoragePath,
        figureStoragePath = "SCFLYArScanImages/",
        runSCFLY = False,
        plotEachSim = False,
        plotSummary = True)

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

    scanConfig_Cu_2 = cfg.SCFLYScan.ScanConfig(
        atomicNumber = 29,
        SCFLYatomicStateNamingFile = "/home/marre55/scflyInput/29_atomicStateNaming.input",
        atomicDataInputFile = "/home/marre55/scfly/atomicdata/FLYCHK_input_files/atomic.inp.29",
        electronTemperatures = np.concatenate([np.arange(1,10)*1e2, (np.arange(10)+1)*1e3]), # eV
        ionDensities = np.concatenate([[8*1e21, 9*1e21], np.arange(1,10)*1e22, (np.arange(10)+1)*1e23]), # 1/cm^3
        timePoints = np.arange(101) * 3.3e-17, # s
        initialStateLevelVector = (2, 8, 14, 0, 0, 0, 0, 0, 0, 0),
        outputBasePath = "/home/marre55/scflyInput/",
        SCFLYBinaryPath = "/home/marre55/scfly/code/exe/scfly",
        outputFileName = "xout",
        dataSeriesName ="Cu_2_recombination_IPD",
        numberStatesToPlot = 870,
        colorMap = plt.cm.tab20b,
        numColorsInColorMap = 20,
        processedDataStoragePath = processedDataStoragePath,
        figureStoragePath = "SCFLY_Cu_2_Recombination_IPD/",
        runSCFLY = False,
        plotEachSim = False,
        plotSummary = True)

    scans = [scanConfig_Cu_2, scanConfig_Cu, scanConfig_Ar]
    runScanList(scans, chunkSize)

