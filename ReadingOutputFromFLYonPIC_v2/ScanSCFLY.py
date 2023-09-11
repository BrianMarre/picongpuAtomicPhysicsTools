import pydantic
import typeguard
import typing

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import SCFlyTools
import PlotAtomicPopulations as plotter
import PlotSummaryScan as summary
import Config as cfg

@typeguard.typechecked
def generateBaseConfigs(config : cfg.Scan.ScanConfig):
    baseConfigs = []
    conditions = []

    for i, electronTemperature in enumerate(tqdm(config.electronTemperatures)):
        for j, ionDensity in enumerate(config.ionDensities):
            # create config for case
            comparisonFLYonPIC_Ar = SCFlyTools.Config.FLYonPICComparison.FLYonPICComparison(
                atomicNumber = config.atomicNumber,
                electronTemperature = electronTemperature, # eV
                ionDensity = ionDensity, # 1/cm^3
                timePoints = config.timePoints, # s
                initialStateLevelVector = config.initialStateLevelVector,
                SCFLYatomicStateNamingFile = config.SCFLYatomicStateNamingFile,
                atomicDataInputFile = config.atomicDataInputFile,
                outputFileName = config.outputFileName,
                basePath = config.outputBasePath,
                folderName = config.dataSeriesName + "_" + str(i) + "_Temp_" + str(j) + "_Density")

            # generate setup
            generatedSetup = comparisonFLYonPIC_Ar.get()

            # store SCFLY BaseConfig
            baseConfigs.append(generatedSetup)
            conditions.append((i,j))

    axisDictConditions = {"electronTemperature":0, "ionDensity":1}
    return baseConfigs, conditions, axisDictConditions

@typeguard.typechecked
def generateSetups(tasks : list[SCFlyTools.BaseConfig.BaseConfig]):
    for setup in tqdm(tasks):
        # generate setup and execute SCFLY
        setup.generateSCFLYSetup()

@typeguard.typechecked
def runSCFLYScan(config : cfg.Scan.ScanConfig, tasks : list[SCFlyTools.BaseConfig.BaseConfig]):
    for setup in tasks:
        # generate setup and execute SCFLY
        setup.execute(config.SCFLYBinaryPath)

@typeguard.typechecked
def generatePlottingConfigs(
        config : cfg.Scan.ScanConfig,
        tasks : list[SCFlyTools.BaseConfig.BaseConfig_TimeDependent]) -> list[cfg.AtomicPopulationPlot.AtomicPopulationPlotConfig]:
    # list of PlottingConfig instances
    plotConfigs = []

    for setup in tasks:
        # create plotting config
        plotConfig = cfg.AtomicPopulationPlot.AtomicPopulationPlotConfig(
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
            loadRaw =                           True)

        # store plotting config
        plotConfigs.append(plotConfig)

    return plotConfigs

@typeguard.typechecked
def plotEachSCFLYScan(tasks : list[cfg.AtomicPopulationPlot.AtomicPopulationPlotConfig], FLYonPICInitialChargeState : int):
    for plotConfig in tasks:
        # plot SCFLY data
        plotter.plot_all([plotConfig], [], [plotConfig], FLYonPICInitialChargeState)

if __name__=="__main__":
    processedDataStoragePath = "preProcessedData/"

    FLYonPICInitialChargeState = 2
    scanConfig_Ar = cfg.Scan.ScanConfig(
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
        plotSummary = True,
        loadRaw = True)

    scanConfig_Cu = cfg.Scan.ScanConfig(
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
        plotSummary = True,
        loadRaw = True)

    scans = [scanConfig_Cu]#, scanConfig_Ar]

    for scanConfig in scans:
        # create scan tasks
        tasks, conditions, axisDict_conditions = generateBaseConfigs(scanConfig)

        # run scfly
        if scanConfig.runSCFLY:
            generateSetups(tasks)
            runSCFLYScan(scanConfig, tasks)

        # plot results
        plotConfigs = generatePlottingConfigs(scanConfig, tasks)
        if scanConfig.plotEachSim:
            plotEachSCFLYScan(plotConfigs, FLYonPICInitialChargeState)
        if scanConfig.plotSummary:
            summary.plotSummary([scanConfig], 1e-7, scanConfig.dataSeriesName)

        # save temperatures and ionDensities for reference
        np.savetxt(scanConfig.figureStoragePath + "electronTemperatures.txt", scanConfig.electronTemperatures)
        np.savetxt(scanConfig.figureStoragePath + "ionDensities.txt", scanConfig.ionDensities)
