import pydantic
import typeguard
import typing

import GenerateSCFLYSetups as generator
import PlotAtomicPopulations as plotter
import AtomicPopulationPlotConfig as cfg

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class ScanConfig(pydantic.BaseModel):
    atomicNumber : int
    SCFLYatomicStateNamingFile : str
    atomicDataInputFile : str

    # eV
    electronTemperatures : list[float]
    # 1/cm^3
    ionDensities : list[float]
    # s
    timePoints : list[float]
    # initial state occupation number vector
    initialStateLevelVector : tuple[int, ...]

    outputBasePath : str
    outputFileName : str
    SCFLYBinaryPath : str
    dataSeriesName : str

    numberStatesToPlot : int
    colorMap : typing.Any
    numColorsInColorMap : int
    processedDataStoragePath : str
    figureStoragePath : str

@typeguard.typechecked
def createSCFLYBaseConfigs(config : ScanConfig) -> list[generator.BaseConfig_SCFLY]:
    SCFLYconfigs = []

    for i, electronTemperature in enumerate(tqdm(config.electronTemperatures)):
        for j, ionDensity in enumerate(config.ionDensities):
            # create config for case
            comparisonFLYonPIC_Ar = generator.Config_SCFLY_FLYonPICComparison(
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
            SCFLYconfigs.append(generatedSetup)
    return SCFLYconfigs

@typeguard.typechecked
def generateSCFLYSetups(tasks : list[generator.BaseConfig_SCFLY]):
    for setup in tqdm(tasks):
        # generate setup and execute SCFLY
        setup.generateSCFLYSetup(config.SCFLYBinaryPath)

@typeguard.typechecked
def runSCFLYScan(config : ScanConfig, tasks : list[generator.BaseConfig_SCFLY]):
    for setup in tasks:
        # generate setup and execute SCFLY
        setup.execute(config.SCFLYBinaryPath)

@typeguard.typechecked
def plotSCFLYScan(config : ScanConfig, tasks : list[generator.BaseConfig_SCFLY_TimeDependent], FLYonPICInitialChargeState : int) -> list[cfg.AtomicPopulationPlotConfig]:
    plotConfigs = []
    for setup in tasks:
        # create plotting config
        plotConfig = cfg.AtomicPopulationPlotConfig(
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

        # plot SCFLY data
        plotter.plot_all([plotConfig], [], [plotConfig], FLYonPICInitialChargeState)

if __name__=="__main__":
    FLYonPICInitialChargeState = 2

    numberStatesToPlot = 470

    # colourmap
    colorMap = plt.cm.tab10
    numColorsInColorMap = 10

    processedDataStoragePath = "preProcessedData/"
    figureStoragePath = "SCFLYArScanImages/"

    scanConfig = ScanConfig(
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
        numberStatesToPlot = numberStatesToPlot,
        colorMap = colorMap,
        numColorsInColorMap = numColorsInColorMap,
        processedDataStoragePath = processedDataStoragePath,
        figureStoragePath = figureStoragePath)

    runSCFLY = False

    print(scanConfig.electronTemperatures)
    print(scanConfig.ionDensities)

    # create scan
    tasks = createSCFLYBaseConfigs(scanConfig)
    if runSCFLY:
        generateSCFLYSetups(tasks)
        runSCFLYScan(scanConfig, tasks)
    plotSCFLYScan(scanConfig, tasks, FLYonPICInitialChargeState)

    np.savetxt(config.figureStoragePath + "electronTemperatures.txt", config.electronTemperatures)
    np.savetxt(config.figureStoragePath + "ionDensities.txt", config.ionDensities)
