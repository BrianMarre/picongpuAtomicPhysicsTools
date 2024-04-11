import numpy as np
import matplotlib.pyplot as plt

import Config as cfg
import ScanSCFLY as scan

if __name__ == "__main__":
    processedDataStoragePath = "preProcessedData/"
    chunkSize = 24

    previouslyRunInitialChargeStates = [2, 3, 4, 5, 6, 7, 8]
    unRunInitialChargeStates = [9]

    # create scanConfigs
    scanConfigs = []
    summaryPlotConfigs = []
    for z in previouslyRunInitialChargeStates:
        scanConfigs.append(
            cfg.SCFLYScan.ScanConfig(
                atomicNumber = 29,
                SCFLYatomicStateNamingFile = "/home/marre55/scflyInput/29_atomicStateNaming.input",
                atomicDataInputFile = "/home/marre55/scfly/atomicdata/FLYCHK_input_files/atomic.inp.29",
                electronTemperatures = np.concatenate([
                    np.arange(1,10)*1e2, np.arange(1,10)*1e3, np.arange(1,10)*1e4]), # eV
                ionDensities = np.concatenate([
                    np.arange(1,10)*1e21, np.arange(1,10)*1e22,
                    np.arange(1,10)*1e23]), # 1/cm^3
                timePoints = np.arange(101) * 3.3e-17, # s
                initialStateLevelVector = (2, 8, 19-z, 0, 0, 0, 0, 0, 0, 0),
                outputBasePath = "/home/marre55/scflyInput/",
                SCFLYBinaryPath = "/home/marre55/scfly/code/exe/scfly",
                outputFileName = "xout",
                dataSeriesName ="Cu_recombination_IPD_ScanZ_" + str(z),
                numberStatesToPlot = 870,
                colorMap = plt.cm.tab20b,
                numColorsInColorMap = 20,
                processedDataStoragePath = processedDataStoragePath,
                figureStoragePath = "SCFLY_Cu_ScanZ_recombination_IPD_/",
                runSCFLY = False,
                plotEachSim = False,
                plotSummary = True))
        summaryPlotConfigs.append(cfg.SummarySCFLYScanPlot.PlotConfig(
                        loadRawEachSCLFYSim = False,
                        loadRawSummaryData = False,
                        dataSetName = "Cu " + str(z) + "+"))

    for z in unRunInitialChargeStates:
        scanConfigs.append(
            cfg.SCFLYScan.ScanConfig(
                atomicNumber = 29,
                SCFLYatomicStateNamingFile = "/home/marre55/scflyInput/29_atomicStateNaming.input",
                atomicDataInputFile = "/home/marre55/scfly/atomicdata/FLYCHK_input_files/atomic.inp.29",
                electronTemperatures = np.concatenate([
                    np.arange(1,10)*1e2, np.arange(1,10)*1e3, np.arange(1,10)*1e4]), # eV
                ionDensities = np.concatenate([
                    np.arange(1,10)*1e21, np.arange(1,10)*1e22,
                    np.arange(1,10)*1e23]), # 1/cm^3
                timePoints = np.arange(101) * 3.3e-17, # s
                initialStateLevelVector = (2, 8, 19-z, 0, 0, 0, 0, 0, 0, 0),
                outputBasePath = "/home/marre55/scflyInput/",
                SCFLYBinaryPath = "/home/marre55/scfly/code/exe/scfly",
                outputFileName = "xout",
                dataSeriesName ="Cu_recombination_IPD_ScanZ_" + str(z),
                numberStatesToPlot = 870,
                colorMap = plt.cm.tab20b,
                numColorsInColorMap = 20,
                processedDataStoragePath = processedDataStoragePath,
                figureStoragePath = "SCFLY_Cu_ScanZ_recombination_IPD_/",
                runSCFLY = True,
                plotEachSim = False,
                plotSummary = True))
        summaryPlotConfigs.append(cfg.SummarySCFLYScanPlot.PlotConfig(
                        loadRawEachSCLFYSim = True,
                        loadRawSummaryData = True,
                        dataSetName = "Cu " + str(z) + "+"))

    # plot results
    scan.runScanList(scanConfigs, chunkSize, plotCombined=True,
                     summaryPlotConfigs=summaryPlotConfigs)
