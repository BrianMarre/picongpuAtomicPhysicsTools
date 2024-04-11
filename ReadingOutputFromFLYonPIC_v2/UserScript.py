"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 PIConGPU contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import Config as cfg
import ScanSCFLY as scan
import matplotlib.pypolt as plt

import PlotTimingData
import PlotSummarySCFLYScan
import PlotAtomicPopulations


def plotTimingData():
    basePath_Cu = "/home/marre55/picInputs/scflyComparison_Cu/"
    files_Cu = ["output_compare_30ppc_1.result", "output_compare_30ppc_2.result", "output_compare_30ppc_3.result", "output_compare_30ppc_4.result"]

    config_Copper = cfg.TimingDataPlot.TimingDataPlot(
        [basePath_Cu],
        [files_Cu],
        ["60ppc"],
        "TimingData_Cu_PressureIonization_IPD_60ppc_alpha_01",
        # plot init and finalize times
        False)

    PlotTimingData.plot(config_Copper)

def plotSCFLYScanSummary():
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

    summaryConfig_Cu = cfg.SummarySCFLYScanPlot.PlotConfig(
        loadRawEachSCLFYSim = False,
        loadRawSummaryData = True,
        dataSetName = "Cu Initial: 2+")

    # create scan baseConfigs
    baseConfigs, conditions, axisDict_conditions = scan.generateBaseConfigs(
        scanConfig_Cu)

    PlotSummarySCFLYScan.plotSummary(
        [scanConfig_Cu],
        [(baseConfigs, conditions, axisDict_conditions)],
        summaryConfig_Cu)

def plotAtomicPopulations():
    # base paths to FLYonPIC simulation openPMD output
    basePath_30ppc_Cu = "/home/marre55/picInputs/scflyComparison_Cu/openPMD_30ppc_Cu/"

    # fileName regexes
    fileNames_30ppc_Cu = ["simOutput_compare_1_%T.bp", "simOutput_compare_2_%T.bp",
                          "simOutput_compare_3_%T.bp", "simOutput_compare_4_%T.bp"]

    # FLYonPIC atomic states input data file
    FLYonPIC_atomicStates_Cu = "/home/marre55/picInputs/scflyComparison_Cu/AtomicStates_Cu.txt"

    # SCFLY filesspeciesName_Cu
    SCFLY_output_Cu_9 = "/home/marre55/scflyInput/Cu_recombination_IPD_ScanZ_9_25_Temp_9_Density/xout"
    SCFLY_output_Cu_2_1keV = "/home/marre55/scflyInput/Cu_recombination_IPD_ScanZ_2_9_Temp_10_Density/xout"
    SCFLY_stateNames_Cu = "/home/marre55/scflyInput/29_atomicStateNaming.input"

    # must be <= numberStates in input data set
    numberStatesToPlot_Cu = 869

    atomicNumber_Cu = 29
    numLevels_Cu = 10
    speciesName_Cu = "Cu"

    # colourmap
    colorMap_Cu = plt.cm.tab20b
    numColorsInColorMap_Cu = 20

    config_FLYonPIC_30ppc_SCFLY_Cu = cfg.AtomicPopulationPlot.PlotConfig(
        FLYonPICAtomicStateInputDataFile =  FLYonPIC_atomicStates_Cu,
        SCFLYatomicStateNamingFile =        SCFLY_stateNames_Cu,
        FLYonPICOutputFileNames =           fileNames_30ppc_Cu,
        FLYonPICBasePath =                  basePath_30ppc_Cu,
        SCFLYOutputFileName =               SCFLY_output_Cu_9,
        numberStatesToPlot =                numberStatesToPlot_Cu,
        colorMap =                          colorMap_Cu,
        numColorsInColorMap =               numColorsInColorMap_Cu,
        speciesName =                       speciesName_Cu,
        atomicNumber=                       atomicNumber_Cu,
        numLevels =                         numLevels_Cu,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "FLYonPIC_30ppc_SCFLY_Cu_PressureIonization_IPD_60ppc_alpha_01",
        loadRaw =                           True)

    config_FLYonPIC_60ppc_SCFLY_Cu_IPD_TestCase = cfg.AtomicPopulationPlot.PlotConfig(
        FLYonPICAtomicStateInputDataFile =  FLYonPIC_atomicStates_Cu,
        SCFLYatomicStateNamingFile =        SCFLY_stateNames_Cu,
        FLYonPICOutputFileNames =           fileNames_30ppc_Cu,
        FLYonPICBasePath =                  basePath_30ppc_Cu,
        SCFLYOutputFileName =               SCFLY_output_Cu_2,
        numberStatesToPlot =                numberStatesToPlot_Cu,
        colorMap =                          colorMap_Cu,
        numColorsInColorMap =               numColorsInColorMap_Cu,
        speciesName =                       speciesName_Cu,
        atomicNumber=                       atomicNumber_Cu,
        numLevels =                         numLevels_Cu,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "FLYonPIC_60ppc_alpha_01_SCFLY_Cu_IPDInIonization",
        loadRaw =                           True)

    tasks_general = [config_FLYonPIC_60ppc_SCFLY_Cu_IPD_TestCase]
    tasks_diff = [config_FLYonPIC_60ppc_SCFLY_Cu_IPD_TestCase]

    PlotAtomicPopulations.plot_all(tasks_general, tasks_diff, [])

if __name__ == "__main__":
    plotTimingData()
    #plotSCFLYScanSummary()
    plotAtomicPopulations()
