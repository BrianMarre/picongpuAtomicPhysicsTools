import numpy as np
import GenerateSCFLYSetups as generator
import PlotAtomicPopulations as plotter

import matplotlib.pyplot as plt
import json

electronTemperatures = np.concatenate([np.arange(1,10)*1e2, (np.arange(10)+1)*1e3]) # eV
ionDensities = np.concatenate([np.arange(1,10)*1e21, (np.arange(10)+1)*1e22]) # 1/cm^3

timePoints = np.arange(101) * 3.3e-17 # s

atomicNumber = 18

initialStateLevelVector = (2, 8, 6, 0, 0, 0, 0, 0, 0, 0)
SCFLYatomicStateNamingFile = "/home/marre55/scflyInput/18_atomicStateNaming.input"
atomicDataInputFile = "/home/marre55/scfly/atomicdata/FLYCHK_input_files/atomic.inp.18"

basePath = "/home/marre55/scflyInput/"
outputFileName = "xout"
dataSeriesName = "Ar_"

SCFLYBinaryPath = "/home/marre55/scfly/code/exe/scfly"

numberStatesToPlot = 470

# colourmap
colorMap = plt.cm.tab20b
numColorsInColorMap = 20

storagePath = "preProcessedData/"

SCFLYconfigs = []
plotConfigs = []

for i, electronTemperature in enumerate(electronTemperatures):
    for j, ionDensity in enumerate(ionDensities):

        # create config for case
        comparisonFLYonPIC_Ar = generator.Config_SCFLY_FLYonPICComparison(
            atomicNumber = atomicNumber,
            electronTemperature = electronTemperature, # eV
            ionDensity = ionDensity, # 1/cm^3
            timePoints = timePoints, # s
            initialStateLevelVector = initialStateLevelVector,
            SCFLYatomicStateNamingFile = SCFLYatomicStateNamingFile,
            atomicDataInputFile = atomicDataInputFile,
            outputFileName = outputFileName,
            basePath = basePath,
            folderName = dataSeriesName + "_" + str(i) + "_Temp_" + str(j) + "_Density")

        # store SCFLY config
        SCFLYconfigs.append(comparisonFLYonPIC_Ar)

        # generate setup and execute SCFLY
        #comparisonFLYonPIC_Ar.get().generateSCFLYSetup().execute(SCFLYBinaryPath)

        # create plotting config
        plotConfig = plotter.Config(
            "",
            SCFLYatomicStateNamingFile,
            [],
            "",
            comparisonFLYonPIC_Ar.basePath + comparisonFLYonPIC_Ar.folderName + "/" + comparisonFLYonPIC_Ar.outputFileName,
            numberStatesToPlot,
            colorMap,
            numColorsInColorMap,
            "",
            comparisonFLYonPIC_Ar.atomicNumber,
            len(comparisonFLYonPIC_Ar.initialStateLevelVector),
            "preProcessedData/",
            "SCFLY_" + comparisonFLYonPIC_Ar.folderName,
            loadRaw = True)

        # store plotting config
        plotConfigs.append(plotConfig)

        # plot SCFLY data
        #plotter.plot_all([plotConfig], [])

print(electronTemperatures)
print(ionDensities)
