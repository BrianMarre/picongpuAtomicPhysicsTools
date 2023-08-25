import numpy as np
import GenerateSCFLYSetups as generator
import PlotAtomicPopulations as plotter

def generate

electronTemperatures = np.concatenate([np.arange(1,10)*1e2, (np.arange(10)+1)*1e3]) # eV
ionDensities = np.concatenate([np.arange(1,10)*1e21, (np.arange(10)+1)*1e22]) # 1/cm^3

timePoints = np.arange(101) * 3.3e-17 # s

atomicNumber = 18
initialStateLevelVector = (2, 8, 6, 0, 0, 0, 0, 0, 0, 0)
SCFLYatomicStateNamingFile = "/home/marre55/scflyInput/18_atomicStateNaming.input"
atomicDataInputFile = "/home/marre55/scfly/atomicData/FLYCHK_input_files/atomic.inp.18"
basePath = "/home/marre55/scflyInput"

configs = []

for electronTemperature in electronTemperatures:
    for ionDensity in ionDensities:
        comparisonFLYonPIC_Ar = generator.Config_SCFLY_FLYonPICComparison(
            atomicNumber = atomicNumber,
            electronTemperature = electronTemperature, # eV
            ionDensity = ionDensity, # 1/cm^3
            timePoints = timePoints, # s
            initialStateLevelVector = initialStateLevelVector,
            SCFLYatomicStateNamingFile="/home/marre55/scflyInput/18_atomicStateNaming.input",
            atomicDataInputFile="/home/marre55/scfly/atomicData/FLYCHK_input_files/atomic.inp.18",
            outputFileName = "xout",
            basePath = "/home/marre55/scflyInput",
            folderName = "Ar_" + str(electronTemperature) + "eV_" + str(ionDensity) "PerCmCube")

        generatedSetup = comparisonFLYonPIC_Ar.get().generateSCFLYSetup()
        generatedSetup.execute("/home/marre55/scfly/code/exe/scfly")

        # store configs
        configs.append(comparisonFLYonPIC_Ar)


        # @todo
        configPlot = plotter.Config(
            "",
            SCFLY_stateNames_Ar,
            [],
            "",
            SCFLY_output_Ar,
            numberStatesToPlot_Ar,
            colorMap_Ar,
            numColorsInColorMap_Ar,
            speciesName_Ar,
            atomicNumber_Ar,
            numLevels_Ar,
            "preProcessedData/",
            "SCFLY_Ar",
            loadRaw=False)