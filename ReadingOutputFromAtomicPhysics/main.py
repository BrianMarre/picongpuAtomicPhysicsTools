import DrawEnergySim as energy
import DrawStateDistributionFlyCHK as flyCHK_population
import DrawStateDistributionSCFly as SCFly_population

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_FlyCHK_Plot():
    # specify ion species
    speciesName = "Cu"
    atomicNumber = 29

    chargeIon = 22

    filenamePIConGPU_output = "/home/marre55/picongpuOutput/testSolver13/simOutput/openPMD/simOutput_%T.bp"
    ppc = str(80)

    #text
    caption = ("picongpu test Simulation: " +speciesName + "-density: 1e20 cm^(-3), T_e = 0.6 keV\n"
        + "- "+ ppc + "ppc for all species\n")
    title = "atomic states of charge " + str(chargeIon) + " " + speciesName + " ions"

    #file output
    filename = "flyCHK_comparison_" + speciesName + "_1e20cm-3_600eV_"+ ppc + "ppc_2.png"

    # picongpu numLevels used in simulation
    numLevels = 10

    # set name of output
    filenameFlyCHK_output="/home/marre55/picongpuOutput/Analysis_Scripts/FlyCHK_Cu_600eV_1e20cm-3/flyspect_data_atomicStates.txt"
    filenameAtomicStateData="/home/marre55/flylite/data/atomic.inp."

    # width of bars in plot
    width = 1
    flyCHKplot_iteration = 120
    flyCHKplot_width = 10

    # colormap
    colormap = plt.cm.tab20b
    numColorsInColorMap = 20

    population.DrawStateDistributionFlyCHK(speciesName, atomicNumber, numLevels, chargeIon,
        filenameAtomicStateData, filenameFlyCHK_output, filenamePIConGPU_output,
        title, caption, width, flyCHKplot_iteration, flyCHKplot_width,
        colormap, numColorsInColorMap,
        filename)

def createEnergyPlot():
    simName = "testSolver13"
    speciesNames = ["Cu", "eth"]

    outputPath = "/home/marre55/picongpuOutput"
    title = "80ppc"
    pictureFileName = title + "Energy"

    drawEnergyOverTime(speciesNames, simName, title, pictureFileName, outputPath)

def main():
    SCFly_population.DrawStateDistributionSCFly()

if __name__ == "__main__":
    main()