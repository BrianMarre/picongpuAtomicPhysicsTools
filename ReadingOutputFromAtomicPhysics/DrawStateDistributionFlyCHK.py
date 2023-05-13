import openpmd_api as io
import numpy as np
import math

import matplotlib.pyplot as plt
import plottingHelper as util

import ConfigNumberConversion as trafo
import openPMD_Reader as openPMD
import flyCHK_Reader as flyCHK

def DrawStateDistributionFlyCHK(speciesName, atomicNumber, numLevels, chargeIon,
    filenameAtomicStateData, filenameFlyCHK_output, filenamePIConGPU_output,
    title, caption, width, flyCHKplot_iteration, flyCHKplot_width,
    colormap, numColorsInColorMap,
    filename ):

    # get actual results
    resultsPIConGPU = openPMD.getAtomicStateData(filenamePIConGPU_output, speciesName, atomicNumber)
    resultsFlyCHK = flyCHK.getAtomicStateData(filenameFlyCHK_output, filenameAtomicStateData, atomicNumber)

    # open series for iteration data
    series = io.Series(filenamePIConGPU_output, io.Access.read_only)

    # create bar plot
    fig = plt.figure(dpi=400)
    ax = fig.add_axes((0.1,0.25,0.8,0.7))
    plt.title(title)
    plt.xlim((-5,200))
    plt.ylim((0,1))

    ax.set_ylabel("relative abundance")
    ax.set_xlabel('iteration')

    # assign each state a constant color, sorted by excitation level

    # 1.) find all occupied states

    # 1.1) in picongpu output
    states = {}     # dictionary(<configNumber>:<numElectrons>)
    # iterate over time steps
    for dict in resultsPIConGPU:
        # iterate over occupied states in current time step
        for state in dict.keys():
            numElectrons = states.get(state)
            if numElectrons is None:
                states[state] = np.sum(trafo.getLevelVector(state, atomicNumber, numLevels))

    # 1.2) in flyChk output
    for state in resultsFlyCHK.keys():
        numElectrons = states.get(state)
        if (numElectrons is None):
            numElectrons = np.sum(trafo.getLevelVector(state, atomicNumber, numLevels))
            # only states of a given ion charge
            if(numElectrons == (atomicNumber - chargeIon)):
                states[state] = numElectrons

    # 2.) get number of electrons of each state, and store them in specific order
    occupiedStates = np.empty(len(states.keys()), dtype="u8" )
    numElectrons = np.empty(len(states.keys()))

    i = 0
    for state in states.keys():
        occupiedStates[i] = state
        numElectrons[i] = states[state]
        i += 1

    # 3.) sort occupied states by numElectrons and secondary configNumber(approx. energy)
    indices = np.lexsort((occupiedStates, 18-numElectrons))
    occupiedStates = occupiedStates[indices]

    del numElectrons
    del states

    # color map
    colors = iter([colormap(i) for i in range(numColorsInColorMap)])

    # 4.) assign colors to the occupied states
    colorStates = {}
    for state in occupiedStates:
        try:
            colorStates[state] = next(colors)
        except StopIteration:
            colors = iter([colormap(i) for i in range(numColorsInColorMap)])
            colorStates[state] = next(colors)

    # plotting of picongpu results
    i = 0
    lastIteration = 0
    # for each time step in series
    for iteration in series.iterations:
        # get corresponding dictionary
        dict = resultsPIConGPU[i]

        # normalization quotient
        totalWeight = math.fsum(dict.values())
        """for weight in sorted(dict.values()):
            totalWeight += weight"""

        # offset for plotting bars over each other
        yOffset = 0
        # follow consistent ordering
        for state in occupiedStates:
            weight = dict.get(state)
            if not (weight is None):
                ax.bar(iteration, weight/totalWeight, width, bottom=yOffset,
                    label=str(trafo.getLevelVector(state, atomicNumber, numLevels)),
                    color=colorStates[state])
                yOffset += weight/totalWeight
        i += 1
        lastIteration = iteration

    # plotting of flyCHK results

    # normalization quotient
    numberDensitiesList = np.empty(len(occupiedStates))
    i = 0
    for state in occupiedStates:
        numberDensitiesList[i] = resultsFlyCHK[state]
        i += 1

    totalNumberDensity = math.fsum(numberDensitiesList)

    # offset for plotting bars over each other
    yOffset = 0
    for state in occupiedStates:
        numberDensity = resultsFlyCHK.get(state)

        if (not (numberDensity is None)):
            relativeAbundance = numberDensity/totalNumberDensity

            if (relativeAbundance >= 1e-3):
                labelBar = label=str(trafo.getLevelVector(state, atomicNumber, numLevels))
            else:
                labelBar = None

            ax.bar(flyCHKplot_iteration, relativeAbundance, flyCHKplot_width, bottom=yOffset,
                    label=labelBar,
                    color=colorStates[state], edgecolor="r")
            yOffset +=relativeAbundance

        else:
            print("state " + str(state) + " not represented in flyCHK?")

    # create plot entry for red border
    ax.bar(flyCHKplot_iteration,0, flyCHKplot_width, bottom=yOffset,
                    label="FlyCHK results with red border",
                    color="w", edgecolor="r")

    util.legend_without_duplicate_labels(ax)
    fig.text(0.1, 0.05, caption)
    plt.savefig(filename)
