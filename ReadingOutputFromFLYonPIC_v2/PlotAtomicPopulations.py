"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import typeguard

import numpy as np
import math
import json
from tqdm import tqdm

from . import Reader
from .SCFlyTools import AtomicConfigNumberConversion as conv
from . import Config as cfg
from . import ChargeStateColors

from . import LoadFLYonPICData
from . import LoadSCFLYData
from . import ReduceToPerChargeState

import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.scale as scale

from labellines import labelLines

@typeguard.typechecked
def loadNew(config : cfg.AtomicPopulationPlot.PlotConfig):
    """pre process raw data, store pre processed data at config specified path and return preprocessed data"""
    mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC = LoadFLYonPICData.loadFLYonPICData(config.OpenPMDReaderConfig)
    atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = LoadSCFLYData.loadSCFLYdata(config.OpenPMDReaderConfig)

    # write pre-processed data to file
    ## FLYonPIC
    if((type(mean) == np.ndarray)
        and (type(stdDev) == np.ndarray)
        and (type(timeSteps_FLYonPIC) == np.ndarray)
        and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray)
        and (axisDict_FLYonPIC != None)):
        np.savetxt(config.processedDataStoragePath + "mean_" + config.dataName + ".data", mean)
        np.savetxt(config.processedDataStoragePath + "stdDev_" + config.dataName + ".data", stdDev)
        np.savetxt(config.processedDataStoragePath + "atomicConfigNumbers_FLYonPIC_" + config.dataName + ".data",
                   atomicConfigNumbers_FLYonPIC)
        np.savetxt(config.processedDataStoragePath + "timeSteps_FLYonPIC_" + config.dataName + ".data",
                   timeSteps_FLYonPIC)
        with open(config.processedDataStoragePath + "axisDict_FLYonPIC_" + config.dataName
                + ".dict", 'w') as File:
            json.dump(axisDict_FLYonPIC, File)
    ## SCFLY
    if((type(atomicPopulationData) == np.ndarray)
        and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
        and (type(timeSteps_SCFLY) == np.ndarray)
        and (axisDict_SCFLY != None)):
        np.savetxt(config.processedDataStoragePath + "atomicPopulationData_" + config.dataName + ".data",
                atomicPopulationData)
        np.savetxt(config.processedDataStoragePath + "atomicConfigNumbers_" + config.dataName + ".data",
                atomicConfigNumbers_SCFLY)
        np.savetxt(config.processedDataStoragePath + "timeSteps_SCFLY_" + config.dataName + ".data",
                timeSteps_SCFLY)
        with open(config.processedDataStoragePath + "axisDict_SCFLY_" + config.dataName
                + ".dict", 'w') as File:
            json.dump(axisDict_SCFLY, File)

    # mark data as previously loaded
    config.loadRaw = False

    return mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
        atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY

@typeguard.typechecked
def loadPrevious(config : cfg.AtomicPopulationPlot.PlotConfig):
    """load previously processed atomic population data from file"""

    ## FLYonPIC
    if(config.FLYonPICAtomicStateInputDataFile == ""):
        mean = None
        stdDev = None
        axisDict_FLYonPIC = None
        atomicConfigNumbers_FLYonPIC = None
        timeSteps_FLYonPIC = None
    elif(len(config.FLYonPICOutputFileNames) == 0):
        mean = None
        stdDev = None
        axisDict_FLYonPIC = None
        atomicConfigNumbers_FLYonPIC = None
        timeSteps_FLYonPIC = None
    else:
        mean = np.loadtxt(config.processedDataStoragePath + "mean_" + config.dataName + ".data")
        stdDev = np.loadtxt(config.processedDataStoragePath + "stdDev_" + config.dataName + ".data")
        atomicConfigNumbers_FLYonPIC = np.loadtxt(
            config.processedDataStoragePath + "atomicConfigNumbers_FLYonPIC_" + config.dataName + ".data",)
        timeSteps_FLYonPIC = np.loadtxt(
            config.processedDataStoragePath + "timeSteps_FLYonPIC_" + config.dataName + ".data")
        with open(config.processedDataStoragePath + "axisDict_FLYonPIC_" + config.dataName + ".dict", 'r') as File:
            axisDict_FLYonPIC = json.load(File)

    ## SCFLY
    if(config.SCFLYatomicStateNamingFile == ""):
        atomicPopulationData = None
        axisDict_SCFLY = None
        atomicConfigNumbers_SCFLY = None
        timeSteps_SCFLY = None
    elif(config.SCFLYOutputFileName == ""):
        atomicPopulationData = None
        axisDict_SCFLY = None
        atomicConfigNumbers_SCFLY = None
        timeSteps_SCFLY = None
    else:
        atomicPopulationData = np.loadtxt(config.processedDataStoragePath + "atomicPopulationData_" + config.dataName + ".data")
        atomicConfigNumbers_SCFLY = np.loadtxt(config.processedDataStoragePath + "atomicConfigNumbers_" + config.dataName + ".data")
        timeSteps_SCFLY = np.loadtxt(config.processedDataStoragePath + "timeSteps_SCFLY_" + config.dataName + ".data")
        with open(config.processedDataStoragePath + "axisDict_SCFLY_" + config.dataName + ".dict", 'r') as File:
            axisDict_SCFLY = json.load(File)

    return mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
        atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY

@typeguard.typechecked
def plot_additive(config : cfg.AtomicPopulationPlot.PlotConfig,
                  mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
                  atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY) :
    """plot atomic populations as stacked line plot on linear scale"""
    colorChargeStates = ChargeStateColors.getChargeStateColors(config, [-1])

    # prepare plot
    figure = plt.figure(dpi=300)
    axes = figure.add_subplot(111)
    axes.set_title("AtomicPopulation Data: " + config.dataName)
    axes.set_xlabel("time[s]")
    axes.set_ylabel("relative abundance, additive")
    axes.set_ylim((0,1.1))

    maxTime = 0

    haveFLYonPICdata = ((type(mean) == np.ndarray)
       and (type(stdDev) == np.ndarray)
       and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray)
       and (type(timeSteps_FLYonPIC) == np.ndarray)
       and (axisDict_FLYonPIC != None))

    if haveFLYonPICdata:
        numberAtomicStates = np.shape(mean)[axisDict_FLYonPIC['atomicState']]

        maxTime = max(maxTime, np.max(timeSteps_FLYonPIC))

        assert(axisDict_FLYonPIC['atomicState'] == 1), "wrong axis ordering in FLYonPIC data according to axisDict_FLYonPIC"
        assert(axisDict_FLYonPIC['timeStep'] == 0), "wrong axis ordering in FLYonPIC data according to axisDict_FLYonPIC"

        ## calculate charge states
        chargeStates = np.fromiter(map(lambda configNumber: conv.getChargeState(configNumber, config.atomicNumber, config.numLevels), atomicConfigNumbers_FLYonPIC), dtype=np.uint8)

        ## create for mask unwanted chargeStates
        chargeStateMask = np.isin(chargeStates, np.array(config.chargeStatesToPlot, dtype=np.uint8))

        ## remove chargeStates we do not want to plot
        maskedMean = mean[:, chargeStateMask]
        maskedStdDev = stdDev[:, chargeStateMask]
        maskedAtomicConfigNumbers_FLYonPIC = atomicConfigNumbers_FLYonPIC[chargeStateMask]
        maskedChargeStates = chargeStates[chargeStateMask]
        maskedNumberAtomicStates = np.count_nonzero(chargeStateMask)

        del mean
        del stdDev
        del atomicConfigNumbers_FLYonPIC
        del chargeStates
        del chargeStateMask
        del numberAtomicStates

        ## prepare non standard states data, if we would plot at less than if we plot all
        if(config.numberStatesToPlot < (maskedNumberAtomicStates - 2)):
            # find numberStatesToPlot highest abundance states of the last iteration
            lastIteration = maskedMean[-1, :]
            sortedIndexationLastIteration = np.argsort(lastIteration)
            collectionIndicesOfPlotStates = sortedIndexationLastIteration[-config.numberStatesToPlot:]

            # find initial state with highest abundance
            collectionIndexInitialMaxAbundanceState = np.argmax(maskedMean[0, :])

            # remove initial state from list of standard plot states
            collectionIndicesOfPlotStates = np.where(
                collectionIndicesOfPlotStates != collectionIndexInitialMaxAbundanceState,
                collectionIndicesOfPlotStates,
                sortedIndexationLastIteration[-(config.numberStatesToPlot+1)])

            del sortedIndexationLastIteration

            # calculate other state density

            ## create mask for other states
            otherStateMask = np.full(maskedNumberAtomicStates, True, dtype=np.bool_)

            ### remove all plot states
            for i in range(config.numberStatesToPlot):
                otherStateMask = np.logical_and(np.arange(maskedNumberAtomicStates) != collectionIndicesOfPlotStates[i], otherStateMask)

            ### remove initial state
            otherStateMask = np.logical_and(np.arange(maskedNumberAtomicStates) != collectionIndexInitialMaxAbundanceState, otherStateMask)

            ## sum over all other states according to mask
            maskedMean_other = np.fromiter(
                map(lambda iteration : math.fsum(iteration), maskedMean[:, otherStateMask]), dtype='f8')
            maskedStdDev_other= np.sqrt(np.fromiter(
                map(lambda stdDevValue : math.fsum(stdDevValue**2), maskedStdDev[:, otherStateMask]),
                dtype='f8'))
            del otherStateMask
        else:
            # @attention we assume atomic State input file to be valid, i.e. already correctly sorted
            collectionIndicesOfPlotStates = np.arange(maskedNumberAtomicStates)

        ## plot standard states
        print()
        print("plotting FLYonPIC additive ...")
        widthBars = np.empty_like(timeSteps_FLYonPIC)
        widthBars[:-1] = timeSteps_FLYonPIC[1:] - timeSteps_FLYonPIC[:-1]
        widthBars[-1] = widthBars[-2]

        offset = 0
        for collectionIndex in tqdm(collectionIndicesOfPlotStates):
            chargeState = maskedChargeStates[collectionIndex]

            ### plot mean value
            axes.plot(timeSteps_FLYonPIC, maskedMean[:, collectionIndex] + offset, linewidth=1, alpha=0.5,
                    color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState))
            offset += maskedMean[:, collectionIndex]

            ### plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * maskedStdDev[:, collectionIndex], width=widthBars, bottom = offset - maskedStdDev[:, collectionIndex],
                align='center', color = colorChargeStates[chargeState], alpha=0.2)

        ## plot non standard states
        if(config.numberStatesToPlot < (maskedNumberAtomicStates - 2)):
            #plot initial state
            ## plot mean value
            chargeState = maskedChargeStates[collectionIndexInitialMaxAbundanceState]

            axes.plot(timeSteps_FLYonPIC, maskedMean[:, collectionIndexInitialMaxAbundanceState] + offset, linewidth=1,
                        alpha=0.5, color=colorChargeStates[chargeState],
                        label="[FLYonPIC] chargeState " + str(chargeState))

            offset += maskedMean[:, collectionIndexInitialMaxAbundanceState]

            ## plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * maskedStdDev[:, collectionIndexInitialMaxAbundanceState], width=widthBars,
                        bottom = offset - maskedStdDev[:, collectionIndexInitialMaxAbundanceState],
                        align='center', color=colorChargeStates[chargeState], alpha=0.2)

            # plot other state
            ## plot mean state
            axes.plot(timeSteps_FLYonPIC, maskedMean_other + offset, color=colorChargeStates[-1], linewidth=1, alpha=0.5,
                      label="state other")
            offset += maskedMean_other
            ## plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * maskedStdDev_other, width=widthBars, bottom = offset - maskedStdDev_other,
                align='center', color=colorChargeStates[-1], alpha=0.2)

    haveSCFLYData = ((type(atomicPopulationData) == np.ndarray)
       and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
       and (type(timeSteps_SCFLY) == np.ndarray)
       and (axisDict_SCFLY != None))

    if haveSCFLYData:
        maxTime = max(maxTime, np.max(timeSteps_SCFLY))

        assert(axisDict_SCFLY['atomicState'] == 1), "wrong axis ordering in SCFLY data"
        assert(axisDict_SCFLY['timeStep'] == 0), "wrong axis ordering in SCFLY data"

        # number Iterations
        numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

        print("plotting SCFLY additive ...")

        offset = np.cumsum(atomicPopulationData, axis=1)
        offset[:,1:] = offset[:,:-1]
        offset[:, 0] = 0

        # for each atomic state
        for i, configNumber in enumerate(atomicConfigNumbers_SCFLY):
            chargeState = conv.getChargeState(configNumber, config.atomicNumber, config.numLevels)

            if chargeState in config.chargeStatesToPlot:
                axes.plot(timeSteps_SCFLY, atomicPopulationData[:, i] + offset[:, i], linewidth=1, alpha=0.5,
                        linestyle="--",
                        color=colorChargeStates[chargeState], label="[SCFLY] chargeState " + str(int(chargeState)))

    axes.set_xlim((0,maxTime))
    handles, labels = axes.get_legend_handles_labels()
    uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

    print("saving...")
    plt.savefig(config.figureStoragePath + "AtomicPopulationData_additive_" + config.dataName,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(figure)

@typeguard.typechecked
def plot_absolute(config : cfg.AtomicPopulationPlot.PlotConfig,
                  mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
                  atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY):
    """plot atomic populations on logarithmic scale"""
    colorChargeStates = ChargeStateColors.getChargeStateColors(config)

    # prepare plot
    figure = plt.figure(dpi=300)
    axes = figure.add_subplot(111)
    axes.set_title("AtomicPopulation Data: " + config.dataName)
    axes.set_xlabel("time[s]")
    axes.set_ylabel("relative abundance")
    axes.set_yscale('log')
    axes.set_ylim((1e-7,1))

    maxTime = 0

    haveFLYonPICdata = ((type(mean) == np.ndarray)
       and (type(stdDev) == np.ndarray)
       and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray)
       and (type(timeSteps_FLYonPIC) == np.ndarray)
       and (axisDict_FLYonPIC != None))

    if haveFLYonPICdata:
        numberAtomicStates = np.shape(mean)[axisDict_FLYonPIC['atomicState']]
        maxTime = max(maxTime, np.max(timeSteps_FLYonPIC))

        assert(axisDict_FLYonPIC['atomicState'] == 1), "wrong axis ordering in FLYonPIC data"

        print()
        print("plotting FLYonPIC absolute ...")
        widthBars = np.empty_like(timeSteps_FLYonPIC)
        widthBars[:-1] = timeSteps_FLYonPIC[1:] - timeSteps_FLYonPIC[:-1]
        widthBars[-1] = widthBars[-2]

        for collectionIndex in tqdm(range(numberAtomicStates)):
            chargeState = conv.getChargeState(atomicConfigNumbers_FLYonPIC[collectionIndex],
                                              config.atomicNumber, config.numLevels)

            if chargeState in config.chargeStatesToPlot:
                ### plot mean value
                axes.plot(timeSteps_FLYonPIC, mean[:, collectionIndex], linewidth=1, alpha=0.5,
                        color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState))

                ### plot standard deviation
                axes.bar(timeSteps_FLYonPIC, 2 * stdDev[:, collectionIndex], width=widthBars,
                    bottom = mean[:, collectionIndex] - stdDev[:, collectionIndex],
                    align='center', color=colorChargeStates[chargeState], alpha=0.2)

    haveSCFLYData = ((type(atomicPopulationData) == np.ndarray)
       and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
       and (type(timeSteps_SCFLY) == np.ndarray)
       and (axisDict_SCFLY != None))

    if haveSCFLYData:

        maxTime = max(maxTime, np.max(timeSteps_SCFLY))

        assert(axisDict_SCFLY['atomicState'] == 1), "wrong axis ordering in SCFLY data"
        assert(axisDict_SCFLY['timeStep'] == 0), "wrong axis ordering in SCFLY data"

        # number Iterations
        numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

        print("plotting SCFLY absolute ...")

        # for each atomic state
        for i, configNumber in enumerate(atomicConfigNumbers_SCFLY):
            chargeState = conv.getChargeState(configNumber, config.atomicNumber, config.numLevels)

            if chargeState in config.chargeStatesToPlot:
                axes.plot(timeSteps_SCFLY, atomicPopulationData[:, i], linewidth=1, alpha=0.5, linestyle="--",
                        color=colorChargeStates[chargeState], label="[SCFLY] chargeState " + str(int(chargeState)))

    axes.set_xlim((0,maxTime))
    handles, labels = axes.get_legend_handles_labels()
    uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

    print("saving...")
    plt.savefig(config.figureStoragePath + "AtomicPopulationData_absolute_" + config.dataName,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(figure)

@typeguard.typechecked
def plotRelativeAbundanceOverall(config : cfg.AtomicPopulationPlot.PlotConfig,
                     mean, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC):
    """plot relative abundances for each atomic state and each step"""

    haveFLYonPICdata = ((type(mean) == np.ndarray)
       and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray)
       and (type(timeSteps_FLYonPIC) == np.ndarray)
       and (axisDict_FLYonPIC != None))
    if not haveFLYonPICdata:
        return

    print("plotting relative abundance overall ...")

    numberAtomicStates = np.shape(mean)[axisDict_FLYonPIC['atomicState']]
    numberTimeSteps = np.shape(mean)[axisDict_FLYonPIC['timeStep']]

    ## only plot states not always zero
    notAlwaysZero = np.any(mean != 0, axis=0)
    numberStatesToPlot = np.count_nonzero(notAlwaysZero)

    Y, X = np.meshgrid(np.arange(0,numberStatesToPlot), np.arange(0,numberTimeSteps))

    # prepare plot
    figure = plt.figure(dpi=200, figsize=(20,30))
    axes = figure.add_subplot(111)
    title = axes.set_title("relative abundances of atomic states: " + config.dataName)
    axes.set_xlabel("PIC step")
    axes.set_ylabel("atomic states")

    yticks = np.arange(0, numberStatesToPlot)
    ylabels = list(map(
        lambda atomicConfigNumber: str(conv.getLevelVector(atomicConfigNumber, config.atomicNumber, config.numLevels)),
        atomicConfigNumbers_FLYonPIC[notAlwaysZero]))
    axes.set_yticks(yticks, ylabels)
    axes.yaxis.set_tick_params(labelsize=2)

    xticks = np.arange(0, numberTimeSteps)[::5]
    axes.set_xticks(xticks, xticks)

    print("saving ...\n")
    plt.pcolormesh(X, Y, mean[:, notAlwaysZero], cmap=plt.cm.plasma, norm=color.LogNorm(vmin=1.e-5, vmax=2.))
    plt.colorbar()
    figure.tight_layout()
    plt.savefig(config.figureStoragePath + "AtomicPopulations_" + config.dataName)
    plt.close(figure)

@typeguard.typechecked
def plotSurplusByState(config : cfg.AtomicPopulationPlot.PlotConfig,
                     mean, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
                     atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY):
    """plot difference (FLYonPIC - SCFLY) for each atomic state and each step"""

    # check number atomic states is equal
    assert(np.shape(mean)[axisDict_FLYonPIC['atomicState']] == np.shape(atomicPopulationData)[axisDict_SCFLY['atomicState']])
    # check number time steps is equal
    assert(np.shape(mean)[axisDict_FLYonPIC['timeStep']] == np.shape(atomicPopulationData)[axisDict_SCFLY['timeStep']])

    # check that timeSteps ~equal
    assert(np.all(np.fromiter(map(
        lambda valueFLYonPIC, valueSCFLY: \
            ((np.abs(valueFLYonPIC - valueSCFLY) / valueSCFLY) <= 1.e-7) if valueSCFLY > 0 else valueFLYonPIC == 0,
        timeSteps_FLYonPIC, timeSteps_SCFLY), dtype=np.bool_)))

    numberAtomicStates = np.shape(mean)[axisDict_FLYonPIC['atomicState']]
    numberTimeSteps = np.shape(mean)[axisDict_FLYonPIC['timeStep']]

    assert((axisDict_FLYonPIC['atomicState'] == axisDict_SCFLY['atomicState'])
           and (axisDict_FLYonPIC['timeStep'] == axisDict_SCFLY['timeStep'])), "inconsistent axis between FLYonPIC and SCFLY"

    ## cut diff to resolution limit
    diff = mean - atomicPopulationData
    Y, X = np.meshgrid(np.arange(0,numberAtomicStates), np.arange(0,numberTimeSteps))

    # prepare plot
    figure = plt.figure(dpi=200, figsize=(20,30))
    axes = figure.add_subplot(111)
    title = axes.set_title("Difference of relative abundances of atomic states FLYonPIC vs SCFLY: " + config.dataName)
    axes.set_xlabel("PIC step")
    axes.set_ylabel("atomic states")

    yticks = np.arange(0, numberAtomicStates)
    ylabels = list(map(
        lambda atomicConfigNumber: str(conv.getLevelVector(atomicConfigNumber, config.atomicNumber, config.numLevels)),
        atomicConfigNumbers_SCFLY))
    axes.set_yticks(yticks, ylabels)
    axes.yaxis.set_tick_params(labelsize=2)

    xticks = np.arange(0, numberTimeSteps)[::5]
    axes.set_xticks(xticks, xticks)

    plt.pcolormesh(X, Y, diff, cmap=plt.cm.RdBu_r, norm=color.SymLogNorm(linthresh=1e-8),)
    plt.colorbar()
    figure.tight_layout()
    plt.savefig(config.figureStoragePath + "AtomicPopulation_diff_" + config.dataName)
    plt.close(figure)

@typeguard.typechecked
def plot_StepDiff(plotTimeSteps : list[int], config : cfg.AtomicPopulationPlot.PlotConfig,
                  mean, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
                  atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY):

    numberAtomicStates = np.shape(mean)[axisDict_FLYonPIC['atomicState']]
    numberTimeSteps = np.shape(mean)[axisDict_FLYonPIC['timeStep']]

    atomicStateCollectionIndices = np.arange(0, numberAtomicStates)

    assert((axisDict_FLYonPIC['atomicState'] == axisDict_SCFLY['atomicState'])
           and (axisDict_FLYonPIC['timeStep'] == axisDict_SCFLY['timeStep'])), "inconsistent axis between FLYonPIC and SCFLY"
    diff = mean - atomicPopulationData #(timeStep, atomicState)

    numberFigures = len(plotTimeSteps)

    # prepare figure
    figure, axes = plt.subplots(numberFigures, 1, dpi=200, figsize=(20,20))
    title = figure.suptitle("FLYonPIC vs SCFLY relative abundances: " + config.dataName)

    maxAbsDiff = np.max(np.abs(diff[plotTimeSteps])) * 1.1

    assert(axisDict_FLYonPIC['timeStep'] == 0)

    # plot all time steps
    for i, stepIdx in enumerate(plotTimeSteps):
        axePair = axes[i]
        axePair.set_title("step: " + str(stepIdx))
        axePair.plot(atomicStateCollectionIndices, diff[stepIdx], linestyle="-", marker="x")
        axePair.set_xticks([])
        axePair.set_xlim((0, numberAtomicStates))
        axePair.set_ylim((-maxAbsDiff, maxAbsDiff))
        axePair.set_yscale(scale.SymmetricalLogScale(axePair.yaxis, linthresh=1e-8))

    xlabels = np.fromiter(
        map(lambda collectionIndex: str(conv.getLevelVector(atomicConfigNumbers_FLYonPIC[collectionIndex],
                                                            config.atomicNumber, config.numLevels)),
        atomicStateCollectionIndices), dtype='U20')
    axePair.set_xticks(atomicStateCollectionIndices, xlabels)
    axePair.set_ylabel("(FLYonPIC - SCFLY) relative abundance")
    axePair.set_xlabel("atomic states")
    plt.setp(axePair.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    axePair.xaxis.set_tick_params(labelsize=2)

    print("saving stepDiff...")
    plt.tight_layout()
    plt.savefig(config.figureStoragePath + "AtomicPopulation_stepDiff_" + config.dataName, bbox_extra_artists=(title,))
    plt.close(figure)

@typeguard.typechecked
def plotRecombinationImportance(config : cfg.AtomicPopulationPlot.PlotConfig, FLYonPICInitialChargeState : int,
                                atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps):
    """plot SCFLY charge state populations below FLYonPIC initial charge state over time"""
    colorChargeStates = ChargeStateColors.getChargeStateColors(config, additionalIndices = [-1])

    # prepare plot
    figure = plt.figure(dpi=300)
    axes = figure.add_subplot(111)
    axes.set_title("RecombinationPlot: " + config.dataName)
    axes.set_xlabel("time[s]")
    axes.set_ylabel("relative abundance")
    axes.set_yscale('log')
    axes.set_ylim((1e-7,1))

    maxTime = np.max(timeSteps)

    # number Iterations
    numberIterations_SCFLY = np.shape(timeSteps)[0]
    numberAtomicStates = np.shape(atomicPopulationData)[axisDict['atomicState']]

    chargeStates = np.fromiter(
        map(lambda atomicConfigNumber: conv.getChargeState(atomicConfigNumber, config.atomicNumber, config.numLevels),
            atomicConfigNumbers), dtype='u1')

    #sum atomic states for each charge states below FLYonPIC charge state
    belowFLYonPICInitial = np.zeros((numberIterations_SCFLY, FLYonPICInitialChargeState))
    firstIndexInitialChargeState = 0
    for i in range(numberAtomicStates):
        chargeState = chargeStates[i]

        if chargeState < FLYonPICInitialChargeState:
            belowFLYonPICInitial[:, chargeState] += atomicPopulationData[:, i]
            firstIndexInitialChargeState = i+1
        elif chargeState == FLYonPICInitialChargeState:
            break

    # sum all atomic states above initial charge state
    aboveFLYonPICInitial = np.zeros(numberIterations_SCFLY)
    lastIndexInitialChargeState = firstIndexInitialChargeState
    for i in range(firstIndexInitialChargeState, numberAtomicStates):
        chargeState = chargeStates[i]
        if chargeState == FLYonPICInitialChargeState:
            lastIndexInitialChargeState = i
        if chargeState > FLYonPICInitialChargeState:
            aboveFLYonPICInitial += atomicPopulationData[:, i]

    print()
    print("plotting SCFLY recombination importance ...")

    # plot initial FLYonPIC charge state atomic states
    atomicStateLines = []
    for i in range(firstIndexInitialChargeState, lastIndexInitialChargeState + 1):
        atomicConfigNumber = atomicConfigNumbers[i]
        z = chargeStates[i]
        line = axes.plot(timeSteps, atomicPopulationData[:, i], linewidth=1, alpha=0.5, linestyle="--",
                    color=colorChargeStates[z], label=str(
                        conv.getLevelVector(atomicConfigNumber, config.atomicNumber, config.numLevels)))
        atomicStateLines.append(line[0])
    # plot below charge states
    for z in range(FLYonPICInitialChargeState):
        axes.plot(timeSteps, belowFLYonPICInitial[:, z], linewidth=1, alpha=0.5, linestyle="--",
                    color=colorChargeStates[z], label="[SCFLY] chargeState " + str(z))
    # plot other
    axes.plot(timeSteps, aboveFLYonPICInitial, linewidth=1, alpha=0.5, linestyle="--",
                color=colorChargeStates[-1], label="[SCFLY] other")

    axes.set_xlim((0,maxTime))
    # legend entries for each charge state
    handles, labels = axes.get_legend_handles_labels()
    uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]][-4:]
    lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

    # line annotation with level vectors
    labelLines(atomicStateLines, zorder=2.5, fontsize=4, backgroundcolor='none', alpha=0.5)

    print("saving recombinationImportance...")
    plt.savefig(config.figureStoragePath + "RecombinationImportance_" + config.dataName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(figure)
    print()


@typeguard.typechecked
def plotChargeStates(config : cfg.AtomicPopulationPlot.PlotConfig,
                     mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
                     atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY):
    """plot charge states relative abundance on logarithmic scale"""
    colorChargeStates = ChargeStateColors.getChargeStateColors(config)

    # prepare plot
    figure = plt.figure(dpi=300)
    axes = figure.add_subplot(111)
    axes.set_title("ChargeState population Data: " + config.dataName)
    axes.set_xlabel("time[s]")
    axes.set_ylabel("relative abundance")
    axes.set_yscale('log')
    axes.set_ylim((1e-5,1))

    maxTime = 0

    haveFLYonPICdata = ((type(mean) == np.ndarray) and (type(stdDev) == np.ndarray)
        and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray) and (type(timeSteps_FLYonPIC) == np.ndarray)
        and (axisDict_FLYonPIC != None))

    if haveFLYonPICdata:
        numberAtomicStates = np.shape(mean)[0]
        maxTime = max(maxTime, np.max(timeSteps_FLYonPIC))

        print()
        print("plotting chargeStates FLYonPIC absolute ...")
        widthBars = np.empty_like(timeSteps_FLYonPIC)
        widthBars[:-1] = timeSteps_FLYonPIC[1:] - timeSteps_FLYonPIC[:-1]
        widthBars[-1] = widthBars[-2]

        assert(axisDict_FLYonPIC['atomicState'] == 1), "wrong axis ordering in FLYonPIC data"

        chargeStateData, axisDict_ChargeState = ReduceToPerChargeState.reduceToPerChargeState(
            config, mean, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC)

        assert(axisDict_ChargeState['timeStep'] == 0)
        assert(axisDict_ChargeState['chargeState'] == 1)

        for chargeState in tqdm(range(config.atomicNumber + 1)):
            if chargeState in config.chargeStatesToPlot:
                ### plot mean value
                axes.plot(timeSteps_FLYonPIC, chargeStateData[:, chargeState], linewidth=1, alpha=0.5,
                        color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState))

    haveSCFLYData = ((type(atomicPopulationData) == np.ndarray)
       and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
       and (type(timeSteps_SCFLY) == np.ndarray)
       and (axisDict_SCFLY != None))

    if haveSCFLYData:
        maxTime = max(maxTime, np.max(timeSteps_SCFLY))

        # number Iterations
        numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

        print("plotting chargeStates SCFLY absolute ...")

        assert(axisDict_SCFLY['atomicState'] == 1), "wrong axis ordering in SCFLY data"

        chargeStateData, axisDict_ChargeState = ReduceToPerChargeState.reduceToPerChargeState(
            config, atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY)

        assert(axisDict_ChargeState['timeStep'] == 0)
        assert(axisDict_ChargeState['chargeState'] == 1)

        for chargeState in tqdm(range(config.atomicNumber + 1)):
            if chargeState in config.chargeStatesToPlot:
                axes.plot(timeSteps_SCFLY, chargeStateData[:, chargeState], linewidth=1, alpha=0.5, linestyle="--",
                        color=colorChargeStates[chargeState], label="[SCFLY] chargeState " + str(chargeState))

    axes.set_xlim((0,maxTime))
    handles, labels = axes.get_legend_handles_labels()
    uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

    print("saving...")
    plt.savefig(config.figureStoragePath + "ChargeStateData_absolute_" + config.dataName,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(figure)
    print()


@typeguard.typechecked
def plot_all(
    tasks_general : list[cfg.AtomicPopulationPlot.PlotConfig],
    tasks_diff : list[cfg.AtomicPopulationPlot.PlotConfig],
    tasks_recombination : list[cfg.AtomicPopulationPlot.PlotConfig],
    FLYonPICInitialChargeState : int = 0):

    # plot additive and absolute for atomic states and absolute for charge states
    for config in tasks_general:
        print(config.dataName)
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadNew(config)
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadPrevious(config)

        """plot_additive(config,
             mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)"""
        """plot_absolute(config,
             mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)"""
        plotChargeStates(config,
             mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)
        plotRelativeAbundanceOverall(config,
             mean, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC)

    # plot diff plots between FLYonPIC and SCFLY
    for config in tasks_diff:
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadNew(config)
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadPrevious(config)

        plotSurplusByState(config,
             mean, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)
        plotStepList = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plot_StepDiff(plotStepList, config,
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
            atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)

    # plot recombination importance from SCFLY scan
    for config in tasks_recombination:
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadNew(config)
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadPrevious(config)

        plotRecombinationImportance(config, FLYonPICInitialChargeState, atomicPopulationData,
                                    axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)
