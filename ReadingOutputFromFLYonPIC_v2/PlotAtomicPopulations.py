import openPMD_Reader as readerOpenPMD
import SCFLY_Reader as readerSCFLY
import ConfigNumberConversion as conv
import AtomicPopulationPlotConfig as cfg
import ChargeStateColors

import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.scale as scale

from labellines import labelLines

import typeguard
from tqdm import tqdm
import numpy as np
import math
import json

@typeguard.typechecked
def loadFLYonPICData(config : cfg.AtomicPopulationPlotConfig):
    if(config.FLYonPICAtomicStateInputDataFile == ""):
        print("SKIPPING FLYonPIC: missing FLYonPIC atomic state data input file")
        return None, None, None, None, None
    if(len(config.FLYonPICOutputFileNames) == 0):
        print("SKIPPING FLYonPIC: missing FLYonPIC_fileNames")
        return None, None, None, None, None

    # load atomic input Data for common indexation of atomic states
    atomicStates = np.loadtxt(
        config.FLYonPICAtomicStateInputDataFile,
        dtype=[('atomicConfigNumber', 'u8'), ('excitationEnergy', 'f4')])['atomicConfigNumber']

    state_to_collectionIndex = {}
    collectionIndex_to_atomicConfigNumber = {}
    for i, state in enumerate(atomicStates):
        state_to_collectionIndex[state] = i
        collectionIndex_to_atomicConfigNumber[i] = int(state)
        assert(int(state) == state)
    del atomicStates

    # load in FLYonPIC data
    sampleListAtomicPopulationData = []
    sampleListTimeSteps = []
    for fileName in config.FLYonPICOutputFileNames:
        sampleAtomicPopulationData, sampleTimeSteps = readerOpenPMD.getAtomicPopulationData(
            config.FLYonPICBasePath + fileName, config.speciesName)
        sampleListAtomicPopulationData.append(sampleAtomicPopulationData)
        sampleListTimeSteps.append(sampleTimeSteps)

    numberSamples = len(config.FLYonPICOutputFileNames)
    numberAtomicStates = len(state_to_collectionIndex.keys())
    numberIterations = len(sampleListTimeSteps[0])

    for sampleTimeSteps in sampleListTimeSteps[1:]:
        if np.any(sampleTimeSteps != sampleListTimeSteps[0]):
            raise RuntimeError("inconsistent time steps in samples")

    timeSteps = np.array(sampleListTimeSteps[0])
    del sampleListTimeSteps

    # get array with all atomicConfigNumbers
    atomicConfigNumbers = np.fromiter(
        map(lambda i : collectionIndex_to_atomicConfigNumber[i], np.arange(numberAtomicStates)), dtype = 'u8')
    del collectionIndex_to_atomicConfigNumber

    # convert popualtion data to array
    data = np.empty((numberSamples, numberIterations, numberAtomicStates), dtype='f8')
    for i, sample in enumerate(sampleListAtomicPopulationData):
        for state, index in state_to_collectionIndex.items():
            data[i, :, index] = np.fromiter(
                map(lambda iteration: 0 if (iteration.get(state) is None) else iteration[state], sample), dtype='f8')

    # calculate total density
    totalDensity = np.empty((numberSamples, numberIterations), dtype='f8')
    for i in range(numberSamples):
        for j in range(numberIterations):
            totalDensity[i,j] = math.fsum(data[i, j])

    # convert to relative abundances
    data = data / totalDensity[:, :, np.newaxis]

    # calculate mean abundance and standard deviation
    mean = np.mean(data, axis = 0)

    if (numberSamples > 1):
        stdDev = np.std(data, axis = 0, ddof = 1)
    else:
        stdDev = np.zeros((numberIterations, numberAtomicStates))

    axisDict = {'timeStep':0, 'atomicState':1}
    return mean, stdDev, axisDict, atomicConfigNumbers, timeSteps

@typeguard.typechecked
def loadSCFLYdata(config : cfg.AtomicPopulationPlotConfig):
    if(config.SCFLYatomicStateNamingFile == ""):
        print("SKIPPING SCFLY: missing SCFLY_stateNames file")
        return None, None, None, None
    if(config.SCFLYOutputFileName == ""):
        print("SKIPPING SCFLY: missing SCFLY_output file")
        return None, None, None, None

    # load data
    atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps = readerSCFLY.getSCFLY_Data(
        config.SCFLYOutputFileName,
        readerSCFLY.readSCFLYNames(config.SCFLYatomicStateNamingFile, config.atomicNumber, config.numLevels)[0])

    # calculate total densities
    assert((len(np.shape(atomicPopulationData)) == 2) and (axisDict['timeStep'] == 0))
    totalDensity = np.fromiter(map(lambda timeStep: math.fsum(timeStep) , atomicPopulationData), dtype='f8')
    # calculate relative abundances
    atomicPopulationData = atomicPopulationData / totalDensity[:, np.newaxis]

    # sort data according to FLYonPIC sorting
    chargeStates = np.fromiter(map(
        lambda atomicConfigNumber : conv.getChargeState(atomicConfigNumber, config.atomicNumber, config.numLevels),
        atomicConfigNumbers), dtype = 'u1')
    sortedIndices = np.lexsort((atomicConfigNumbers, chargeStates))
    del chargeStates

    atomicConfigNumbersSorted = atomicConfigNumbers[sortedIndices]
    atomicPopulationDataSorted = atomicPopulationData[:, sortedIndices]
    del atomicConfigNumbers
    del atomicPopulationData

    return atomicPopulationDataSorted, axisDict, atomicConfigNumbersSorted, timeSteps

@typeguard.typechecked
def reduceToPerChargeState(config : cfg.AtomicPopulationPlotConfig, populationData, axisDict, atomicConfigNumbers):
    shape = np.shape(populationData)
    numberTimeSteps = shape[axisDict['timeStep']]
    numberAtomicStates = shape[axisDict['atomicState']]
    del shape

    assert(numberAtomicStates == np.shape(atomicConfigNumbers)[0]), "shape of populationData not consistent with atomicConfigNumbers"
    assert(axisDict['timeStep'] == 0)
    assert(axisDict['atomicState'] == 1)


    # reduce t per charge state
    chargeStateData = np.zeros((numberTimeSteps, config.atomicNumber + 1))
    for i in range(numberAtomicStates):
        atomicConfigNumber = atomicConfigNumbers[i]
        chargeState = conv.getChargeState(atomicConfigNumber, config.atomicNumber, config.numLevels)
        chargeStateData[:, int(chargeState)] += populationData[:, i]

    axisDict = {'timeStep' : 0, 'chargeState' : 1}
    return chargeStateData, axisDict

@typeguard.typechecked
def preProcess(config : cfg.AtomicPopulationPlotConfig):
    """pre process raw data, store pre processed data at config specified path and return preprocessed data"""
    mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC = loadFLYonPICData(config)
    atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadSCFLYdata(config)

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

    return mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
        atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY

@typeguard.typechecked
def loadProcessed(config : cfg.AtomicPopulationPlotConfig):
    """load previously processed atomic population data from file"""

    ## FLYonPIC
    if(config.FLYonPICAtomicStateInputDataFile == ""):
        print("SKIPPING FLYonPIC: missing FLYonPIC atomic state data input file")
        mean = None
        stdDev = None
        axisDict_FLYonPIC = None
        atomicConfigNumbers_FLYonPIC = None
        timeSteps_FLYonPIC = None
    elif(len(config.FLYonPICOutputFileNames) == 0):
        print("SKIPPING FLYonPIC: missing FLYonPIC_fileNames")
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
        print("SKIPPING SCFLY: missing SCFLY_stateNames file")
        atomicPopulationData = None
        axisDict_SCFLY = None
        atomicConfigNumbers_SCFLY = None
        timeSteps_SCFLY = None
    elif(config.SCFLYOutputFileName == ""):
        print("SKIPPING SCFLY: missing SCFLY_output file")
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
def plot_additive(config : cfg.AtomicPopulationPlotConfig,
                  mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
                  atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY):
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

    # if have FLYonPIC data, plot it
    if((type(mean) == np.ndarray)
       and (type(stdDev) == np.ndarray)
       and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray)
       and (type(timeSteps_FLYonPIC) == np.ndarray)
       and (axisDict_FLYonPIC != None)):
        numberAtomicStates = np.shape(mean)[axisDict_FLYonPIC['atomicState']]

        maxTime = max(maxTime, np.max(timeSteps_FLYonPIC))

        assert(axisDict_FLYonPIC['atomicState'] == 1), "wrong axis ordering in FLYonPIC data"

        ## prepare non standard states data, if we would plot at less than if we plot all
        if(config.numberStatesToPlot < (numberAtomicStates - 2)):
            # find numberStatesToPlot highest abundance states of the last iteration
            lastIteration = mean[:,-1]
            sortedIndexationLastIteration = np.argsort(lastIteration)
            collectionIndicesOfPlotStates = sortedIndexationLastIteration[-numberStatesToPlot:]

            # find initial state with highest abundance
            if axisDict_FLYonPIC['timeStep'] == 0:
                collectionIndexInitialMaxAbundanceState = np.argmax(mean[0, :])
            elif axisDict_FLYonPIC['timeStep'] == 1:
                collectionIndexInitialMaxAbundanceState = np.argmax(mean[:, 0])
            else:
                raise RuntimeError("invalid mean shape / axisDict_FLYonPIC")

            # remove initial state from list of standard plot states
            collectionIndicesOfPlotStates = np.where(
                collectionIndicesOfPlotStates != collectionIndexInitialMaxAbundanceState,
                collectionIndicesOfPlotStates,
                sortedIndexationLastIteration[-(config.numberStatesToPlot+1)])

            del sortedIndexationLastIteration

            # calculate other state density

            ## create mask for other states
            otherStateMask = np.full(numberAtomicStates, True, dtype='b')

            ### remove all plot states
            for i in range(numberStatesToPlot):
                otherStateMask = np.logical_and(np.arange(numberAtomicStates) != collectionIndicesOfPlotStates[i], otherStateMask)

            ### remove initial state
            otherStateMask = np.logical_and(np.arange(numberAtomicStates) != collectionIndexInitialMaxAbundanceState, otherStateMask)

            ## sum over all other states according to mask
            mean_other = np.fromiter(
                map(lambda iteration : math.fsum(iteration), np.transpose(mean[:, otherStateMask])), dtype='f8')
            stdDev_other= np.sqrt(np.fromiter(
                map(lambda stdDevValue : math.fsum(stdDevValue**2), np.transpose(stdDev[:, otherStateMask])),
                dtype='f8'))
        else:
            # @attention we assume atomic State input file to be valid, i.e. already correctly sorted
            collectionIndicesOfPlotStates = np.arange(numberAtomicStates)

        ## plot standard states
        print("plotting FLYonPIC additive ...")
        widthBars = np.empty_like(timeSteps_FLYonPIC)
        widthBars[:-1] = timeSteps_FLYonPIC[1:] - timeSteps_FLYonPIC[:-1]
        widthBars[-1] = widthBars[-2]

        offset = 0
        for collectionIndex in tqdm(collectionIndicesOfPlotStates):
            chargeState = conv.getChargeState(atomicConfigNumbers_FLYonPIC[collectionIndex], config.atomicNumber, config.numLevels)

            ### plot mean value
            axes.plot(timeSteps_FLYonPIC, mean[:, collectionIndex] + offset, linewidth=1, alpha=0.5,
                    color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState))
            offset += mean[:, collectionIndex]

            ### plot standard deviation
            axes.bar( timeSteps_FLYonPIC, 2 * stdDev[:, collectionIndex], width=widthBars, bottom = offset - stdDev[:, collectionIndex],
                align='center', color=colorChargeStates[chargeState], alpha=0.2)

        ## plot non standard states
        if(config.numberStatesToPlot < (numberAtomicStates - 2)):
            #plot initial state
            ## plot mean value
            chargeState = conv.getChargeState(atomicConfigNumbers[collectionIndexInitialMaxAbundanceState],
                                                config.atomicNumber, config.numLevels)
            axes.plot(timeSteps_FLYonPIC, mean[:, collectionIndexInitialMaxAbundanceState] + offset, linewidth=1,
                        alpha=0.5, color=colorChargeStates[chargeState],
                        label="[FLYonPIC] chargeState " + str(chargeState))
            offset += mean[:, collectionIndexInitialMaxAbundanceState]
            ## plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * stdDev[:, collectionIndexInitialMaxAbundanceState], width=widthBars,
                        bottom = offset - stdDev[:, collectionIndexInitialMaxAbundanceState],
                        align='center', color=colorChargeStates[chargeState], alpha=0.2)

            # plot other state
            ## plot mean state
            axes.plot(timeSteps_FLYonPIC, mean_other + offset, color=colorChargeStates[-1], linewidth=1, alpha=0.5,
                      label="state other")
            offset += mean_other
            ## plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * stdDev_other, width=widthBars, bottom = offset - stdDev_other,
                align='center', color=colorChargeStates[-1], alpha=0.2)

    # if have SCFLY data, plot
    if((type(atomicPopulationData) == np.ndarray)
       and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
       and (type(timeSteps_SCFLY) == np.ndarray)
       and (axisDict_SCFLY != None)):

        maxTime = max(maxTime, np.max(timeSteps_SCFLY))

        assert(axisDict_SCFLY['atomicState'] == 1), "wrong axis ordering in SCFLY data"

        # number Iterations
        numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

        print("plotting SCFLY additive ...")

        offset = np.cumsum(atomicPopulationData, axis=1)
        offset[:,1:] = offset[:,:-1]
        offset[:, 0] = 0

        # for each atomic state
        for i, configNumber in enumerate(atomicConfigNumbers_SCFLY):
            chargeState = conv.getChargeState(configNumber, config.atomicNumber, config.numLevels)

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
    print()

@typeguard.typechecked
def plot_absolute(config : cfg.AtomicPopulationPlotConfig,
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

 # if have FLYonPIC data, plot it
    if((type(mean) == np.ndarray)
       and (type(stdDev) == np.ndarray)
       and (type(timeSteps_FLYonPIC) == np.ndarray)
       and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray)
       and (axisDict_FLYonPIC != None)):

        numberAtomicStates = np.shape(mean)[axisDict_FLYonPIC['atomicState']]
        maxTime = max(maxTime, np.max(timeSteps_FLYonPIC))

        assert(axisDict_FLYonPIC['atomicState'] == 1), "wrong axis ordering in FLYonPIC data"

        print("plotting FLYonPIC absolute ...")
        widthBars = np.empty_like(timeSteps_FLYonPIC)
        widthBars[:-1] = timeSteps_FLYonPIC[1:] - timeSteps_FLYonPIC[:-1]
        widthBars[-1] = widthBars[-2]

        for collectionIndex in tqdm(range(numberAtomicStates)):
            chargeState = conv.getChargeState(atomicConfigNumbers_FLYonPIC[collectionIndex],
                                              config.atomicNumber, config.numLevels)

            ### plot mean value
            axes.plot(timeSteps_FLYonPIC, mean[:, collectionIndex], linewidth=1, alpha=0.5,
                      color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState))

            ### plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * stdDev[:, collectionIndex], width=widthBars,
                bottom = mean[:, collectionIndex] - stdDev[:, collectionIndex],
                align='center', color=colorChargeStates[chargeState], alpha=0.2)

    # if have SCFLY data, plot
    if((type(atomicPopulationData) == np.ndarray)
       and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
       and (type(timeSteps_SCFLY) == np.ndarray)
       and (axisDict_SCFLY != None)):

        maxTime = max(maxTime, np.max(timeSteps_SCFLY))

        assert(axisDict_SCFLY['atomicState'] == 1), "wrong axis ordering in SCFLY data"

        # number Iterations
        numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

        print("plotting SCFLY absolute ...")

        # for each atomic state
        for i, configNumber in enumerate(atomicConfigNumbers_SCFLY):
            chargeState = conv.getChargeState(configNumber, config.atomicNumber, config.numLevels)

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
    print()

@typeguard.typechecked
def plot_DiffByState(config : cfg.AtomicPopulationPlotConfig,
                     mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
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
    figure = plt.figure(dpi=300, figsize=(20,30))
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
def plot_StepDiff(plotTimeSteps : list[int], config : cfg.AtomicPopulationPlotConfig,
                  mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
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

    print("saving...")
    plt.tight_layout()
    plt.savefig(config.figureStoragePath + "AtomicPopulation_stepDiff_" + config.dataName, bbox_extra_artists=(title,))
    plt.close(figure)

@typeguard.typechecked
def plotRecombinationImportance(config : cfg.AtomicPopulationPlotConfig, FLYonPICInitialChargeState : int,
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

    maxTime = np.max(timeSteps_SCFLY)

    # number Iterations
    numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]
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

    print("plotting SCFLY recombination importance ...")

    # plot initial FLYonPIC charge state atomic states
    atomicStateLines = []
    for i in range(firstIndexInitialChargeState, lastIndexInitialChargeState + 1):
        atomicConfigNumber = atomicConfigNumbers[i]
        z = chargeStates[i]
        line = axes.plot(timeSteps_SCFLY, atomicPopulationData[:, i], linewidth=1, alpha=0.5, linestyle="--",
                    color=colorChargeStates[z], label=str(
                        conv.getLevelVector(atomicConfigNumber, config.atomicNumber, config.numLevels)))
        atomicStateLines.append(line[0])
    # plot below charge states
    for z in range(FLYonPICInitialChargeState):
        axes.plot(timeSteps_SCFLY, belowFLYonPICInitial[:, z], linewidth=1, alpha=0.5, linestyle="--",
                    color=colorChargeStates[z], label="[SCFLY] chargeState" + str(z))
    # plot other
    axes.plot(timeSteps_SCFLY, aboveFLYonPICInitial, linewidth=1, alpha=0.5, linestyle="--",
                color=colorChargeStates[-1], label="[SCFLY] other")

    axes.set_xlim((0,maxTime))
    # legend entries for each charge state
    handles, labels = axes.get_legend_handles_labels()
    uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]][-4:]
    lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

    # line annotation with level vectors
    labelLines(atomicStateLines, zorder=2.5, fontsize=4, backgroundcolor='none', alpha=0.5)

    print("saving...")
    plt.savefig(config.figureStoragePath + "RecombinationImportance_" + config.dataName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(figure)
    print()

@typeguard.typechecked
def plotChargeStates(config : cfg.AtomicPopulationPlotConfig,
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
    axes.set_ylim((1e-7,1))

    maxTime = 0

 # if have FLYonPIC data, plot it
    if((type(mean) == np.ndarray)
       and (type(stdDev) == np.ndarray)
       and (type(atomicConfigNumbers_FLYonPIC) == np.ndarray)
       and (type(timeSteps_FLYonPIC) == np.ndarray)
       and (axisDict_FLYonPIC != None)):

        numberAtomicStates = np.shape(mean)[0]
        maxTime = max(maxTime, np.max(timeSteps_FLYonPIC))

        print("plotting FLYonPIC absolute ...")
        widthBars = np.empty_like(timeSteps_FLYonPIC)
        widthBars[:-1] = timeSteps_FLYonPIC[1:] - timeSteps_FLYonPIC[:-1]
        widthBars[-1] = widthBars[-2]

        assert(axisDict_FLYonPIC['atomicState'] == 1), "wrong axis ordering in FLYonPIC data"

        chargeStateData, axisDict_ChargeState = reduceToPerChargeState(
            config, mean, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC)

        assert(axisDict_ChargeState['timeStep'] == 0)
        assert(axisDict_ChargeState['chargeState'] == 1)

        for chargeState in tqdm(range(config.atomicNumber + 1)):
            ### plot mean value
            axes.plot(timeSteps_FLYonPIC, chargeStateData[:, chargeState], linewidth=1, alpha=0.5,
                      color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState))

    # if have SCFLY data, plot
    if((type(atomicPopulationData) == np.ndarray)
       and (type(atomicConfigNumbers_SCFLY) == np.ndarray)
       and (type(timeSteps_SCFLY) == np.ndarray)
       and (axisDict_SCFLY != None)):

        maxTime = max(maxTime, np.max(timeSteps_SCFLY))

        # number Iterations
        numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

        print("plotting SCFLY absolute ...")

        assert(axisDict_SCFLY['atomicState'] == 1), "wrong axis ordering in SCFLY data"

        chargeStateData, axisDict_ChargeState = reduceToPerChargeState(
            config, atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY)

        assert(axisDict_ChargeState['timeStep'] == 0)
        assert(axisDict_ChargeState['chargeState'] == 1)

        for chargeState in tqdm(range(config.atomicNumber + 1)):
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
def plot_all(tasks_general : list[cfg.AtomicPopulationPlotConfig], tasks_diff : list[cfg.AtomicPopulationPlotConfig], tasks_recombination : list[cfg.AtomicPopulationPlotConfig], FLYonPICInitialChargeState : int = 0):
    # plot additive and absolute for atomic states and absolute for charge states
    for config in tasks_general:
        print(config.dataName)
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = preProcess(config)
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadProcessed(config)

        plot_additive(config,
             mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)
        plot_absolute(config,
             mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)
        plotChargeStates(config,
             mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)

    # plot diff plots between FLYonPIC and SCFLY
    for config in tasks_diff:
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = preProcess(config)
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadProcessed(config)

        plot_DiffByState(config,
             mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)
        plotStepList = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plot_StepDiff(plotStepList, config,
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
            atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)

    # plot recombination importance from SCFLY scan
    for config in tasks_recombination:
        if(config.loadRaw):
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = preProcess(config)
        else:
            mean, stdDev, axisDict_FLYonPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY = loadProcessed(config)

        plotRecombinationImportance(config, FLYonPICInitialChargeState, atomicPopulationData,
                                    axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)

if __name__ == "__main__":
    # base paths to FLYonPIC simulation openPMD output
    basePath_30ppc_Ar = "/home/marre55/picInputs/testSCFlyComparison_Ar/openPMD_30ppc/"
    basePath_60ppc_Ar = "/home/marre55/picInputs/testSCFlyComparison_Ar/openPMD_60ppc/"
    basePath_30ppc_Li = "/mnt/data1/marre55/testSCFLYComparison/openPMD_30ppc/"

    # fileName regexes
    fileNames_30ppc_Ar = ["simOutput_compare_2_%T.bp", "simOutput_compare_3_%T.bp", "simOutput_compare_4_%T.bp"]
    fileNames_60ppc_Ar = ["simOutput_compare_1_%T.bp", "simOutput_compare_2_%T.bp", "simOutput_compare_3_%T.bp",
                          "simOutput_compare_4_%T.bp"]
    fileNames_30ppc_Li = ["simOutput_compare_1_%T.bp"]

    # FLYonPIC atomic states input data file
    FLYonPIC_atomicStates_Ar = "/home/marre55/picInputs/testSCFlyComparison_Ar/AtomicStates_Ar.txt"
    FLYonPIC_atomicStates_Li = "/home/marre55/picInputs/testSCFlyComparison_Li/AtomicStates_Li.txt"

    # SCFLY files
    SCFLY_output_Ar = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Ar/xout"
    SCFLY_stateNames_Ar = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Ar/atomicStateNaming.input"

    SCFLY_output_Li = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Li/xout"
    SCFLY_stateNames_Li = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Li/atomicStateNaming.input"

    # must be < numberStates in input data set
    numberStatesToPlot_Ar = 470
    numberStatesToPlot_Li = 48

    atomicNumber_Ar = 18
    numLevels_Ar = 10
    speciesName_Ar = "Ar"

    atomicNumber_Li = 3
    numLevels_Li = 10
    speciesName_Li = "Li"

    # colourmap
    colorMap_Ar = plt.cm.tab20b
    numColorsInColorMap_Ar = 20

    colorMap_Li = plt.cm.tab10
    numColorsInColorMap_Li = 10

    config_FLYonPIC_30ppc_Ar = cfg.AtomicPopulationPlotConfig(
        FLYonPICAtomicStateInputDataFile =  FLYonPIC_atomicStates_Ar,
        SCFLYatomicStateNamingFile =        "",
        FLYonPICOutputFileNames =           fileNames_30ppc_Ar,
        FLYonPICBasePath =                  basePath_30ppc_Ar,
        SCFLYOutputFileName =               "",
        numberStatesToPlot =                numberStatesToPlot_Ar,
        colorMap =                          colorMap_Ar,
        numColorsInColorMap =               numColorsInColorMap_Ar,
        speciesName =                       speciesName_Ar,
        atomicNumber=                       atomicNumber_Ar,
        numLevels =                         numLevels_Ar,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "FLYonPIC_30ppc_Ar",
        loadRaw =                           True)

    config_FLYonPIC_60ppc_Ar = cfg.AtomicPopulationPlotConfig(
        FLYonPICAtomicStateInputDataFile =  FLYonPIC_atomicStates_Ar,
        SCFLYatomicStateNamingFile =        "",
        FLYonPICOutputFileNames =           fileNames_60ppc_Ar,
        FLYonPICBasePath =                  basePath_60ppc_Ar,
        SCFLYOutputFileName =               "",
        numberStatesToPlot =                numberStatesToPlot_Ar,
        colorMap =                          colorMap_Ar,
        numColorsInColorMap =               numColorsInColorMap_Ar,
        speciesName =                       speciesName_Ar,
        atomicNumber=                       atomicNumber_Ar,
        numLevels =                         numLevels_Ar,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "FLYonPIC_60ppc_Ar",
        loadRaw =                           True)

    config_FLYonPIC_60ppc_SCFLY_Ar = cfg.AtomicPopulationPlotConfig(
        FLYonPICAtomicStateInputDataFile =  FLYonPIC_atomicStates_Ar,
        SCFLYatomicStateNamingFile =        SCFLY_stateNames_Ar,
        FLYonPICOutputFileNames =           fileNames_60ppc_Ar,
        FLYonPICBasePath =                  basePath_60ppc_Ar,
        SCFLYOutputFileName =               SCFLY_output_Ar,
        numberStatesToPlot =                numberStatesToPlot_Ar,
        colorMap =                          colorMap_Ar,
        numColorsInColorMap =               numColorsInColorMap_Ar,
        speciesName =                       speciesName_Ar,
        atomicNumber=                       atomicNumber_Ar,
        numLevels =                         numLevels_Ar,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "FLYonPIC_60ppc_SCFLY_Ar",
        loadRaw =                           True)

    config_SCFLY_Ar = cfg.AtomicPopulationPlotConfig(
        FLYonPICAtomicStateInputDataFile =  "",
        SCFLYatomicStateNamingFile =        SCFLY_stateNames_Ar,
        FLYonPICOutputFileNames =           [],
        FLYonPICBasePath =                  "",
        SCFLYOutputFileName =               SCFLY_output_Ar,
        numberStatesToPlot =                numberStatesToPlot_Ar,
        colorMap =                          colorMap_Ar,
        numColorsInColorMap =               numColorsInColorMap_Ar,
        speciesName =                       speciesName_Ar,
        atomicNumber=                       atomicNumber_Ar,
        numLevels =                         numLevels_Ar,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "SCFLY_Ar",
        loadRaw =                           True)

    config_FLYonPIC_30ppc_SCFLY_Li = cfg.AtomicPopulationPlotConfig(
        FLYonPICAtomicStateInputDataFile =  FLYonPIC_atomicStates_Li,
        SCFLYatomicStateNamingFile =        SCFLY_stateNames_Li,
        FLYonPICOutputFileNames =           fileNames_30ppc_Li,
        FLYonPICBasePath =                  basePath_30ppc_Li,
        SCFLYOutputFileName =               SCFLY_output_Li,
        numberStatesToPlot =                numberStatesToPlot_Li,
        colorMap =                          colorMap_Li,
        numColorsInColorMap =               numColorsInColorMap_Li,
        speciesName =                       speciesName_Li,
        atomicNumber=                       atomicNumber_Li,
        numLevels =                         numLevels_Li,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "FLYonPIC_30ppc_SCFLY_Li",
        loadRaw =                           True)

    config_SCFLY_Li = cfg.AtomicPopulationPlotConfig(
        FLYonPICAtomicStateInputDataFile =  "",
        SCFLYatomicStateNamingFile =        SCFLY_stateNames_Li,
        FLYonPICOutputFileNames =           [],
        FLYonPICBasePath =                  "",
        SCFLYOutputFileName =               SCFLY_output_Li,
        numberStatesToPlot =                numberStatesToPlot_Li,
        colorMap =                          colorMap_Li,
        numColorsInColorMap =               numColorsInColorMap_Li,
        speciesName =                       speciesName_Li,
        atomicNumber=                       atomicNumber_Li,
        numLevels =                         numLevels_Li,
        processedDataStoragePath =          "preProcessedData/",
        figureStoragePath =                 "",
        dataName =                          "SCFLY_Li",
        loadRaw =                           True)

    tasks_general = [config_FLYonPIC_30ppc_SCFLY_Li, config_SCFLY_Li, config_SCFLY_Ar, config_FLYonPIC_30ppc_Ar, config_FLYonPIC_60ppc_Ar, config_FLYonPIC_60ppc_SCFLY_Ar]
    tasks_diff = [config_FLYonPIC_60ppc_SCFLY_Ar]

    plot_all(tasks_general, tasks_diff, [])