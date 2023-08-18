import openPMD_Reader as readerOpenPMD
import SCFLY_Reader as readerSCFLY

import ConfigNumberConversion as conv

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import math

import json

def loadFLYonPICData(config):
    if(config.FLYonPIC_atomicStates == ""):
        print("SKIPPING FLYonPIC: missing FLYonPIC atomic state data input file")
        return None, None, None, None
    if(len(config.FLYonPIC_fileNames) == 0):
        print("SKIPPING FLYonPIC: missing FLYonPIC_fileNames")
        return None, None, None, None

    # load atomic input Data for common indexation of atomic states
    atomicStates = np.loadtxt(
        config.FLYonPIC_atomicStates, dtype=[('atomicConfigNumber', 'u8'), ('excitationEnergy', 'f4')])['atomicConfigNumber']

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
    for fileName in config.FLYonPIC_fileNames:
        sampleAtomicPopulationData, sampleTimeSteps = readerOpenPMD.getAtomicPopulationData(config.FLYonPIC_basePath + fileName, config.speciesName)
        sampleListAtomicPopulationData.append(sampleAtomicPopulationData)
        sampleListTimeSteps.append(sampleTimeSteps)

    numberSamples = len(config.FLYonPIC_fileNames)
    numberAtomicStates = len(state_to_collectionIndex.keys())
    numberIterations = len(sampleListTimeSteps[0])

    for sampleTimeSteps in sampleListTimeSteps[1:]:
        if np.any(sampleTimeSteps != sampleListTimeSteps[0]):
            raise RuntimeError("inconsistent time steps in samples")

    timeSteps = np.array(sampleListTimeSteps[0])
    del sampleListTimeSteps

    # convert to array
    data = np.empty((numberSamples, numberAtomicStates, numberIterations), dtype='f8')
    for i, sample in enumerate(sampleListAtomicPopulationData):
        for state, index in state_to_collectionIndex.items():
            data[i, index] = np.fromiter(
                map(lambda iteration: 0 if (iteration.get(state) is None) else iteration[state], sample), dtype='f8')

    # calculate total density
    totalDensity = np.empty((numberSamples, numberIterations), dtype='f8')
    for i in range(numberSamples):
        for j in range(numberIterations):
            totalDensity[i,j] = math.fsum(data[i, :, j])

    # convert to relative abundances
    data = data / totalDensity[:, np.newaxis, :]

    # calculate mean abundance and standard deviation
    mean = np.mean(data, axis = 0)

    if (numberSamples > 1):
        stdDev = np.std(data, axis = 0, ddof = 1)
    else:
        stdDev = np.zeros((numberAtomicStates, numberIterations))

    return mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps

def loadSCFLYdata(config):
    if(config.SCFLY_stateNames == ""):
        print("SKIPPING SCFLY: missing SCFLY_stateNames file")
        return None, None, None, None
    if(config.SCFLY_output == ""):
        print("SKIPPING SCFLY: missing SCFLY_output file")
        return None, None, None, None

    # load state names
    SCFLY_to_FLYonPIC = readerSCFLY.readSCFLYNames(config.SCFLY_stateNames, config.numLevels, config.atomicNumber)

    # load data
    atomicPopulationData, axisDict, atomicConfigNumbers, timeData = readerSCFLY.getSCFLY_Data(
        config.SCFLY_output, SCFLY_to_FLYonPIC)

    # calculate total densities
    assert((len(np.shape(atomicPopulationData)) == 2) and (axisDict['timeStep'] == 0))
    totalDensity = np.fromiter(map(lambda timeStep: math.fsum(timeStep) , atomicPopulationData), dtype='f8')

    # calculate relative abundances
    atomicPopulationData = atomicPopulationData / totalDensity[:, np.newaxis]

    return atomicPopulationData, axisDict, atomicConfigNumbers, timeData

def preProcess(config):
    """pre process raw data, store pre processed data at config specified path and return preprocessed data"""
    mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps_FLYonPIC = loadFLYonPICData(config)
    atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps_SCFLY = loadSCFLYdata(config)

    # write pre-processed data to file
    ## FLYonPIC
    if((type(mean) == np.ndarray)
        and (type(stdDev) == np.ndarray)
        and (type(timeSteps_FLYonPIC) == np.ndarray)
        and (collectionIndex_to_atomicConfigNumber != None)):
        np.savetxt(config.processedDataStoragePath + "mean_" + config.dataName + ".data", mean)
        np.savetxt(config.processedDataStoragePath + "stdDev_" + config.dataName + ".data", stdDev)
        np.savetxt(config.processedDataStoragePath + "timeSteps_FLYonPIC_" + config.dataName + ".data",
                    timeSteps_FLYonPIC)
        with open(config.processedDataStoragePath + "collectionIndex_to_ConfigNumber_" + config.dataName
                + ".dict", 'w') as File:
            json.dump(collectionIndex_to_atomicConfigNumber, File)
    ## SCFLY
    if((type(atomicPopulationData) == np.ndarray)
        and (type(atomicConfigNumbers) == np.ndarray)
        and (type(timeSteps_SCFLY) == np.ndarray)
        and (axisDict != None)):
        np.savetxt(config.processedDataStoragePath + "atomicPopulationData_" + config.dataName + ".data",
                atomicPopulationData)
        np.savetxt(config.processedDataStoragePath + "atomicConfigNumbers_" + config.dataName + ".data",
                atomicConfigNumbers)
        np.savetxt(config.processedDataStoragePath + "timeSteps_SCFLY_" + config.dataName + ".data",
                timeSteps_SCFLY)
        with open(config.processedDataStoragePath + "axis_" + config.dataName
                + ".dict", 'w') as File:
            json.dump(axisDict, File)

    return mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps_FLYonPIC, \
        atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps_SCFLY

def loadPreProcessed(config):
    """load pre-processed data from file"""

    ## FLYonPIC
    if(config.FLYonPIC_atomicStates == ""):
        print("SKIPPING FLYonPIC: missing FLYonPIC atomic state data input file")
        mean = None
        stdDev = None
        timeSteps_FLYonPIC = None
        collectionIndex_to_atomicConfigNumber = None
    elif(len(config.FLYonPIC_fileNames) == 0):
        print("SKIPPING FLYonPIC: missing FLYonPIC_fileNames")
        mean = None
        stdDev = None
        timeSteps_FLYonPIC = None
        collectionIndex_to_atomicConfigNumber = None
    else:
        mean = np.loadtxt(config.processedDataStoragePath + "mean_" + config.dataName + ".data")
        stdDev = np.loadtxt(config.processedDataStoragePath + "stdDev_" + config.dataName + ".data")
        timeSteps_FLYonPIC = np.loadtxt(config.processedDataStoragePath + "timeSteps_FLYonPIC_" + config.dataName + ".data")
        with open(config.processedDataStoragePath + "collectionIndex_to_ConfigNumber_" + config.dataName + ".dict", 'r') as File:
            conversionDictionary = json.load(File)

        # convert string keys back to int
        collectionIndex_to_atomicConfigNumber = {}
        for key, value in conversionDictionary.items():
            collectionIndex_to_atomicConfigNumber[int(key)] = value

    ## SCFLY
    if(config.SCFLY_stateNames == ""):
        print("SKIPPING SCFLY: missing SCFLY_stateNames file")
        atomicPopulationData = None
        axisDict = None
        atomicConfigNumbers = None
        timeSteps_SCFLY = None
    elif(config.SCFLY_output == ""):
        print("SKIPPING SCFLY: missing SCFLY_output file")
        atomicPopulationData = None
        axisDict = None
        atomicConfigNumbers = None
        timeSteps_SCFLY = None
    else:
        atomicPopulationData = np.loadtxt(config.processedDataStoragePath + "atomicPopulationData_" + config.dataName + ".data")
        atomicConfigNumbers = np.loadtxt(config.processedDataStoragePath + "atomicConfigNumbers_" + config.dataName + ".data")
        timeSteps_SCFLY = np.loadtxt(config.processedDataStoragePath + "timeSteps_SCFLY_" + config.dataName + ".data")
        with open(config.processedDataStoragePath + "axis_" + config.dataName + ".dict", 'r') as File:
            axisDict = json.load(File)

    return mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps_FLYonPIC, \
        atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps_SCFLY

def plot_additive(config,
         mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps_FLYonPIC,
         atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps_SCFLY):

    # colors
    colors = iter([config.colorMap(i) for i in range(config.numColorsInColorMap)])

    ## assign all chargeStates a color
    colorChargeStates = {}
    for z in range(config.atomicNumber + 1):
        try:
            colorChargeStates[z] = next(colors)
        except StopIteration:
            colors = iter([config.colorMap(i) for i in range(numColorsInColorMap)])
            colorChargeStates[z] = next(colors)

    # prepare plot
    figure = plt.figure(dpi=300)
    axes = figure.add_subplot(111)
    axes.set_title("AtomicPopulation Data: " + config.dataName)
    axes.set_ylim(bottom=0)
    axes.set_xlabel("PIC step")
    axes.set_ylabel("relative abundance, additive")
    axes.set_ylim((0,1.1))

    maxTime = 0

    # if have FLYonPIC data, plot it
    if((type(mean) == np.ndarray)
       and (type(stdDev) == np.ndarray)
        and (type(timeSteps_FLYonPIC) == np.ndarray)
       and (collectionIndex_to_atomicConfigNumber != None)):
        numberAtomicStates = np.shape(mean)[0]

        maxTime = max(maxTime, np.max(timeSteps_FLYonPIC))

        ## prepare non standard states data
        if(config.numberStatesToPlot < (numberAtomicStates - 2)):
            # find numberStatesToPlot highest abundance states of the last iteration
            lastIteration = mean[:,0]
            sortedIndexationLastIteration = np.argsort(lastIteration)
            collectionIndicesOfPlotStates = sortedIndexationLastIteration[-numberStatesToPlot:]

            # find initial state with highest abundance
            collectionIndexInitialMaxAbundanceState = np.argmax(mean[:,0])

            # remove initial state from list of standard plot states
            collectionIndicesOfPlotStates = np.where(collectionIndicesOfPlotStates != collectionIndexInitialMaxAbundanceState, collectionIndicesOfPlotStates, sortedIndexationLastIteration[-(config.numberStatesToPlot+1)])
            del sortedIndexationLastIteration

            # calculate other state density
            otherStateMask = np.full(numberAtomicStates, True, dtype='b')
            ## remove all plot states
            for i in range(numberStatesToPlot):
                otherStateMask = np.logical_and(np.arange(numberAtomicStates) != collectionIndicesOfPlotStates[i], otherStateMask)
            ## remove initial state
            otherStateMask = np.logical_and(np.arange(numberAtomicStates) != collectionIndexInitialMaxAbundanceState, otherStateMask)
            ## sum over all other states
            mean_other = np.fromiter(map(lambda iteration : math.fsum(iteration), np.transpose(mean[otherStateMask, :])), dtype='f8')
            stdDev_other= np.fromiter(map(lambda iteration : math.fsum(iteration), np.transpose((stdDev[otherStateMask, :]))), dtype='f8')

            try:
                colorChargeStates[-1] = next(colors)
            except StopIteration:
                colors = iter([config.colorMap(i) for i in range(numColorsInColorMap)])
                colorChargeStates[-1] = next(colors)
        else:
            # we assume atomic State input file to be valid, i.e. correctly sorted
            collectionIndicesOfPlotStates = np.arange(numberAtomicStates)

        ## plot standard states
        print("plotting FLYonPIC ...")
        widthBars = np.empty_like(timeSteps_FLYonPIC)
        widthBars[:-1] = timeSteps_FLYonPIC[1:] - timeSteps_FLYonPIC[:-1]
        widthBars[-1] = widthBars[-2]

        offset = 0
        for collectionIndex in tqdm(collectionIndicesOfPlotStates):
            chargeState = conv.getChargeState(collectionIndex_to_atomicConfigNumber[collectionIndex], config.atomicNumber, config.numLevels)

            ### plot mean value
            axes.plot(timeSteps_FLYonPIC, mean[collectionIndex, :] + offset, drawstyle='steps-mid',
                      color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState), linewidth=1, alpha=0.5)
            offset += mean[collectionIndex, :]

            ### plot standard deviation
            axes.bar( timeSteps_FLYonPIC, 2 * stdDev[collectionIndex, :], width=widthBars, bottom = offset - stdDev[collectionIndex, :],
                align='center', color=colorChargeStates[chargeState], alpha=0.2)

        ## plot non standard states
        if(config.numberStatesToPlot < (numberAtomicStates - 2)):
            #plot initial state
            ## plot mean value
            chargeState = conv.getChargeState(collectionIndex_to_atomicConfigNumber[collectionIndexInitialMaxAbundanceState], config.atomicNumber, config.numLevels)
            axes.plot(timeSteps_FLYonPIC, mean[collectionIndexInitialMaxAbundanceState, :] + offset, drawstyle='steps-mid',
                      color=colorChargeStates[chargeState], label="[FLYonPIC] chargeState " + str(chargeState), linewidth=1,
                      alpha=0.5)
            offset += mean[collectionIndexInitialMaxAbundanceState, :]
            ## plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * stdDev[collectionIndexInitialMaxAbundanceState, :], width=widthBars,
                     bottom = offset - stdDev[collectionIndexInitialMaxAbundanceState, :],
                     align='center', color=colorChargeStates[chargeState], alpha=0.2)

            # plot other state
            ## plot mean state
            axes.plot(timeSteps_FLYonPIC, mean_other + offset, drawstyle='steps-mid', color=colorChargeStates[-1],
                      label="state other", linewidth=1, alpha=0.5)
            offset += mean_other
            ## plot standard deviation
            axes.bar(timeSteps_FLYonPIC, 2 * stdDev_other, width=widthBars, bottom = offset - stdDev_other,
                align='center', color=colorChargeStates[-1], alpha=0.2)

    # if have SCFLY data, plot
    if((type(atomicPopulationData) == np.ndarray)
       and (type(atomicConfigNumbers) == np.ndarray)
       and (type(timeSteps_SCFLY) == np.ndarray)
       and (axisDict != None)):

        maxTime = max(maxTime, np.max(timeSteps_SCFLY))

        # number Iterations
        numberIterations_SCFLY = np.shape(timeSteps_SCFLY)[0]

        # sort states according to primary chargeState, secondary atomicConfigNumber
        chargeStates = np.fromiter(map(
            lambda atomicConfigNumber : conv.getChargeState(atomicConfigNumber, config.atomicNumber, config.numLevels),
            atomicConfigNumbers), dtype = 'u1')
        sortedIndices = np.lexsort((atomicConfigNumbers, chargeStates))
        del chargeStates

        atomicConfigNumbersSorted = atomicConfigNumbers[sortedIndices]
        atomicPopulationDataSorted = atomicPopulationData[:, sortedIndices]

        print("plotting SCFLY ...")

        offset = 0
        # for each atomic state
        for i, configNumber in enumerate(atomicConfigNumbersSorted):
            chargeState = conv.getChargeState(configNumber, config.atomicNumber, config.numLevels)

            axes.plot(timeSteps_SCFLY, atomicPopulationDataSorted[:, i] + offset, drawstyle='steps-mid',
                      color=colorChargeStates[chargeState], label="[SCFLY] chargeState " + str(int(chargeState)), linewidth=1)

            offset += atomicPopulationData[:, i]

    axes.set_xlim((0,maxTime))
    handles, labels = axes.get_legend_handles_labels()
    uniqueHandles = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    lgd = axes.legend(*zip(*uniqueHandles), loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')

    print("saving...")
    plt.savefig("AtomicPopulationData_" + config.dataName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print()

class Config:
    def __init__(
        self,
        FLYonPIC_atomicStates,
        SCFLY_stateNames,
        FLYonPIC_fileNames,
        FLYonPIC_basePath,
        SCFLY_output,
        numberStatesToPlot,
        colorMap,
        numColorsInColorMap,
        speciesName,
        atomicNumber,
        numLevels,
        processedDataStoragePath,
        dataName):
        self.FLYonPIC_atomicStates = FLYonPIC_atomicStates
        self.SCFLY_stateNames = SCFLY_stateNames
        self.FLYonPIC_fileNames = FLYonPIC_fileNames
        self.FLYonPIC_basePath = FLYonPIC_basePath
        self.SCFLY_output = SCFLY_output
        self.speciesName = speciesName
        self.atomicNumber = atomicNumber
        self.numLevels = numLevels
        self.numberStatesToPlot = numberStatesToPlot
        self.colorMap = colorMap
        self.numColorsInColorMap = numColorsInColorMap
        self.processedDataStoragePath = processedDataStoragePath
        self.dataName = dataName

if __name__ == "__main__":
    # base paths to FLYonPIC simulation openPMD output
    basePath_30ppc_Ar = "/home/marre55/picInputs/testSCFlyComparison_Ar/openPMD_30ppc/"
    basePath_60ppc_Ar = "/home/marre55/picInputs/testSCFlyComparison_Ar/openPMD_60ppc/"

    # fileName regexes
    fileNames_30ppc_Ar = ["simOutput_compare_2_%T.bp", "simOutput_compare_3_%T.bp", "simOutput_compare_4_%T.bp"]
    fileNames_60ppc_Ar = ["simOutput_compare_1_%T.bp", "simOutput_compare_2_%T.bp", "simOutput_compare_3_%T.bp",
                       "simOutput_compare_4_%T.bp"]

    # FLYonPIC atomic states input data file
    FLYonPIC_atomicStates_Ar = "/home/marre55/picInputs/testSCFlyComparison_Ar/AtomicStates_Ar.txt"

    # SCFLY files
    SCFLY_output_Ar = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Ar/xout"
    SCFLY_stateNames_Ar = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Ar/atomicStateNaming.input"

    SCFLY_output_Li = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Li/xout"
    SCFLY_stateNames_Li = "/home/marre55/scflyInput/testCase_ComparisonToFLYonPIC_Li/atomicStateNaming.input"

    # must be < numberStates in input data set
    numberStatesToPlot_Ar = 470

    atomicNumber_Ar = 18
    numLevels_Ar = 10
    speciesName_Ar = "Ar"

    atomicNumber_Li = 3
    numLevels_Li = 10
    speciesName_Li = "Li"

    # colourmap
    colorMap = plt.cm.tab20b
    numColorsInColorMap = 20

    loadRaw = True

    config_SCFLY_Ar = Config(
        "",
        SCFLY_stateNames_Ar,
        [],
        "",
        SCFLY_output_Ar,
        numberStatesToPlot_Ar,
        colorMap,
        numColorsInColorMap,
        speciesName_Ar,
        atomicNumber_Ar,
        numLevels_Ar,
        "preProcessedData/",
        "SCFLY_Ar")

    config_30ppc_Ar = Config(
        FLYonPIC_atomicStates_Ar,
        "",
        fileNames_30ppc_Ar,
        basePath_30ppc_Ar,
        "",
        numberStatesToPlot_Ar,
        colorMap,
        numColorsInColorMap,
        speciesName_Ar,
        atomicNumber_Ar,
        numLevels_Ar,
        "preProcessedData/",
        "FLYonPIC_30ppc_Ar")

    config_60ppc_Ar = Config(
        FLYonPIC_atomicStates_Ar,
        "",
        fileNames_60ppc_Ar,
        basePath_60ppc_Ar,
        "",
        numberStatesToPlot_Ar,
        colorMap,
        numColorsInColorMap,
        speciesName_Ar,
        atomicNumber_Ar,
        numLevels_Ar,
        "preProcessedData/",
        "FLYonPIC_60ppc_Ar")

    config_Ar_FLYonPIC_SCFLY = Config(
        FLYonPIC_atomicStates_Ar,
        SCFLY_stateNames_Ar,
        fileNames_60ppc_Ar,
        basePath_60ppc_Ar,
        SCFLY_output_Ar,
        numberStatesToPlot_Ar,
        colorMap,
        numColorsInColorMap,
        speciesName_Ar,
        atomicNumber_Ar,
        numLevels_Ar,
        "preProcessedData/",
        "FLYonPIC_60ppc_SCFLY_Ar")

    config_SCFLY_Li = Config(
        "",
        SCFLY_stateNames_Li,
        [],
        "",
        SCFLY_output_Li,
        numberStatesToPlot_Ar,
        colorMap,
        numColorsInColorMap,
        speciesName_Li,
        atomicNumber_Li,
        numLevels_Li,
        "preProcessedData/",
        "SCFLY_Li")

    tasks = [config_SCFLY_Li]#, config_SCFLY_Ar, config_30ppc_Ar, config_60ppc_Ar, config_Ar_FLYonPIC_SCFLY]

    for config in tasks:
        print(config.dataName)
        if loadRaw:
            mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps_SCFLY = preProcess(config)
        else:
            mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps_FLYonPIC, \
                atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps_SCFLY = loadPreProcessed(config)

        plot_additive(config,
             mean, stdDev, collectionIndex_to_atomicConfigNumber, timeSteps_FLYonPIC,
             atomicPopulationData, axisDict, atomicConfigNumbers, timeSteps_SCFLY)