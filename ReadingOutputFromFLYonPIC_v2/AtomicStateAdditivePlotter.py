
    # additive plot removed for now, @todo convert to readers, Brian Marre, 2024
    #plot_additive(config,
    #    mean, stdDev, axisDict_FLY´onPIC, atomicConfigNumbers_FLYonPIC, timeSteps_FLYonPIC,
    #    atomicPopulationData, axisDict_SCFLY, atomicConfigNumbers_SCFLY, timeSteps_SCFLY)

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
        chargeStates = np.fromiter(map(
                lambda configNumber: conv.getChargeState(
                    configNumber,
                    config.openPMDReaderConfig.atomicNumber,
                    config.openPMDReaderConfig.numLevels),
                atomicConfigNumbers_FLYonPIC),
            dtype=np.uint8)

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
            chargeState = conv.getChargeState(configNumber, config.openPMDReaderConfig.atomicNumber, config.openPMDReaderConfig.numLevels)

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