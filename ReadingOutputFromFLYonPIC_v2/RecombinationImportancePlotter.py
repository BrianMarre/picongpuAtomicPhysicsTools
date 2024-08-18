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
        map(lambda atomicConfigNumber: conv.getChargeState(atomicConfigNumber, config.openPMDReaderConfig.atomicNumber,
                                                           config.openPMDReaderConfig.numLevels),
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
                        conv.getLevelVector(atomicConfigNumber, config.openPMDReaderConfig.atomicNumber, config.openPMDReaderConfig.numLevels)))
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
