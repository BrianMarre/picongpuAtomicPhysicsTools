


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
                                                            config.openPMDReaderConfig.atomicNumber, config.openPMDReaderConfig.numLevels)),
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
        lambda atomicConfigNumber: str(conv.getLevelVector(
            atomicConfigNumber, config.openPMDReaderConfig.atomicNumber, config.openPMDReaderConfig.numLevels)),
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