 
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

        for chargeState in tqdm(range(config.openPMDReaderConfig.atomicNumber + 1)):
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

        for chargeState in tqdm(range(config.openPMDReaderConfig.atomicNumber + 1)):
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