 
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
        lambda atomicConfigNumber: str(conv.getLevelVector(atomicConfigNumber, config.openPMDReaderConfig.atomicNumber, config.openPMDReaderConfig.numLevels)),
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