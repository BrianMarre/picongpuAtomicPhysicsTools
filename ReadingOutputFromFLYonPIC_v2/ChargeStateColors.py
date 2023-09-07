import typeguard
import PlottingConfig.AtomicPopulationPlot as cfg

@typeguard.typechecked
def getChargeStateColors(config : cfg.AtomicPopulationPlotConfig, additionalIndices : list[int] = []):
    """@return dictionary assigning one color to each charge state"""
    colors = iter([config.colorMap(i) for i in range(config.numColorsInColorMap)])

    ## assign all chargeStates a color
    colorChargeStates = {}
    for z in range(config.atomicNumber + 1):
        try:
            colorChargeStates[z] = next(colors)
        except StopIteration:
            colors = iter([config.colorMap(i) for i in range(config.numColorsInColorMap)])
            colorChargeStates[z] = next(colors)

    for index in additionalIndices:
        try:
            colorChargeStates[index] = next(colors)
        except StopIteration:
            colors = iter([config.colorMap(i) for i in range(config.numColorsInColorMap)])
            colorChargeStates[index] = next(colors)

    return colorChargeStates
