import pydantic
import typing

class ScanConfig(pydantic.BaseModel):
    atomicNumber : int
    SCFLYatomicStateNamingFile : str
    atomicDataInputFile : str

    # eV
    electronTemperatures : list[float]
    # 1/cm^3
    ionDensities : list[float]
    # s
    timePoints : list[float]
    # initial state occupation number vector
    initialStateLevelVector : tuple[int, ...]

    outputBasePath : str
    outputFileName : str
    SCFLYBinaryPath : str
    dataSeriesName : str

    numberStatesToPlot : int
    colorMap : typing.Any
    numColorsInColorMap : int

    processedDataStoragePath : str
    figureStoragePath : str

    runSCFLY : bool
    loadRaw : bool
    plotEachSim : bool
    plotSummary : bool
