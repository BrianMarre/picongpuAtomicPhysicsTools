import pydantic
import typing
import matplotlib.colors as color


class AtomicPopulationPlotConfig(pydantic.BaseModel):
    """Config object for plotting atomic populations"""
    # name of ion species in openPMD output
    speciesName : str
    # atomic number of ion species
    atomicNumber : int
    # maximum principal quantum number used in SCFLY and FLYonPIC
    numLevels : int

    # path of file FLYonPIC atomic state data input file
    FLYonPICAtomicStateInputDataFile : str
    # path to atomciStateNaming.input file, contains for each SCFLY state its corresponding occupation number vector
    SCFLYatomicStateNamingFile : str

    # base path for FLYonPIC output fileNames
    FLYonPICBasePath : str
    # FLYonPIC output fileNames, each a regex describing openPMD naming of openPMD output files
    FLYonPICOutputFileNames : list[str]

    # path of SCFLY output file
    SCFLYOutputFileName : str
    # name of states to plot
    numberStatesToPlot : int
    # colormap to use
    colorMap : typing.Any
    # number of colors in colormap
    numColorsInColorMap : int

    # path for storing processed input data
    processedDataStoragePath : str
    # path for storing plots
    figureStoragePath : str
    # descriptive name of data set, used for plot labeling and storage naming, must be unique
    dataName : str
    # True: read form raw simulation output, False: load previously processed data from processedDataStoragePath
    loadRaw : bool
