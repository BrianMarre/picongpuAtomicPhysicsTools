"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
import typing

class ReaderConfig(pydantic.BaseModel):
    """Config object for reading FLYonPIC openPMD particle output"""

    # name of ion species in openPMD output
    speciesName : str
    # atomic number of ion species
    atomicNumber : int
    # maximum principal quantum number used
    numLevels : int

    # number of threads to use for reading
    numberWorkers : int
    # number of particles to pass to each thread
    chunkSize : int

    # path of file FLYonPIC atomic state input data file, used for converting collection Index to configNumber
    FLYonPICAtomicStateInputDataFile : str

    # base path for openPMD ouput
    FLYonPICBasePath : str
    # openPMD output fileNames, a regex describing openPMD naming of openPMD output files, see openPMD-api for details
    FLYonPICOutputFileNames : list[str]

    # path for storing processed data
    processedDataStoragePath : str
