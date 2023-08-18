import openpmd_api as io
import numpy as np
import math

def getAtomicStateData(filename, speciesName, atomicNumber):
    """ returns the atomic state data of an openPMD particle output of a simulation

        Paramters:
            filename(string-like) ... filenames of data input
            speciesName(string-like) ... string identifier of species
            atomicNumber(uint < 128) ... atomic Number of Ion-Species

        return value: list of dictionaries, each entry of the list corresponds to one
            iteration and each dictionary contains all occupied states and their
            associated weight as key-value pair

        @note this function is exessively commented, such that it may be used as an
        introduction to using the openPMD-api.
    """
    # open a series of adios [.bp](hdf[.h5]) files, in read only mode
    series = io.Series(filename, io.Access.read_only)

    listIterationData = []

    for i in series.iterations:
        # select simulation iteration
        step = series.iterations[i]

        # get one specific species from all species of particles
        species = step.particles[speciesName]

        # define data to be requested later
        atomicConfigNumber = species["atomicConfigNumber"][io.Mesh_Record_Component.SCALAR]
        weighting = species["weighting"][io.Mesh_Record_Component.SCALAR]

        # mark to be loaded
        dataAtomicConfigNumber = atomicConfigNumber.load_chunk()
        dataWeightings = weighting.load_chunk()

        # load data
        series.flush()

        # bin existing states
        states = {}

        # for all macro particles
        for i in range(0,np.shape(dataAtomicConfigNumber)[0]):
            configIndex = dataAtomicConfigNumber[i]

            # search in dictionary
            weighting = states.get(configIndex)

            # exists => add weight
            if weighting:
                states[configIndex].append(dataWeightings[i])
            # does not exist create new key with weight as initial value
            else:
                states[configIndex] = [dataWeightings[i]]

        for state in states.keys():
            states[state] = math.fsum(states[state])

        # store data
        listIterationData.append(states)

    # delete data
    del series

    return listIterationData
