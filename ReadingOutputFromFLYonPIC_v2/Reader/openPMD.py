import openpmd_api as io
import numpy as np
import math
from tqdm import tqdm

def getAtomicPopulationData(filename, speciesName):
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
    # open a series of adios [.bp] files, in read only mode
    series = io.Series(filename, io.Access.read_only)

    #particle data available output, debug code
    #step = series.iterations[0]
    #for particleSpecies in step.particles:
    #    print("\t {0}".format(particleSpecies))
    #    print("With records:")
    #    for record in step.particles[particleSpecies]:
    #        print("\t {0}".format(record))

    listIterationData = []
    listTimeSteps = []

    print(filename + " iteration:")
    for i in tqdm(series.iterations):

        # select simulation iteration
        step = series.iterations[i]

        # get one specific species from all species of particles
        species = step.particles[speciesName]

        listTimeSteps.append(step.get_attribute("time") * step.get_attribute("timeUnitSI"))

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

        # for all macro particles, @todo parallelise in lockstep variety
        for j in range(0,np.shape(dataAtomicConfigNumber)[0]):
            atomicConfigNumber = dataAtomicConfigNumber[j]

            # search in dictionary
            weighting = states.get(atomicConfigNumber)

            # exists => add weight
            if weighting:
                states[atomicConfigNumber].append(dataWeightings[j])
            # does not exist create new key with weight as initial value
            else:
                states[atomicConfigNumber] = [dataWeightings[j]]

        for state in states.keys():
            states[state] = math.fsum(states[state])

        # store data
        listIterationData.append(states)

    # delete data
    del series

    return listIterationData, listTimeSteps
