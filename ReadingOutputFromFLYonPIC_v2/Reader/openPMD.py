"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

import openpmd_api as io
import numpy as np
import math
import typeguard
from tqdm import tqdm

@typeguard.typechecked
def getAtomicPopulationData(filename : str, speciesName : str, collectionIndex_to_atomicConfigNumber : dict[int, int]):
    """ returns the atomic state data of an openPMD particle output of a simulation

        Paramters:
            filename(string-like) ... filenames of data input
            speciesName(string-like) ... string identifier of species
            collectionIndex_to_atomicConfigNumber .. dictionary[collectionIndex:atomicConfigNumber] as in FYLonPIC input

        return value: list of dictionaries, each entry of the list corresponds to one
            iteration and each dictionary contains all occupied states and their
            associated weight as key-value pair

        @note this function is exessively commented, such that it may be used as an
        introduction to using the openPMD-api.
    """
    # open a series of adios [.bp] files, in read only mode
    series = io.Series(filename, io.Access.read_only)

    listIterationData = []
    listTimeSteps = []

    print(filename + " iteration:")
    for i in tqdm(series.iterations):

        # select simulation iteration
        step = series.iterations[i]

        # get one specific species from all species of particles
        species = step.particles[speciesName]

        print(species.particle_patches["numParticles"])

        #listTimeSteps.append(step.get_attribute("time") * step.get_attribute("timeUnitSI"))

        # define data to be requested later
        atomicStateCollectionIndex = species["atomicStateCollectionIndex"][io.Mesh_Record_Component.SCALAR]
        weighting = species["weighting"][io.Mesh_Record_Component.SCALAR]

        # mark to be loaded
        dataAtomicStateCollectionIndex = atomicStateCollectionIndex.load_chunk()
        dataWeightings = weighting.load_chunk()

        # load data
        series.flush()

        # bin existing states
        states = {}

        # for all macro particles, @todo parallelise in lockstep variety
        for j in range(0,np.shape(dataAtomicStateCollectionIndex)[0]):
            atomicConfigNumber = collectionIndex_to_atomicConfigNumber[dataAtomicStateCollectionIndex[j]]

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
