import numpy as np
import ConfigNumberConversion as trafo

def getAtomicStateData(filenameFlyCHK_output, filenameAtomicStateTable, Z):
    """ reads the atomic State data from a flyCHK spectrum output file

        parameters:
            - filenameFlyCHK_output ... path to flyCHK output file
            - filenameAtomicStateTable ... path to flyCHK atomic input data for index conversion
            - Z ... atomic number of the element, simulated
    """
    # flyCHK always uses 10 levels
    numLevels = 10

    # load output data:[(<ionizationState>, <Population of lower state>, <Population upper state>, <local state index lower>, <local state index upper>)]
    dataOccupation = np.loadtxt(filenameFlyCHK_output, dtype=[("Iso", 'u1'), ("Pop_lower", 'f8'), ("Pop_upper", 'f4'),
        ("localIndex_lower", 'u4'), ("localIndex_upper", 'u4')], skiprows=26, usecols=[0,8,9,10,11])

    # uniqueEntries is dictionary(iso:dictionary(localIndex:Pop)), containing all occupied states once
    uniqueEntries = {}
    for entry in dataOccupation:
        iso = entry["Iso"]
        uniqueEntriesIonizationState = uniqueEntries.get(iso)

        if uniqueEntriesIonizationState:
            lowerLocalIndex = entry["localIndex_lower"]
            upperLocalIndex = entry["localIndex_upper"]

            uniqueEntryState = uniqueEntriesIonizationState.get(lowerLocalIndex)
            if not uniqueEntryState:
                uniqueEntriesIonizationState[lowerLocalIndex]=entry["Pop_lower"]

            uniqueEntryState = uniqueEntriesIonizationState.get(upperLocalIndex)
            if not uniqueEntryState:
                uniqueEntriesIonizationState[upperLocalIndex]=entry["Pop_upper"]
        else:
            uniqueEntries[iso] = { entry["localIndex_lower"]:entry["Pop_lower"], entry["localIndex_upper"]:entry["Pop_upper"] }

    del dataOccupation

    # load atomic state data, taken from flylite level reader@Axel Huebl
    # regex that matches floats and scientific floats
    rg_flt = "([-\+[0-9]+\.[0-9]*[Ee]*[\+-]*[0-9]*)"
    # read data from file
    dtype = np.dtype([
            #      iz                j (in charge_state)      ilev
            ('charge_state', 'int'), ('state_idx', 'int'), ('total_idx', 'int'),
            ('name', '|S8'), # kname(ilev): unique naming of the state
            ('energy', np.float), # elev(ilev): energy from ground state [eV]
            # glev(ilev): number of permutations within state
            #             (spin up/down, angular momentum) [large int]
            ('statistical_weight', np.float),
            ('charge_screened', 'float'), # qlev(ilev)
            ('ionization_potential', 'float'), # evtoip(ilev) [eV]
            ('destruction_rate', 'float'), # gamx: destruction rate [1/s]
            # ncont(i,j): index of its own ionization level
            ('ioniz_lvl_idx', 'int'),
            # mobtot(ilev): maximum principle quantum number with population != 0
            #               (K=1, L=2, ...)
            ('nmax', 'int'),
            # noctot(ilev,k=1-10): shell population
            #   careful, those are not space separated but only aligned to 2 fields
            ('n_1', 'int'),
            ('n_2', 'int'), ('n_3', 'int'), ('n_4', 'int'),
            ('n_5', 'int'), ('n_6', 'int'), ('n_7', 'int'),
            ('n_8', 'int'), ('n_9', 'int'), ('n_10', 'int')
        ])

    #load dataStates
    dataStates = np.fromregex(filenameAtomicStateTable + "{:02d}".format(Z), # assure fixed width of Z("01" instead of "1")
        "\s*(\d+)\s+(\d+)\s+(\d+)\s+"
        "(........)\s+" +
        rg_flt + "\s+" +
        rg_flt + "\s+" +
        rg_flt + "\s+" + rg_flt + "\s+" + rg_flt + "\s+"
        "(\d+)\s+(\d+)\s+"
        "(\d+)(.\d)(.\d)(.\d)(.\d)(.\d)(.\d)(.\d)(.\d)(.\d)\s*\n",
        dtype=dtype)

    # dictionary(<configNumber>:<number density>)
    statePopulations = {}

    for state in dataStates:
        uniqueEntriesIonizationState = uniqueEntries.get( state["charge_state"] )

        if uniqueEntriesIonizationState:
            populationState = uniqueEntriesIonizationState.get( state["state_idx"] )

            if populationState:
                levelVector = np.empty(numLevels)

                for i in range(0, numLevels):
                    levelVector[i] = state["n_" + str(i+1)]

                configNumber = trafo.getConfigNumber(levelVector, Z)
                statePopulations[configNumber]=populationState

    return statePopulations