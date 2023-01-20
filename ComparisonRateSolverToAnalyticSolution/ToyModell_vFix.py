import numpy as np
import numpy.linalg as linalg
import numpy.random as rng

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as tick

# diagonal entries of rate matrix from non-diagonal entries
def fillDiagonalEntries(rateMatrix, numStates):
    for i in range(numStates):
        rateMatrix[i,i] = 0

        for j in range(numStates):
            if (j != i):
                rateMatrix[i,i] += -rateMatrix[j,i]


# takes rate matrix and calculates a steady state solution,
# with the sum of all state densities normalized to one
def steadyStateSolution(rateMatrix):
    shape = np.shape(rateMatrix)

    # verification of assumptions
    if len(shape) != 2:
        raise("rateMatrix must be 2 dimensional")
    if shape[0] != shape[1]:
        raise("rate Matrix must be square")

    numStates = shape[0]
    del shape

    # copy to avoid side effects
    rateMatrixLocal = np.copy(rateMatrix)

    #replace last row with condition that all sum(N_i) = 1
    for i in range(numStates):
        rateMatrixLocal[numStates-1, i] = 1

    #solution vector of steady state condition rateMatrix * N_steadyState = (0,...,0,1)
    solution = np.zeros(numStates)
    solution[numStates-1] = 1

    steadyStateSolution = linalg.solve(rateMatrixLocal, solution)
    return steadyStateSolution


# calculate time dependent analytic solution for constant rate amtrix
def analyticTimeDependentSolution(initStateVector, rateMatrix, numTimePoints, startTime, endTime, numStates):
    # get eigen vectors  and values of rate matrix
    eigenValues, eigenVectors = linalg.eig(rateMatrix)

    # determine initial state coefficients
    initialCoefficients = linalg.solve(eigenVectors, initStateVector)

    times = np.linspace(startTime, endTime, num=numTimePoints)
    relativeDensity = np.empty((numTimePoints, numStates), dtype=np.csingle)
    # analytic time dependent solution
    for i in range(numDrawPoints):
        # get a current state vector for a given time
        relativeDensity[i] = np.dot(eigenVectors, ( initialCoefficients * np.exp(eigenValues * (times[i]-startTime) )))
    return times, relativeDensity.real


# do one solver step on the given particle
def processIonExponential(rateMatrix, numStates, particle):
    if particle['timeRemaining'] <= 0:
        return

    newState = rng.randint(0,numStates)
    oldState = particle['atomicState']

    randomNumber = rng.random()

    timeRemaining = particle['timeRemaining']

    if newState == oldState:
        # no state change solver
        quasiProbability = 1 + rateMatrix[oldState, oldState] * timeRemaining
                             # different sign since already negated

        if quasiProbability == 1:
            print("remark: isolated state encountered")
            'test'
        elif quasiProbability <= 0:
            # too busy
            timeRemaining = 1/(- rateMatrix[oldState, oldState])
        # remaining case 0 < quasiProbability < 1, do nothing

        probability = np.exp( rateMatrix[oldState, oldState] * timeRemaining)
                             # different sign since already negated

        if randomNumber <= probability:
            # succesful no-change transition
            particle['timeRemaining'] -= timeRemaining

    else:
        # standard solver case
        quasiProbability = rateMatrix[newState, oldState] * timeRemaining

        if quasiProbability >= 1:
            timeRemaining = 1/rateMatrix[newState, oldState]
        # remaining case 0 < quasiProbability < 1, do nothing

        probability = 1 - np.exp( - rateMatrix[newState, oldState] * timeRemaining)

        if randomNumber <= probability:
            particle['atomicState'] = newState
            particle['timeRemaining'] -= timeRemaining

    return particle


# do one solver step on the given particle
def processIonLinear(rateMatrix, numStates, particle):
    if particle['timeRemaining'] <= 0:
        return

    newState = rng.randint(0,numStates)
    oldState = particle['atomicState']

    randomNumber = rng.random()


    if newState == oldState:
        # no state change solver
        quasiProbability = 1 + rateMatrix[oldState, oldState] * particle['timeRemaining']
                            # different sign since already negated

        if quasiProbability > 1:
            print("error: negative time encountered in no state change")
            particle['timeRemaining'] = 0

        elif quasiProbability == 1:
            print("remark: isolated state encountered")

        elif quasiProbability < 0:
            'test'
            # will change more than once in time remaining

        elif randomNumber <= quasiProbability:
            particle['timeRemaining'] = 0
            # succesful no-change transition

    else:
        # standard solver case
        quasiProbability = rateMatrix[newState, oldState] * particle['timeRemaining']

        if quasiProbability >= 1:
            # would happen more than once
            particle['atomicState'] = newState
            particle['timeRemaining'] -= 1/rateMatrix[newState, oldState]

            if rateMatrix[newState, oldState] < 0:
                particle['timeRemaining'] = 0
                print("error: negative time encountered in state change")

        elif quasiProbability < 0:
            particle['timeRemaining'] = 0
            print("error: negative rate or time encountered")

        elif randomNumber <= quasiProbability:
            # would happen less than once
            particle['atomicState'] = newState
            particle['timeRemaining'] = 0
    return particle


# do one solver step of given step length
def solverStep(macroParticles, rateMatrix, timeStep, numStates, processIonFunctor):
    macroParticles['timeRemaining'] = timeStep # unit: t_PIC

    fullyProcessed = False
    while(not(fullyProcessed)):
        fullyProcessed = True

        for particle in macroParticles:
            if particle['timeRemaining'] > 0: 
                processIonFunctor(rateMatrix, numStates, particle)

                if particle['timeRemaining'] > 0:
                    fullyProcessed = False


def solve(
    macroParticles, rateMatrix, timeStepLength, numTimeSteps, numStates, processIonFunctor,
    consoleOutputInterval = None):
    macroParticleHistograms = []

    bins = range(numStates+1)

    # do solver steps
    for i in range(numTimeSteps):
        solverStep(macroParticles, rateMatrix, timeStepLength, numStates, approximationMethods[approximationMethod])

        if not(consoleOutputInterval is None) and ((i + 1) % consoleOutputInterval == 0):
            print("\t timeStep: " + str(i+1))


        # state binning
        macroParticleHistogram, binEdges = np.histogram(macroParticles['atomicState'], bins=bins, weights=macroParticles['weight'])
        macroParticleHistograms.append(macroParticleHistogram)

    if not(consoleOutputInterval is None):
        print("\t timeStep: " + str(i+1))

    return np.array(macroParticleHistograms)


approximationMethods = {"exponential" : processIonExponential , "linear" : processIonLinear}


def createEqualWeightMacroParticles(initStateVector, numMacroParticlesPerState):
    """create equal weight macro particles from given initial state veector
    """
    macroParticleType = np.dtype([('weight', 'f4'), ('atomicState', 'u1'), ('timeRemaining', 'f8')])

    # create macro particles from initial state distribution
    initMacroParticles = []
    for state, density in enumerate(initStateVector):
        # only >0 densities
        if density > 0:
            # for each spawn numMacroParticlesPerState new macroParticles
            for i in range(numMacroParticlesPerState[state]):
                #                (weight = total density/num, state, timeRemaining)
                initMacroParticles.append( (density/numMacroParticlesPerState[state], state, 0.) )
    return np.array(initMacroParticles, dtype=macroParticleType)

def sampleSolver(initMacroParticles, numStates, rateMatrix,
    timeStepLength, numTimeSteps, approximationMethod,
    numSolverSamples, consoleOutputIntervalSample, consoleOutputIntervalSolver):

    bins = range(numStates+1)
    initMacroParticleHistogram, binEdges = np.histogram(initMacroParticles['atomicState'], bins=bins, weights=initMacroParticles['weight'])

    print("start solver")
    samples = []
    # solver samples, linear
    for n in range(numSolverSamples):
        macroParticles = np.copy(initMacroParticles)

        if ((n+1) % consoleOutputIntervalSample == 0):
            print("sample: " + str(n+1))
            consoleOutputInterval = consoleOutputIntervalSolver
        else:
            consoleOutputInterval = None

        macroParticleHistograms = solve(
            macroParticles, rateMatrix, timeStepLength, numTimeSteps, numStates,
            approximationMethods[approximationMethod],
            consoleOutputInterval=consoleOutputInterval)

        samples.append(macroParticleHistograms)

    mean = np.empty((numTimeSteps+1, numStates))
    mean[1:] = np.mean(samples, axis = 0)
    mean[0] = initMacroParticleHistogram
    stdDev = np.empty((numTimeSteps+1, numStates))
    stdDev[1:] = np.std(samples, axis = 0)
    stdDev[0] = np.zeros(numStates)
    del samples

    return mean, stdDev

def plotSolverAnalyticComparison(startTime, endTime, timeStepLength, numTimeSteps,
    numStates, rateMatrix, times, timeDependentDensities, steadyStateDensities, mean, stdDev,
    approximationMethod, numSolverSamples, numMacroParticlesPerState, L_fileName, plotSolverResults=False):
        # plotting results
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    ax.set_title("picongpu solver for constant rate matrix")
    ax.set_xlim((startTime, startTime + timeStepLength * (numTimeSteps) + timeStepLength * (endTime - startTime)/30))
    ax.set_ylim((0,1.1))
    ax.set_xlabel(r"time [$\Delta{}t_{\mathrm{PIC}}$]")
    ax.set_ylabel("relative density of atomic states, additive")

    ml = tick.MultipleLocator(1)
    ax.xaxis.set_minor_locator(ml)

    # define colors of states
    colormap = plt.cm.tab10
    numColorsInColorMap = 10

    colors = iter([colormap(i) for i in range(numColorsInColorMap)])

    colorStates = {}
    for i in range(numStates):
        try:
            colorStates[i] = next(colors)
        except StopIteration:
            colors = iter([colormap(i) for i in range(numColorsInColorMap)])
            colorStates[state] = next(colors)

    # plotting analytic time dependent
    offset = 0
    for state in range(numStates):
        if (plotSolverResults):
            ax.plot(times, timeDependentDensities[:,state]+offset, color=colorStates[state],
                label=r"state " + str(state+1) +", analytically/solver mean $\pm$ std Dev", linewidth=2)
        else:
            ax.plot(times, timeDependentDensities[:,state]+offset, color=colorStates[state],
                label=r"state " + str(state+1) +", analytically", linewidth=2)
        offset += timeDependentDensities[:,state]

    # plot steady state
    offset = 0
    for state in range(numStates):
        ax.bar( (startTime + timeStepLength * (numTimeSteps)), steadyStateDensities[state],
            width=timeStepLength * (endTime - startTime)/30, bottom=offset, align='edge', color=colorStates[state],
            label="state "+ str(state+1) + ", steady state analytically")
        offset += steadyStateDensities[state]

    if (plotSolverResults):
        # plot solver
        timeSteps = np.arange(numTimeSteps+1) * timeStepLength + startTime
        offset = 0
        for state in range(numStates):
            # plot mean value
            ax.plot(timeSteps, mean[:, state] + offset, drawstyle='steps-mid', color=colorStates[state],
                #label="state " + str(state) + ", solver",
                linewidth=1)
            offset += mean[:,state]
            # plot standard deviation
            ax.bar( timeSteps, 2 * stdDev[:, state], width=timeStepLength, bottom = offset - stdDev[:, state],
                align='center', color=colorStates[state], alpha=0.5)

    #create legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize='small')
    if plotSolverResults:
        text = ax.text(1.05, 0.1, "Approximation type: " + approximationMethod + "\n"
            + r"rate matrix $\left[\frac{1}{\Delta{}t_{\mathrm{PIC}}}\right]$" + ":\n"
            + np.array2string(rateMatrix, formatter={'float_kind':lambda x: "% .3f" % x}) + "\n\n"
            + r"time step length solver[$\Delta{}t_{\mathrm{PIC}}$]: " + str(timeStepLength) + "\n"
            + "number of macro particles: " + str(numMacroParticlesPerState[0]) + "\n"
            + "number of solver runs: " + str(numSolverSamples),
            transform=ax.transAxes)
    else:
        text = ax.text(1.05, 0.1,
            r"rate matrix $\left[\frac{1}{\Delta{}t_{\mathrm{PIC}}}\right]$" + ":\n"
            + np.array2string(rateMatrix, formatter={'float_kind':lambda x: "% .3f" % x}),
            transform=ax.transAxes)

    fig.savefig("testSolver_"+L_fileName, bbox_extra_artists=(lgd,text), bbox_inches='tight')
    plt.close()

#def createSampleRateMatrix(numStates, rateMatricesEntries):
#    """creates several rate matrices with every combination of the rateMatricesEntries
#    """
#    rateMatrices = np.empty(((numStates**2 - numStates) * len(rateMatricesEntries), numStates, numStates))


if __name__ == "__main__" :
    # defining rateMatrix
    # number of states
    numStates = 4

    rateMatrix = np.zeros((numStates, numStates))
    # set non diagonal rate matrix entries
    # R_j,i = R(i -> j), in units of 1/t_PIC, must be >= 0
    rateMatrix[1,0] = 0.005         # R_2,1 = R_1 -> 2
    rateMatrix[2,0] = 0.004         # R_3,1 = R_1 -> 3
    rateMatrix[3,0] = 0.080         # R_4,1 = R_1 -> 4
    rateMatrix[0,1] = 0.010         # R_1,2 = R_2 -> 1
    rateMatrix[2,1] = 0.006         # R_3,2 = R_2 -> 3
    rateMatrix[3,1] = 0.010         # R_4,2 = R_2 -> 4
    rateMatrix[0,2] = 0.003         # R_1,3 = R_3 -> 1
    rateMatrix[1,2] = 0.002         # R_2,3 = R_3 -> 2
    rateMatrix[3,2] = 0.030         # R_4,3 = R_3 -> 4
    rateMatrix[0,3] = 0.020         # R_1,4 = R_4 -> 1
    rateMatrix[1,3] = 0.002         # R_2,4 = R_4 -> 2
    rateMatrix[2,3] = 0.070         # R_3,4 = R_4 -> 3

    # initial state vector, BEWARE: sum(n_i) == 1
    initStateVector = np.zeros(numStates)
    initStateVector[0] = 1.

    # analytic parameters
    numDrawPoints = 1000
    startTime = 0 #unit: t_PIC
    endTime = 50 #unit: t_PIC

    # solver parameters
    numMacroParticles = 200
    timeStepLength = 1. #unit: t_PIC
    approximationMethod = 'exponential'

    # solver variation
    numSolverSamples = 200

    #output
    filename = "test_4level"

    # creature comforts
    consoleOutputIntervalSample = 10
    consoleOutputIntervalSolver = 30

    # fill diagonal entries
    fillDiagonalEntries(rateMatrix, numStates)

    print("outputFile: " + filename)
    print("rateMatrix:")
    print(rateMatrix)

    # number of macro particles to create for each atomic state
    numMacroParticlesPerState = np.full(numStates, numMacroParticles) 
    numTimeSteps = int((endTime - startTime) / timeStepLength)

    # steady state solution
    steadyStateDensities = steadyStateSolution(rateMatrix)

    # time dependent analytic solution
    times, timeDependentDensities = analyticTimeDependentSolution(initStateVector, rateMatrix, numDrawPoints,
        startTime, endTime, numStates)

    # solver
    initMacroParticles = createEqualWeightMacroParticles(initStateVector, numMacroParticlesPerState)

    mean, stdDev = sampleSolver(initMacroParticles, numStates, rateMatrix,
        timeStepLength, numTimeSteps, approximationMethod,
        numSolverSamples, consoleOutputIntervalSample, consoleOutputIntervalSolver)

    # plotting results
    plotSolverAnalyticComparison(startTime, endTime, timeStepLength, numTimeSteps,
        numStates, rateMatrix, times, timeDependentDensities, steadyStateDensities, mean, stdDev,
        approximationMethod, numSolverSamples, numMacroParticlesPerState, filename+"_onlyAnalytic", False)
    plotSolverAnalyticComparison(startTime, endTime, timeStepLength, numTimeSteps,
        numStates, rateMatrix, times, timeDependentDensities, steadyStateDensities, mean, stdDev,
        approximationMethod, numSolverSamples, numMacroParticlesPerState, filename, True)

    print("outputFile: " + filename)