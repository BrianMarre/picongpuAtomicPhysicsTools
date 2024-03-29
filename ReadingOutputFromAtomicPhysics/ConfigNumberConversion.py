import numpy as np

def g(n):
    return 2*n**2

def numberOfOccupationNumbersInShell(n, atomicNumber):
    return min(g(n), atomicNumber)+1

def stepLength(n, atomicNumber):
     stepLength = 1
     for i in range(1,n):
        stepLength *= numberOfOccupationNumbersInShell(i, atomicNumber)

     return stepLength

def getLevelVector(x, atomicNumber, numLevels):
    levelVector = np.empty(numLevels)

    for n in range(numLevels, 0, -1):
        # calculate current stepLength
        currentStepLength = stepLength(n, atomicNumber)

        # get occupation number
        levelVector[n-1] = x // currentStepLength

        # remove contribution of current level
        x -= currentStepLength * levelVector[n - 1]

    return levelVector

# tests
def getConfigNumber(levelVector, atomicNumber):
    configNumber = 0
    for i in range(0,len(levelVector)):
        configNumber += stepLength(i+1, atomicNumber) * levelVector[i]
    return configNumber

def testGetLevelVector():
    assert(getLevelVector(11, 6, 10) == np.array((2,3,0,0,0,0,0,0,0,0))), "getLevelVector test failed"

def testGetConfigNumber():
    assert(getConfigNumber(np.array((2,1,1,0,0,0,0,1,0,0)), 18) == 352973), "getConfigNumber test failed"
    assert(getConfigNumber(np.array((2,3,0,0,0,0,0,0,0,0)), 18) == 11), "getConfigNumber test failed"
    assert(getConfigNumber(np.array((0,1,0,0,0,0,0,0,0,1)), 18) == 24134536956), "getConfigNumber test failed"

def testAll():
    testGetConfigNumber()
    testGetLevelVector()
