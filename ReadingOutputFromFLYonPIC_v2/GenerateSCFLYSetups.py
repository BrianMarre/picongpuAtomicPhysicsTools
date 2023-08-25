import typing
import pydantic

import numpy as np
import ConfigNumberConversion as conv
import os
import shutil
import subprocess
import time

def readSCFLYNames(fileName, Z, numLevels):
    """read SCFly atomic state names from file and convert them to FLYonPIC atomicConfigNumbers

    @attention states must be uniquely described by their shell occupation number vector!

    @param fileName path of file describing SCFLY atomic states,
        file must give in each line name of one state and occupation number for each shell for state

        Exp.: h_10001 1 0 0 0 0 0 0 0 0 0\n ...
    @param Z atomic number of the element for which atomic states are contained
    @param numLevels number of shells included in the data set

     @returns dict of {SCFLYname : FLYonPIC_atomicConfigNumber}
    """

    # prepare result datatype description
    occupationNumberCollums = []
    ids = []
    for i in range(numLevels):
        occupationNumberCollums.append((str(i), 'u1'))
        ids.append(str(i))

    dtype = [("state", 'U10')]
    dtype.extend(occupationNumberCollums)

    # load file content
    conversionTable = np.loadtxt(fileName, dtype=dtype)

    # create translation dictionary
    configNumbers = []
    for levelVector in conversionTable[ids]:
         configNumbers.append(conv.getConfigNumber(levelVector, Z))

    atomicConfigNumber_to_StateName = dict(zip(configNumbers, conversionTable["state"]))

    # add completely ionized state by hand
    atomicConfigNumber_to_StateName[0] = str(Z)+"+"

    return atomicConfigNumber_to_StateName

class BaseConfig_SCFLY_TimeDependent(pydantic.BaseModel):
    # Z of element
    atomicNumber : int
    # time points for input and output for/from SCFLY
    timePoints : list[float]
    # constant number density of ions, 1/cm^3
    ionDensity : list[float]
    # electron temperature, eV
    electronTemperature : list[float]
    # list of atomicStates, by SCFLY Name, and starting fraction for each state
    initialCondition : list[tuple[str, float]]
    # minimum charge state to include in calculation
    minChargeState : int
    # maximum charge state to include in calculation
    maxChargeState : int
    # number of atomic shells to include in calculation
    numLevels : int
    # path to SCFLY atomicData input file
    atomicDataInputFile : str
    # output file name, written by SCFLY, @attention will overwrite existing files
    outputFileName : str
    # generated setup
    generated : bool = False
    # basePath for scfly setup, path to folder containing scfly setups
    basePath : str
    # name of folder containing the generated SCFLY setup
    folderName : str

    def get(self):
        return self

    def generateSCFLYSetup(self):
        # create folder
        os.makedirs(self.basePath + self.folderName)

        # copy atomicData input file to execution folder
        shutil.copyfile(self.atomicDataInputFile, self.basePath + self.folderName + "/" + "atomic.inp."+ str(self.atomicNumber))

        # create history file
        historyData = np.empty((len(self.timePoints), 3))
        historyData[:,0] = self.timePoints
        historyData[:,1] = self.ionDensity
        historyData[:,2] = self.electronTemperature
        np.savetxt(self.basePath + self.folderName + "/iondensity", historyData, fmt = '%.6E', header = "time nt te", comments='')

        # create initial state file
        with open(self.basePath + self.folderName + "/initialstate", "w") as initialstate:
            for entry in self.initialCondition:
                initialstate.write(entry[0] + " {0:.6E}\n".format(entry[1]))


        # create runfile
        with open(self.basePath + self.folderName + "/runfile.txt", "w") as runfile:
            runfile.write(
                "z " + str(self.atomicNumber)
                + " 0\n" + "evolve td\n"
                + "initial file initialstate\n"
                + "history iondensity nt\n"
                + "outfile " + self.outputFileName + "\n"
                + "isos " + str(self.minChargeState) + " " + str(self.maxChargeState) + " " + str(self.numLevels) + "\n"
                + "end\n")

        # mark as generated
        self.generated = True
        return self

    def execute(self, SCFLYBinaryPath: str):
        assert(self.generated), "setup must be generated before executing"

        print(["cd "+ self.basePath + self.folderName, "&&", SCFLYBinaryPath, "runfile.txt",
                        "&&", "cd /home/marre55/scflyTools"])
        subprocess.run([SCFLYBinaryPath, "runfile.txt"], cwd=(self.basePath+self.folderName))

class Config_SCFLY_FLYonPICComparison(pydantic.BaseModel):
    # Z of element
    atomicNumber : int
    # constant electron temperature, eV
    electronTemperature : float
    # constant number density of ions, 1/cm^3
    ionDensity : float
    # time points for input and output for/from SCFLY, in s
    timePoints : list[float]
    # initial state occupation number level vector, (n_1, n_2, ...)
    initialStateLevelVector : tuple[int, ...]
    # path to atomciStateNaming.input file, contains for each SCFLY state its corresponding occupation number vector
    SCFLYatomicStateNamingFile : str
    # path to SCFLY atomicData input file
    atomicDataInputFile : str
    # output file fileName
    outputFileName : str
    # basePath for scfly setup, path to folder containing scfly setups
    basePath : str
    # name of folder containing the generated SCFLY setup
    folderName : str

    def get(self) -> BaseConfig_SCFLY_TimeDependent:
        # determine  number of levels to track form input
        numLevels = len(self.initialStateLevelVector)

        # get translation from FLYonPIC to SCFLY
        atomicConfigNumber_to_StateName = readSCFLYNames(self.SCFLYatomicStateNamingFile, self.atomicNumber, numLevels)

        return BaseConfig_SCFLY_TimeDependent(
            atomicNumber = self.atomicNumber,
            timePoints = self.timePoints,
            ionDensity = np.full_like(self.timePoints, self.ionDensity),
            electronTemperature = np.full_like(self.timePoints, self.electronTemperature),
            initialCondition = [(atomicConfigNumber_to_StateName[conv.getConfigNumber(self.initialStateLevelVector, self.atomicNumber)], 1.)],
            minChargeState = 0,
            maxChargeState = self.atomicNumber,
            numLevels = numLevels,
            atomicDataInputFile = self.atomicDataInputFile,
            outputFileName = self.outputFileName,
            basePath = self.basePath,
            folderName = self.folderName)

if __name__ == "__main__":
    comparisonFLYonPIC_Ar = Config_SCFLY_FLYonPICComparison(
        atomicNumber = 18,
        electronTemperature = 1e3, # eV
        ionDensity = 1e22, # 1/cm^3
        timePoints = np.arange(101) * 3.3e-17, # s
        initialStateLevelVector = (2, 8, 6, 0, 0, 0, 0, 0, 0, 0),
        SCFLYatomicStateNamingFile="/home/marre55/scflyInput/18_atomicStateNaming.input",
        atomicDataInputFile="/home/marre55/scflyInput/atomic.inp.18",
        outputFileName = "xout",
        basePath = "/home/marre55/scflyTools/",
        folderName = "test")

    generatedSetup = comparisonFLYonPIC_Ar.get().generateSCFLYSetup()
    generatedSetup.execute("/home/marre55/scfly/code/exe/scfly")
