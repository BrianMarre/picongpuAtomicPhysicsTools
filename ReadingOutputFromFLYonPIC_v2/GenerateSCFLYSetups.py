import typing
import pydantic

import numpy as np
import ConfigNumberConversion as conv
import os
import shutil
import subprocess

""" @file generates and runs SCFLY simulations

    Basic Workflow:
    1.) create Config instance for SCFLY simulation
    2.) call get()method on it to convert to a BaseConfig instance for SCFLY
    3.) call generateSCFLYSetup() on BaseConfig instance to generate a SCFLY setup
    4.) call execute() on previously generated BaseConfig to run SCFLY simulation setup

    see `if __name__=="__main__":` for example
"""

class BaseConfig_SCFLY_TimeDependent(pydantic.BaseModel):
    """BaseConfig for creating optical thin, no radiation temperature, time dependent SCFLY simulations"""
    # Z of element
    atomicNumber : int
    # time points for input and output for/from SCFLY
    timePoints : list[float]
    # number density of ions over time, 1/cm^3
    ionDensity : list[float]
    # electron temperature over time, eV
    electronTemperature : list[float]
    # list of atomicStates, by SCFLY Name, and starting fraction for each state
    initialCondition : list[tuple[str, float]]
    # minimum charge state to include in calculation
    minChargeState : int
    # maximum charge state to include in calculation
    maxChargeState : int
    # number of atomic shells to include in calculation, <=10
    numLevels : int
    # path to SCFLY atomicData input file, will be copied to SCFLY setup directory
    atomicDataInputFile : str
    # output file name, written by SCFLY, @attention will overwrite existing files
    outputFileName : str
    # whether setup has been generated already
    _generated : bool = False
    # basePath for scfly setup, path to folder containing scfly setups
    basePath : str
    # name of folder containing the generated SCFLY setup
    folderName : str

    def get(self):
        """convert Config to BaseConfig"""
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
                "z " + str(self.atomicNumber) + " 0\n"
                + "evolve td\n"
                + "initial file initialstate\n"
                + "history iondensity nt\n"
                + "outfile " + self.outputFileName + "\n"
                + "isos " + str(self.minChargeState) + " " + str(self.maxChargeState) + " " + str(self.numLevels) + "\n"
                + "end\n")

        # mark as generated
        self._generated = True
        return self

    def execute(self, SCFLYBinaryPath: str):
        """run scfly"""
        # check setup previously generated
        assert(self._generated), "setup must be generated before executing"

        # print location to console
        print(["cd "+ self.basePath + self.folderName, "&&", SCFLYBinaryPath, "runfile.txt",
                        "&&", "cd /home/marre55/scflyTools"])

        subprocess.run([SCFLYBinaryPath, "runfile.txt"], cwd=(self.basePath+self.folderName))

class Config_SCFLY_FLYonPICComparison(pydantic.BaseModel):
    """configuration for SCFLY simulations for comparison to FLYonPIC simulations"""
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
        """convert Config to BaseConfig"""
        # determine  number of levels to track form input
        numLevels = len(self.initialStateLevelVector)

        # get translation from FLYonPIC to SCFLY
        , atomicConfigNumber_to_StateName = readSCFLYNames(self.SCFLYatomicStateNamingFile, self.atomicNumber, numLevels)

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
    # example of use

    # create config object
    comparisonFLYonPIC_Ar = Config_SCFLY_FLYonPICComparison(
        atomicNumber = 18,
        electronTemperature = 1e3, # eV
        ionDensity = 1e22, # 1/cm^3
        timePoints = np.arange(101) * 3.3e-17, # s
        initialStateLevelVector = (2, 8, 6, 0, 0, 0, 0, 0, 0, 0),
        SCFLYatomicStateNamingFile="/home/marre55/scflyInput/18_atomicStateNaming.input",
        atomicDataInputFile="/home/marre55/scfly/atomicData/FLYCHK_input_files/atomic.inp.18",
        outputFileName = "xout",
        basePath = "/home/marre55/scflyInput",
        folderName = "test")

    # convert ConfigObject to BaseConfig and generate SCFLY setup from BaseConfig
    generatedSetup = comparisonFLYonPIC_Ar.get().generateSCFLYSetup()

    # run SCFLY for setup
    generatedSetup.execute("/home/marre55/scfly/code/exe/scfly")
