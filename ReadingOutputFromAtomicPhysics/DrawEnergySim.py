import numpy as np
import matplotlib.pyplot as plt

def drawEnergyOverTime(speciesNames, simName, title, pictureFilename, outputPath):
    # load data
    filepath = outputPath + "/" + simName + "/simOutput/"
    dataParticles = {}

    dtype = np.dtype([("step", "u4"), ("E_kin_Joule", "f8"), ("E_Joule", "f8")])

    for entry in speciesNames:
        dataParticles[entry]=np.loadtxt(filepath + entry + "_energy_all.dat", dtype=dtype)

    dtype = np.dtype([("step", "u4"), ("total", "f8"), ("B_x", "f8"), ("B_y", "f8"), ("B_z", "f8"), ("E_x", "f8"), ("E_y", "f8"), ("E_z", "f8")])
    dataFields = np.loadtxt(filepath + "fields_energy.dat", dtype=dtype)

    totalEnergy = np.copy(dataFields["total"])
    for species in dataParticles.keys():
        totalEnergy += dataParticles[species]["E_kin_Joule"]


    # actual plotting
    fig = plt.figure(dpi=200)

    # plot total field energy
    plt.plot(dataFields["step"], dataFields["total"], label="FieldEnergy")

    # plot particle energy
    for species in dataParticles.keys():
        plt.plot(dataParticles[species]["step"], dataParticles[species]["E_kin_Joule"], label=species)

    # plot total energy
    plt.plot(dataFields["step"], totalEnergy, label = "total energy")

    plt.xlabel("iteration")
    plt.ylabel("Energy [J]")
    plt.ylim((-0.1e-10,4e-10))
    plt.title("energy of simulation, " + title)
    plt.legend()
    plt.savefig(pictureFileName)
    plt.close()


simName = "testSolver13"
speciesNames = ["Cu", "eth"]

outputPath = "/home/marre55/picongpuOutput"
title = "80ppc"
pictureFileName = title + "Energy"


drawEnergyOverTime(speciesNames, simName, title, pictureFileName, outputPath)
