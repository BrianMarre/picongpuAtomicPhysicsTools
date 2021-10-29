import openpmd_api as io
import numpy as np

filename = "/home/marre55/picongpuOutput/verificationArgon/simOutput/openPMD/simOutput_%T.bp"

series = io.Series(filename, io.Access.read_only)

#particle data available output, debug code
step = series.iterations[0]
for particleSpecies in step.particles:
    print("\t {0}".format(particleSpecies))
    print("With records:")
    for record in step.particles[particleSpecies]:
        print("\t {0}".format(record))

