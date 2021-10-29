import openpmd_api as io
import numpy as np
import math

filename = "/home/marre55/picongpuOutput/testMacroCount/simOutput/openPMD/simOutput_%T.bp"
speciesName = "C"

series = io.Series(filename, io.Access.read_only)

totWeightingNumpy = {}
totWeightingFsum = {}
totWeightingDumb = {}
totWeightingSorted = {}

for iteration in series.iterations:
    step = series.iterations[iteration]

    weighting = step.particles[speciesName]["weighting"][io.Mesh_Record_Component.SCALAR]
    dataWeightings = weighting.load_chunk()
    series.flush()

    totWeightingNumpy[iteration] = np.sum(dataWeightings)
    totWeightingFsum[iteration] = math.fsum(dataWeightings)

    sum = 0
    for weight in sorted(dataWeightings):
        sum += weight

    totWeightingSorted[iteration] = sum

    sum = 0
    for weight in dataWeightings:
        sum += weight

    totWeightingDumb[iteration] = sum

print(totWeightingNumpy)
print(totWeightingFsum)
print(totWeightingSorted)
print(totWeightingDumb)