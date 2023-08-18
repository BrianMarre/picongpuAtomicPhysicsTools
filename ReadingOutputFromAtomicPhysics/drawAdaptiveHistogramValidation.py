import numpy as np
import openpmd_api as io
import matplotlib.pyplot as plt
import AdaptiveHistogram_Reader as reader

filepath = "/home/marre55/picongpuOutput/testAdaptiveHistogramOutput/simOutput/C_adaptiveHistogramPerSuperCell/"
filename = filepath + "adaptiveHistogramPerSuperCell_%T.bp"

particleOutputFilename = "/home/marre55/picongpuOutput/testAdaptiveHistogramOutput/simOutput/openPMD/simOutput_%T.bp"
electronSpeciesName = 'eth'

# read adaptiveHistogramOutput from file
timeSteps, histograms, argumentUNIT = reader.readAccumulatedAdaptiveHistogram(filename)
# open particle data output
seriesParticleOutput = io.Series(particleOutputFilename, io.Access.read_only)

# adaptive histogram output
for j in range(len(timeSteps)):
    iteration = timeSteps[j] # iteration index of current entry
    histogram = histograms[j] # histogram of iteration index

    leftBoundaries = np.empty(len(histogram)) # number of entries on dictionary
    weights = np.empty(len(histogram))
    widths = np.empty(len(histogram))

    # transcribe to numpy 1d-arrays
    i = 0
    for k,v in histogram.items():
        leftBoundaries[i] = k
        weights[i] = v[0]
        widths[i] = v[1]
        i+= 1

    fig = plt.figure(dpi=400)
    plt.title("adaptiveHistogram, iteration" + str(iteration))
    plt.xlabel("physical kinetic energy electrons[Ry]")
    plt.ylabel("macro particle weight / binWidth")

    # plot adaptive histogram of current time step
    plt.bar(
        leftBoundaries,
        weights/widths,
        widths,
        align = 'edge')


    # plot binned particle output with same bins
    # get corresponding particle data
    step = seriesParticleOutput.iterations[iteration]
    electronSpecies = step.particles[electronSpeciesName]

    # prepare buffers
    momentum_x = electronSpecies["momentum"]['x']
    momentum_y = electronSpecies["momentum"]['y']
    momentum_z = electronSpecies["momentum"]['z']
    weighting = electronSpecies["weighting"][io.Mesh_Record_Component.SCALAR]

    dataWeighting = weighting.load_chunk()
    dataMomentum_x = momentum_x.load_chunk()
    dataMomentum_y = momentum_y.load_chunk()
    dataMomentum_z = momentum_z.load_chunk()

    # flush data to prepared buffers
    seriesParticleOutput.flush()

    # convert to SI units
    momentumUnit_x_SI = momentum_x.unit_SI
    momentumUnit_y_SI = momentum_y.unit_SI
    momentumUnit_z_SI = momentum_z.unit_SI
    weightingUnit_SI = weighting.unit_SI

    dataWeighting = dataWeighting * weightingUnit_SI

    # remove scaling of momentum
    dataMomentum_x = dataMomentum_x / dataWeighting
    dataMomentum_y = dataMomentum_y / dataWeighting
    dataMomentum_z = dataMomentum_z / dataWeighting

    # physical constants
    c = 299792458 #speed of light, m/s
    m_e = 9.1093837015e-31 #electron mass, kg

    # debug only
    electronVolt = 1.602176634e-19 #J/eV

    # bin particles
    # E_kin [eV]= (sqrt(p^2*c^2 + m^2*c^4) - m*c^2) / J/eV
    p2 = np.empty(len(dataMomentum_x), dtype=np.double)
    for i in range(len(dataMomentum_x)):
        p2[i] = (
            (np.double(dataMomentum_x[i]) * momentumUnit_x_SI)**2
            + (np.double(dataMomentum_y[i]) * momentumUnit_y_SI)**2
            + (np.double(dataMomentum_z[i]) * momentumUnit_z_SI)**2) # impulse norm squared, (kg * m/s)^2

    E_kin = (np.sqrt( m_e**2 * c**4 + p2 * c**2 ) - (m_e * c**2) ) / argumentUNIT # kinetic energy, AU
    # (sqrt(kg^2 * m^4/s^4) = kg * m^2/s^2 = Nm = J )/ (J/AU) = AU
    # checked m_e * c^2 ~ 511keV

    binBoundaries = sorted(np.unique(np.append(leftBoundaries, leftBoundaries + widths)))
    binCenters = (np.array(binBoundaries[1:]) + np.array(binBoundaries[:-1]))/2.
    binWidths = np.array(binBoundaries[1:]) - np.array(binBoundaries[:-1])

    # bin particles, with same bins
    hist, bins = np.histogram(E_kin, bins=binBoundaries, weights=dataWeighting) # same bins
    # plot binned particles as comparison
    plt.plot( binCenters, hist/binWidths, 'xr' )
    plt.savefig( str(iteration) + "_adaptiveHistogramOutput.png" )

    # free binning
    histFree, binsFree = np.histogram(E_kin, bins=8, weights=dataWeighting) # free binning
    centerBinsFree = binsFree/2 + np.roll(binsFree, -1, 0)/2
    centerBinsFree = centerBinsFree[:-1]
    widthBinsFree = binsFree[1:] - binsFree[:-1]

    fig = plt.figure(dpi=400)
    plt.title("histogram of electrons, iteration " + str(iteration))
    plt.xlabel("physical kinetic energy electrons[Ry]")
    plt.ylabel("macro particle weight / widthBin")

    # plot
    plt.plot( centerBinsFree, histFree/widthBinsFree, 'x-r' )
    plt.savefig(str(iteration) + "_adaptiveHistogramOutput_freeBinning.png")

print("Plot adaptive histogram completed")
del timeSteps
del histograms
