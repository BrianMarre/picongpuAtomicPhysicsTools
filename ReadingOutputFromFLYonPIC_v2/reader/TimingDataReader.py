"""
This file is part of the FLYonPIC_Eval.
Copyright 2024 FLYonPIC_Eval contributors

Distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Authors: Brian Edward Marre
License: GPLv3+
"""

from . import Reader

import numpy as np
import numpy.typing as npt

import pydantic
import typeguard

@typeguard.typechecked
class TimingDataReader(Reader):
    """read timing data from stdout file of PIConGPU run"""

    # picongpu stdout output file name, may be path absolute or relative to execution location
    outputFileName : str

    def read(self) -> npt.NDArray[np.dtype([('step', 'i4'), ('time_total[msec]', 'u8'), ('time_step[msec]', 'u8')])]:
        """
        @returns np.NDArray(numbertimeSteps+2) = [initTime , time for steps, cleanUpTime]
        """

        # time steps and calculation runtime
        regex_timeStep = r"\s{0,2}\d+ % =\s+(\d+) \| time elapsed:\s*(\d+h)?\s*(\d+min)?\s*(\d+sec)?\s+(\d+msec) \| avg time per step:\s*(\d+h)?\s*(\d+min)?\s*(\d+sec)?\s+(\d+msec)\n"
        result_template_timeStep = [('step', 'u4'), ('total_h', 'U10'), ('total_min', 'U10'), ('total_sec', 'U10'), ('total_msec', 'U10'), ('step_h', 'U10'), ('step_min', 'U10'), ('step_sec', 'U10'), ('step_msec', 'U10')]

        ## read in data from file
        output = np.fromregex(self.outputFileName, regex_timeStep, result_template_timeStep)

        numberSteps = np.shape(output)[0] - 1

        ## remove units
        timeUnits = ['h', 'min', 'sec', 'msec']
        for timeUnit in timeUnits:
            output['total_' + timeUnit] = np.char.rstrip(output['total_' + timeUnit], timeUnit)
            output['step_' + timeUnit] = np.char.rstrip(output['step_' + timeUnit], timeUnit)

        ## unify h, minute, seconds and millisecond columns to single time column
        result = np.empty(len(output)+2, dtype = [('step', 'i4'), ('time_total[msec]', 'u8'), ('time_step[msec]', 'u8')])

        result['step'][1:-1] = output['step']
        result['time_total[msec]'][1:-1] = np.fromiter(map(lambda entry : (0 if entry[1] == '' else int(entry[1])) * 60 * 60 * 1000
                                                    + (0 if entry[2] == '' else int(entry[2])) * 60 * 1000
                                                    + (0 if entry[3] == '' else int(entry[3])) * 1000
                                                    + (0 if entry[4] == '' else int(entry[4])),
                                                    output), dtype='u8')
        result['time_step[msec]'][1:-1] = np.fromiter(map(lambda entry : (0 if entry[5] == '' else int(entry[5])) * 60 * 60 * 1000
                                                    + (0 if entry[6] == '' else int(entry[6])) * 60 * 1000
                                                    + (0 if entry[7] == '' else int(entry[7])) * 1000
                                                    + (0 if entry[8] == '' else int(entry[8])),
                                                    output), dtype='u8')
        del output

        # initTime
        regex_initTime = r"initialization time:\s*(\d+h)?\s*(\d+min)?\s*(\d+sec)?\s+(\d+msec) = (\d+.\d+) sec\n"
        result_template_initTime = [('h', 'U10'), ('min', 'U10'), ('sec', 'U10'), ('msec', 'U10'), ('inSeconds', 'f8')]

        output = np.fromregex(self.outputFileName, regex_initTime, result_template_initTime)

        timeUnits = ['h', 'min', 'sec', 'msec']
        for timeUnit in timeUnits:
            output[timeUnit] = np.char.rstrip(output[timeUnit], timeUnit)

        initTime = ((0 if output[0][0] == '' else int(output[0][0])) * 60 * 60 * 1000
                    + (0 if output[0][1] == '' else int(output[0][1])) * 60 * 1000
                    + (0 if output[0][2] == '' else int(output[0][2])) * 1000
                    + (0 if output[0][3] == '' else int(output[0][3]))) # millisecond

        del output

        ## store init time as first entry
        result[0][0] = -1
        result[0][1] = 0
        result[0][2] = initTime

        #full simulation Time
        regex_fullSimulationTime = r"full simulation time:\s*(\d+h)?\s*(\d+min)?\s*(\d+sec)?\s+(\d+msec) = (\d+.\d+) sec\n"
        result_template_fullSimulationTime = [('h', 'U10'), ('min', 'U10'), ('sec', 'U10'), ('msec', 'U10'), ('inSeconds', 'f8')]

        output = np.fromregex(self.outputFileName, regex_fullSimulationTime, result_template_fullSimulationTime)

        timeUnits = ['h', 'min', 'sec', 'msec']
        for timeUnit in timeUnits:
            output[timeUnit] = np.char.rstrip(output[timeUnit], timeUnit)

        try:
            fullSimulationTime = ((0 if output[0][0] == '' else int(output[0][0])) * 60 * 60 * 1000
                                    + (0 if output[0][1] == '' else int(output[0][1])) * 60 * 1000
                                    + (0 if output[0][2] == '' else int(output[0][2])) * 1000
                                    + (0 if output[0][3] == '' else int(output[0][3]))) # millisecond
        except IndexError:
            # if not in file set to last time step
            fullSimulationTime = result[-2][1]

        ## calculate and store cleanup time as last entry
        result[-1][0] = numberSteps + 1
        result[-1][1] = fullSimulationTime
        result[-1][2] = fullSimulationTime - result['time_total[msec]'][-2]

        del output

        return result