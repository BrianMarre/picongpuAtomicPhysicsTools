import pydantic

class PlotConfig(pydantic.BaseModel):
    """plotting config for summary plot"""
    loadRawEachSCLFYSim : bool
    """switch for loading each SCFLY simulation results from raw data"""
    loadRawSummaryData : bool
    """switch for loading summary plot data results from raw data"""
    additionalDataName : str
    """name specialization for different zeroCutoffLimits"""
    seriesName : str
    """general name for set of scan"""
