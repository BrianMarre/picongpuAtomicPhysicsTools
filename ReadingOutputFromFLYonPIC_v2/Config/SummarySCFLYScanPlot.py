import pydantic

class PlotConfig(pydantic.BaseModel):
    """plotting config for summary plot"""
    loadRawEachSCLFYSim : bool
    """switch for loading each SCFLY simulation results from raw data"""
    loadRawSummaryData : bool
    """switch for loading summary plot data results from raw data"""
    dataSetName : str
    """name to use for data set"""
