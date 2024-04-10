import typeguard

@typeguard.typechecked
class TimingDataPlot:
    def __init__(
            self,
            caseBasePaths : list[str],
            caseFileNames : list[list[str]],
            caseDataNames : list[str],
            fileName : str,
            plotCleanUpAndInit : bool):
        self.caseBasePaths = caseBasePaths
        self.caseFileNames = caseFileNames
        self.caseDataNames = caseDataNames
        self.fileName = fileName
        self.plotCleanUpAndInit = plotCleanUpAndInit
