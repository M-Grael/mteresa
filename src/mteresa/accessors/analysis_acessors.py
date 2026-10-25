import pandas as pd
from mteresa.analysis.peak_deviation_analysis import peak_deviation_analysis as _peak_deviation_analysis

@pd.api.extensions.register_dataframe_accessor("mteresa")
class MTeresaAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def peak_deviation_analysis(self,
        timestamp_column: str = 'timestamp',
        burst_cutoff: float = 6.0,
        min_irts: int = 25,
        grouping_index_column: str = 'session_id',
        events_column: str = 'event_alias',
        filter_events: str | list[str] | None = None,
        transform: bool = None):
        return _peak_deviation_analysis(self._obj, timestamp_column=timestamp_column, burst_cutoff=burst_cutoff, 
                                        min_irts=min_irts, grouping_index_column=grouping_index_column, events_column=events_column,
                                        filter_events=filter_events, 
                                        transform=transform)