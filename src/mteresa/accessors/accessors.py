import pandas as pd
from mteresa.utils.validation import validate_df as _validate_df
from mteresa.utils.data_handling import filter_events as _filter_events
from mteresa.utils.data_handling import summary_events as _summary_events
from mteresa.analysis.peak_deviation_analysis import peak_deviation_analysis as _peak_deviation_analysis

@pd.api.extensions.register_dataframe_accessor("mteresa")
class MTeresaAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def validate_df(self, schema=None) -> pd.DataFrame:
        """Validate the current Dataframe using mteresa rules."""
        _validate_df(self._obj, schema=schema)
        return self._obj

    def filter_events(
        self,
        events_to_keep: str | list[str] | None = None,
        event_column: str = "event_alias",
    ) -> pd.DataFrame:
        return _filter_events(
            self._obj, events_to_keep=events_to_keep, event_column=event_column
        )

    def summary_events(
        self,
        session_id_column: str = "session_id",
        event_column: str = "event_alias",
        events_to_keep: str | list[str] | None = None,
        events_to_drop: str | list[str] | None = None,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        return _summary_events(
            self._obj,
            session_id_column=session_id_column,
            event_column=event_column,
            events_to_keep=events_to_keep,
            events_to_drop=events_to_drop,
            prefix=prefix,
        )
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