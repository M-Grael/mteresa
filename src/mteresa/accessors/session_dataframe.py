import pandas as pd
from typing import List
from mteresa.utils.validation import validate_df as _validate_df
from mteresa.utils.data_handling import filter_events as _filter_events
from mteresa.utils.data_handling import filter_events as _summary_events

@pd.api.extensions.register_dataframe_accessor("mteresa")
class MTeresaAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def validate_df(self, schema=None):
        """Validate the current Dataframe using mteresa rules."""
        _validate_df(self._obj, schema=schema)
        return self._obj

    def filter_events(self, events_to_keep: str | List[str] = None, event_column: str = 'event_alias'):
        return _filter_events(self._obj, events_to_keep=events_to_keep, event_column=event_column)

    def summary_events(self, 
        df: pd.DataFrame, 
        event_column: str = "event_alias", 
        events_to_keep: str | list[str] | None = None, 
        events_to_drop: str | list[str] | None = None,
    ) -> pd.DataFrame:    
        return _summary_events(self._obj, df=df, event_column=event_column, events_to_keep=events_to_keep, events_to_drop=events_to_drop)