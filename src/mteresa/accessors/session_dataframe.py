import pandas as pd
from mteresa.utils.validation import validate_df as _validate_df
from mteresa.utils.data_handling import filter_events as _filter_events
from mteresa.utils.data_handling import summary_events as _summary_events


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