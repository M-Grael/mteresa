import pandas as pd

def filter_events(
    df: pd.DataFrame,
    events_to_keep: str | list[str] | None = None,
    event_column: str = "event_alias",
) -> pd.DataFrame:
    """Filter a DataFrame to include only rows with specified event names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    events_to_keep : str | list[str] | None, default None
        A single event string or a list of event strings to retain.
        event_column : str, default 'event_alias'
        The column name containing event names.

    Returns
    -------
    pd.DataFrame
        A filtered copy of the original DataFrame.

    Raises
    ------
    ValueError
        If `events_to_keep` is None.
    KeyError
        If `event_column` is not present in `df.columns`.
    """
    if events_to_keep is None:
        raise ValueError(
            "`events_to_keep` must be specified. Pass a single event string or a"
            " list of event strings."
        )

    if event_column not in df.columns:
        raise KeyError(
        f"Column '{event_column}' not found in DataFrame. Available columns:"
        f" {list(df.columns)}"
        )

    if isinstance(events_to_keep, str):
        events_to_keep = [events_to_keep]

    return df[df[event_column].isin(events_to_keep)].copy()


import pandas as pd


import pandas as pd


def summary_events(
    df: pd.DataFrame,
    session_id_column: str = 'session_id',
    event_column: str = 'event_alias',
    events_to_keep: str | list[str] | None = None,
    events_to_drop: str | list[str] | None = None,
    prefix: str = None
) -> pd.DataFrame:
    """Summary events into a wide-format DataFrame with event counts per session."""

    if session_id_column not in df.columns:
        raise KeyError(
            f'Column "{session_id_column}" not found in DataFrame. Available columns: '
            f'{list(df.columns)}'
        )

    if event_column not in df.columns:
        raise KeyError(
            f"Column '{event_column}' not found in DataFrame. Available columns: "
            f'{list(df.columns)}'
        )

    if events_to_keep is not None and events_to_drop is not None:
        raise ValueError(
            '`events_to_keep` and `events_to_drop` cannot be specified simultaneously.'
        )


    if isinstance(events_to_keep, str):
        events_to_keep = [events_to_keep]
    if isinstance(events_to_drop, str):
        events_to_drop = [events_to_drop]

    print(df[event_column].unique())
    print(df.columns.values)
    print(df[event_column])

    df_filtered = df.copy()

    if events_to_keep is not None:
        df_filtered = df_filtered[
            df_filtered[event_column].isin(events_to_keep)
        ]
    elif events_to_drop is not None:
        df_filtered = df_filtered[
            ~df_filtered[event_column].isin(events_to_drop)
        ]

    pivoted = df_filtered.pivot_table(
        index=[session_id_column],
        columns=event_column,
        aggfunc='size',
        fill_value=0,
    )

    if events_to_keep is not None:
        pivoted = pivoted.reindex(columns=events_to_keep, fill_value=0)

    if prefix is not None:
        pivoted = pivoted.add_prefix(prefix=prefix)

    return pivoted.rename_axis(None, axis=1).reset_index()

if __name__ == '__main__':
    import mteresa as mts
    df = mts.read_medpc(folder_path='sandbox\\data_test\\sofia_teste_data',
                        msn_folder_path='sandbox\\data_test\\sofia_teste_data',
                        guess_event=True, guess_session=True)

    print(summary_events(df))
