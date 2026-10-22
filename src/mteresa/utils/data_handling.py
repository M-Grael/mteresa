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

def summary_events(
    df: pd.DataFrame,
    event_column: str = "event_alias",
    events_to_keep: str | list[str] | None = None,
    events_to_drop: str | list[str] | None = None,
) -> pd.DataFrame:
 
    """Summary .

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    events_to_keep : str | list[str] | None, default None
        A single event string or a list of event strings to retain.
    events_to_drop : str | list[str] | None, default None
        A single event string or a list of event strings to drop.
    event_column : str, default 'event_alias'
        The column name containing event names.

    Returns
    -------
    pd.DataFrame
        A wide melted copy of the original DataFrame.

    Raises
    ------
    ValueError
        If `events_to_keep` and `events_to_drop` are not None
        simultaneously.
    KeyError
        If `event_column` is not present in `df.columns`.
    """
    
    return df.columns.values

if __name__ == '__main__':
    df = pd.read_parquet('sandbox\\sofia.parquet')

    print(summary_events(df))
