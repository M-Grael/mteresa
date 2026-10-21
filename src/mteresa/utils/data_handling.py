import pandas as pd
from typing import List

def filter_events(df, events_to_keep: str | List[str] = None, event_column: str = 'event_alias'):  
    """
    Params:
        df: the dataframe being filtred. Could be a dataframe from cleaning functions
        events_to_be_keeped: events to be keeped. cleaning functions save them in
            ```event_alias``` column
        event_column: column that the event are saved. in mteresa heuristics:,
            clean functions save ```event_column``` as ```event_alias```
            if, for some reason, the user renames the event_column after
            cleaning the data, the function can handle the new name
            with event_column parameter. 

    Returns:
        A pd.Dataframe containg a copy of previous dataframe with only the selected events
    """

    if events_to_keep is None:
        raise ValueError('events_to_keep param should be informed. ' \
        'Plese, make sure to pass the events to be filtred. ' \
        'The argument can be single value string or a list of strings.')

    if event_column not in df.columns.values:
        raise KeyError('Event column informed not in dataframe columns. ' \
        'Please, make sure to confirm if it is the correct column name')

    if isinstance(events_to_keep, str):
        events_to_keep = [events_to_keep]

    return df[df[event_column].isin(events_to_keep)].copy()