import pandas as pd
import numpy as np

def random_irt(q, w, b, rng=np.random.default_rng, size = None):
    q, w, b = np.broadcast_arrays(q, w, b)
    shape = size if size is not None else q.shape
    is_between = rng.binomial(1, q, size=shape)
    between_draw = rng.exponential(1 / b, size=shape)
    within_draw = rng.exponential(1 / w, size=shape)
    return np.where(is_between, between_draw, within_draw)

class BiExponentialRefactorialModel:
    def __init__(
            self, 
            df,
            timestamp_column='timestamp',
            subject_column='subject',
            grouping_column='group',
            event_column='event_alias',
            session_index='session_id',
            filter_events = None):
        
        self.df = df.copy()

        if isinstance(filter_events, str):
            filter_events = [filter_events]
            self.df = self.df[self.df[event_column].isin(filter_events)]

        self.df['IRT'] = (
            self.df
            .groupby(session_index)['timestamp']
            .diff()
            .dropna()
            .reset_index(drop=True)
            )
    
        self.df['D'] = (
            self.df
            .groupby(session_index)['IRT']
            .tranform('min')
            )
    
        self.y = self.df[timestamp_column]
        self.grouping = self.df[grouping_column]
        self.subject = self.df[subject_column]

    def _prepare_data(self):
        pass
    
    def _build_model(self):
        pass

    def fit(self):
        pass
