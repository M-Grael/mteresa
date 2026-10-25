import pandas as pd
import numpy as np
from os.path import join

pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 20

def output_transformation():
    return None

def peak_deviation_analysis(
    df: pd.DataFrame,
    timestamp_column: str = 'timestamp',
    burst_cutoff: float = 6.0,
    min_irts: int = 25,
    grouping_index_column: str = 'session_id',
    events_column: str = 'event_alias',
    filter_events: str | list[str] | None = None,
    transform: bool = False
) -> pd.DataFrame:
    """
    Implements Peak Deviation Analysis (Richards et al., 1993).
    """
#   STEP 0
#   ================================
#         Dataframe Validation
#   ================================
    required_cols = [grouping_index_column, timestamp_column, events_column]
    fails = [c for c in required_cols if c not in df.columns]
    if fails:
        raise KeyError(f"Columns {fails} not found in dataframe.")
#   STEP 1
#   ================================
#       Clean and prepare data
#   ================================
    df_pda = df.copy()
    df_pda = df_pda.dropna(subset=grouping_index_column)
    
    if df_pda[grouping_index_column].dtype == 'int64':
        df_pda = df_pda.sort_values(by=grouping_index_column, kind='stable')
    
    if isinstance(filter_events, str):
        filter_events = [filter_events]
    if not filter_events is None:
        df_pda = df_pda[df_pda[events_column].isin(filter_events)]
    
    df_pda[timestamp_column] = pd.to_numeric(df_pda[timestamp_column], errors='coerce')
    df_pda = df_pda[[grouping_index_column, timestamp_column]]

    pda_analysis = []
    for idx in df_pda[grouping_index_column].unique():
        df_analysis  = df_pda[df_pda[grouping_index_column] == idx].reset_index(drop=True)        
        if len(df_analysis) < min_irts:
            pda_analysis.append({grouping_index_column: idx, 'BR': np.nan, 'PkA': np.nan, 'PkL': np.nan})
        else:
            df_analysis['IRT'] = df_analysis[timestamp_column].diff()
            df_analysis = df_analysis.dropna().reset_index(drop=True)
#   STEP 2
#   ================================
#      Clarify Burst and Pauses
#   ================================
            df_analysis['irt_type'] = np.where(df_analysis['IRT'] < burst_cutoff, 'burst', 'pause')
            NB = (df_analysis['irt_type'] == 'burst').sum()
            N = (df_analysis['irt_type'] == 'pause').sum()
            M = df_analysis.loc[df_analysis['irt_type'] == 'pause', 'IRT'].mean()
            
            df_analysis['nb'] = NB
            df_analysis['n'] = N
            df_analysis['m'] = M

            pause_df = (
                df_analysis
                .query('irt_type == "pause"')
                .sort_values('IRT', ascending=False)
                .reset_index(drop=True)
                [[grouping_index_column, 'IRT', 'nb', 'n', 'm']]
                )

#   STEP 3
#   ================================
#     Burst Ratio (BR) calculation
#   ================================
            if N == 0 or M <= burst_cutoff:
                pda_analysis.append({grouping_index_column: idx, 'BR': np.nan, 'PkA': np.nan, 'PkL': np.nan})
                continue

            BR = NB / (N * (np.exp(burst_cutoff / (M - burst_cutoff)) - 1))

#   STEP 4
#   ================================
#      Pause Dataframe Stats
#   ================================

            pause_df['i'] = pause_df.index + 1
            pause_df['S(i)'] = pause_df['i'] / pause_df['n']
            # Fo(t) = exp(-(t - cut) / (M - cut))
            pause_df['Fo(t)'] = np.exp(-(pause_df['IRT'] - burst_cutoff) / (M - burst_cutoff))

            # Gibbons, 1985, p. 63
            cond = np.absolute(pause_df['S(i)'] - pause_df['Fo(t)']) > np.absolute(pause_df['S(i)'] - pause_df['Fo(t)'].shift())
            pause_df['Dn[t(i)]'] = np.where(cond, pause_df['S(i)'] - pause_df['Fo(t)'], pause_df['S(i)'] - pause_df['Fo(t)'].shift())

#   STEP 5
#   ================================
#      PkA and PkL segments
#   ================================           
            # t_pos is equal to the ``IRT`` row which ``Dn[t(i)]`` is max.
            # t_neg is equal to the ``IRT`` row which ``Dn[t(i)]`` is min.

            dn_max = pause_df['Dn[t(i)]'].max()
            dn_min = pause_df['Dn[t(i)]'].min()

            t_pos = pause_df.loc[pause_df['Dn[t(i)]'] == dn_max, 'IRT'].values[0]
            t_neg = pause_df.loc[pause_df['Dn[t(i)]'] == dn_min, 'IRT'].values[0]

#   STEP 6
#   ================================
#      PkA and multimodal check
#   ================================
            if t_neg > t_pos:
                PkA = dn_max + np.abs(dn_min)
                d_pos = dn_max
            else:
                pos_challenge_df = pause_df[pause_df['IRT'] <= t_neg].copy()
                dn_max_challenge = pos_challenge_df['Dn[t(i)]'].max()
                t_pos_challenge = pos_challenge_df.loc[pos_challenge_df['Dn[t(i)]'] == dn_max_challenge, 'IRT'].values[0]

                neg_challenge_df = pause_df[pause_df['IRT'] >= t_pos].copy()
                dn_min_challenge = neg_challenge_df['Dn[t(i)]'].min()
                t_neg_challenge = neg_challenge_df.loc[neg_challenge_df['Dn[t(i)]'] == dn_min_challenge, 'IRT'].values[0]

                PkA_first_candidate = dn_max + np.abs(dn_min_challenge)
                PkA_second_candidate = dn_max_challenge + np.abs(dn_min)

                PkA = max(PkA_first_candidate, PkA_second_candidate)
                segment_winner = np.argmax([PkA_first_candidate, PkA_second_candidate])
                d_pos = dn_max if segment_winner == 0 else dn_max_challenge

#   STEP 7
#   ================================
#               PkL 
#   ================================
            if t_neg > t_pos:
                pkl_segment_df = pause_df[pause_df['IRT'].between(t_pos, t_neg)].copy()
                pkl_segment_df['target'] = np.abs(d_pos - pkl_segment_df['Dn[t(i)]'] - 0.5*PkA)
                PkL = pkl_segment_df.loc[pkl_segment_df['target'].idxmin(), 'IRT']
            
            else: # Else block for multimodal peak checking
                if segment_winner == 0:
                    pkl_segment_df = pause_df[pause_df['IRT'].between(t_pos, t_neg_challenge)].copy()
                    pkl_segment_df['target'] = np.abs(d_pos - pkl_segment_df['Dn[t(i)]'] - 0.5*PkA)
                    PkL = pkl_segment_df.loc[pkl_segment_df['target'].idxmin(), 'IRT']
                else:
                    pkl_segment_df = pause_df[pause_df['IRT'].between(t_pos_challenge, t_neg)].copy()
                    pkl_segment_df['target'] = np.abs(d_pos - pkl_segment_df['Dn[t(i)]'] - 0.5*PkA)
                    PkL = pkl_segment_df.loc[pkl_segment_df['target'].idxmin(), 'IRT']

            pda_analysis.append({grouping_index_column: idx, 'BR': BR, 'PkA': PkA, 'PkL': PkL})
            

    return pd.DataFrame(pda_analysis)    

if __name__ == '__main__':
    PATH = join('sandbox','graja.csv')
    df = pd.read_csv(PATH)

    print(peak_deviation_analysis(df, burst_cutoff=6, min_irts=20))