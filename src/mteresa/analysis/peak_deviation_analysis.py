import pandas as pd
import numpy as np
from os.path import join
from typing import Optional

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
    Performs inter-response time (IRT) peak deviation analysis on behavioral event data.
    """
#   ==============================
#        Dataframe Validation
#   ==============================
    required_cols = [grouping_index_column, timestamp_column, events_column]
    fails = [c for c in required_cols if c not in df.columns]
    if fails:
        raise KeyError(f"Columns {fails} not found in dataframe.")
    
    
    return None

# def peak_deviation_analysis(irt_column, timestamp_column = None, dataframe=None, burst_cutoff=6, min_irts=25):
#     '''
#     Implements Peak Deviation Analysis (Richards et al., 1993).
    
#     :param df: Dataframe with 'irt_list' column (string of list or list)
#     :param burst_cutoff: Cutoff value to separate bursts from pauses (default 6s)
#     :return: Dataframe with BR, PkA, and PkL metrics for each session
#     '''
#     # 1. Clean and Prepare Data
#     df = df[df['sessao'] != 0].copy()
    
#     # Create a skeleton of all sessions to ensure none are dropped if they lack pauses
#     all_sessions = df[['sessao', 'sujeito']].drop_duplicates()

#     # Explode IRT lists
#     df['irt_list'] = df['irt_list'].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) else x
#     )   
#     df_long = df.explode('irt_list', ignore_index=True)
#     df_long['irt'] = pd.to_numeric(df_long['irt_list'], errors='coerce')
#     df_long = df_long.dropna(subset=['irt'])
    
#     # 2. Classify Bursts vs Pauses
#     df_long['irt_type'] = np.where(df_long['irt'] >= burst_cutoff, 'pause', 'burst')
    
#     # Calculate Burst Counts (Nb)
#     burst_counts = (
#         df_long[df_long['irt_type'] == 'burst']
#         .groupby(['sessao', 'sujeito'])
#         .size()
#         .rename('Nb')
#         .reset_index()
#     )
    
#     # Prepare Pause Data
#     pause_df = df_long[df_long['irt_type'] == 'pause'].copy()
    
#     # Sort descending for Survivor Function calculation (Step 1 in paper)
#     pause_df = pause_df.sort_values(
#         by=['sujeito', 'sessao', 'irt'], 
#         ascending=[True, True, False], 
#         kind='mergesort'
#     )
    
#     # 3. Calculate Global Stats (N, M) and Survivor Functions (Sn, Fo, Dn)
#     # We do this before applying the custom function to leverage vectorization
#     stats = pause_df.groupby(['sessao', 'sujeito']).agg(
#         N=('irt', 'size'),
#         M=('irt', 'mean')
#     ).reset_index()
    
#     pause_df = pause_df.merge(stats, on=['sessao', 'sujeito'])
    
#     # Calculate Sn (Empirical Survivor Function)
#     # i goes from 1 to N
#     pause_df['i'] = pause_df.groupby(['sessao', 'sujeito']).cumcount() + 1
#     pause_df['Sn'] = pause_df['i'] / pause_df['N']
    
#     # Calculate Fo (Corresponding Negative Exponential Survivor Function)
#     # Fo(t) = exp(-(t - cut) / (M - cut))
#     valid_mask = pause_df['M'] > burst_cutoff
#     pause_df['Fo'] = np.nan
#     pause_df.loc[valid_mask, 'Fo'] = np.exp(
#         -(pause_df.loc[valid_mask, 'irt'] - burst_cutoff) / 
#         (pause_df.loc[valid_mask, 'M'] - burst_cutoff)
#     )
    
#     # Calculate Dn (Difference)
#     pause_df['Dn'] = pause_df['Sn'] - pause_df['Fo']
    
#     # 4. Define Logic for Peak Finding (PkA, PkL)
#     def calculate_peak_metrics(group):
#         if group.empty or pd.isna(group.name): # Safety check
#             return pd.Series({'PkA': np.nan, 'PkL': np.nan})
        
#         # Helper to get scalar values safely
#         Dn = group['Dn'].values
#         irt = group['irt'].values
        
#         # Find global extremes indices (relative to the group array)
#         idx_pos = np.argmax(Dn) # Index of max positive deviation
#         idx_neg = np.argmin(Dn) # Index of max negative deviation
        
#         D_pos = Dn[idx_pos]
#         D_neg = Dn[idx_neg]
        
#         # RICHARDS (1993) MULTIMODAL CHECK
#         # Note: Array is sorted by IRT DESCENDING. 
#         # So index 0 is Largest IRT, index -1 is Smallest IRT.
#         # Paper says: "When t(neg) > t(pos) ... correct."
#         # t(neg) > t(pos) means t(neg) is a LARGER IRT, so it should appear EARLIER in the array (Lower index).
#         # Correct case: idx_neg < idx_pos
        
#         best_pos_idx = idx_pos
#         best_neg_idx = idx_neg
        
#         if idx_neg > idx_pos:
#             # Incorrect location: t(neg) < t(pos). This indicates multiple peaks.
#             # We must search for secondary extrema.
            
#             # 1. Search for new Max (Pos') between global Min (idx_neg) and end of array (Smallest IRT)
#             # Range: [idx_neg : ]
#             sub_dn_1 = Dn[idx_neg:]
#             if len(sub_dn_1) > 0:
#                 idx_pos_prime = idx_neg + np.argmax(sub_dn_1)
#                 PkA_1 = Dn[idx_pos_prime] - Dn[idx_neg]
#             else:
#                 PkA_1 = -1 # Invalid
            
#             # 2. Search for new Min (Neg') between start of array (Largest IRT) and global Max (idx_pos)
#             # Range: [ : idx_pos + 1]
#             sub_dn_2 = Dn[:idx_pos+1]
#             if len(sub_dn_2) > 0:
#                 idx_neg_prime = np.argmin(sub_dn_2)
#                 PkA_2 = Dn[idx_pos] - Dn[idx_neg_prime]
#             else:
#                 PkA_2 = -1 # Invalid
                
#             # Compare and choose the largest Peak Area
#             if PkA_1 > PkA_2:
#                 best_pos_idx = idx_pos_prime
#                 best_neg_idx = idx_neg
#             else:
#                 best_pos_idx = idx_pos
#                 best_neg_idx = idx_neg_prime

#         # Calculate Final Metrics using the identified Peak Segment
#         final_D_pos = Dn[best_pos_idx]
#         final_D_neg = Dn[best_neg_idx]
        
#         # PkA: Sum of max positive excursion and max negative excursion (absolute)
#         PkA = final_D_pos - final_D_neg 
        
#         # PkL: Median of the Peak Area
#         # We look for the time t within the segment [neg_idx : pos_idx] 
#         # where Dn is closest to the midpoint of the excursion.
#         # Midpoint Value = D_neg + (PkA / 2) = (D_pos + D_neg) / 2
        
#         target_dn = (final_D_pos + final_D_neg) / 2.0
        
#         # Extract the segment arrays
#         # Ensure indices are ordered (neg appears before pos in sorted list)
#         start, end = min(best_pos_idx, best_neg_idx), max(best_pos_idx, best_neg_idx)
        
#         segment_dn = Dn[start : end+1]
#         segment_irt = irt[start : end+1]
        
#         # Find index in segment closest to target
#         if len(segment_dn) > 0:
#             local_idx = np.abs(segment_dn - target_dn).argmin()
#             PkL = segment_irt[local_idx]
#         else:
#             PkL = np.nan

#         return pd.Series({'PkA': PkA, 'PkL': PkL})

#     # Apply the peak finding logic to each session
#     peak_metrics = pause_df.groupby(['sessao', 'sujeito']).apply(calculate_peak_metrics).reset_index()
    
#     # 5. Calculate Burst Ratio (BR)
#     # BR = Nb / [ N * ( exp(cut / (M - cut)) - 1 ) ]
    
#     # Merge N, M, Nb into one dataframe for BR calc
#     br_df = all_sessions.merge(stats, on=['sessao', 'sujeito'], how='left')
#     br_df = br_df.merge(burst_counts, on=['sessao', 'sujeito'], how='left').fillna({'Nb': 0, 'N': 0})
    
#     def compute_br(row, cut):
#         N = row['N']
#         Nb = row['Nb']
#         M = row['M']
        
#         # Guard against division by zero or invalid Mean
#         if N == 0 or M <= cut:
#             return np.nan
            
#         denominator = N * (np.exp(cut / (M - cut)) - 1)
#         if denominator == 0:
#             return np.nan
            
#         return Nb / denominator

#     br_df['BR'] = br_df.apply(compute_br, axis=1, cut=burst_cutoff)
    
#     # 6. Final Merge
#     final_df = all_sessions.merge(br_df[['sessao', 'sujeito', 'BR']], on=['sessao', 'sujeito'], how='left')
#     final_df = final_df.merge(peak_metrics, on=['sessao', 'sujeito'], how='left')
#     final_df = df.merge(final_df,  on=['sessao', 'sujeito'], how='left')
    
#     # Merge back to original df structure if needed, or return summary
#     # Here we merge the metrics back to the ORIGINAL user df (summarized)
#     # Since the input 'df' was raw data, usually users want the metrics per session.
#     # We will return the metrics dataframe.
    
#     return final_df
    

if __name__ == '__main__':
    import mteresa as mts
    PATH = join('sandbox','graja.csv')
    df = pd.read_csv(PATH)

    peak_deviation_analysis(df)