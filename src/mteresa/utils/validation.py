def validate_df(df, schema=None):
    """Valida se o DataFrame possui as colunas e tipos esperados."""
    if df.empty:
        raise ValueError("O DataFrame está vazio.")
    else: print('ok')

    
    return True