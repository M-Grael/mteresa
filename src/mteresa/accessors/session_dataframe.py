import pandas as pd
from mteresa.utils.validation import validate_df as _validate_df

@pd.api.extensions.register_dataframe_accessor("mteresa")
class MTeresaAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def validate_df(self, schema=None):
        """Valida o DataFrame atual usando as regras do mteresa."""
        _validate_df(self._obj, schema=schema)
        return self._obj  # Retornar self._obj permite encadear métodos (method chaining)