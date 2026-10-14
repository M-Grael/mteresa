import pandas as pd

@pd.api.extensions.register_dataframe_accessor("mteresa")
class MTeresaDataFrameAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def info_resumida(self) -> dict:
        return {
            "linhas": len(self._obj),
            "colunas": list(self._obj.columns)
        }