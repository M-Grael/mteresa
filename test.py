import mteresa as mts

print(f"mteresa version: {mts.__version__}")
import pandas as pd


df = pd.DataFrame({"session_id": [1, 2], "value": [10.5, 20.1]})

# Chamada encadeada simples
df.mteresa.validate_df()