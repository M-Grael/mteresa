import pandas as pd
import mteresa

# Testando a importação do pacote
print(f"Versão do mteresa: {mteresa.__version__}")

# Testando o accessor do Pandas
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(df.mteresa.info_resumida())