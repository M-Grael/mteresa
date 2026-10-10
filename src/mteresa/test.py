import os
import pandas as pd
from cmdstanpy import CmdStanModel, set_cmdstan_path

# 1. Configuração do ambiente RTools e CmdStan
rtools_base = r'D:\03_renovacao\ASU\cmdstan\RTools40'
os.environ["PATH"] = f"{rtools_base}\\usr\\bin;{rtools_base}\\mingw64\\bin;{rtools_base};" + os.environ.get("PATH", "")

set_cmdstan_path(r'D:\03_renovacao\ASU\cmdstan\cmdstan-2.39.0')

caminho_do_modelo = r'D:\03_renovacao\ASU\graja\teste_data\berm_test\berm.stan'
caminho_dos_dados = r'D:\03_renovacao\ASU\graja\teste_data\berm_test\Phase2IRT.xlsx' 

# 2. Leitura dos dados
df = pd.read_excel(caminho_dos_dados)

print("Compilando o modelo...")
# force_compile=True garante que ele vai recompilar a sua correção no arquivo .stan
model = CmdStanModel(stan_file=caminho_do_modelo, force_compile=True)
print("Modelo compilado com sucesso!")

# 3. Tratamento dos dados (Garante alinhamento perfeito entre ID e Grupo)
df['subject_id'] = pd.factorize(df['Subject'])[0] + 1
G_por_sujeito = df.groupby('subject_id', sort=True)['G'].first().tolist()

stan_data = {
    'N': len(df), 
    'S': df['subject_id'].nunique(),
    'subject': df['subject_id'].tolist(),
    'Y': df['IRT'].tolist(),
    'd': df['d'].tolist(),
    'G': G_por_sujeito 
}

# 4. Amostragem MCMC
print("Iniciando a amostragem (MCMC)...")
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=500, 
    iter_sampling=500, 
    show_progress=True 
)

print("\nConcluído com sucesso!")

# 5. Tenta gerar o resumo oficial; se o stansummary.exe falhar, calcula via Pandas
try:
    summary_df = fit.summary()
    summary_df.to_csv(r'analise_summary.csv')
    print("-> Resumo salvo via CmdStanPy em 'analise_summary.csv'")
except Exception as e:
    print(f"\n[Aviso] fit.summary() falhou ({e}). Gerando resumo alternativo via Pandas...")

# 6. Extrai TODAS as amostras posteriores diretamente (funciona mesmo se stansummary falhar)
posterior_draws = fit.draws_pd()
posterior_draws.to_csv(r'analise_posteriores.csv', index=False)
print("-> Distribuição posterior completa salva em 'analise_posteriores.csv'")

# Se o summary() oficial falhou, gera a média e desvio padrão manualmente:
if 'summary_df' not in locals():
    manual_summary = posterior_draws.describe().T[['mean', 'std', 'min', '50%', 'max']]
    manual_summary.to_csv(r'analise_summary.csv')
    print("-> Resumo estatístico manual salvo em 'analise_summary.csv'")

# 7. Diagnósticos do amostrador NUTS no console
print("\n--- Diagnósticos do Amostrador (NUTS/HMC) ---")
print(fit.diagnose())

# 8. Exportação opcional via ArviZ (NetCDF e Métricas adicionais)
try:
    import arviz as az
    
    idata = az.from_cmdstanpy(posterior=fit, observed_data={"Y": df['IRT'].values})
    
    # Salva o arquivo NetCDF (estrutura completa do ArviZ)
    idata.to_netcdf(r'analise_completa.nc')
    print("-> Objeto ArviZ salvo em 'analise_completa.nc'")
    
    # Salva diagnósticos de amostragem específicos (se existirem)
    if hasattr(idata, 'sample_stats'):
        df_stats = idata.sample_stats.to_dataframe().reset_index()
        df_stats.to_csv('sample_stats.csv', index=False)
        print("-> Métricas de amostragem salvas em 'sample_stats.csv'")

except ImportError:
    print("\n[Aviso] ArviZ não instalado. Caso deseje salvar o arquivo .nc, execute: pip install arviz")