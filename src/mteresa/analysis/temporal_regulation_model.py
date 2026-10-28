
import numpy as np

class TemporalRegulationModel:
    def __init__(self, alpha_prior=(2, 2), beta_prior=(2, 2)):
        """
        1. Hiperparâmetros e configurações da classe.
        Aqui definimos as configurações dos priors e do modelo.
        """

        
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Atributos preenchidos apenas após o fit
        self.model = None
        self.trace_ = None

    def _build_model(self, data):
        """
        2. Método privado que constrói o Grafo Probabilístico.
        Separa a construção da lógica de amostragem.
        """
        with pm.Model() as model:
            # Prior para a taxa/regulação inicial
            alpha = pm.Gamma('alpha', alpha=self.alpha_prior[0], beta=self.alpha_prior[1])
            
            
            # Likelihood (Verossimilhança com base nos dados observados)
            like = pm.Mixture('obs', mu=alpha, observed=data)
            
        return model

    def fit(self, data, draws=2000, tune=1000, chains=2):
        """
        3. O Método fit(): executa o motor de inferência (MCMC).
        Armazena os resultados no próprio objeto (self.trace_).
        """
        # Constrói o grafo com os dados fornecidos
        self.model = self._build_model(data)
        
        # Executa a amostragem
        with self.model:
            self.trace_ = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                return_inferencedata=True
            )
        
        return self

    def summary(self):
        """
        4. Métodos utilitários de diagnóstico usando ArviZ.
        """
        if self.trace_ is None:
            raise ValueError("O modelo precisa ser ajustado com .fit() antes do resumo.")
        
        return az.summary(self.trace_)


# --- Exemplo de Uso ---
if __name__ == "__main__":
    import pymc as pm
    import arviz as az
    # Dados simulados (ex: eventos temporais)
    dados = np.random.poisson(lam=5.0, size=100)
    
    # Instancia o modelo
    trm = TemporalRegulationModel(alpha_prior=(2, 1))
    
    # Ajusta o modelo aos dados
    trm.fit(dados, draws=1000, tune=500)
    
    # Exibe métricas estatísticas posteriores (Média, Intervalo de Credibilidade, R-hat)
    print(trm.summary())