
import numpy as np

class TemporalRegulationModel:
    def __init__(self, df, w_prior=(2, 2), beta_prior=(2, 2)):
        """
        1. Hiperparâmetros e configurações da classe.
        Aqui definimos as configurações dos priors e do modelo.
        """

        self.df = df
        self.w_prior = w_prior
        self.beta_prior = beta_prior
        
        # Atributos preenchidos apenas após o fit
        self.model = None
        self.trace_ = None
    
    def _random(self, q, w, k, t, rng=np.random.default_rng, size=None):
        q, w, k, t = np.broadcast_arrays(q, w, k, t)

        shape = size if size is not None else q.shape
        is_timed = rng.binomial(1, q, size=shape)
        timed_draw = rng.gamma(shape=k, scale=t, size=shape)
        exponential_draw = rng.exponential(scale= 1 / w, size=shape)
        return np.where(is_timed, timed_draw, exponential_draw)

        


    def _build_model(self, data):
        """
        2. Método privado que constrói o Grafo Probabilístico.
        Separa a construção da lógica de amostragem.
        """
        with pm.Model() as model:
            # Prior para a taxa/regulação inicial
            gamma = pm.Gamma('gamma', alpha=self.alpha_prior[0], beta=self.alpha_prior[1])
            w = pm.Exponential('w', lam=self.w_prior)
            
            
            # Likelihood (Verossimilhança com base nos dados observados)
            like = pm.CustomDist(
                'obs', 
                random=self._random(0.2, self.w_prior, 2, 1), 
                observed=data)
            
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
    dados = r'sandbox\graja.csv'    

    trm = TemporalRegulationModel(alpha_prior=(2, 1))
    
    trm.fit(dados, draws=200, tune=500)
    
    print(trm.summary())