import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Funções da verossimilhança (ordem dos parâmetros: q, w, b)
# ---------------------------------------------------------------
def random_irt(q, w, b, rng=None, size=None):
    """Sorteia IRTs (na escala de Y - d) da mistura de duas exponenciais."""
    if rng is None:
        rng = np.random.default_rng()
    q, w, b = np.broadcast_arrays(q, w, b)
    shape = size if size is not None else q.shape
    is_between = rng.binomial(1, q, size=shape).astype(bool)
    between_draw = rng.exponential(1.0 / b, size=shape)  # numpy usa escala = 1/taxa
    within_draw = rng.exponential(1.0 / w, size=shape)
    return np.where(is_between, between_draw, within_draw)


def biexp_logp(value, q, w, b):
    """Log-densidade da mistura: q*Exp(b) + (1-q)*Exp(w)."""
    import pytensor.tensor as pt

    log_between = pt.log(q) + pt.log(b) - b * value
    log_within = pt.log1p(-q) + pt.log(w) - w * value
    return pt.logaddexp(log_between, log_within)


def _require(df, columns, why):
    """Falha cedo, com mensagem clara, se faltar coluna."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Colunas ausentes ({why}): {missing}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )


# ---------------------------------------------------------------
# Classe do modelo
# ---------------------------------------------------------------
class BiExponentialRefactorialModel:
    """
    Dois formatos de entrada são aceitos:

    A) Eventos brutos: irt_column=None. O IRT é calculado a partir de
       timestamp_column, agrupado por session_index (event_column só é
       necessária se filter_events for usado).

    B) IRTs prontos (ex.: matthew.csv): irt_column='IRT'. Não precisa de
       timestamp, sessão nem evento.

    refactorial_period_column: coluna com o período refratário d (como no
    Stan). Se None, d é estimado como o menor IRT de cada sujeito.
    """

    def __init__(
        self,
        df,
        timestamp_column='timestamp',
        subject_column='subject',
        grouping_column='group',
        event_column='event_alias',
        session_index='session_id',
        irt_column=None,
        refactorial_period_column=None,
        filter_events=None,
        w_mode='fmax',
    ):
        if w_mode not in ("fmax", "offset"):
            raise ValueError("w_mode should be 'fmax' or 'offset'.")
        self.w_mode = w_mode

        df = df.copy()
        _require(df, [subject_column, grouping_column], "sempre necessárias")

        # Filtro de eventos (opcional)
        if filter_events is not None:
            _require(df, [event_column], "usada em filter_events")
            if isinstance(filter_events, str):
                filter_events = [filter_events]
            df = df[df[event_column].isin(filter_events)]

        # ---- IRT ----
        if irt_column is None:
            _require(df, [timestamp_column, session_index], "cálculo do IRT")
            df = df.sort_values([session_index, timestamp_column])
            df["IRT"] = df.groupby(session_index)[timestamp_column].diff()
            df = df.dropna(subset=["IRT"])
        else:
            # Formato B: IRTs já calculados
            _require(df, [irt_column], "irt_column")
            df["IRT"] = df[irt_column].astype(float)

        # ---- d (período refratário) ----
        if refactorial_period_column is None:
            df["D"] = df.groupby(subject_column)["IRT"].transform("min")
        else:
            _require(df, [refactorial_period_column], "refactorial_period_column")
            df["D"] = df[refactorial_period_column].astype(float)

        df = df.reset_index(drop=True)

        self.df = df
        self.y = df["IRT"].to_numpy(dtype=float)
        self.d = df["D"].to_numpy(dtype=float)
        y_shift = self.y - self.d
        # Tolerância para erro de ponto flutuante; negativos reais são erro nos dados
        if np.any(y_shift < -1e-9):
            n_bad = int((y_shift < -1e-9).sum())
            raise ValueError(f'{n_bad} observações com IRT < d.')
        self.y_shift = np.maximum(y_shift, 0.0)

        # Índice de sujeito por observação (tamanho N)
        self.subject_idx, self.subject_labels = pd.factorize(df[subject_column])
        
        self.group_idx, self.group_labels = pd.factorize(df[grouping_column], sort=True)

        self.coords = {
            'subject': np.asarray(self.subject_labels),
            'group': np.asarray(self.group_labels),
            'obs': np.arange(len(df)),
        }

        self.model = None
        self.idata = None

    def _build_model(self):
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model(coords=self.coords) as model:
            subj = pm.Data('subj', self.subject_idx, dims='obs')
            grp = pm.Data('grp', self.group_idx, dims='obs')

            # --- L (log-odds de within sobre between) ---
            sigma_L = pm.HalfNormal('sigma_L', 5)
            L_group = pm.Normal('L_group', 0, 10, dims='group')
            L_offset = pm.Normal('L_offset', 0, 1, dims='subject')
            L_subj_cond = pm.Deterministic(
                'L_subject_group',
                L_group[None, :] + L_offset[:, None] * sigma_L,
                dims=('subject', "group"),
)

            # --- w (log da taxa within-bout) ---
            sigma_w = pm.HalfNormal('sigma_w', 5)
            w_group = pm.Normal('w_group', 0, 10, dims='group')
            w_offset = pm.Normal('w_offset', 0, 1, dims='subject')
            w_subj_cond = pm.Deterministic(
                'w_subject_group',
                w_group[None, :] + w_offset[:, None] * sigma_w,
                dims=('subject', 'group'),
            )

            # --- b (log da taxa between-bout) ---
            sigma_b = pm.HalfNormal('sigma_b', 5)
            b_group = pm.Normal('b_group', 0, 10, dims='group')
            b_offset = pm.Normal('b_offset', 0, 1, dims='subject')
            b_subj_cond = pm.Deterministic(
                'b_subject_group',
                b_group[None, :] + b_offset[:, None] * sigma_b,
                dims=('subject', 'group'),
            )
            # --- Parâmetros por sujeito na escala natural ---
            q_s = pm.math.sigmoid(-L_subj_cond)  # = 1 / (exp(L) + 1)
            b_s = pt.exp(b_subj_cond)
            if self.w_mode == 'fmax':
                w_s = pt.maximum(b_s, pt.exp(w_subj_cond))
            else:
                w_s = b_s + pt.exp(w_subj_cond)

            pm.CustomDist(
                'y',
                q_s[subj, grp],
                w_s[subj, grp],
                b_s[subj, grp],
                logp=biexp_logp,
                random=random_irt,
                observed=self.y_shift,
                dims='obs',
            )

        self.model = model
        return model

    def fit(
        self,
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.95,
        max_treedepth=15,
        nuts_sampler='pymc',
        random_seed=42,
    ):
        import pymc as pm

        if self.model is None:
            self._build_model()

        kwargs = {}
        if nuts_sampler == "pymc":
            # No Stan: max_treedepth = 15 (repassado ao passo NUTS)
            kwargs["max_treedepth"] = max_treedepth

        with self.model:
            self.idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                target_accept=target_accept, nuts_sampler=nuts_sampler,
                random_seed=random_seed, **kwargs,
            )
            pm.compute_log_likelihood(self.idata)
            pm.sample_posterior_predictive(
                self.idata, extend_inferencedata=True, random_seed=random_seed
            )
        return self

    def summary(self):
        import arviz as az

        if self.idata is None:
            raise ValueError("Rode .fit() antes de .summary().")
        var_names = [
            "L_group", "w_group", "b_group",
            "sigma_L", "sigma_w", "sigma_b",
        ]
        return az.summary(self.idata, var_names=var_names)

    def summary_natural(self):
        """
        Resumo na escala natural, calculado sobre as AMOSTRAS da posterior
        (nunca sobre a média do resumo em escala log).
 
        - q_between: probabilidade de uma resposta vir do between-bout
        - b_rate, w_rate: taxas por unidade de tempo (exp do log da taxa)
        - *_ratio: razão de taxas de cada grupo em relação ao primeiro
        - sigma_*_fator: fator multiplicativo de variação entre sujeitos
          (exp(sigma)); em L, sigma segue em log-odds
 
        Obs.: exp(w_group) é a taxa within de um sujeito típico antes do
        fmax; o w efetivo é max(b, exp(w_subject)).
        """
        import arviz as az
        import xarray as xr
 
        if self.idata is None:
            raise ValueError("Rode .fit() antes de .summary_natural().")
        post = self.idata.posterior
 
        data = {
            "q_between": 1.0 / (1.0 + np.exp(post["L_group"])),
            "b_rate": np.exp(post["b_group"]),
            "w_rate": np.exp(post["w_group"]),
            "sigma_L": post["sigma_L"],
            "sigma_w_fator": np.exp(post["sigma_w"]),
            "sigma_b_fator": np.exp(post["sigma_b"]),
        }
        table = az.summary(xr.Dataset(data))
 
        # Razões de grupo em tabela à parte (dimensão 'group' diferente)
        if post.sizes["group"] > 1:
            ratios = {}
            for name, var in [("b", "b_group"), ("w", "w_group")]:
                diff = post[var].isel(group=slice(1, None)) - post[var].isel(
                    group=0, drop=True
                )
                ratios[f"{name}_ratio_vs_ref"] = np.exp(diff)
            table = pd.concat([table, az.summary(xr.Dataset(ratios))])
 
        return table
 

    def predictive_irt(self):
        """Amostras preditivas na escala original (soma d de volta)."""
        if self.idata is None:
            raise ValueError("Rode .fit() antes.")
        y_rep = self.idata.posterior_predictive["y"]
        return y_rep + self.d
